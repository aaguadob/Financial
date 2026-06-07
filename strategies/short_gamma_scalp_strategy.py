"""
strategies/short_gamma_scalp_strategy.py
-----------------------------------------
Pure short-gamma scalping: sell ATM options and delta-hedge daily whenever
the Volatility Risk Premium (VRP = IV − HV) exceeds the threshold.

Edge: SPY options historically trade at an implied vol premium of 3-5 pp
above realised vol (the "variance risk premium").  Selling that premium and
delta-hedging neutralises directional risk, leaving a theta/VRP harvest that
is positive as long as realised vol stays below the entry IV.

Signal
------
  VRP = IV − HV  >  vrp_threshold  →  SHORT gamma

Position lifecycle
------------------
  Entry : sell ATM option, immediately delta-hedge by buying the shares.
  Daily : rebalance the hedge to the new BS delta (using *constant* entry IV).
  Exit  : option expires after `maturity_days` trading days; close the hedge.

P&L accounting (no financing flows)
-------------------------------------
  cum_pnl starts at −(entry commission + entry spread).
  Each day: opt_pnl + hedge_pnl − rebal_comm.
  At expiry: deduct the exit spread.
  pnl_pct = cum_pnl / entry_premium  (return on premium received).

Using constant entry IV throughout isolates the pure gamma-scalp result:
  Σ(−½γdS² + θdt) − costs
which is positive exactly when realised vol < entry IV.
"""

import logging
import numpy as np
import pandas as pd

from strategies.gamma_scalp_strategy import GammaScalping, _Trade, _load_iv

logger = logging.getLogger(__name__)


class Strategy:
    """
    Pure short-gamma harvest: sell vol when IV > HV.

    Parameters
    ----------
    hv_window     : lookback (trading days) for rolling HV       [default 20]
    vrp_threshold : minimum IV − HV spread to enter              [default 0.03]
    maturity_days : holding period in trading days               [default 21]
    risk_free     : annualised risk-free rate                     [default 0.05]
    option_type   : "call" or "put"                              [default "call"]
    iv_ticker     : Yahoo Finance ticker for IV proxy            [default '^VIX']
    iv_fixed      : constant IV override (float)                 [default None]
    spread_pct    : bid/ask as fraction of entry premium         [default 0.01]
    """

    direction = "short"

    def __init__(
        self,
        df: pd.DataFrame,
        hv_window:     int   = 20,
        vrp_threshold: float = 0.03,
        maturity_days: int   = 21,
        risk_free:     float = 0.05,
        option_type:   str   = "call",
        iv_ticker:     str   = "^VIX",
        iv_fixed:      float = None,
        spread_pct:    float = 0.01,
    ):
        self._df       = df
        self._hv_w     = hv_window
        self._thresh   = vrp_threshold
        self._mat      = maturity_days
        self._r        = risk_free
        self._opt_type = option_type
        self._iv       = _load_iv(df.index, iv_ticker, iv_fixed)
        self._spread   = spread_pct

    def _hv(self, idx: int) -> float:
        if idx < self._hv_w + 1:
            return np.nan
        closes = self._df["close"].iloc[idx - self._hv_w - 1 : idx + 1].values
        log_r  = np.log(closes[1:] / closes[:-1])
        return float(log_r[-self._hv_w :].std(ddof=1) * np.sqrt(252))

    def generate_signals(self) -> pd.DataFrame:
        """
        Emit one short-gamma signal per non-overlapping maturity window.
        Only fires when IV > HV by at least vrp_threshold.
        """
        rows, last_idx = [], -self._mat

        for i, date in enumerate(self._df.index):
            hv  = self._hv(i)
            iv  = float(self._iv.iloc[i]) if not pd.isna(self._iv.iloc[i]) else np.nan
            vrp = iv - hv if not (np.isnan(iv) or np.isnan(hv)) else np.nan

            sig = (
                not np.isnan(vrp)
                and vrp >= self._thresh          # IV must exceed HV (sell rich vol)
                and (i - last_idx) >= self._mat
            )
            if sig:
                last_idx = i

            rows.append({
                "signal":      sig,
                "iv":          iv,
                "hv":          hv,
                "vrp":         vrp,
                "direction":   "short",
                "stop_loss":   np.nan,
                "take_profit": np.nan,
            })

        return pd.DataFrame(rows, index=self._df.index)

    def simulate(
        self,
        df:              pd.DataFrame,
        initial_capital: float,
        size_pct:        float,
        max_open:        int,
        commission:      float,
    ) -> tuple[pd.Series, list]:
        signals = self.generate_signals()
        n_sigs  = int(signals["signal"].sum())
        print(f"   Signals fired: {n_sigs}")

        cash     = float(initial_capital)
        open_pos: list[dict] = []
        closed:   list[_Trade] = []
        equity_vals: list[float] = []

        for i, (date, row) in enumerate(df.iterrows()):
            S        = float(row["close"])
            iv_today = float(self._iv.iloc[i]) if not pd.isna(self._iv.iloc[i]) else None

            # 1. Advance open positions
            for pos in open_pos:
                pos["gs"].step(S, iv=iv_today)

            # 2. Close expired positions
            remaining = []
            for pos in open_pos:
                gs = pos["gs"]
                if gs.expired:
                    pnl_pct = max(gs.cum_pnl / gs.net_capital, -1.0)
                    cash   += pos["alloc"] * (1.0 + pnl_pct)
                    closed.append(_Trade(
                        pos["entry_date"], date, "short", pos["alloc"], pnl_pct,
                    ))
                    print(
                        f"\n{'='*60}\n"
                        f"  [SHORT GAMMA] Position closed\n"
                        f"  Exit date      : {date.date()}\n"
                        f"  Entry premium  : {gs.entry_premium:.4f}\n"
                        f"  cum_pnl        : {gs.cum_pnl:.4f}\n"
                        f"  pnl_pct        : {pnl_pct:+.4f}  ({pnl_pct*100:+.2f}%)\n"
                        f"  Total comm     : {gs.total_comm:.4f}\n"
                        f"  Total spread   : {gs.total_spread:.4f}\n"
                        f"{'='*60}\n"
                    )
                else:
                    remaining.append(pos)
            open_pos = remaining

            # 3. Open new position on signal
            sig = signals.at[date, "signal"]
            iv  = signals.at[date, "iv"]
            hv  = signals.at[date, "hv"]
            vrp = signals.at[date, "vrp"]

            if sig and len(open_pos) < max_open and not np.isnan(iv):
                try:
                    gs = GammaScalping(
                        stock       = S,
                        strike      = S,
                        risk_free   = self._r,
                        iv          = float(iv),
                        maturity    = self._mat / 252.0,
                        option_type = self._opt_type,
                        direction   = "short",
                        commission  = commission,
                        spread_pct  = self._spread,
                    )
                except Exception as exc:
                    logger.warning("GammaScalping init failed at %s: %s", date, exc)
                    gs = None

                if gs is not None:
                    alloc = cash * size_pct
                    if alloc > 0:
                        cash -= alloc * (1.0 + commission)
                        open_pos.append({
                            "gs":         gs,
                            "entry_date": date,
                            "alloc":      alloc,
                        })
                        print(
                            f"\n{'='*60}\n"
                            f"  [SHORT GAMMA] Entry signal fired\n"
                            f"  Date           : {date.date()}\n"
                            f"  Spot (S)       : {S:.4f}\n"
                            f"  IV             : {iv:.4f}  ({iv*100:.2f}%)\n"
                            f"  HV             : {hv:.4f}  ({hv*100:.2f}%)\n"
                            f"  VRP            : {vrp:+.4f}  (IV > HV by {vrp*100:.2f}pp)\n"
                            f"  Entry premium  : {gs.entry_premium:.4f}\n"
                            f"  Allocation     : {alloc:,.2f}\n"
                            f"{'='*60}"
                        )

            # 4. Mark-to-market equity
            mtm = sum(
                pos["alloc"] * (1.0 + max(pos["gs"].cum_pnl / pos["gs"].net_capital, -1.0))
                for pos in open_pos
            )
            equity_vals.append(cash + mtm)

        # Force-close anything open at the last bar
        for pos in open_pos:
            gs      = pos["gs"]
            pnl_pct = max(gs.cum_pnl / gs.net_capital, -1.0)
            cash   += pos["alloc"] * (1.0 + pnl_pct)
            closed.append(_Trade(
                pos["entry_date"], df.index[-1], "short", pos["alloc"], pnl_pct,
            ))

        return pd.Series(equity_vals, index=df.index, name="equity"), closed
