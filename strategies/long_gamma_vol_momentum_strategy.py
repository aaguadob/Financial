"""
strategies/long_gamma_vol_momentum_strategy.py
-----------------------------------------------
Long-gamma scalping with a volatility MOMENTUM entry signal.

Why HV > IV is a bad long-gamma signal
---------------------------------------
Historical volatility (HV) is backward-looking: it reflects the last N days.
By the time HV > IV (VIX), the volatility spike has already peaked, VIX has
started normalising, and the next 21 days tend to be CALMER — the exact
opposite of what a long-gamma trade needs.

Better signal: fire DURING the acceleration phase
---------------------------------------------------
We enter long gamma when VIX has risen sharply over a short window
(momentum_window days), indicating that volatility is currently expanding —
not after it has already peaked.

Signal conditions (all must be true):
  1. VIX momentum : VIX_today / VIX_{N_days_ago} − 1  ≥  momentum_threshold
     (e.g. VIX up ≥ 15 % over the last 5 trading days)
  2. VIX ceiling  : VIX_today  ≤  vix_ceiling
     (avoid chasing vol that is already extreme; premium too expensive)
  3. VIX floor    : VIX_today  ≥  vix_floor
     (ignore micro-spikes in ultra-low-vol regimes where gamma gains are tiny)
  4. Non-overlap  : at least maturity_days bars since the last signal

Intuition
---------
VIX jumps from 15 → 18 % (+20 %) in 5 days:
  • Vol is expanding RIGHT NOW.
  • Realised vol over the NEXT 21 days is likely to stay ≥ entry IV (~18 %).
  • Long-gamma delta-hedging will harvest Σ(½γdS² − θdt) > 0.
  • This is the opposite of entering after the spike (when vol reverts).

P&L accounting
--------------
Identical to short_gamma_scalp_strategy: constant entry IV, no cash-flow
noise from the hedge. pnl_pct = cum_pnl / entry_premium.
"""

import logging
import numpy as np
import pandas as pd

from strategies.gamma_scalp_strategy import GammaScalping, _Trade, _load_iv

logger = logging.getLogger(__name__)


class Strategy:
    """
    Long-gamma harvest triggered by VIX momentum (vol acceleration).

    Parameters
    ----------
    momentum_window     : bars over which to measure VIX % rise  [default 5]
    momentum_threshold  : minimum VIX % rise to trigger          [default 0.15]
    vix_ceiling         : don't enter above this VIX level       [default 0.45]
    vix_floor           : don't enter below this VIX level       [default 0.12]
    maturity_days       : holding period in trading days         [default 21]
    risk_free           : annualised risk-free rate               [default 0.05]
    option_type         : "call" or "put"                        [default "call"]
    iv_ticker           : Yahoo Finance ticker for IV proxy      [default '^VIX']
    iv_fixed            : constant IV override (float)           [default None]
    spread_pct          : bid/ask as fraction of entry premium   [default 0.01]
    """

    direction = "long"

    def __init__(
        self,
        df: pd.DataFrame,
        momentum_window:    int   = 5,
        momentum_threshold: float = 0.15,
        vix_ceiling:        float = 0.45,
        vix_floor:          float = 0.12,
        maturity_days:      int   = 21,
        risk_free:          float = 0.05,
        option_type:        str   = "call",
        iv_ticker:          str   = "^VIX",
        iv_fixed:           float = None,
        spread_pct:         float = 0.01,
    ):
        self._df        = df
        self._mom_w     = momentum_window
        self._mom_thr   = momentum_threshold
        self._ceil      = vix_ceiling
        self._floor     = vix_floor
        self._mat       = maturity_days
        self._r         = risk_free
        self._opt_type  = option_type
        self._iv        = _load_iv(df.index, iv_ticker, iv_fixed)
        self._spread    = spread_pct

    def _vix_momentum(self, i: int) -> float:
        """Fractional change in VIX over the last momentum_window bars."""
        if i < self._mom_w:
            return np.nan
        iv_now  = self._iv.iloc[i]
        iv_prev = self._iv.iloc[i - self._mom_w]
        if pd.isna(iv_now) or pd.isna(iv_prev) or iv_prev <= 0:
            return np.nan
        return float(iv_now / iv_prev - 1.0)

    def generate_signals(self) -> pd.DataFrame:
        """
        Emit one long-gamma signal per non-overlapping maturity window.

        Fires when:
          • VIX has risen ≥ momentum_threshold over momentum_window days
          • VIX is within [vix_floor, vix_ceiling]
          • At least maturity_days bars since the previous signal
        """
        rows, last_idx = [], -self._mat

        for i, date in enumerate(self._df.index):
            iv  = float(self._iv.iloc[i]) if not pd.isna(self._iv.iloc[i]) else np.nan
            mom = self._vix_momentum(i)

            sig = (
                not np.isnan(iv)
                and not np.isnan(mom)
                and mom  >= self._mom_thr        # VIX is rising fast
                and iv   <= self._ceil           # not already at panic levels
                and iv   >= self._floor          # not a micro-blip in low-vol
                and (i - last_idx) >= self._mat
            )
            if sig:
                last_idx = i

            rows.append({
                "signal":      sig,
                "iv":          iv,
                "vix_mom":     mom if not np.isnan(mom) else 0.0,
                "direction":   "long",
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
                        pos["entry_date"], date, "long", pos["alloc"], pnl_pct,
                    ))
                    print(
                        f"\n{'='*60}\n"
                        f"  [LONG GAMMA] Position closed\n"
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
            sig     = signals.at[date, "signal"]
            iv      = signals.at[date, "iv"]
            vix_mom = signals.at[date, "vix_mom"]

            if sig and len(open_pos) < max_open and not np.isnan(iv):
                try:
                    gs = GammaScalping(
                        stock       = S,
                        strike      = S,
                        risk_free   = self._r,
                        iv          = float(iv),
                        maturity    = self._mat / 252.0,
                        option_type = self._opt_type,
                        direction   = "long",
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
                            f"  [LONG GAMMA] Entry signal fired\n"
                            f"  Date           : {date.date()}\n"
                            f"  Spot (S)       : {S:.4f}\n"
                            f"  IV (VIX)       : {iv:.4f}  ({iv*100:.2f}%)\n"
                            f"  VIX momentum   : {vix_mom:+.4f}  ({vix_mom*100:+.2f}% over {self._mom_w} days)\n"
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
                pos["entry_date"], df.index[-1], "long", pos["alloc"], pnl_pct,
            ))

        return pd.Series(equity_vals, index=df.index, name="equity"), closed
