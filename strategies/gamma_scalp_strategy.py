"""
strategies/gamma_scalp_strategy.py
------------------------------------
Delta-neutral gamma scalping strategy driven by the Volatility Risk Premium
(VRP = IV − HV).

  VRP > +threshold  →  SHORT gamma  (sell ATM option, delta-hedge daily)
                        Earns theta + VRP if realised vol stays below IV.
  VRP < −threshold  →  LONG  gamma  (buy  ATM option, delta-hedge daily)
                        Profits when realised vol exceeds IV.

Each position runs for `maturity_days` trading days.  At entry the full
delta-hedge simulation is pre-run on the actual forward price path using
the GammaScalping class from options_strategies.py.  The resulting daily
cumulative P&L series drives the mark-to-market equity curve.

IV source: configurable ticker from Yahoo Finance (default '^VIX' for SPY).
           Pass `iv_fixed=<float>` to bypass the download.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from options_strategies import GammaScalping


# ── minimal trade-result object ───────────────────────────────────────────────

class _Trade:
    __slots__ = ("entry_date", "exit_date", "direction", "allocation", "pnl_pct")

    def __init__(self, entry_date, exit_date, direction, allocation, pnl_pct):
        self.entry_date = entry_date
        self.exit_date  = exit_date
        self.direction  = direction
        self.allocation = allocation
        self.pnl_pct    = pnl_pct


# ── IV data helper ────────────────────────────────────────────────────────────

def _load_iv(index: pd.DatetimeIndex, iv_ticker: str,
             iv_fixed: float | None) -> pd.Series:
    """Return a Series of daily IV values aligned to `index`."""
    if iv_fixed is not None:
        return pd.Series(float(iv_fixed), index=index)
    try:
        start = str(pd.Timestamp(index[0]).date())
        end   = str((pd.Timestamp(index[-1]) + pd.Timedelta(days=10)).date())
        raw   = yf.download(iv_ticker, start=start, end=end,
                            auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        iv = raw["Close"].squeeze()
        if isinstance(iv, pd.DataFrame):
            iv = iv.iloc[:, 0]
        iv = iv / 100.0               # VIX is in percentage-points → decimal
        iv.index = pd.to_datetime([d.date() for d in iv.index])
        df_dates = pd.to_datetime([d.date() for d in index])
        aligned  = iv.reindex(df_dates, method="ffill")
        aligned.index = index
        return aligned
    except Exception as exc:
        print(f"   [gamma_scalp] IV download failed ({exc}). "
              f"Pass iv_fixed=<float> to use a constant IV.")
        return pd.Series(np.nan, index=index)


# ── strategy ──────────────────────────────────────────────────────────────────

class Strategy:
    """
    Delta-neutral gamma scalping  ·  long gamma when IV < HV, short when IV > HV.

    Parameters
    ----------
    hv_window     : lookback (trading days) for rolling HV       [default 20]
    vrp_threshold : minimum |IV − HV| to enter a trade           [default 0.03]
    maturity_days : holding period in trading days (~1 month)     [default 21]
    risk_free     : annualised risk-free rate                     [default 0.05]
    option_type   : "call" or "put" for the hedged leg            [default "call"]
    iv_ticker     : Yahoo Finance ticker for IV proxy             [default '^VIX']
    iv_fixed      : override — use this constant IV (float)       [default None]
    """

    direction = "long"   # placeholder; actual direction is per-trade

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
    ):
        self._df        = df
        self._hv_w      = hv_window
        self._thresh    = vrp_threshold
        self._mat       = maturity_days
        self._r         = risk_free
        self._opt_type  = option_type
        self._iv        = _load_iv(df.index, iv_ticker, iv_fixed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _hv(self, idx: int) -> float:
        """Rolling close-to-close HV (annualised) ending at bar `idx`."""
        if idx < self._hv_w + 1:
            return np.nan
        closes = self._df["close"].iloc[idx - self._hv_w - 1: idx + 1].values
        log_r  = np.log(closes[1:] / closes[:-1])
        return float(log_r[-self._hv_w:].std(ddof=1) * np.sqrt(252))

    # ── public interface ──────────────────────────────────────────────────────

    def generate_signals(self) -> pd.DataFrame:
        """
        Emit one signal per non-overlapping position window.

        Columns: signal (bool), iv, hv, vrp, stop_loss, take_profit.
        stop_loss / take_profit are NaN — exits are handled in simulate().
        """
        rows, last_idx = [], -self._mat

        for i, date in enumerate(self._df.index):
            hv  = self._hv(i)
            iv  = float(self._iv.iloc[i]) if not pd.isna(self._iv.iloc[i]) else np.nan
            vrp = iv - hv if not (np.isnan(iv) or np.isnan(hv)) else np.nan

            sig = (
                not np.isnan(vrp)
                and abs(vrp) > self._thresh
                and (i - last_idx) >= self._mat
            )
            if sig:
                last_idx = i

            rows.append({
                "signal":      sig,
                "iv":          iv,
                "hv":          hv,
                "vrp":         vrp,
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
        """
        Custom simulation.

        At each signal bar, GammaScalping is pre-run on the actual forward
        price path (next maturity_days closes).  The resulting daily
        cumulative P&L series is used for mark-to-model until expiry.

        P&L normalisation:
            pnl_pct = final_pnl / entry_premium  (return on premium committed)
        MTM on day d of the trade:
            mtm = allocation × (1 + cum_pnl[d] / entry_premium)
        """
        signals = self.generate_signals()
        n_sigs  = int(signals["signal"].sum())
        print(f"   Signals fired: {n_sigs}")

        cash      = float(initial_capital)
        open_pos  = []                  # list of dicts (see below)
        closed: list[_Trade] = []
        equity_vals: list[float] = []

        for i, (date, row) in enumerate(df.iterrows()):
            S = float(row["close"])

            # ── check exits ───────────────────────────────────────────────
            remaining = []
            for pos in open_pos:
                day_in = i - pos["entry_i"]
                # Expired when we've consumed the full cum_pnl series
                if day_in >= len(pos["cum_pnl"]) - 1:
                    final_pnl  = float(pos["cum_pnl"][-1])
                    pnl_pct    = final_pnl / pos["entry_premium"] - 2 * commission
                    # Cap loss at full allocation
                    pnl_pct    = max(pnl_pct, -(1.0 - commission))
                    cash      += pos["alloc"] * (1 + pnl_pct + commission)
                    closed.append(_Trade(
                        pos["entry_date"], date,
                        pos["direction"], pos["alloc"], pnl_pct,
                    ))
                else:
                    remaining.append(pos)
            open_pos = remaining

            # ── check entries ─────────────────────────────────────────────
            sig = signals.at[date, "signal"]
            iv  = signals.at[date, "iv"]
            hv  = signals.at[date, "hv"]
            vrp = signals.at[date, "vrp"]

            if (sig and len(open_pos) < max_open
                    and not np.isnan(iv) and not np.isnan(hv)):
                end_idx = min(i + self._mat, len(df) - 1)
                closes  = df["close"].iloc[i: end_idx + 1].values
                direction = "short" if vrp > 0 else "long"

                try:
                    gs = GammaScalping(
                        stock         = S,
                        strike        = S,           # ATM
                        risk_free     = self._r,
                        iv            = float(iv),
                        hv            = float(hv),
                        maturity      = self._mat / 252.0,
                        option_type   = self._opt_type,
                        closes        = closes,
                        vrp_threshold = self._thresh,
                        forced_mode   = direction,
                        commission    = commission,  # charged on every hedge rebalancing
                    )
                except Exception:
                    gs = None

                if gs is not None and gs.results is not None and gs.entry_premium > 0:
                    alloc = cash * size_pct
                    cost  = alloc * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        open_pos.append({
                            "entry_i":       i,
                            "entry_date":    date,
                            "direction":     direction,
                            "cum_pnl":       gs.results["cum_pnl"],  # shape (n_days+1,)
                            "entry_premium": gs.entry_premium,
                            "alloc":         alloc,
                        })

            # ── mark-to-market equity ─────────────────────────────────────
            mtm = 0.0
            for pos in open_pos:
                day_in   = min(i - pos["entry_i"], len(pos["cum_pnl"]) - 1)
                cum_pnl  = float(pos["cum_pnl"][day_in])
                # Return on premium, scaled to allocation
                pnl_frac = cum_pnl / pos["entry_premium"]
                mtm     += max(0.0, pos["alloc"] * (1 + pnl_frac))
            equity_vals.append(cash + mtm)

        return pd.Series(equity_vals, index=df.index, name="equity"), closed
