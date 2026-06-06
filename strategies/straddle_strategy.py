"""
strategies/straddle_strategy.py
--------------------------------
ATM straddle strategy driven by the Volatility Risk Premium (VRP = IV − HV).

  VRP > +threshold  →  SHORT straddle (IV expensive: sell vol, collect theta)
  VRP < −threshold  →  LONG  straddle (IV cheap:     buy  vol, profit on move)

Positions are held for `maturity_days` trading days and marked-to-model
daily via Black-Scholes so the equity curve is continuous.  Early exit
triggers when stop-loss or take-profit thresholds (expressed as multiples
of the initial premium) are breached intraday.

IV source: configurable ticker from Yahoo Finance (default '^VIX' for SPY).
           A fixed float can be passed as `iv_fixed` to bypass the download.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from iv_calculator import bs_call, bs_put


# ── Black-Scholes ATM straddle price ──────────────────────────────────────────

def _straddle(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return bs_call(S, K, r, sigma, T) + bs_put(S, K, r, sigma, T)


# ── internal position container ───────────────────────────────────────────────

@dataclass
class _Pos:
    entry_date: object
    exit_date:  object
    K:          float    # ATM strike = entry close
    iv:         float    # implied vol at entry (constant for repricing)
    premium:    float    # straddle price paid/received at entry
    direction:  str      # "long" or "short"
    allocation: float    # capital committed
    r:          float    # risk-free rate
    T0:         float    # initial time to expiry in years
    day:        int = 0  # trading days elapsed since entry


# ── minimal trade-result object (matches what compute_metrics expects) ─────────

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
        # Normalise both index and df.index to date-only for alignment
        iv.index = pd.to_datetime([d.date() for d in iv.index])
        df_dates = pd.to_datetime([d.date() for d in index])
        aligned  = iv.reindex(df_dates, method="ffill")
        aligned.index = index
        return aligned
    except Exception as exc:
        print(f"   [straddle] IV download failed ({exc}). "
              f"Pass iv_fixed=<float> to use a constant IV.")
        return pd.Series(np.nan, index=index)


# ── strategy ──────────────────────────────────────────────────────────────────

class Strategy:
    """
    ATM straddle  ·  sell when IV >> HV, buy when IV << HV.

    Parameters
    ----------
    hv_window      : lookback (trading days) for rolling HV   [default 20]
    vrp_threshold  : minimum |IV − HV| to enter a trade       [default 0.03]
    maturity_days  : holding period in trading days            [default 21]
    risk_free      : annualised risk-free rate                 [default 0.05]
    iv_ticker      : Yahoo Finance ticker for IV proxy         [default '^VIX']
    iv_fixed       : override — use this constant IV (float)   [default None]
    stop_loss_pct  : exit when P&L < −stop_loss_pct × premium [default 1.0]
    take_profit_pct: exit when P&L >  take_profit_pct × premium[default 0.75]
    """

    direction = "long"   # placeholder; actual direction is per-trade

    def __init__(
        self,
        df: pd.DataFrame,
        hv_window:       int   = 20,
        vrp_threshold:   float = 0.03,
        maturity_days:   int   = 21,
        risk_free:       float = 0.05,
        iv_ticker:       str   = "^VIX",
        iv_fixed:        float = None,
        stop_loss_pct:   float = 1.0,
        take_profit_pct: float = 0.75,
    ):
        self._df        = df
        self._hv_w      = hv_window
        self._thresh    = vrp_threshold
        self._mat       = maturity_days
        self._r         = risk_free
        self._sl        = stop_loss_pct
        self._tp        = take_profit_pct
        self._iv        = _load_iv(df.index, iv_ticker, iv_fixed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _hv(self, idx: int) -> float:
        """Rolling close-to-close HV (annualised) ending at bar `idx`."""
        if idx < self._hv_w + 1:
            return np.nan
        closes  = self._df["close"].iloc[idx - self._hv_w - 1: idx + 1].values
        log_r   = np.log(closes[1:] / closes[:-1])
        return float(log_r[-self._hv_w:].std(ddof=1) * np.sqrt(252))

    def _pnl_ratio(self, pos: _Pos, current_val: float) -> float:
        """Signed P&L as a fraction of the entry premium."""
        if pos.direction == "short":
            return (pos.premium - current_val) / pos.premium
        return (current_val - pos.premium) / pos.premium

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
        Custom simulation: Black-Scholes mark-to-model daily.

        Exit triggers:
          • Expiry  (day >= maturity_days)
          • Stop    (P&L < −stop_loss_pct  × premium)
          • Target  (P&L >  take_profit_pct × premium)
        """
        signals = self.generate_signals()
        n_sigs  = int(signals["signal"].sum())
        print(f"   Signals fired: {n_sigs}")

        dt        = 1.0 / 252         # one trading day in years
        cash      = float(initial_capital)
        open_pos: list[_Pos]   = []
        closed:   list[_Trade] = []
        equity_vals: list[float] = []

        for i, (date, row) in enumerate(df.iterrows()):
            S = float(row["close"])

            # ── check exits ───────────────────────────────────────────────
            remaining: list[_Pos] = []
            for pos in open_pos:
                T_rem     = max(pos.T0 - pos.day * dt, 1e-4)
                val_now   = _straddle(S, pos.K, pos.r, pos.iv, T_rem)
                pnl_ratio = self._pnl_ratio(pos, val_now)

                expired   = date >= pos.exit_date
                stopped   = pnl_ratio <= -self._sl
                targeted  = pnl_ratio >= self._tp

                if expired or stopped or targeted:
                    if expired:
                        # Use intrinsic value at expiry
                        intrinsic  = abs(S - pos.K)
                        final_ratio = self._pnl_ratio(pos, intrinsic)
                    else:
                        final_ratio = pnl_ratio

                    # Cap loss at full allocation (margin-call equivalent)
                    net_pnl = max(final_ratio - 2 * commission, -(1.0 - commission))
                    cash   += pos.allocation * (1 + net_pnl + commission)
                    closed.append(_Trade(pos.entry_date, date,
                                        pos.direction, pos.allocation, net_pnl))
                else:
                    pos.day += 1
                    remaining.append(pos)
            open_pos = remaining

            # ── check entries ─────────────────────────────────────────────
            sig = signals.at[date, "signal"]
            iv  = signals.at[date, "iv"]
            hv  = signals.at[date, "hv"]
            vrp = signals.at[date, "vrp"]

            if (sig and len(open_pos) < max_open
                    and not np.isnan(iv) and not np.isnan(hv)):
                T0      = self._mat / 252.0
                premium = _straddle(S, S, self._r, iv, T0)

                if premium > 0:
                    direction = "short" if vrp > 0 else "long"
                    alloc     = cash * size_pct
                    cost      = alloc * (1 + commission)
                    if cost <= cash:
                        end_idx  = min(i + self._mat, len(df) - 1)
                        cash    -= cost
                        open_pos.append(_Pos(
                            entry_date=date,
                            exit_date=df.index[end_idx],
                            K=S, iv=iv, premium=premium,
                            direction=direction, allocation=alloc,
                            r=self._r, T0=T0,
                        ))

            # ── mark-to-market equity ─────────────────────────────────────
            mtm = 0.0
            for pos in open_pos:
                T_rem = max(pos.T0 - pos.day * dt, 1e-4)
                val   = _straddle(S, pos.K, pos.r, pos.iv, T_rem)
                if pos.direction == "short":
                    # Short: MTM rises when straddle decays
                    mtm += max(0.0, pos.allocation * (1 + (pos.premium - val) / pos.premium))
                else:
                    mtm += max(0.0, pos.allocation * val / pos.premium)
            equity_vals.append(cash + mtm)

        return pd.Series(equity_vals, index=df.index, name="equity"), closed
