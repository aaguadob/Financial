"""
strategies/post_earnings_straddle_strategy.py
----------------------------------------------
Post-earnings realized-move straddle (always long).

Buy an ATM straddle `days_before_entry` trading days before earnings, hold it
for `maturity_days` trading days (through and past the announcement).

Rationale: a large post-earnings directional move can more than offset IV crush,
particularly for stocks with a history of high earnings surprise.  Entering just
before the event gives exposure to the move in either direction.

IV repricing: uses daily VIX (or iv_fixed) so post-earnings IV crush is reflected
naturally in the mark-to-model P&L.  If the underlying makes a large move the
intrinsic value of the straddle compensates for the vol compression.

Config requirements
-------------------
    ticker            : must match the backtest ticker (e.g. "AAPL")
    days_before_entry : trading days before earnings to open   [default 2]
    maturity_days     : total holding period in trading days   [default 21]
    risk_free         : annualised risk-free rate              [default 0.05]
    iv_ticker         : Yahoo Finance ticker for IV proxy      [default "^VIX"]
    iv_fixed          : constant IV override (float)           [default None]
    stop_loss_pct     : exit if P&L < −stop_loss_pct          [default 1.0]
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from iv_calculator import bs_call, bs_put


# ── pricing helpers ───────────────────────────────────────────────────────────

def _straddle(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return bs_call(S, K, r, sigma, T) + bs_put(S, K, r, sigma, T)


def _load_iv(index: pd.DatetimeIndex, iv_ticker: str,
             iv_fixed: float | None) -> pd.Series:
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
        iv = iv / 100.0
        iv.index = pd.to_datetime([d.date() for d in iv.index])
        df_dates = pd.to_datetime([d.date() for d in index])
        aligned  = iv.reindex(df_dates, method="ffill")
        aligned.index = index
        return aligned
    except Exception as exc:
        print(f"   [post_earnings_straddle] IV download failed ({exc}). "
              f"Pass iv_fixed=<float> to use a constant IV.")
        return pd.Series(np.nan, index=index)


def _earnings_entry_indices(ticker_sym: str, df_index: pd.DatetimeIndex,
                             n_before: int) -> list[tuple[int, int]]:
    """
    Return a sorted list of (entry_i, earn_i) for every earnings event.

    entry_i : bar to open  (n_before trading days before earnings)
    earn_i  : bar of the earnings announcement
    """
    try:
        ed = yf.Ticker(ticker_sym).get_earnings_dates(limit=100)
        if ed is None or ed.empty:
            print(f"   [post_earnings_straddle] No earnings dates for '{ticker_sym}'.")
            return []
    except Exception as exc:
        print(f"   [post_earnings_straddle] Earnings fetch failed: {exc}")
        return []

    norm = df_index.normalize()

    def _first_bar_on_or_after(d: pd.Timestamp) -> int | None:
        hits = np.where(norm >= d)[0]
        return int(hits[0]) if len(hits) else None

    events = []
    for dt_tz in ed.index:
        earn_date = pd.Timestamp(dt_tz.date())
        earn_i    = _first_bar_on_or_after(earn_date)
        if earn_i is None:
            continue
        entry_i = earn_i - n_before
        if entry_i < 0:
            continue
        events.append((entry_i, earn_i))

    return sorted(set(events), key=lambda x: x[0])


# ── position container ────────────────────────────────────────────────────────

@dataclass
class _Pos:
    entry_date: object
    exit_i:     int      # bar index to close (entry_i + maturity_days)
    K:          float
    r:          float
    T0:         float
    premium:    float
    allocation: float
    day:        int = field(default=0)


class _Trade:
    __slots__ = ("entry_date", "exit_date", "direction", "allocation", "pnl_pct")

    def __init__(self, entry_date, exit_date, direction, allocation, pnl_pct):
        self.entry_date = entry_date
        self.exit_date  = exit_date
        self.direction  = direction
        self.allocation = allocation
        self.pnl_pct    = pnl_pct


# ── strategy ──────────────────────────────────────────────────────────────────

class Strategy:
    """
    Post-earnings realized-move straddle.

    Always LONG: buy straddle `days_before_entry` trading days before earnings,
    hold for `maturity_days` trading days regardless of the announcement date.

    Parameters
    ----------
    ticker            : ticker symbol matching the backtest ticker
    days_before_entry : trading days before earnings to open   [default 2]
    maturity_days     : total holding period in trading days   [default 21]
    risk_free         : annualised risk-free rate              [default 0.05]
    iv_ticker         : Yahoo Finance ticker for IV proxy      [default "^VIX"]
    iv_fixed          : constant IV override                   [default None]
    stop_loss_pct     : exit if P&L < −stop_loss_pct          [default 1.0]
    """

    direction = "long"

    def __init__(
        self,
        df: pd.DataFrame,
        ticker:            str   = "AAPL",
        days_before_entry: int   = 2,
        maturity_days:     int   = 21,
        risk_free:         float = 0.05,
        iv_ticker:         str   = "^VIX",
        iv_fixed:          float = None,
        stop_loss_pct:     float = 1.0,
    ):
        self._df     = df
        self._mat    = maturity_days
        self._r      = risk_free
        self._sl     = stop_loss_pct
        self._iv     = _load_iv(df.index, iv_ticker, iv_fixed)
        self._events = _earnings_entry_indices(ticker, df.index, days_before_entry)
        print(f"   [post_earnings_straddle] {len(self._events)} tradeable events found.")

    def generate_signals(self) -> pd.DataFrame:
        sig = pd.Series(False, index=self._df.index)
        for entry_i, _ in self._events:
            if 0 <= entry_i < len(sig):
                sig.iloc[entry_i] = True
        return pd.DataFrame({
            "signal":      sig,
            "stop_loss":   np.nan,
            "take_profit": np.nan,
        })

    def simulate(
        self,
        df:              pd.DataFrame,
        initial_capital: float,
        size_pct:        float,
        max_open:        int,
        commission:      float,
    ) -> tuple[pd.Series, list]:
        """
        Bar-by-bar simulation with daily IV repricing.

        The straddle is repriced each day using the current VIX as the IV
        proxy.  Post-earnings IV crush is reflected naturally through VIX
        compression; the intrinsic value from the price move compensates.
        Exit on reaching maturity_days or breaching the stop-loss.
        """
        dt   = 1.0 / 252
        cash = float(initial_capital)
        open_pos: list[_Pos]   = []
        closed:   list[_Trade] = []
        equity_vals: list[float] = []

        entry_set: set[int] = {entry_i for entry_i, _ in self._events}
        print(f"   Signals fired: {len(self._events)}")

        for i, (date, row) in enumerate(df.iterrows()):
            S    = float(row["close"])
            iv_t = (float(self._iv.iloc[i])
                    if not pd.isna(self._iv.iloc[i]) else np.nan)
            iv_safe = iv_t if not np.isnan(iv_t) else 0.20

            # ── check exits ───────────────────────────────────────────────
            remaining: list[_Pos] = []
            for pos in open_pos:
                T_rem     = max(pos.T0 - pos.day * dt, 1e-4)
                val_now   = _straddle(S, pos.K, pos.r, iv_safe, T_rem)
                pnl_ratio = (val_now - pos.premium) / pos.premium

                if i >= pos.exit_i or pnl_ratio <= -self._sl:
                    net_pnl = max(pnl_ratio - 2 * commission, -(1.0 - commission))
                    cash   += pos.allocation * (1 + net_pnl + commission)
                    closed.append(_Trade(pos.entry_date, date, "long",
                                         pos.allocation, net_pnl))
                else:
                    pos.day += 1
                    remaining.append(pos)
            open_pos = remaining

            # ── check entries ─────────────────────────────────────────────
            if i in entry_set and len(open_pos) < max_open and not np.isnan(iv_t):
                T0      = self._mat / 252.0
                premium = _straddle(S, S, self._r, iv_t, T0)
                if premium > 0:
                    alloc = cash * size_pct
                    cost  = alloc * (1 + commission)
                    if cost <= cash:
                        exit_i = min(i + self._mat, len(df) - 1)
                        cash  -= cost
                        open_pos.append(_Pos(
                            entry_date=date,
                            exit_i=exit_i,
                            K=S, r=self._r, T0=T0,
                            premium=premium, allocation=alloc,
                        ))

            # ── mark-to-market equity ─────────────────────────────────────
            mtm = 0.0
            for pos in open_pos:
                T_rem = max(pos.T0 - pos.day * dt, 1e-4)
                val   = _straddle(S, pos.K, pos.r, iv_safe, T_rem)
                mtm  += max(0.0, pos.allocation * val / pos.premium)
            equity_vals.append(cash + mtm)

        return pd.Series(equity_vals, index=df.index, name="equity"), closed
