"""
strategies/pre_earnings_straddle_strategy.py
---------------------------------------------
Pre-earnings IV expansion straddle (always long).

Buy an ATM straddle `days_before_entry` trading days before a quarterly earnings
announcement, close it `days_before_exit` trading days before the event.

Rationale: implied volatility tends to expand in the weeks leading up to earnings
("IV run-up"), inflating the straddle's value even without a price move.
Closing before the announcement avoids IV crush on the earnings day.

IV dynamics: the repricing IV is modelled as
    IV(t) = base_IV_t  +  earnings_iv_premium × (days_elapsed / holding_days)
where base_IV_t is the daily VIX (or iv_fixed).  This linear ramp captures the
pre-earnings IV expansion and makes the straddle appreciate as the event nears.

Config requirements
-------------------
    ticker               : must match the backtest ticker (e.g. "AAPL")
    days_before_entry    : trading days before earnings to open     [default 21]
    days_before_exit     : trading days before earnings to close    [default 2]
    earnings_iv_premium  : extra vol accumulated by event date      [default 0.05]
    risk_free            : annualised risk-free rate                [default 0.05]
    iv_ticker            : Yahoo Finance base-IV ticker             [default "^VIX"]
    iv_fixed             : constant base-IV override (float)        [default None]
    stop_loss_pct        : exit if P&L < −stop_loss_pct × premium  [default 1.0]
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
        print(f"   [pre_earnings_straddle] IV download failed ({exc}). "
              f"Pass iv_fixed=<float> to use a constant IV.")
        return pd.Series(np.nan, index=index)


def _earnings_pairs(ticker_sym: str, df_index: pd.DatetimeIndex,
                    n_entry: int, n_exit: int) -> list[tuple[int, int, int]]:
    """
    Return a sorted list of (entry_i, exit_i, earn_i) for every earnings event
    that has a valid entry bar in df_index.

    entry_i : bar to open  (n_entry trading days before earnings)
    exit_i  : bar to close (n_exit  trading days before earnings)
    earn_i  : bar of the earnings announcement
    """
    try:
        ed = yf.Ticker(ticker_sym).get_earnings_dates(limit=100)
        if ed is None or ed.empty:
            print(f"   [pre_earnings_straddle] No earnings dates for '{ticker_sym}'.")
            return []
    except Exception as exc:
        print(f"   [pre_earnings_straddle] Earnings fetch failed: {exc}")
        return []

    norm = df_index.normalize()

    def _first_bar_on_or_after(d: pd.Timestamp) -> int | None:
        hits = np.where(norm >= d)[0]
        return int(hits[0]) if len(hits) else None

    pairs = []
    for dt_tz in ed.index:
        earn_date = pd.Timestamp(dt_tz.date())
        earn_i    = _first_bar_on_or_after(earn_date)
        if earn_i is None:
            continue
        entry_i = earn_i - n_entry
        exit_i  = earn_i - n_exit
        if entry_i < 0 or exit_i <= entry_i:
            continue
        pairs.append((entry_i, exit_i, earn_i))

    return sorted(pairs, key=lambda x: x[0])


# ── position container ────────────────────────────────────────────────────────

@dataclass
class _Pos:
    entry_date:    object
    exit_i:        int      # bar index to close
    holding_days:  int      # exit_i − entry_i
    K:             float
    entry_premium: float
    allocation:    float
    r:             float
    T0:            float
    iv_prem:       float    # earnings_iv_premium param
    day:           int = field(default=0)


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
    Pre-earnings IV expansion straddle.

    Always LONG: buy straddle `days_before_entry` trading days before earnings,
    sell `days_before_exit` trading days before the announcement.

    Parameters
    ----------
    ticker               : ticker symbol matching the backtest ticker
    days_before_entry    : trading days before earnings to enter   [default 21]
    days_before_exit     : trading days before earnings to close   [default 2]
    earnings_iv_premium  : extra vol added linearly as event nears [default 0.05]
    risk_free            : annualised risk-free rate               [default 0.05]
    iv_ticker            : Yahoo Finance ticker for base IV        [default "^VIX"]
    iv_fixed             : constant base-IV override               [default None]
    stop_loss_pct        : exit if P&L < −stop_loss_pct           [default 1.0]
    """

    direction = "long"

    def __init__(
        self,
        df: pd.DataFrame,
        ticker:              str   = "AAPL",
        days_before_entry:   int   = 21,
        days_before_exit:    int   = 2,
        risk_free:           float = 0.05,
        iv_ticker:           str   = "^VIX",
        iv_fixed:            float = None,
        earnings_iv_premium: float = 0.05,
        stop_loss_pct:       float = 1.0,
    ):
        self._df      = df
        self._n_in    = days_before_entry
        self._n_out   = days_before_exit
        self._r       = risk_free
        self._sl      = stop_loss_pct
        self._iv_prem = earnings_iv_premium
        self._iv      = _load_iv(df.index, iv_ticker, iv_fixed)
        self._pairs   = _earnings_pairs(ticker, df.index, days_before_entry, days_before_exit)
        print(f"   [pre_earnings_straddle] {len(self._pairs)} tradeable events found.")

    def generate_signals(self) -> pd.DataFrame:
        sig = pd.Series(False, index=self._df.index)
        for entry_i, _, _ in self._pairs:
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
        Bar-by-bar simulation with IV expansion repricing.

        IV at each bar = base_IV_today + earnings_iv_premium × (day / holding_days).
        Exits on reaching the target close bar or hitting the stop-loss.
        """
        dt   = 1.0 / 252
        cash = float(initial_capital)
        open_pos: list[_Pos]   = []
        closed:   list[_Trade] = []
        equity_vals: list[float] = []

        entry_map: dict[int, tuple[int, int]] = {
            entry_i: (exit_i, exit_i - entry_i)
            for entry_i, exit_i, _ in self._pairs
        }
        print(f"   Signals fired: {len(self._pairs)}")

        for i, (date, row) in enumerate(df.iterrows()):
            S       = float(row["close"])
            base_iv = (float(self._iv.iloc[i])
                       if not pd.isna(self._iv.iloc[i]) else np.nan)
            iv_safe = base_iv if not np.isnan(base_iv) else 0.20

            # ── check exits ───────────────────────────────────────────────
            remaining: list[_Pos] = []
            for pos in open_pos:
                T_rem    = max(pos.T0 - pos.day * dt, 1e-4)
                progress = pos.day / max(pos.holding_days, 1)
                eff_iv   = iv_safe + pos.iv_prem * progress
                val_now  = _straddle(S, pos.K, pos.r, eff_iv, T_rem)
                pnl_ratio = (val_now - pos.entry_premium) / pos.entry_premium

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
            if i in entry_map and len(open_pos) < max_open and not np.isnan(base_iv):
                exit_i, holding_days = entry_map[i]
                T0      = holding_days / 252.0
                premium = _straddle(S, S, self._r, base_iv, T0)
                if premium > 0:
                    alloc = cash * size_pct
                    cost  = alloc * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        open_pos.append(_Pos(
                            entry_date=date,
                            exit_i=exit_i,
                            holding_days=holding_days,
                            K=S, entry_premium=premium,
                            allocation=alloc, r=self._r, T0=T0,
                            iv_prem=self._iv_prem,
                        ))

            # ── mark-to-market equity ─────────────────────────────────────
            mtm = 0.0
            for pos in open_pos:
                T_rem    = max(pos.T0 - pos.day * dt, 1e-4)
                progress = pos.day / max(pos.holding_days, 1)
                eff_iv   = iv_safe + pos.iv_prem * progress
                val      = _straddle(S, pos.K, pos.r, eff_iv, T_rem)
                mtm     += max(0.0, pos.allocation * val / pos.entry_premium)
            equity_vals.append(cash + mtm)

        return pd.Series(equity_vals, index=df.index, name="equity"), closed
