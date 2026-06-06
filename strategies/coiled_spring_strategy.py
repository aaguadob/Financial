"""
Coiled spring breakout (long).
Signal: uptrend + new 3-month high + narrowing range that stays above 50 MA.
Risk management added here since the original strategy has no stop/target.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from buy_strategies import CoiledSpringStrategy as _Impl


class Strategy:
    direction = "long"

    def __init__(self, df, ma_short=20, ma_long=50,
                 stop_distance=0.07, reward_ratio=1.75):
        self._df   = df
        self._impl = _Impl(df, ma_short=ma_short, ma_long=ma_long)
        self._stop = stop_distance
        self._rr   = reward_ratio

    def generate_signals(self):
        s     = self._impl.generate_signals()
        close = self._df["close"]
        mask  = s["buy_signal"].to_numpy()
        sl    = np.where(mask, close * (1 - self._stop), np.nan)
        tp    = np.where(mask, close * (1 + self._stop * self._rr), np.nan)
        return pd.DataFrame(
            {"signal": s["buy_signal"], "stop_loss": sl, "take_profit": tp},
            index=s.index,
        )
