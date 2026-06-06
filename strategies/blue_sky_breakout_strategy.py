"""
Blue-sky breakout (long).
Signal: price breaks 20-day high + new 3-month high + not overextended
from 52-week low + OBV at 3-month high + bullish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from buy_strategies import BlueSkyBreakoutStrategy as _Impl


class Strategy:
    direction = "long"

    def __init__(self, df):
        self._impl = _Impl(df)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"buy_signal": "signal"})
