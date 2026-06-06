"""
Bullish divergence (long).
Signal: long-term uptrend + price makes lower low while 2+ indicators make
higher low + bullish reversal candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from buy_strategies import BullishDivergenceStrategy as _Impl


class Strategy:
    direction = "long"

    def __init__(self, df, ma_short=50, ma_long=200):
        self._impl = _Impl(df, ma_short=ma_short, ma_long=ma_long)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"buy_signal": "signal"})
