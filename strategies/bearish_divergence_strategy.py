"""
Bearish divergence short (short).
Signal: long-term downtrend + price above 50 MA + price makes higher high
while 2+ indicators make lower high + bearish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sell_strategies import BearishDivergenceStrategy as _Impl


class Strategy:
    direction = "short"

    def __init__(self, df, ma_short=50, ma_long=200):
        self._impl = _Impl(df, ma_short=ma_short, ma_long=ma_long)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"sell_signal": "signal"})
