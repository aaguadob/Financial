"""
Blue-sea breakdown short (short).
Signal: price breaks 20-day low + new 3-month low + not more than 2× its
52-week high + OBV at 3-month low + bearish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sell_strategies import BlueSeaBreakdownStrategy as _Impl


class Strategy:
    direction = "short"

    def __init__(self, df):
        self._impl = _Impl(df)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"sell_signal": "signal"})
