"""
Rising-wedge breakdown short (short).
Signal: uptrend + rising wedge (higher highs and lows but tightening range)
+ MACD lower highs + OBV below 20-day low + bearish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sell_strategies import RisingWedgeBreakdownStrategy as _Impl


class Strategy:
    direction = "short"

    def __init__(self, df, ma_short=20, ma_long=50):
        self._impl = _Impl(df, ma_short=ma_short, ma_long=ma_long)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"sell_signal": "signal"})
