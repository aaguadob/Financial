"""
Relief rally short (short).
Signal: downtrend + price rallies into 20 or 50 MA + stochastic overbought
+ bearish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sell_strategies import ReliefRallyShortStrategy as _Impl


class Strategy:
    direction = "short"

    def __init__(self, df, ma_short=20, ma_long=50, trend_period=63):
        self._impl = _Impl(df, ma_short=ma_short, ma_long=ma_long,
                           trend_period=trend_period)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"sell_signal": "signal"})
