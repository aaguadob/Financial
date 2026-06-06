"""
Pullback to moving average (long).
Signal: uptrend + price near MA + stochastic oversold + bullish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from buy_strategies import PullbackStrategy as _Impl


class Strategy:
    direction = "long"

    def __init__(self, df, ma_short=20, ma_long=50, trend_lookback=63):
        self._impl = _Impl(df, ma_short=ma_short, ma_long=ma_long,
                           trend_lookback=trend_lookback)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"buy_signal": "signal"})
