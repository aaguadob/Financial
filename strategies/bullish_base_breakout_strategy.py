"""
Bullish base breakout (long).
Signal: prior downtrend + price consolidates into a tight base + MACD
making higher lows + OBV breakout above its moving average + bullish candle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from buy_strategies import BullishBaseBreakout as _Impl


class Strategy:
    direction = "long"

    def __init__(self, df, ma_short=20, ma_long=50,
                 base_lookback=30, downtrend_lookback=63, obv_ma_period=20):
        self._impl = _Impl(
            df,
            ma_short=ma_short, ma_long=ma_long,
            base_lookback=base_lookback,
            downtrend_lookback=downtrend_lookback,
            obv_ma_period=obv_ma_period,
        )

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"buy_signal": "signal"})
