"""
Gap-down short (short).
Signal: steady uptrend + gap down below prior low; entry triggered when
price then breaks below the gap-day low (confirming the gap holds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sell_strategies import GapDownShortStrategy as _Impl


class Strategy:
    direction = "short"

    def __init__(self, df, ma=50):
        self._impl = _Impl(df, ma=ma)

    def generate_signals(self):
        return self._impl.generate_signals().rename(columns={"sell_signal": "signal"})
