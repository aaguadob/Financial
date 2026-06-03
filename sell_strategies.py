import numpy as np
import pandas as pd


class ReliefRallyShortStrategy:

    def __init__(self, df, ma_short=20, ma_long=50, trend_period=63):
        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.trend_period = trend_period
        self.signals = pd.DataFrame(index=df.index)

    def is_downtrend(self, idx):

        if idx < self.trend_period:
            return False

        highs = self.df["high"].iloc[idx-self.trend_period:idx]

        ma20 = self.df[f"close_MA_{self.ma_short}"].iloc[idx-self.trend_period:idx]
        ma50 = self.df[f"close_MA_{self.ma_long}"].iloc[idx-self.trend_period:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        lower_highs = all(x > y for x, y in zip(highs[:-1], highs[1:]))

        strong = (
            all(ma20_diff < 0)
            and all(ma50_diff < 0)
            and ma20.iloc[-1] < ma50.iloc[-1]
        )

        weak = (
            ma20_diff.mean() < 0
            and ma50_diff.mean() < 0
            and ma20.iloc[-1] < ma50.iloc[-1]
        )

        return lower_highs and (strong or weak)

    def rally_into_ma(self, idx):

        price = self.df["close"].iloc[idx]
        ma20 = self.df[f"close_MA_{self.ma_short}"].iloc[idx]
        ma50 = self.df[f"close_MA_{self.ma_long}"].iloc[idx]

        strong_condition = abs(price - ma20) / ma20 < 0.02
        weak_condition = abs(price - ma50) / ma50 < 0.02

        return strong_condition or weak_condition

    def stochastic_overbought(self, idx):

        return self.df["stoch_k"].iloc[idx] >= 80

    def bearish_candle(self, idx):

        return self.df["close"].iloc[idx] < self.df["open"].iloc[idx]

    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)

        risk = stop_loss - entry

        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.is_downtrend(idx)
                and self.rally_into_ma(idx)
                and self.stochastic_overbought(idx)
                and self.bearish_candle(idx)
            ):

                stop_loss, take_profit = self.risk_management(idx)

                sell_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

            else:

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals
    

class BearishDivergenceStrategy:

    def __init__(self, df, ma_short=50, ma_long=200):

        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.signals = pd.DataFrame(index=df.index)

    def long_term_downtrend(self, idx, lookback=10):

        if idx < lookback:
            return False

        ma50 = self.df[f"close_MA_{self.ma_short}"].iloc[idx]
        ma200 = self.df[f"close_MA_{self.ma_long}"].iloc[idx]

        ma200_series = self.df[f"close_MA_{self.ma_long}"].iloc[idx-lookback:idx]

        ma200_diff = ma200_series.diff().dropna()

        condition1 = ma50 < ma200
        condition2 = ma200_diff.mean() < 0

        return condition1 and condition2


    def price_above_ma50(self, idx):

        return self.df["close"].iloc[idx] > self.df[f"close_MA_{self.ma_short}"].iloc[idx]


    def find_two_highs(self, idx, lookback=60, min_gap=5):

        highs = []

        for i in range(max(1, idx-lookback), idx):

            if (
                self.df["high"].iloc[i] > self.df["high"].iloc[i-1]
                and self.df["high"].iloc[i] > self.df["high"].iloc[i+1]
            ):
                if len(highs) == 0 or i - highs[-1] >= min_gap:
                    highs.append(i)

        if len(highs) >= 2:
            return highs[-2:]

        return None


    def divergence(self, highs):

        h1, h2 = highs

        indicators = [
            "MACD",
            "MACD_hist",
            "stoch_k",
            "obv",
            "cci",
            "rsi",
        ]

        count = 0

        for ind in indicators:

            if ind in self.df.columns:

                if self.df[ind].iloc[h2] < self.df[ind].iloc[h1]:
                    count += 1

        return count >= 2


    def bearish_candle(self, idx):

        return self.df["close"].iloc[idx] < self.df["open"].iloc[idx]


    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)

        risk = stop_loss - entry

        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit


    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.long_term_downtrend(idx)
                and self.price_above_ma50(idx)
            ):

                highs = self.find_two_highs(idx)

                if highs and self.divergence(highs):

                    if self.bearish_candle(idx):

                        stop_loss, take_profit = self.risk_management(idx)

                        sell_signals.append(True)
                        stop_losses.append(stop_loss)
                        take_profits.append(take_profit)

                        continue

            sell_signals.append(False)
            stop_losses.append(np.nan)
            take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals


class GapDownShortStrategy:

    def __init__(self, df, ma=50):

        self.df = df.copy()
        self.ma = ma
        self.signals = pd.DataFrame(index=df.index)

    def steady_uptrend(self, idx, lookback=40):

        if idx < lookback:
            return False

        ma_series = self.df[f"close_MA_{self.ma}"].iloc[idx-lookback:idx]

        ma_diff = ma_series.diff().dropna()

        price_series = self.df["close"].iloc[idx-lookback:idx]

        ma_rising = ma_diff.mean() > 0
        price_above_ma = all(price_series > ma_series)

        return ma_rising and price_above_ma

    def gap_down(self, idx):

        if idx < 1:
            return False

        today_high = self.df["high"].iloc[idx]
        prev_low = self.df["low"].iloc[idx-1]

        return today_high < prev_low

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        gap_low = None
        prev_low = None
        active_setup = False

        for idx in range(len(self.df)):

            if self.steady_uptrend(idx) and self.gap_down(idx):

                gap_low = self.df["low"].iloc[idx]
                prev_low = self.df["low"].iloc[idx-1]
                active_setup = True

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

                continue

            if active_setup:

                price = self.df["close"].iloc[idx]

                # cancel setup if gap filled
                if price >= prev_low:
                    active_setup = False

                # entry trigger
                elif price < gap_low:

                    entry = price
                    stop_loss = entry * 1.07
                    risk = stop_loss - entry
                    take_profit = entry - risk * 1.75

                    sell_signals.append(True)
                    stop_losses.append(stop_loss)
                    take_profits.append(take_profit)

                    active_setup = False
                    continue

            sell_signals.append(False)
            stop_losses.append(np.nan)
            take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals


class BlueSeaBreakdownStrategy:

    def __init__(self, df):

        self.df = df.copy()
        self.signals = pd.DataFrame(index=df.index)

    def pivot_low_recent(self, idx, pivot_window=20):

        if idx < pivot_window:
            return False

        prev_low = self.df["close"].iloc[idx-pivot_window:idx].min()

        return self.df["close"].iloc[idx] < prev_low

    def three_month_low(self, idx, lookback=63):

        if idx < lookback:
            return False

        prev_lows = self.df["close"].iloc[idx-lookback:idx]

        return self.df["close"].iloc[idx] < prev_lows.min()

    def not_overextended(self, idx, lookback=252):

        if idx < lookback:
            return True  # allow if not enough data

        high_52w = self.df["high"].iloc[idx-lookback:idx].max()
        new_low = self.df["close"].iloc[idx]

        ratio = high_52w / new_low

        return ratio <= 2.0

    def obv_confirmation(self, idx, lookback=63):

        if idx < lookback:
            return False

        prev_obv = self.df["obv"].iloc[idx-lookback:idx]

        return self.df["obv"].iloc[idx] <= prev_obv.min()

    def bearish_candle(self, idx):

        return self.df["close"].iloc[idx] < self.df["open"].iloc[idx]

    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)

        risk = stop_loss - entry

        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.pivot_low_recent(idx)
                and self.three_month_low(idx)
                and self.not_overextended(idx)
                and self.obv_confirmation(idx)
                and self.bearish_candle(idx)
            ):

                stop_loss, take_profit = self.risk_management(idx)

                sell_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

            else:

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals
    
class RisingWedgeBreakdownStrategy:

    def __init__(self, df, ma_short=20, ma_long=50):

        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.signals = pd.DataFrame(index=df.index)

    def uptrend(self, idx, lookback=30):

        if idx < lookback:
            return False

        ma20 = self.df[f"close_MA_{self.ma_short}"].iloc[idx-lookback:idx]
        ma50 = self.df[f"close_MA_{self.ma_long}"].iloc[idx-lookback:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        strong = all(ma20_diff > 0) and all(ma50_diff > 0) and ma20.iloc[-1] > ma50.iloc[-1]
        weak = ma20_diff.mean() > 0 and ma20.iloc[-1] > ma50.iloc[-1]

        return strong or weak

    def rising_wedge(self, idx, window=20):

        if idx < window:
            return False

        highs = self.df["high"].iloc[idx-window:idx]
        lows = self.df["low"].iloc[idx-window:idx]

        higher_highs = highs.iloc[-1] > highs.iloc[0]
        higher_lows = lows.iloc[-1] > lows.iloc[0]

        ranges = highs.values - lows.values
        tightening = ranges[-1] < ranges[0]

        return higher_highs and higher_lows and tightening

    def macd_lower_highs(self, idx, window=20):

        if idx < window:
            return False

        macd = self.df["MACD"].iloc[idx-window:idx]

        peaks = macd.nlargest(3)

        if len(peaks) < 3:
            return False

        return peaks.iloc[0] > peaks.iloc[1] > peaks.iloc[2]

    def obv_breakdown(self, idx, window=20):

        if idx < window:
            return False

        obv_series = self.df["obv"].iloc[idx-window:idx]

        return self.df["obv"].iloc[idx] < obv_series.min()

    def bearish_candle(self, idx):

        return self.df["close"].iloc[idx] < self.df["open"].iloc[idx]

    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["close"].iloc[idx]

        stop_loss = entry * 1.07
        risk = stop_loss - entry
        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.uptrend(idx)
                and self.rising_wedge(idx)
                and self.macd_lower_highs(idx)
                and self.obv_breakdown(idx)
                and self.bearish_candle(idx)
            ):

                stop_loss, take_profit = self.risk_management(idx)

                sell_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

            else:

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals
