import pandas as pd

import mplfinance as mpf

def add_moving_average(df, column = "close", window = 20):
    """
    Calculate a moving average for a given column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing financial data
    column : str
        Column name with the values (e.g., 'close')
    window : int
        Moving average period (e.g., 20, 50, 200)

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new moving average column
    """
    
    df = df.copy()
    ma_column = f"{column}_MA_{window}"
    
    df[ma_column] = df[column].rolling(window=window).mean()
    
    return df

def add_macd(df, column="close", fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column with price values (default 'close')
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    signal : int
        Signal EMA period

    Returns
    -------
    pandas.DataFrame
        DataFrame with MACD, Signal, and Histogram
    """

    df = df.copy()

    # EMAs
    df["EMA_fast"] = df[column].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df[column].ewm(span=slow, adjust=False).mean()

    # MACD line
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]

    # Signal line
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    # Histogram
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df


def add_stochastic(df, high="high", low="low", close="close",
                   k_period=5, k_smooth=3, d_period=3):

    df = df.copy()

    # Lowest low and highest high
    lowest_low = df[low].rolling(window=k_period).min()
    highest_high = df[high].rolling(window=k_period).max()

    # Raw %K
    df["stoch_k"] = 100 * (df[close] - lowest_low) / (highest_high - lowest_low)

    # Smoothed %K
    df["stoch_k_smooth"] = df["stoch_k"].rolling(window=k_smooth).mean()

    # %D line
    df["stoch_d"] = df["stoch_k_smooth"].rolling(window=d_period).mean()

    return df

import numpy as np

def add_obv(df, close="close", volume="volume"):
    """
    Calculate On-Balance Volume (OBV)
    """

    df = df.copy()

    # Price difference
    price_diff = df[close].diff()

    # Direction of movement
    direction = np.sign(price_diff)

    # Replace NaN with 0
    direction = direction.fillna(0)

    # OBV calculation
    df["obv"] = (direction * df[volume]).cumsum()

    return 

def add_rsi(df, column="close", period=5):
    """
    Calculate Relative Strength Index (RSI)
    """

    df = df.copy()

    delta = df[column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def add_cci(df, high="high", low="low", close="close", period=20):
    """
    Calculate Commodity Channel Index (CCI)
    """

    df = df.copy()

    # Typical price
    tp = (df[high] + df[low] + df[close]) / 3

    # Moving average of TP
    sma = tp.rolling(window=period).mean()

    # Mean deviation
    mean_dev = tp.rolling(window=period).apply(
        lambda x: (abs(x - x.mean())).mean(),
        raw=False
    )

    # CCI
    df["cci"] = (tp - sma) / (0.015 * mean_dev)

    return df

def plot_candles(df, title="Candlestick Chart"):
    """
    Plot candlestick chart from a dataframe.

    Required columns:
    open, high, low, close
    Index must be datetime
    """

    df = df.copy()

    if not isinstance(df.index, type(df.index)):
        df.index = df.index

    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=title,
        volume=True
    )

def plot_full_chart(df, save_path="chart.png"):
    """
    Plot candlestick chart with indicators and save it.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain: date, open, high, low, close, volume
    save_path : str
        File path where the image will be saved
    """

    df = df.copy()
    df = df.sort_values("date")
    df = df.set_index("date")

    # --- indicators ---
    df = add_moving_average(df, "close", 20)
    df = add_moving_average(df, "close", 50)

    df = add_macd(df)
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_obv(df)
    df = add_cci(df)

    apds = [

        mpf.make_addplot(df["close_MA_20"]),
        mpf.make_addplot(df["close_MA_50"]),

        # MACD
        mpf.make_addplot(df["MACD"], panel=2),
        mpf.make_addplot(df["MACD_signal"], panel=2),
        mpf.make_addplot(df["MACD_hist"], panel=2, type="bar"),

        # RSI
        mpf.make_addplot(df["rsi"], panel=3),

        # Stochastic
        mpf.make_addplot(df["stoch_k"], panel=4),
        mpf.make_addplot(df["stoch_d"], panel=4),
    ]

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        addplot=apds,
        volume=True,
        figsize=(14,10),
        panel_ratios=(6,2,2,2,2),
        title="Technical Analysis Chart",
        savefig=save_path
    )

import yfinance as yf

ticker = "^GSPC"
data = pd.DataFrame(yf.download(ticker, start='2020-01-01', end='2025-01-01'))
returns = data.pct_change().dropna()

print(data)

data["moving_average_20"] = add_moving_average(data, "close", 20)
data["moving_average_50"] = add_moving_average(data, "close", 50)
data["moving_average_200"] = add_moving_average(data, "close", 200)
data["MACD"] = add_macd(data, "close")
data["stochastics"] = add_stochastic(data, "high", "low", "close", 5, 3, 3)
data["OBV"] = add_obv(data, "close", "volume")
data["RSI"] = add_rsi(data, "close", 5)
data["CCI"] = add_cci(data, "high", "low", "close", 20)

plot_full_chart(data, "analysis_chart.png")