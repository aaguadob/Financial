"""
hv_calculator.py
----------------
Calculates Historical Volatility (HV) from a closing price series.

Supports:
  - Spot HV (single value over the last N days)
  - Rolling HV series (one value per day)
  - Multiple windows simultaneously (e.g. HV20, HV60, HV252)
  - Close-to-close (standard) and Parkinson (high/low) estimators
  - VRP calculation when IV is provided

Usage:
    python hv_calculator.py                          # demo with synthetic data
    python hv_calculator.py --ticker SPY --window 20
    python hv_calculator.py --ticker SPY --iv 0.22   # also prints VRP
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── estimators ────────────────────────────────────────────────────────────────

def hv_close_to_close(
    closes: np.ndarray,
    window: int = 20,
    ddof: int = 1,
    annualise: bool = True,
) -> float:
    """
    Standard close-to-close HV using log returns.
    Uses only the last `window` returns.

    Parameters
    ----------
    closes   : array of closing prices, most recent last
    window   : number of returns to use
    ddof     : degrees of freedom for std (1 = sample, 0 = population)
    annualise: multiply by sqrt(252) to get annualised vol
    """
    if len(closes) < window + 1:
        raise ValueError(f"Need at least {window + 1} closes, got {len(closes)}")
    log_rets = np.log(closes[1:] / closes[:-1])
    hv = log_rets[-window:].std(ddof=ddof)
    return hv * np.sqrt(252) if annualise else hv


def hv_parkinson(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 20,
    annualise: bool = True,
) -> float:
    """
    Parkinson estimator — uses high/low range instead of close-to-close.
    More efficient than close-to-close (uses intraday information).
    Assumes no drift (slightly biased in trending markets).

    Formula: sqrt( 1/(4*ln2*N) * sum( ln(H/L)^2 ) )
    """
    if len(highs) < window:
        raise ValueError(f"Need at least {window} bars, got {len(highs)}")
    hl = np.log(highs[-window:] / lows[-window:])
    hv = np.sqrt(np.sum(hl**2) / (4 * np.log(2) * window))
    return hv * np.sqrt(252) if annualise else hv


# ── rolling series ────────────────────────────────────────────────────────────

def hv_rolling(
    closes: np.ndarray,
    window: int = 20,
    ddof: int = 1,
    annualise: bool = True,
) -> np.ndarray:
    """
    Rolling close-to-close HV series.
    Returns array of length len(closes) - window,
    aligned to closes[window:] (i.e. index i covers closes[i:i+window]).
    """
    log_rets = np.log(closes[1:] / closes[:-1])
    n = len(log_rets)
    if n < window:
        raise ValueError(f"Need at least {window} returns, got {n}")

    hv = np.array([
        log_rets[i - window:i].std(ddof=ddof)
        for i in range(window, n + 1)
    ])
    return hv * np.sqrt(252) if annualise else hv


def hv_multi_window(
    closes: np.ndarray,
    windows: list[int] = [20, 60, 252],
    annualise: bool = True,
) -> dict[int, float]:
    """
    Compute spot HV for multiple windows simultaneously.
    Useful for comparing short-term vs long-term realised vol.
    """
    return {
        w: hv_close_to_close(closes, window=w, annualise=annualise)
        for w in windows
        if len(closes) >= w + 1
    }


# ── VRP ───────────────────────────────────────────────────────────────────────

def vrp_report(iv: float, hv_dict: dict[int, float]):
    """Print VRP across multiple HV windows."""
    print(f"\n── Volatility Risk Premium ──")
    print(f"  IV: {iv:.2%}")
    print(f"  {'Window':<10} {'HV':<10} {'VRP':>10}  {'Signal'}")
    print(f"  {'──────':<10} {'──':<10} {'───':>10}  {'──────'}")
    for w, hv in sorted(hv_dict.items()):
        vrp = iv - hv
        signal = "SELL VOL" if vrp > 0.03 else ("BUY VOL" if vrp < -0.03 else "neutral")
        print(f"  HV{w:<7} {hv:<10.2%} {vrp:>+10.2%}  {signal}")


# ── plot ──────────────────────────────────────────────────────────────────────

STYLE = dict(
    bg="#0d1117", ax_bg="#0d1117", grid="#1e2a38",
    text="#c9d1d9", accent="#58a6ff",
    green="#3fb950", red="#f85149", amber="#d29922",
    colors=["#58a6ff", "#3fb950", "#d29922", "#f85149"],
)

def _apply_style(ax, title=""):
    ax.set_facecolor(STYLE["ax_bg"])
    ax.tick_params(colors=STYLE["text"], labelsize=8)
    ax.spines[:].set_color(STYLE["grid"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.xaxis.label.set_color(STYLE["text"])
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=10, pad=6)
    ax.grid(True, color=STYLE["grid"], linewidth=0.5, linestyle="--")


def plot_hv(
    closes: np.ndarray,
    windows: list[int] = [20, 60, 252],
    iv: float | None = None,
    ticker: str = "",
):
    """
    Two-panel plot:
      Top:    closing price
      Bottom: rolling HV for each window + IV line if provided
    """
    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor(STYLE["bg"])
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.4], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    title = f"{ticker} — " if ticker else ""

    # ── top: price ──
    _apply_style(ax1, f"{title}Closing Price")
    ax1.plot(closes, color=STYLE["accent"], linewidth=1.2)
    ax1.set_ylabel("Price")
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── bottom: rolling HV ──
    _apply_style(ax2, "Historical Volatility (annualised)")
    ax2.set_ylabel("Volatility")

    max_window = max(w for w in windows if len(closes) >= w + 2)
    x_offset = max_window  # align all series to same x axis

    for w, color in zip(windows, STYLE["colors"]):
        if len(closes) < w + 2:
            continue
        hv_series = hv_rolling(closes, window=w)
        # pad with NaNs so all series share the same x axis as closes
        pad = len(closes) - len(hv_series)
        padded = np.concatenate([np.full(pad, np.nan), hv_series])
        ax2.plot(padded, color=color, linewidth=1.2, label=f"HV{w}", alpha=0.9)

    if iv is not None:
        ax2.axhline(iv, color=STYLE["red"], linewidth=1.5,
                    linestyle="--", label=f"IV {iv:.1%}")
        # shade VRP
        hv20 = hv_rolling(closes, window=min(windows))
        pad  = len(closes) - len(hv20)
        padded_hv20 = np.concatenate([np.full(pad, np.nan), hv20])
        ax2.fill_between(range(len(closes)), iv, padded_hv20,
                         where=(padded_hv20 < iv), alpha=0.12,
                         color=STYLE["red"], label="VRP (sell vol zone)")
        ax2.fill_between(range(len(closes)), iv, padded_hv20,
                         where=(padded_hv20 > iv), alpha=0.12,
                         color=STYLE["green"], label="VRP (buy vol zone)")

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(fontsize=8, facecolor=STYLE["bg"], labelcolor=STYLE["text"],
               loc="upper left")
    ax2.set_xlabel("Trading days")

    fig.tight_layout()
    path = f"hv_{ticker or 'chart'}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"\n  Plot saved → {path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical Volatility Calculator")
    parser.add_argument("--ticker",  type=str,   default=None,
                        help="Ticker to fetch via yfinance (e.g. SPY)")
    parser.add_argument("--window",  type=int,   default=20)
    parser.add_argument("--iv",      type=float, default=None,
                        help="Current IV to overlay and compute VRP (e.g. 0.22)")
    parser.add_argument("--period",  type=str,   default="2y",
                        help="yfinance period string (e.g. 1y, 2y, 5y)")
    args = parser.parse_args()

    if args.ticker:
        try:
            import yfinance as yf
            df     = yf.download(args.ticker, period=args.period, progress=False)
            closes = df["Close"].dropna().values
            print(f"Fetched {len(closes)} closes for {args.ticker}")
        except ImportError:
            print("yfinance not installed — using synthetic data. pip install yfinance")
            closes = None
    else:
        closes = None

    if closes is None:
        # synthetic GBM fallback
        np.random.seed(42)
        closes = 100 * np.exp(np.cumsum(np.random.normal(0, 0.18 / np.sqrt(252), 504)))
        ticker = "SYNTHETIC"
    else:
        ticker = args.ticker

    windows = [20, 60, 252]

    # spot HV report
    hv_dict = hv_multi_window(closes, windows=windows)
    print(f"\n── Spot HV Report — {ticker} ──")
    for w, hv in sorted(hv_dict.items()):
        print(f"  HV{w:<4}: {hv:.2%}")

    if args.iv:
        vrp_report(args.iv, hv_dict)

    plot_hv(closes, windows=windows, iv=args.iv, ticker=ticker)