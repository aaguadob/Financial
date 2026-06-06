"""
backtesting/backtest.py
-----------------------
Lean backtesting engine.  The engine has no knowledge of any specific
strategy — it dynamically loads whatever Strategy class is pointed to
by the YAML config, runs the simulation, and reports only the
metrics / plots the user has requested.

Usage:
    python backtesting/backtest.py
    python backtesting/backtest.py --config backtesting/config.yaml
    python backtesting/backtest.py --config backtesting/config.yaml --no-plot
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parent.parent   # project root
sys.path.insert(0, str(ROOT))


# ── data ──────────────────────────────────────────────────────────────────────

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker} [{start} → {end}]")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every indicator required by any strategy in the strategies/ folder."""
    df = df.copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    for p in (20, 50, 200):
        df[f"close_MA_{p}"] = close.rolling(p).mean()

    lo14, hi14 = low.rolling(14).min(), high.rolling(14).max()
    df["stoch_k"] = 100 * (close - lo14) / (hi14 - lo14).replace(0, np.nan)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    df["obv"] = (np.sign(close.diff()).fillna(0) * vol).cumsum()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    tp  = (high + low + close) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))

    return df


# ── dynamic strategy loader ───────────────────────────────────────────────────

def load_strategy_class(path: str):
    """
    Load the Strategy class from a file path (absolute or relative to ROOT).
    The file must define a class named exactly 'Strategy'.
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Strategy file not found: {p}")

    spec = importlib.util.spec_from_file_location("_strategy_module", p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "Strategy"):
        raise AttributeError(f"{p.name} must define a class named 'Strategy'.")
    return mod.Strategy


# ── trade model ───────────────────────────────────────────────────────────────

class Trade:
    __slots__ = ("entry_date", "entry", "stop_loss", "take_profit",
                 "direction", "allocation", "exit_date", "exit_price", "pnl_pct")

    def __init__(self, entry_date, entry: float, stop_loss: float,
                 take_profit: float, direction: str, allocation: float):
        self.entry_date  = entry_date
        self.entry       = entry
        self.stop_loss   = stop_loss
        self.take_profit = take_profit
        self.direction   = direction
        self.allocation  = allocation   # capital committed at entry
        self.exit_date   = None
        self.exit_price  = None
        self.pnl_pct     = None         # net P&L fraction (after commission)

    def close(self, date, price: float, commission: float) -> None:
        self.exit_date  = date
        self.exit_price = price
        raw = (
            (price - self.entry) / self.entry
            if self.direction == "long"
            else (self.entry - price) / self.entry
        )
        self.pnl_pct = raw - 2 * commission   # entry + exit commissions


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    direction: str,
    initial_capital: float,
    size_pct: float,
    max_open: int,
    commission: float,
) -> tuple[pd.Series, list[Trade]]:
    """
    Bar-by-bar simulation driven by a signals DataFrame with columns:
      signal (bool), stop_loss (float), take_profit (float).

    Position size is dynamic (size_pct × current cash), so losses don't
    permanently freeze future entries.
    Stop-loss takes priority when both stop-loss and take-profit are hit
    on the same bar (conservative assumption).
    Remaining open positions are force-closed at the last available price.
    """
    cash: float = initial_capital
    open_trades: list[Trade] = []
    closed: list[Trade]      = []
    equity: list[float]      = []

    for date, row in df.iterrows():
        # ── exits ──────────────────────────────────────────────────────────
        remaining: list[Trade] = []
        for t in open_trades:
            if t.direction == "long":
                hit_sl = row["low"]  <= t.stop_loss
                hit_tp = row["high"] >= t.take_profit
            else:
                hit_sl = row["high"] >= t.stop_loss
                hit_tp = row["low"]  <= t.take_profit

            if hit_sl or hit_tp:
                exit_p = t.stop_loss if hit_sl else t.take_profit
                t.close(date, exit_p, commission)
                # Return: committed capital + net P&L (entry commission already paid)
                cash += t.allocation * (1 + t.pnl_pct + commission)
                closed.append(t)
            else:
                remaining.append(t)
        open_trades = remaining

        # ── entry ───────────────────────────────────────────────────────────
        if date in signals.index:
            sig = signals.at[date, "signal"]
            sl  = signals.at[date, "stop_loss"]
            tp  = signals.at[date, "take_profit"]
        else:
            sig, sl, tp = False, np.nan, np.nan

        if sig and len(open_trades) < max_open and not pd.isna(sl) and not pd.isna(tp):
            alloc = cash * size_pct
            if alloc * (1 + commission) <= cash and alloc > 0:
                cash -= alloc * (1 + commission)
                open_trades.append(
                    Trade(date, row["close"], sl, tp, direction, alloc)
                )

        # ── mark-to-market equity ────────────────────────────────────────────
        if direction == "long":
            mtm = sum(t.allocation * row["close"] / t.entry for t in open_trades)
        else:
            mtm = sum(
                t.allocation * (1 + (t.entry - row["close"]) / t.entry)
                for t in open_trades
            )
        equity.append(cash + mtm)

    # Force-close any remaining positions at the last bar's close
    last_date  = df.index[-1]
    last_close = float(df.iloc[-1]["close"])
    for t in open_trades:
        t.close(last_date, last_close, commission)
        closed.append(t)

    return pd.Series(equity, index=df.index, name="equity"), closed


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series, trades: list[Trade],
                    initial_capital: float) -> dict:
    rets    = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_r = equity.iloc[-1] / initial_capital - 1
    cagr    = (1 + total_r) ** (1 / max(n_years, 1e-9)) - 1
    sharpe  = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    dd      = (equity - equity.cummax()) / equity.cummax()
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd < 0 else np.inf

    done   = [t for t in trades if t.pnl_pct is not None]
    wins   = [t for t in done if t.pnl_pct > 0]
    losses = [t for t in done if t.pnl_pct <= 0]
    win_r  = len(wins) / len(done) if done else 0.0
    avg_w  = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
    avg_l  = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    pf     = (avg_w * len(wins)) / abs(avg_l * len(losses)) if losses and avg_l else np.inf

    return dict(
        total_return  = total_r,
        cagr          = cagr,
        sharpe        = sharpe,
        max_drawdown  = max_dd,
        calmar        = calmar,
        n_trades      = len(done),
        win_rate      = win_r,
        avg_win_pct   = avg_w,
        avg_loss_pct  = avg_l,
        profit_factor = pf,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

STYLE = dict(
    bg="#0d1117", ax_bg="#0d1117", grid="#1e2a38",
    text="#c9d1d9", accent="#58a6ff",
    green="#3fb950", red="#f85149", amber="#d29922",
)

METRIC_FMT: dict[str, tuple[str, str]] = {
    "total_return":  ("{:+.1%}", "Total Return"),
    "cagr":          ("{:+.1%}", "CAGR"),
    "sharpe":        ("{:.2f}",  "Sharpe Ratio"),
    "max_drawdown":  ("{:.1%}",  "Max Drawdown"),
    "calmar":        ("{:.2f}",  "Calmar Ratio"),
    "n_trades":      ("{:d}",    "Trades"),
    "win_rate":      ("{:.1%}",  "Win Rate"),
    "avg_win_pct":   ("{:+.1%}", "Avg Win"),
    "avg_loss_pct":  ("{:+.1%}", "Avg Loss"),
    "profit_factor": ("{:.2f}",  "Profit Factor"),
}


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(STYLE["ax_bg"])
    ax.tick_params(colors=STYLE["text"], labelsize=8)
    ax.spines[:].set_color(STYLE["grid"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.xaxis.label.set_color(STYLE["text"])
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=10, pad=6)
    ax.grid(True, color=STYLE["grid"], linewidth=0.5, linestyle="--")


def _draw_equity(ax, equity: pd.Series, benchmark: pd.Series, ticker: str) -> None:
    _style(ax, "Equity Curve vs Buy-and-Hold")
    ax.plot(equity.index,    equity    / equity.iloc[0],
            color=STYLE["accent"], lw=1.5, label="Strategy")
    ax.plot(benchmark.index, benchmark / benchmark.iloc[0],
            color=STYLE["amber"],  lw=1.2, ls="--", label=f"{ticker} B&H")
    ax.axhline(1, color=STYLE["grid"], lw=0.8)
    ax.set_ylabel("Normalised value")
    ax.legend(fontsize=9, facecolor=STYLE["bg"], labelcolor=STYLE["text"])


def _draw_drawdown(ax, equity: pd.Series) -> None:
    _style(ax, "Drawdown")
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    ax.fill_between(dd.index, dd, 0, color=STYLE["red"], alpha=0.45)
    ax.plot(dd.index, dd, color=STYLE["red"], lw=0.7)
    ax.set_ylabel("Drawdown %")


def _draw_trade_pnl(ax, trades: list[Trade]) -> None:
    _style(ax, "Trade P&L (%)")
    done = [t for t in trades if t.pnl_pct is not None]
    if done:
        pnls   = sorted(t.pnl_pct * 100 for t in done)
        colors = [STYLE["green"] if p > 0 else STYLE["red"] for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.80, width=0.85)
        ax.axhline(0, color=STYLE["text"], lw=0.8)
        ax.set_xlabel("Trade (sorted)")
        ax.set_ylabel("P&L %")
    else:
        ax.text(0.5, 0.5, "No closed trades", ha="center", va="center",
                color=STYLE["text"], transform=ax.transAxes)


def _fmt_metric(key: str, value) -> str:
    fmt, _ = METRIC_FMT[key]
    if isinstance(value, float) and np.isinf(value):
        return "∞"
    return fmt.format(int(value) if fmt == "{:d}" else value)


def plot_results(equity: pd.Series, benchmark: pd.Series, trades: list[Trade],
                 metrics: dict, cfg: dict) -> None:
    out        = cfg.get("output", {})
    requested  = out.get("plots", ["equity_curve", "drawdown", "trade_pnl"])
    save_dir   = out.get("save_dir", "backtesting/results")
    ticker     = cfg["ticker"]
    strat_name = Path(cfg["strategy"]["path"]).stem

    has_eq  = "equity_curve" in requested
    bottom  = [p for p in ("drawdown", "trade_pnl") if p in requested]
    n_bot   = len(bottom)

    if not has_eq and n_bot == 0:
        return

    fig = plt.figure(figsize=(14, 9 if (has_eq and n_bot) else 5))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle(
        f"{strat_name}  ·  {ticker}  [{cfg['start_date']} → {cfg['end_date']}]",
        color=STYLE["text"], fontsize=12, y=0.99,
    )

    if has_eq and n_bot:
        gs    = gridspec.GridSpec(2, max(n_bot, 1), hspace=0.42, wspace=0.30)
        ax_eq = fig.add_subplot(gs[0, :])
        baxes = [fig.add_subplot(gs[1, i]) for i in range(n_bot)]
    elif has_eq:
        ax_eq = fig.add_subplot(111)
        baxes = []
    else:
        gs    = gridspec.GridSpec(1, n_bot, wspace=0.30)
        ax_eq = None
        baxes = [fig.add_subplot(gs[0, i]) for i in range(n_bot)]

    if has_eq:
        _draw_equity(ax_eq, equity, benchmark, ticker)

    for i, key in enumerate(bottom):
        if key == "drawdown":
            _draw_drawdown(baxes[i], equity)
        elif key == "trade_pnl":
            _draw_trade_pnl(baxes[i], trades)

    # Metrics text box (filtered to what the user requested)
    m_keys = out.get("metrics", list(METRIC_FMT))
    lines  = [
        f"{METRIC_FMT[k][1]:<18}: {_fmt_metric(k, metrics[k])}"
        for k in m_keys if k in METRIC_FMT and k in metrics
    ]
    if lines:
        fig.text(
            0.985, 0.01, "\n".join(lines),
            ha="right", va="bottom", fontsize=8.5,
            color=STYLE["text"], fontfamily="monospace",
            bbox=dict(facecolor=STYLE["ax_bg"], edgecolor=STYLE["grid"],
                      boxstyle="round,pad=0.5"),
        )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{save_dir}/{strat_name}_{ticker}_{cfg['start_date']}_{cfg['end_date']}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"  Chart → {fname}")
    plt.show()


# ── engine ────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Load a YAML config, dynamically import the Strategy class, download
    and prepare data, run the simulation, then report / plot results.

    Usage:
        bt = Backtester(config_dict)
        bt.run().report().plot()
    """

    def __init__(self, config: dict):
        self.cfg        = config
        self._equity    = None
        self._trades    = None
        self._metrics   = None
        self._bench     = None
        self._risk_data = None   # populated by .risk() for use by .analyse()

    def run(self) -> "Backtester":
        cfg        = self.cfg
        ticker     = cfg["ticker"]
        start      = str(cfg["start_date"])
        end        = str(cfg["end_date"])
        strat_path = cfg["strategy"]["path"]
        params     = cfg["strategy"].get("params", {})
        capital    = float(cfg.get("initial_capital", 100_000))
        commission = float(cfg.get("commission", 0.001))
        size_pct   = float(cfg.get("position", {}).get("size_pct", 0.95))
        max_open   = int(cfg.get("position", {}).get("max_open", 1))

        print(f"\n── {Path(strat_path).stem}  ·  {ticker}  [{start} → {end}] ──")
        print(f"   Capital ${capital:,.0f}  |  Pos size {size_pct:.0%}  "
              f"|  Max open {max_open}  |  Commission {commission:.2%}")

        print("   Downloading data …")
        df          = compute_indicators(download_data(ticker, start, end))
        self._bench = df["close"].copy()

        StratClass = load_strategy_class(strat_path)
        strat      = StratClass(df, **params)

        # Strategies may override simulate() (e.g. options strategies that
        # need custom P&L logic).  Fall back to the generic equity simulator.
        if hasattr(strat, "simulate"):
            print("   Running strategy-level simulation …")
            self._equity, self._trades = strat.simulate(
                df, capital, size_pct, max_open, commission,
            )
        else:
            direction = strat.direction
            print("   Generating signals …")
            signals = strat.generate_signals()
            print(f"   Signals fired: {int(signals['signal'].sum())}")
            print("   Simulating …")
            self._equity, self._trades = simulate(
                df, signals, direction, capital, size_pct, max_open, commission,
            )

        self._metrics = compute_metrics(self._equity, self._trades, capital)
        print("   Done.\n")
        return self

    def report(self) -> "Backtester":
        self._require_run()
        requested = self.cfg.get("output", {}).get("metrics", list(METRIC_FMT))
        print("── Performance ───────────────────────────────────────")
        for key in requested:
            if key in METRIC_FMT and key in self._metrics:
                label = METRIC_FMT[key][1]
                print(f"  {label:<18}: {_fmt_metric(key, self._metrics[key])}")
        print()
        return self

    def plot(self) -> "Backtester":
        self._require_run()
        plot_results(self._equity, self._bench, self._trades, self._metrics, self.cfg)
        return self

    def risk(self) -> "Backtester":
        """Run risk analysis when 'risk.enabled: true' in config."""
        self._require_run()
        risk_cfg = self.cfg.get("risk", {})
        if not risk_cfg.get("enabled", False):
            return self

        risk_path = Path(__file__).parent / "risk_analysis.py"
        spec      = importlib.util.spec_from_file_location("_risk_module", risk_path)
        risk_mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(risk_mod)

        save_dir   = self.cfg.get("output", {}).get("save_dir", "backtesting/results")
        strat_name = Path(self.cfg["strategy"]["path"]).stem
        ticker     = self.cfg["ticker"]

        print("── Risk Analysis ─────────────────────────────────────")
        ra = risk_mod.RiskAnalyser(
            self._trades, risk_cfg, save_dir, strat_name, ticker
        )
        ra.run().report()
        if risk_cfg.get("plots"):
            ra.plot()
        # Preserve internal risk state for use by .analyse()
        self._risk_data = {"fits": ra._fits, "res": ra._res}
        return self

    def analyse(self) -> "Backtester":
        """
        Export results to JSON and request a written strategy analysis from Claude.
        Enabled when 'ai_analysis.enabled: true' in config.
        Reads the API key from the ANTHROPIC_API_KEY environment variable
        (or optionally from 'ai_analysis.api_key' in config).
        """
        self._require_run()
        ai_cfg = self.cfg.get("ai_analysis", {})
        if not ai_cfg.get("enabled", False):
            return self

        ai_path  = Path(__file__).parent / "ai_analysis.py"
        spec     = importlib.util.spec_from_file_location("_ai_module", ai_path)
        ai_mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ai_mod)

        print("── AI Analysis ───────────────────────────────────────")
        analyser = ai_mod.AIAnalyser(
            equity    = self._equity,
            trades    = self._trades,
            metrics   = self._metrics,
            risk_data = self._risk_data,   # None if .risk() was not called
            cfg       = self.cfg,
            cfg_ai    = ai_cfg,
        )
        analyser.run().report()
        return self

    # read-only access for use as a library
    @property
    def equity(self)  -> pd.Series:     self._require_run(); return self._equity
    @property
    def trades(self)  -> list[Trade]:   self._require_run(); return self._trades
    @property
    def metrics(self) -> dict:          self._require_run(); return self._metrics

    def _require_run(self):
        if self._equity is None:
            raise RuntimeError("Call .run() before accessing results.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run a backtest from a YAML config file."
    )
    ap.add_argument("--config", default="backtesting/config.yaml",
                    help="Path to config YAML  (default: backtesting/config.yaml)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip chart output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_plot:
        cfg.setdefault("output", {})["plots"] = []
        cfg.setdefault("risk", {})["plots"]   = []

    bt = Backtester(cfg)
    bt.run().report()
    if cfg.get("output", {}).get("plots"):
        bt.plot()
    bt.risk()
    bt.analyse()
