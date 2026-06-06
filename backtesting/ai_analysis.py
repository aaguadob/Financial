"""
backtesting/ai_analysis.py
---------------------------
Exports full backtest results to JSON and asks the Claude CLI
(`claude -p`) for a written quantitative analysis of the strategy.

No API key needed — the call goes through the same Claude Code session
that is already authenticated.  The `claude` binary must be in PATH
(it is, if you are running this from within Claude Code).

The analysis covers:
  • Statistical validity (sample size, significance)
  • Risk-reward profile (Sharpe, Calmar, profit factor, win rate)
  • Tail-risk assessment (VaR, CVaR, skewness, kurtosis)
  • P&L distribution shape and best-fit distribution
  • Red flags, inconsistencies, and key concerns
  • Overall verdict: genuine edge or artefact?

Output:
  • JSON file   — <save_dir>/<strategy>_<ticker>_results.json
  • Text report — <save_dir>/<strategy>_<ticker>_ai_analysis.txt
    (also printed to the terminal)
"""

import json
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st


# ── JSON serialisation helpers ────────────────────────────────────────────────

def _to_py(obj):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj.date()) if hasattr(obj, "date") else str(obj)
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj


def _equity_monthly(equity: pd.Series) -> list[dict]:
    """Resample equity to month-end for a compact curve summary."""
    monthly = equity.resample("ME").last()
    return [{"date": str(d.date()), "equity": round(float(v), 2)}
            for d, v in monthly.items()]


# ── assemble results dict ─────────────────────────────────────────────────────

def build_results_dict(
    equity:    pd.Series,
    trades:    list,
    metrics:   dict,
    risk_data: dict | None,
    cfg:       dict,
) -> dict:
    """
    Assemble a self-contained results dict ready for JSON serialisation
    and for transmission to Claude.
    """
    strat_name = Path(cfg["strategy"]["path"]).stem
    ticker     = cfg["ticker"]
    start      = str(cfg["start_date"])
    end        = str(cfg["end_date"])
    capital    = float(cfg.get("initial_capital", 100_000))
    comm       = float(cfg.get("commission", 0.001))
    pos        = cfg.get("position", {})
    n_years    = (equity.index[-1] - equity.index[0]).days / 365.25

    # Trade list
    trade_rows = []
    for t in trades:
        if getattr(t, "pnl_pct", None) is None:
            continue
        trade_rows.append({
            "entry":     str(getattr(t, "entry_date", "?")),
            "exit":      str(getattr(t, "exit_date",  "?")),
            "direction": getattr(t, "direction", "?"),
            "pnl_pct":   round(float(t.pnl_pct), 6),
        })

    pnl_arr = np.array([r["pnl_pct"] for r in trade_rows]) if trade_rows else np.array([])

    # Risk block (when RiskAnalyser has been run)
    risk_block: dict = {}
    if risk_data and pnl_arr.size >= 5:
        fits  = risk_data.get("fits", [])
        res   = risk_data.get("res", {})
        confs = list(res.get("historical", {}).keys())

        risk_block = {
            "descriptive_stats": {
                "n_trades":        int(pnl_arr.size),
                "mean":            round(float(pnl_arr.mean()), 4),
                "std":             round(float(pnl_arr.std()),  4),
                "min":             round(float(pnl_arr.min()),  4),
                "max":             round(float(pnl_arr.max()),  4),
                "skewness":        round(float(st.skew(pnl_arr)),     4),
                "excess_kurtosis": round(float(st.kurtosis(pnl_arr)), 4),
            },
            "distribution_fit": {
                "best_fit": fits[0]["name"] if fits else "n/a",
                "ranked": [
                    {
                        "name":    f["name"],
                        "aic":     round(f["aic"],     2),
                        "bic":     round(f["bic"],     2),
                        "ks_stat": round(f["ks_stat"], 4),
                        "ks_pval": round(f["ks_pval"], 4),
                    }
                    for f in fits
                ],
            },
            "var_cvar": {},
        }
        for conf in confs:
            key = f"confidence_{int(conf * 100)}"
            risk_block["var_cvar"][key] = {}
            for method in ("historical", "parametric", "monte_carlo"):
                if conf in res.get(method, {}):
                    var, cvar = res[method][conf]
                    risk_block["var_cvar"][key][method] = {
                        "var":  round(float(var),  4),
                        "cvar": round(float(cvar), 4),
                    }

    result = {
        "metadata": {
            "strategy":  strat_name,
            "ticker":    ticker,
            "period":    {"start": start, "end": end, "years": round(n_years, 2)},
            "capital":   {"initial": capital,
                          "final": round(float(equity.iloc[-1]), 2)},
            "commission": comm,
            "position":  {
                "size_pct": float(pos.get("size_pct", 0.1)),
                "max_open": int(pos.get("max_open", 1)),
            },
        },
        "performance":          {k: _to_py(v) for k, v in metrics.items()},
        "equity_curve_monthly": _equity_monthly(equity),
        "trades":               trade_rows,
    }
    if risk_block:
        result["risk_analysis"] = risk_block

    return _to_py(result)


# ── Claude CLI call ───────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are an independent quantitative finance analyst reviewing a systematic \
trading strategy backtest.  Your role is to give an honest, rigorous \
assessment — including both strengths and weaknesses — so that a portfolio \
manager can decide whether to allocate real capital to this strategy.

Be sceptical of overly good results; flag potential overfitting, small sample \
size, look-ahead bias, or dependence on a few outlier trades.  Interpret the \
numbers rather than just repeating them.

Backtest results:
```json
{json_data}
```

Structure your response as:
1. **Executive Summary** (2-3 sentences)
2. **Performance Assessment** — risk-adjusted returns, win rate, profit factor
3. **Statistical Validity** — sample size, significance, distribution shape
4. **Tail Risk & Downside** — VaR, CVaR, skewness, worst-case scenarios
5. **Concerns & Red Flags** — anything suspicious or worth investigating further
6. **Overall Verdict** — genuine edge or artefact? Would you allocate capital?\
"""


def call_claude_cli(results_dict: dict, model: str | None = None,
                    timeout: int = 120) -> str:
    """
    Call the Claude CLI (`claude -p`) with the results JSON embedded in a
    quantitative analysis prompt.  Returns the model's text response.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return (
            "ERROR: `claude` binary not found in PATH.\n"
            "Make sure you are running this from within Claude Code."
        )

    json_str = json.dumps(results_dict, indent=2)
    prompt   = _PROMPT_TEMPLATE.format(json_data=json_str)

    cmd = [
        claude_bin, "-p", prompt,
        "--output-format", "text",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 and proc.stderr:
            return f"ERROR from claude CLI:\n{proc.stderr.strip()}"
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"ERROR: Claude CLI timed out after {timeout}s."
    except Exception as exc:
        return f"ERROR calling claude CLI: {exc}"


# ── main class ────────────────────────────────────────────────────────────────

class AIAnalyser:
    """
    Orchestrates JSON export and Claude CLI analysis.

    Parameters
    ----------
    equity     : equity curve Series
    trades     : list of closed trade objects with a .pnl_pct attribute
    metrics    : performance metrics dict from compute_metrics()
    risk_data  : internal state dict from RiskAnalyser (or None)
    cfg        : full config dict
    cfg_ai     : the 'ai_analysis' sub-dict from config
    """

    def __init__(self, equity, trades, metrics, risk_data, cfg, cfg_ai):
        self._equity    = equity
        self._trades    = trades
        self._metrics   = metrics
        self._risk_data = risk_data
        self._cfg       = cfg
        self._cfg_ai    = cfg_ai
        self._save_dir  = cfg.get("output", {}).get("save_dir", "backtesting/results")
        self._strat     = Path(cfg["strategy"]["path"]).stem
        self._ticker    = cfg["ticker"]
        self._analysis  = ""

    def run(self) -> "AIAnalyser":
        cfg_ai  = self._cfg_ai
        model   = cfg_ai.get("model")     # None → CLI default (Sonnet)
        timeout = int(cfg_ai.get("timeout_seconds", 120))

        print("   Building results JSON …")
        self._results = build_results_dict(
            self._equity, self._trades, self._metrics,
            self._risk_data, self._cfg,
        )

        # Always save the JSON
        Path(self._save_dir).mkdir(parents=True, exist_ok=True)
        json_path = (f"{self._save_dir}/"
                     f"{self._strat}_{self._ticker}_results.json")
        with open(json_path, "w") as f:
            json.dump(self._results, f, indent=2)
        print(f"   Results JSON → {json_path}")

        print("   Calling Claude CLI for strategy analysis …")
        self._analysis = call_claude_cli(self._results, model=model, timeout=timeout)
        return self

    def report(self) -> "AIAnalyser":
        width = 72
        sep   = "─" * width
        print(f"\n── AI Strategy Analysis ─────────────────────────────────────────────")
        for line in self._analysis.splitlines():
            indent = len(line) - len(line.lstrip())
            prefix = " " * indent
            if len(line) <= width:
                print(line)
            else:
                for wrapped in textwrap.wrap(line.strip(), width - indent,
                                              subsequent_indent=prefix):
                    print(prefix + wrapped)
        print(sep)

        if self._cfg_ai.get("save_report", True):
            rpt_path = (f"{self._save_dir}/"
                        f"{self._strat}_{self._ticker}_ai_analysis.txt")
            with open(rpt_path, "w") as f:
                f.write(f"Strategy : {self._strat}\n")
                f.write(f"Ticker   : {self._ticker}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(sep + "\n\n")
                f.write(self._analysis)
                f.write("\n")
            print(f"  Analysis saved → {rpt_path}")

        return self
