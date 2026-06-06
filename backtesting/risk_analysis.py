"""
backtesting/risk_analysis.py
-----------------------------
Risk analysis module for backtested trade P&L distributions.

Three layers of analysis:

  1. Distribution fitting
     Fits candidate distributions to the empirical trade P&L series and
     ranks them by AIC / BIC and KS goodness-of-fit test.

  2. Risk metrics at configurable confidence levels (e.g. 95%, 99%)
     ┌─────────────┬──────────────────────────────────────────────────┐
     │ Historical  │ Direct empirical percentile / conditional mean   │
     │ Parametric  │ Derived analytically from the best-fit PDF       │
     │ Monte Carlo │ Simulated from the best-fit PDF (n_simulations)  │
     └─────────────┴──────────────────────────────────────────────────┘
     Each method produces VaR (Value at Risk) and CVaR (Expected Shortfall).

  3. Next-trade Monte Carlo
     Simulates n_simulations possible P&L outcomes of one additional trade
     drawn from the fitted distribution, showing the full range of risk.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import integrate

# ── appearance ────────────────────────────────────────────────────────────────

_STYLE = dict(
    bg="#0d1117", ax_bg="#0d1117", grid="#1e2a38",
    text="#c9d1d9", accent="#58a6ff",
    green="#3fb950", red="#f85149", amber="#d29922",
    purple="#bc8cff",
)

_PDF_COLORS = [_STYLE["green"], _STYLE["amber"], _STYLE["red"],
               _STYLE["purple"], "#ff7c7c", "#7ecfff"]

# ── candidate distributions ───────────────────────────────────────────────────

_DISTS: dict = {
    "norm":      st.norm,
    "t":         st.t,
    "skewnorm":  st.skewnorm,
    "laplace":   st.laplace,
    "gev":       st.genextreme,
    "johnsonsu": st.johnsonsu,
}

DEFAULT_DISTRIBUTIONS = ["norm", "t", "skewnorm", "laplace", "gev"]


# ── distribution fitting ──────────────────────────────────────────────────────

def fit_distributions(pnl: np.ndarray, dist_names: list[str]) -> list[dict]:
    """
    Fit each named distribution to `pnl` and compute goodness-of-fit metrics.

    Returns a list of result dicts sorted ascending by AIC (best first).
    Each dict contains: name, dist, params, log_ll, aic, bic, ks_stat, ks_pval.
    """
    n       = len(pnl)
    results = []
    for name in dist_names:
        dist = _DISTS.get(name)
        if dist is None:
            continue
        try:
            params   = dist.fit(pnl)
            log_ll   = float(np.sum(dist.logpdf(pnl, *params)))
            k        = len(params)
            aic      = 2 * k - 2 * log_ll
            bic      = k * np.log(n) - 2 * log_ll
            ks_s, ks_p = st.kstest(pnl, lambda x, p=params: dist.cdf(x, *p))
            results.append({
                "name":    name,
                "dist":    dist,
                "params":  params,
                "log_ll":  log_ll,
                "aic":     aic,
                "bic":     bic,
                "ks_stat": float(ks_s),
                "ks_pval": float(ks_p),
            })
        except Exception:
            pass   # fitting failed (e.g. degenerate data) — skip silently
    return sorted(results, key=lambda r: r["aic"])


# ── risk metrics ──────────────────────────────────────────────────────────────

def historical_var_cvar(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """Empirical VaR and CVaR at `confidence` level (e.g. 0.95)."""
    alpha = 1.0 - confidence
    var   = float(np.percentile(pnl, alpha * 100))
    tail  = pnl[pnl <= var]
    cvar  = float(tail.mean()) if len(tail) else var
    return var, cvar


def parametric_var_cvar(fit: dict, confidence: float) -> tuple[float, float]:
    """
    Parametric VaR and CVaR from the fitted distribution.

      VaR  = dist.ppf(1 − confidence)
      CVaR = (1 / (1−confidence)) × ∫_{−∞}^{VaR} x f(x) dx
             computed via numerical integration (valid for any distribution).
    """
    alpha  = 1.0 - confidence
    dist   = fit["dist"]
    params = fit["params"]
    var    = float(dist.ppf(alpha, *params))
    lower  = float(dist.ppf(1e-9, *params))   # practical lower bound
    numer, _ = integrate.quad(
        lambda x: x * dist.pdf(x, *params), lower, var, limit=200
    )
    return var, float(numer / alpha)


def monte_carlo_var_cvar(
    fit: dict, n_sim: int, confidence: float, seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Monte Carlo VaR, CVaR, and the raw sample array.

    Draws n_sim P&L values from the fitted distribution to represent
    the possible outcomes of future trades.
    """
    rng     = np.random.default_rng(seed)
    alpha   = 1.0 - confidence
    samples = fit["dist"].rvs(*fit["params"], size=n_sim, random_state=rng)
    var     = float(np.percentile(samples, alpha * 100))
    tail    = samples[samples <= var]
    cvar    = float(tail.mean()) if len(tail) else var
    return var, cvar, samples


# ── plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(_STYLE["ax_bg"])
    ax.tick_params(colors=_STYLE["text"], labelsize=8)
    ax.spines[:].set_color(_STYLE["grid"])
    ax.yaxis.label.set_color(_STYLE["text"])
    ax.xaxis.label.set_color(_STYLE["text"])
    if title:
        ax.set_title(title, color=_STYLE["text"], fontsize=10, pad=6)
    ax.grid(True, color=_STYLE["grid"], linewidth=0.5, linestyle="--")


def _draw_distribution(ax, pnl: np.ndarray, fits: list[dict],
                        conf_levels: list[float],
                        var_hist: dict[float, float]) -> None:
    """Histogram of trade P&Ls with all fitted PDF curves overlaid."""
    _style_ax(ax, "P&L Distribution — Empirical vs Fitted PDFs")
    n_bins = max(10, len(pnl) // 3)
    ax.hist(pnl * 100, bins=n_bins, density=True,
            color=_STYLE["accent"], alpha=0.30, label="Empirical")

    # x grid in decimal space for PDF, converted to % for display
    pad = 0.15 * (pnl.max() - pnl.min())
    x_dec = np.linspace(pnl.min() - pad, pnl.max() + pad, 400)
    x_pct = x_dec * 100

    for fit, col in zip(fits, _PDF_COLORS):
        y = fit["dist"].pdf(x_dec, *fit["params"]) / 100   # Jacobian: dec → pct
        star = "★ " if fit is fits[0] else ""
        ax.plot(x_pct, y, color=col, lw=1.5,
                label=f"{star}{fit['name']}  AIC {fit['aic']:.1f}")

    ls_cycle = ["--", ":"]
    for lvl, ls in zip(conf_levels, ls_cycle):
        var = var_hist[lvl]
        ax.axvline(var * 100, color=_STYLE["red"], lw=1.3, ls=ls,
                   label=f"Hist VaR {lvl:.0%}: {var:+.1%}")

    ax.set_xlabel("Trade P&L (%)")
    ax.set_ylabel("Density (per %)")
    ax.legend(fontsize=7.5, facecolor=_STYLE["bg"], labelcolor=_STYLE["text"],
              loc="upper left")


def _draw_qq(ax, pnl: np.ndarray, best: dict) -> None:
    """Q-Q plot of empirical quantiles vs the best-fit theoretical distribution."""
    _style_ax(ax, f"Q-Q Plot — {best['name']} (lowest AIC)")
    n     = len(pnl)
    probs = (np.arange(1, n + 1) - 0.5) / n
    emp   = np.sort(pnl) * 100
    theo  = best["dist"].ppf(probs, *best["params"]) * 100
    ax.scatter(theo, emp, color=_STYLE["accent"], s=35, alpha=0.85, zorder=3)
    lo, hi = min(theo.min(), emp.min()), max(theo.max(), emp.max())
    ax.plot([lo, hi], [lo, hi], color=_STYLE["amber"], lw=1.3, ls="--",
            label="Perfect fit")
    ax.set_xlabel("Theoretical quantiles (%)")
    ax.set_ylabel("Empirical quantiles (%)")
    ax.legend(fontsize=8, facecolor=_STYLE["bg"], labelcolor=_STYLE["text"])


def _draw_mc(ax, mc_samples: dict[float, np.ndarray],
             var_mc: dict[float, float], cvar_mc: dict[float, float],
             conf_levels: list[float], dist_name: str) -> None:
    """
    Monte Carlo next-trade outcome distribution.
    Samples are drawn from the best-fit distribution (lowest AIC).
    Uses the first confidence level's sample set for the histogram;
    marks VaR and CVaR lines for all requested confidence levels.
    """
    _style_ax(ax, f"Monte Carlo — Next-Trade P&L  [{dist_name}]")
    samples = mc_samples[conf_levels[0]]
    ax.hist(samples * 100, bins=100, density=True, color=_STYLE["purple"],
            alpha=0.40, label=f"{len(samples):,} draws from {dist_name}")

    p_loss = float((samples < 0).mean())
    exp_pnl = float(samples.mean())
    ax.text(0.97, 0.96,
            f"P(loss) = {p_loss:.1%}\nE[P&L] = {exp_pnl:+.2%}",
            ha="right", va="top", fontsize=8.5,
            color=_STYLE["text"], fontfamily="monospace",
            transform=ax.transAxes,
            bbox=dict(facecolor=_STYLE["ax_bg"], edgecolor=_STYLE["grid"],
                      boxstyle="round,pad=0.4"))

    ls_var  = ["--", "-.", ":"]
    ls_cvar = [":",  "--", "-."]
    clr_cyc = [_STYLE["amber"], _STYLE["red"]]
    for lvl, lsv, lsc, col in zip(conf_levels, ls_var, ls_cvar,
                                   clr_cyc + [_STYLE["red"]] * 10):
        ax.axvline(var_mc[lvl]  * 100, color=col, lw=1.5, ls=lsv,
                   label=f"VaR  {lvl:.0%}: {var_mc[lvl]:+.2%}")
        ax.axvline(cvar_mc[lvl] * 100, color=col, lw=1.5, ls=lsc,
                   label=f"CVaR {lvl:.0%}: {cvar_mc[lvl]:+.2%}")

    ax.axvline(0, color=_STYLE["text"], lw=0.9, alpha=0.6)
    ax.set_xlabel("Simulated next-trade P&L (%)")
    ax.set_ylabel("Density (per %)")
    ax.legend(fontsize=7.5, facecolor=_STYLE["bg"], labelcolor=_STYLE["text"])


# ── main analyser ─────────────────────────────────────────────────────────────

class RiskAnalyser:
    """
    Fit the trade P&L distribution, compute risk metrics, and plot results.

    Parameters
    ----------
    trades     : list of trade objects with a .pnl_pct attribute
    cfg_risk   : dict from the 'risk:' section of config.yaml
    save_dir   : directory for saving output charts
    strat_name : strategy name (used in filenames / titles)
    ticker     : ticker symbol (used in filenames / titles)
    """

    def __init__(self, trades, cfg_risk: dict,
                 save_dir: str, strat_name: str, ticker: str):
        done       = [t for t in trades if getattr(t, "pnl_pct", None) is not None]
        self._pnl  = np.array([t.pnl_pct for t in done], dtype=float)
        self._cfg  = cfg_risk
        self._dir  = save_dir
        self._sn   = strat_name
        self._tk   = ticker
        # populated by run()
        self._fits: list[dict]           = []
        self._res:  dict[str, dict]      = {}   # {method: {conf: (var, cvar)}}
        self._mc_samples: dict[float, np.ndarray] = {}

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> "RiskAnalyser":
        pnl = self._pnl
        if len(pnl) < 5:
            print("   [risk] Too few closed trades (need ≥ 5) — skipping.")
            return self
        if len(pnl) < 20:
            print(f"   [risk] Note: small sample (n={len(pnl)}) — "
                  "distribution estimates carry wide uncertainty.")

        dist_names = self._cfg.get("distributions", DEFAULT_DISTRIBUTIONS)
        confs      = [float(c) for c in self._cfg.get("confidence_levels", [0.95, 0.99])]
        n_sim      = int(self._cfg.get("n_simulations", 10_000))

        print("   Fitting distributions …")
        self._fits = fit_distributions(pnl, dist_names)
        if not self._fits:
            print("   [risk] All distribution fits failed — check the P&L data.")
            return self

        best = self._fits[0]
        self._res = {"historical": {}, "parametric": {}, "monte_carlo": {}}

        print("   Computing VaR / CVaR …")
        for conf in confs:
            hv, hc      = historical_var_cvar(pnl, conf)
            pv, pc      = parametric_var_cvar(best, conf)
            mv, mc, smp = monte_carlo_var_cvar(best, n_sim, conf)
            self._res["historical"][conf]   = (hv, hc)
            self._res["parametric"][conf]   = (pv, pc)
            self._res["monte_carlo"][conf]  = (mv, mc)
            self._mc_samples[conf]          = smp

        return self

    # ── report ────────────────────────────────────────────────────────────────

    def report(self) -> "RiskAnalyser":
        if not self._fits:
            return self

        pnl   = self._pnl
        confs = list(self._res["historical"].keys())

        print("── Distribution Fit ──────────────────────────────────────────")
        print(f"  {'Distribution':<12}  {'AIC':>9}  {'BIC':>9}  "
              f"{'KS stat':>8}  {'KS p-val':>9}  Rank")
        print("  " + "─" * 65)
        for i, fit in enumerate(self._fits):
            rank = f"  ★ best" if i == 0 else f"  #{i+1}"
            print(f"  {fit['name']:<12}  {fit['aic']:>9.2f}  {fit['bic']:>9.2f}  "
                  f"  {fit['ks_stat']:>6.4f}    {fit['ks_pval']:>7.4f}{rank}")

        best = self._fits[0]
        print(f"\n  Skewness: {float(st.skew(pnl)):+.3f}   "
              f"Excess kurtosis: {float(st.kurtosis(pnl)):+.3f}   "
              f"n = {len(pnl)} trades")

        n_sim = int(self._cfg.get("n_simulations", 10_000))
        print()
        print("── VaR & CVaR ────────────────────────────────────────────────")
        print(f"  Parametric and Monte Carlo both sample from: "
              f"{best['name']} (best-fit, lowest AIC)")
        print(f"  Monte Carlo: {n_sim:,} draws from fitted {best['name']}")
        print()
        col_w  = 22
        header = f"  {'Method':<28}"
        for c in confs:
            header += f"  VaR({c:.0%})   CVaR({c:.0%})"
        print(header)
        print("  " + "─" * (30 + col_w * len(confs)))
        labels = {
            "historical":  "Historical (empirical)",
            "parametric":  f"Parametric ({best['name']})",
            "monte_carlo": f"Monte Carlo ({best['name']}, {n_sim:,}×)",
        }
        for method in ("historical", "parametric", "monte_carlo"):
            row = f"  {labels[method]:<28}"
            for c in confs:
                var, cvar = self._res[method][c]
                row += f"  {var:>+7.2%}   {cvar:>+7.2%}"
            print(row)

        print(f"\n  Best-fit: {best['name']}  "
              f"(AIC {best['aic']:.2f} | KS p-val {best['ks_pval']:.4f})")
        print(f"  Trade P&L — mean: {pnl.mean():+.2%}  "
              f"std: {pnl.std():.2%}  "
              f"min: {pnl.min():+.2%}  max: {pnl.max():+.2%}")
        print()
        return self

    # ── plot ──────────────────────────────────────────────────────────────────

    def plot(self) -> "RiskAnalyser":
        if not self._fits:
            return self

        requested = self._cfg.get("plots",
                                   ["pnl_distribution", "qq_plot", "monte_carlo"])
        plot_keys = [k for k in ("pnl_distribution", "qq_plot", "monte_carlo")
                     if k in requested]
        if not plot_keys:
            return self

        pnl   = self._pnl
        confs = list(self._res["historical"].keys())

        var_hist  = {c: self._res["historical"][c][0]  for c in confs}
        var_mc    = {c: self._res["monte_carlo"][c][0] for c in confs}
        cvar_mc   = {c: self._res["monte_carlo"][c][1] for c in confs}

        n   = len(plot_keys)
        fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.5))
        fig.patch.set_facecolor(_STYLE["bg"])
        fig.suptitle(
            f"Risk Analysis  ·  {self._sn}  ·  {self._tk}",
            color=_STYLE["text"], fontsize=12, y=1.01,
        )
        if n == 1:
            axes = [axes]

        ax_iter = iter(axes)
        for key in plot_keys:
            ax = next(ax_iter)
            if key == "pnl_distribution":
                _draw_distribution(ax, pnl, self._fits, confs, var_hist)
            elif key == "qq_plot":
                _draw_qq(ax, pnl, self._fits[0])
            elif key == "monte_carlo":
                _draw_mc(ax, self._mc_samples, var_mc, cvar_mc, confs,
                         self._fits[0]["name"])

        plt.tight_layout()
        Path(self._dir).mkdir(parents=True, exist_ok=True)
        fname = f"{self._dir}/risk_{self._sn}_{self._tk}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
        print(f"  Risk chart → {fname}")
        plt.show()
        return self
