# Machine Learning in Finance

A collection of quantitative finance tools covering volatility analysis, options pricing, options strategies, and ML-based equity strategies.

---

## Structure

```
├── Volatility & Options
│   ├── iv_calculator.py          # Implied volatility from market option prices
│   ├── hv_calculator.py          # Historical volatility across multiple lookbacks
│   ├── options_strategies.py     # Vol strategies + gamma scalping simulation
│   ├── vol_strategy_selector.py  # End-to-end IV vs HV pipeline
│   └── spy_gamma_scalp.py        # Live yfinance runner for gamma scalping
│
├── Equity Strategies
│   ├── buy_strategies.py         # Pullback-to-MA long strategy
│   ├── sell_strategies.py        # Relief-rally short strategy
│   └── utils.py                  # Moving averages, candlestick charts
│
└── Machine Learning
    ├── models.py                 # Deep neural network (PyTorch)
    ├── regression_models.py      # Polynomial, PCA, ElasticNet, XGBoost
    └── trainer.py                # Training pipeline (OpenBB + sklearn + torch)
```

Notebooks (`*.ipynb`) contain exploratory analysis, MPT, backtests, and assignment work.

---

## Volatility & Options

### Core idea

The **volatility risk premium (VRP)** is the gap between implied vol (IV) and historical vol (HV):

```
VRP = IV − HV
  > 0  →  options expensive  →  sell vol
  < 0  →  options cheap      →  buy vol
```

### iv_calculator.py

Computes annualised IV from a market option price using Newton-Raphson (with Brent fallback).

```bash
python3 iv_calculator.py --stock 100 --strike 100 --call_price 3.50 --maturity 0.082 --risk_free 0.05 --window 20
```

### hv_calculator.py

Estimates HV from 1m / 3m / 6m / 1y lookbacks, all scaled to the same target window W via `σ_W = σ_daily × √W`. Flags anomalies (recent vol spike vs long-run level).

```bash
python3 hv_calculator.py --ticker SPY --window 20 --iv 0.22
```

### options_strategies.py

Five strategies auto-selected by the VRP signal:

| Signal | Strategy |
|---|---|
| IV >> HV | `ShortStraddle` — sell ATM call + put |
| IV >> HV | `ShortStrangle` — sell OTM call + put |
| IV << HV | `LongStraddle` — buy ATM call + put |
| IV << HV | `LongStrangle` — buy OTM call + put |
| Either | `GammaScalping` — single ATM option + daily delta hedge |

Each strategy exposes `.describe()`, `.greeks()`, `.pnl_at_expiry()`, and `.plot()`.

#### GammaScalping

Holds a single ATM call (or put) and rebalances a delta hedge at end of each trading day:

```
IV > HV  →  SHORT gamma: sell option, delta-hedge daily
            Daily P&L ≈ +Θ·Δt − ½Γ·(ΔS)²   (earn theta, pay on big moves)

IV < HV  →  LONG gamma:  buy option, delta-hedge daily
            Daily P&L ≈ −Θ·Δt + ½Γ·(ΔS)²   (pay theta, earn on big moves)
```

All prices and P&L are **per share**. Multiply by 100 for one standard equity contract.

```python
from options_strategies import GammaScalping

gs = GammaScalping(
    stock=100, strike=100, risk_free=0.05,
    iv=0.25, hv=0.18, maturity=21/252,
    option_type="call",   # or "put"
    closes=closes,        # historical closes used as the realized path
)
gs.describe()
gs.plot()
```

### vol_strategy_selector.py

Full pipeline: market price → IV → HV estimates → VRP score → strategy selection + gamma scalp.

```bash
python3 vol_strategy_selector.py --ticker SPY --stock 520 --strike 520 \
    --call_price 8.50 --maturity 0.082 --risk_free 0.05 --window 20
```

### spy_gamma_scalp.py

Live runner that fetches data from yfinance, picks the ATM option nearest to a target expiry, and runs the full pipeline.

```bash
# Auto-select signal from IV vs HV
python3 spy_gamma_scalp.py --ticker SPY

# Force long gamma on TSLA regardless of VRP
python3 spy_gamma_scalp.py --ticker TSLA --mode long

# Force short gamma on QQQ using a put
python3 spy_gamma_scalp.py --ticker QQQ --mode short --option_type put

# Target a specific expiry window
python3 spy_gamma_scalp.py --ticker NVDA --target_days 21
```

---

## Equity Strategies

### buy_strategies.py — `PullbackStrategy`

Long signal: stock in uptrend (MA20 > MA50, higher lows over 63 days) pulling back to the 20 or 50 MA with stochastic oversold confirmation.

### sell_strategies.py — `ReliefRallyShortStrategy`

Short signal: stock in downtrend (MA20 < MA50, lower highs) bouncing into the 20 or 50 MA — fade the relief rally.

---

## Machine Learning

### models.py — `DNN`

Feedforward neural network in PyTorch for return prediction. Configurable depth and hidden size.

### regression_models.py

Sklearn pipeline wrappers: polynomial regression, PCA dimensionality reduction, ElasticNet regularisation, and XGBoost. Used for factor-based return forecasting.

### trainer.py

End-to-end training script: pulls OHLCV data via OpenBB, engineers features, trains and evaluates all models.

---

## Installation

```bash
pip install numpy scipy matplotlib yfinance scikit-learn xgboost torch openbb mplfinance
```

---

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib` — core maths and plotting
- `yfinance` — live market data for options and prices
- `scikit-learn`, `xgboost`, `torch` — ML models
- `openbb` — data pipeline in trainer.py
- `mplfinance` — candlestick charts in utils.py
