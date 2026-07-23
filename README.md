# Alpha Research Lab

A research-oriented Python framework for constructing, validating, backtesting, and monitoring systematic equity factors.

> **Project status:** active development.
> The data, factor-research, backtesting, and neutralisation layers are implemented. Multi-factor construction, additional signals, and monitoring tools are planned.

The project follows an end-to-end workflow:

```text
Data
→ Factor construction
→ Signal validation
→ Portfolio construction
→ Backtesting
→ Exposure analysis
→ Risk controls
```

The current implementation focuses on daily US large-cap equities, using the present-day S&P 100 as the research universe and SPY as the market benchmark.

---

## Research objective

The project is designed to answer a practical question:

> How does a statistically promising factor behave after realistic portfolio construction, trading costs, and systematic risk exposures are considered?

Rather than stopping at factor correlations, the framework tests each signal through progressively stricter stages:

1. Does the factor predict future cross-sectional returns?
2. Is the relationship stable across horizons and subperiods?
3. Does it survive a rebalance-based backtest?
4. How much performance comes from market beta or sector allocation?
5. What remains after neutralisation?

---

## Current workflow

### 1. Data acquisition and validation

The pipeline downloads daily equity and benchmark data, standardises it into a tidy panel, and checks:

* missing dates and tickers;
* non-positive prices;
* extreme returns and price gaps;
* available history by ticker;
* sector coverage;
* benchmark alignment.

The processed equity panel includes adjusted prices, returns, forward returns, volume, dollar volume, and security metadata.

### 2. Factor construction

The initial factor library includes:

* **12–1 month momentum**
* **3-month momentum**
* **1-month reversal**
* **63-day realised volatility**

Raw factors are winsorised and transformed into:

* cross-sectional z-scores;
* percentile ranks;
* sector-neutral z-scores.

### 3. Signal validation

Factors are evaluated using:

* Pearson and Spearman information coefficients;
* IC summary statistics;
* factor-return quantiles;
* top-minus-bottom quantile spreads;
* 1-day and 5-day forward-return horizons;
* subperiod analysis;
* non-overlapping forward-return samples;
* rolling IC.

### 4. Portfolio backtesting

The backtester supports:

* configurable rebalance frequency and offset;
* equal-weight long and short quantiles;
* persistent holdings between rebalances;
* daily portfolio returns;
* long- and short-leg decomposition;
* turnover;
* transaction costs;
* cumulative returns;
* Sharpe ratio and drawdown analysis;
* subperiod and rebalance-offset robustness tests.

### 5. Exposure diagnosis and neutralisation

The risk layer measures:

* market beta and alpha;
* rolling beta;
* benchmark correlation and (R^2);
* sector exposures;
* long- and short-book exposures.

The framework currently supports:

* sector-neutral factor scores;
* rolling stock-beta estimates;
* SPY-based beta hedging;
* combined sector- and beta-neutral strategies.

---

## Current findings

The initial research produced two main factor candidates.

### Realised volatility

The raw realised-volatility strategy was the strongest initial long-short result:

| Metric                | Raw strategy |
| --------------------- | -----------: |
| Annualised return     |        15.8% |
| Annualised volatility |        24.0% |
| Sharpe ratio          |         0.73 |
| Maximum drawdown      |       −45.3% |
| Realised market beta  |         0.82 |

However, exposure analysis showed that much of the result was associated with:

* positive market beta;
* a large long Technology position;
* short exposure to defensive sectors.

Sector neutralisation produced a more balanced result:

| Metric                | Sector-neutral strategy |
| --------------------- | ----------------------: |
| Annualised return     |                   11.8% |
| Annualised volatility |                   15.4% |
| Sharpe ratio          |                    0.80 |
| Maximum drawdown      |                  −27.7% |
| Realised market beta  |                    0.48 |

This was the strongest risk-adjusted version tested so far.

A fully sector- and beta-neutral version retained positive performance, but at a materially lower return:

| Metric                | Sector + beta neutral |
| --------------------- | --------------------: |
| Annualised return     |                  3.6% |
| Annualised volatility |                 12.5% |
| Sharpe ratio          |                  0.34 |
| Maximum drawdown      |                −26.2% |
| Realised market beta  |                 −0.06 |

The main lesson is that the raw result contained both stock-selection information and substantial systematic exposure.

### 12–1 month momentum

The factor showed positive and reasonably stable IC, but its symmetric long-short portfolio was weak:

* the highest-momentum stocks performed well;
* the lowest-momentum stocks were ineffective short candidates;
* most useful performance came from the long book.

This suggests momentum may be more suitable as a long-side ranking or portfolio-tilt signal than as a simple top-minus-bottom strategy.

---

## Project structure

```text
alpha-research-lab/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_analysis.ipynb
│   └── 03_backtest_analysis.ipynb
├── scripts/
│   ├── download_data.py
│   ├── build_processed_panel.py
│   ├── build_factor_panel.py
│   └── run_factor_backtests.py
├── src/
│   └── alpha_research/
│       ├── config/
│       ├── universe.py
│       ├── data_loader.py
│       ├── data_checks.py
│       ├── returns.py
│       ├── factors.py
│       ├── signal_processing.py
│       ├── validation.py
│       ├── backtest.py
│       └── risk.py
├── tests/
├── reports/
├── docs/
├── pyproject.toml
└── requirements.txt
```

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/THUdzc14/alpha-research-lab.git
cd alpha-research-lab

python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and the local package:

```powershell
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```powershell
pytest -v
```

---

## Running the research pipeline

Download the data:

```powershell
python scripts/download_data.py
```

Build the processed equity panel:

```powershell
python scripts/build_processed_panel.py
```

Construct and process factors:

```powershell
python scripts/build_factor_panel.py
```

Run the initial factor backtests:

```powershell
python scripts/run_factor_backtests.py
```

The notebooks provide the main exploratory and diagnostic analysis.

---

## Research principles

The project follows several principles:

* factor definitions should be economically interpretable;
* signal validation should precede portfolio backtesting;
* overlapping forward returns should be treated cautiously;
* transaction costs and turnover should be included;
* long and short books should be analysed separately;
* dollar neutrality should not be confused with beta neutrality;
* systematic exposures should be measured before results are described as alpha;
* negative and inconclusive findings should be retained.

---

## Known limitations

The current framework has several important limitations:

* the historical universe uses current S&P 100 constituents and therefore contains survivorship bias;
* the universe is not reconstructed point in time;
* borrow availability and short-borrow fees are not modelled;
* transaction costs use a simplified fixed-basis-point assumption;
* market impact is not modelled;
* delisting returns are not included;
* beta is estimated using a single-market-factor rolling model;
* sector classifications are treated as historically constant;
* results are in-sample research results rather than live or fully out-of-sample performance.

These limitations are documented rather than hidden and will guide future improvements.

---

## Roadmap

Planned next steps include:

* liquidity and illiquidity factors;
* rolling beta as a standalone factor;
* idiosyncratic volatility;
* risk-adjusted momentum;
* factor correlation and redundancy analysis;
* multi-factor score construction;
* constrained portfolio optimisation;
* improved transaction-cost modelling;
* point-in-time universe data;
* walk-forward and out-of-sample validation;
* portfolio and factor monitoring;
* automated research reports.

---

## Documentation

Additional notes are available in:

* [`docs/progress_report.md`](docs/progress_report.md)
* [`docs/methodology.md`](docs/methodology.md)

---

## Disclaimer

This repository is an educational and research project. It is not investment advice, and the results should not be interpreted as evidence of future investment performance.
