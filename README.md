# Systematic Alpha Research Framework

Factor construction, signal validation, backtesting, portfolio construction, and risk monitoring for a daily US large-cap equity universe.

## Project objective

This is a personal quant research portfolio project. The goal is not to claim a live profitable strategy, but to build a clean and reproducible alpha research workflow:

```text
Data -> Factors -> Signal validation -> Portfolio construction -> Backtesting -> Risk monitoring -> Report
```

## Initial scope

- Universe: S&P 100 current constituents, with SPY as benchmark
- Frequency: daily
- Data source: `yfinance`
- Storage: local Parquet files
- Initial period: 2015-01-01 to latest available date
- MVP strategy type: long-short cross-sectional factor portfolios
- Rebalancing: weekly/monthly
- Costs: simplified transaction-cost model

## Important limitations

This MVP uses public data and a current index universe. This introduces important limitations:

- survivorship bias from using current constituents;
- possible missing or inconsistent historical data from public sources;
- simplified transaction-cost assumptions;
- no borrow costs, shorting constraints, market impact, or execution model;
- no live trading or broker integration.

These limitations are intentional for the MVP and should be documented in any results.

## Repository structure

```text
alpha-research-lab/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
├── src/alpha_research/
│   ├── config/
│   ├── data_loader.py
│   ├── universe.py
│   ├── factors.py
│   ├── signal_processing.py
│   ├── validation.py
│   ├── portfolio.py
│   ├── backtester.py
│   ├── costs.py
│   ├── metrics.py
│   ├── monitoring.py
│   └── visualisation.py
├── scripts/
├── reports/
├── tests/
└── dashboard/
```

## Setup

```bash
git clone https://github.com/THUdzc14/alpha-research-lab.git
cd alpha-research-lab

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## First data download

```bash
python scripts/download_data.py --universe sp100 --start 2015-01-01
```

This creates:

```text
data/raw/sp100_prices.parquet
data/raw/spy_benchmark.parquet
data/raw/sp100_universe.csv
```

## Current roadmap

### Week 1 — Project setup and data pipeline

- [x] Repository skeleton
- [x] Requirements file
- [x] Universe loader
- [x] Data download script
- [ ] Data validation checks
- [ ] Adjusted return panel

### Week 2 — Factor library

- [ ] Momentum factors
- [ ] Reversal factor
- [ ] Realised volatility
- [ ] Downside volatility
- [ ] Liquidity proxy
- [ ] Rolling beta

### Week 3 — Signal validation

- [ ] Rank IC
- [ ] Pearson IC
- [ ] Quantile returns
- [ ] Rolling IC
- [ ] Factor decay
- [ ] Factor correlations

## Design principles

- Shift signals before applying returns.
- Separate signal validation from portfolio backtesting.
- Include turnover and transaction costs.
- Prefer simple, interpretable factors first.
- Analyse instability and failed results honestly.
- Document assumptions and limitations.
