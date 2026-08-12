# Alpha Research Lab

An end-to-end research framework for constructing, testing, combining, attributing and monitoring systematic equity factors.

> **Project status:** the core research workflow is complete.  
> The next stage is to extract reusable analytics into `src/alpha_research/` and build a Streamlit research and monitoring dashboard.

The project follows a deliberately layered workflow:

```text
Data quality
→ Factor construction
→ Signal validation
→ Portfolio backtesting
→ Multi-factor construction
→ Portfolio optimisation
→ Robustness and capacity
→ Performance and risk attribution
→ Monitoring
→ Final strategy decision
```

The current implementation uses daily US large-cap equity data, the present-day S&P 100 as the research universe and SPY as the market benchmark.

---

## Research objective

The project addresses a practical research question:

> How does a statistically promising factor behave after portfolio construction, transaction costs, robustness testing and systematic risk exposures are considered?

The workflow is designed to distinguish:

- predictive signal quality from portfolio performance;
- gross performance from implementable net performance;
- stock-selection effects from market and sector exposures;
- full-sample results from phase, subperiod and rolling robustness;
- nominal diversification from risk and contribution diversification; and
- useful complexity from optimisation that does not survive out of sample.

---

## Research scope

| Item | Current scope |
|---|---|
| Universe | 101 present-day S&P 100 securities |
| Equity data | 2 January 2015 to 2 July 2026 |
| Final portfolio sample | 7 January 2016 to 1 July 2026 |
| Benchmark | SPY |
| Return frequency | Daily |
| Primary forward horizon | 5 trading days for signal validation |
| Portfolio construction | Equal-weight long and short quintiles |
| Baseline transaction cost | 10 bps per unit of turnover |
| Final candidates | Composite Score, Fixed 50/50 Sleeves and Pure Inverse Volatility |

The use of current index constituents introduces survivorship bias. Results are research evidence rather than production or investable performance.

---

## Retained factors

The initial factor library included:

- 12–1 month momentum;
- 3-month momentum;
- 1-month reversal; and
- 63-day realised volatility.

After predictive validation, portfolio testing and redundancy analysis, two factors were retained.

| Factor | Mean 5-day rank IC | IC t-statistic | Research role |
|---|---:|---:|---|
| 12–1 month momentum | 0.0197 | 3.65 | Complementary trend signal |
| 63-day realised volatility | 0.0253 | 4.61 | Strongest standalone portfolio signal |

Additional candidates—including idiosyncratic volatility, liquidity and risk-adjusted momentum—were investigated but not retained because they were redundant, unstable or insufficiently predictive in the current framework.

A central finding is that signal quality and portfolio quality are different. Momentum has positive cross-sectional predictive information, but its lowest-ranked securities form a weak short basket. Realised volatility translates more strongly into portfolio returns, although much of its raw performance is associated with market beta and sector exposure.

---

## Final portfolio hierarchy

### Primary implementation

**Composite Score, rebalanced every 21 trading days**

The strategy averages the processed momentum and realised-volatility scores before ranking securities. It is retained because it has:

- the highest historical return among the final candidates;
- the highest candidate Sharpe ratio;
- the strongest transaction-cost resilience;
- positive results across all rebalance phases;
- the highest median rolling Sharpe ratio; and
- the lowest average turnover.

### Defensive alternative

**Pure Inverse Volatility, rebalanced every 10 trading days**

The strategy combines independent momentum and realised-volatility sleeves using inverse trailing sleeve volatility. It provides:

- the lowest whole-sample volatility;
- the least severe maximum drawdown;
- the strongest result during 2019–2022; and
- the best lower-tail rolling stability.

### Transparent benchmark

**Fixed 50/50 Sleeves, rebalanced every 10 trading days**

The portfolio gives equal capital allocations to the two independent factor sleeves. It remains the clearest reference for evaluating whether dynamic sleeve allocation adds value.

All final implementations use offset zero and transaction costs of 10 basis points per unit of turnover.

---

## Final historical results

| Portfolio | Net ann. return | Ann. volatility | Sharpe | Max drawdown | Mean daily turnover |
|---|---:|---:|---:|---:|---:|
| Composite Score | 16.13% | 21.17% | 0.813 | −29.24% | 5.09% |
| Fixed 50/50 Sleeves | 10.84% | 16.77% | 0.698 | −25.68% | 6.24% |
| Pure Inverse Volatility | 10.96% | 16.25% | 0.722 | −21.77% | 6.60% |
| SPY context | 15.55% | 17.83% | 0.900 | −33.72% | — |

Composite Score slightly exceeds SPY’s annualised return but not its Sharpe ratio. SPY is a long-only contextual benchmark rather than an exposure-matched alternative.

---

## Robustness findings

### Rebalance phases

Every rebalance phase at the selected frequencies produces a positive annualised return and Sharpe ratio.

Composite Score’s phase-averaged results are:

- annualised return: 15.84%;
- Sharpe ratio: 0.802;
- worst-phase annualised return: 13.46%; and
- worst-phase Sharpe ratio: 0.701.

Its selected offset-zero result is close to its phase average and is therefore not driven by a favourable rebalance date.

### Transaction costs

The candidate hierarchy is unchanged across costs of 0, 5, 10, 20 and 50 basis points per unit of turnover.

At 50 basis points, phase-averaged annualised returns remain positive:

| Portfolio | Ann. return | Sharpe |
|---|---:|---:|
| Composite Score | 10.00% | 0.556 |
| Fixed 50/50 Sleeves | 3.83% | 0.308 |
| Pure Inverse Volatility | 3.99% | 0.321 |

### Subperiods

Performance is positive but materially regime-dependent.

| Subperiod | Composite | Fixed 50/50 | Pure inverse volatility |
|---|---:|---:|---:|
| 2016–2018 | 7.65% | 3.59% | 3.05% |
| 2019–2022 | 2.86% | 1.67% | 5.32% |
| 2023–present | 41.45% | 28.56% | 26.02% |

The strong post-2022 period contributes substantially to the full-sample results. None of the strategies should be described as uniformly strong across market regimes.

### Capacity

Under a 1% maximum participation assumption, worst-phase fifth-percentile capacity is estimated at:

- Composite Score: $24.8 million;
- Fixed 50/50 Sleeves: $40.2 million; and
- Pure Inverse Volatility: $38.3 million.

These are scenario-based research estimates, not deployable capital limits.

---

## Optimisation findings

Walk-forward global minimum-variance allocation produced stable, moderate sleeve weights but did not improve materially on simpler allocations.

Maximum-Sharpe mean-variance optimisation was rejected because:

- 61.7% of allocations were boundary solutions;
- estimated and realised sleeve-return spreads were effectively uncorrelated;
- the return-ranking hit rate was 48.3%;
- MVO underperformed equal weighting by 14.3 bps per allocation period; and
- increasing MVO intensity monotonically reduced return and Sharpe while increasing turnover and drawdown.

The result illustrates a broader lesson:

> A more sophisticated allocation method is useful only when its additional estimates contain reliable information.

---

## Attribution and monitoring

The completed attribution layer covers:

- portfolio, sleeve and long/short performance;
- transaction-cost contributions;
- holdings-implied and realised market beta;
- sector exposures;
- rolling volatility and drawdowns;
- security and sector return contributions;
- position and beta-contribution concentration;
- turnover and security-level trade capacity; and
- missing-return exposure.

The latest monitoring state is **WARNING**, not **BREACH**.

As of 1 July 2026:

| Portfolio | Holdings-implied beta | 126-day volatility | Largest absolute sector net exposure |
|---|---:|---:|---:|
| Composite Score | 1.83 | 36.7% | 67.9% |
| Fixed 50/50 Sleeves | 1.41 | 30.1% | 44.9% |
| Pure Inverse Volatility | 1.41 | 30.2% | 44.9% |

Signal health and implementation diagnostics pass. The warnings arise from elevated beta, volatility and concentrated sector, beta and realised-return contributions.

The selected portfolios are long/short by capital construction, but they are not market-neutral or sector-neutral.

---

## Notebook workflow

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Data coverage, quality and universe diagnostics |
| `02_factor_analysis.ipynb` | Factor construction and predictive validation |
| `03_backtest_analysis.ipynb` | Standalone factor backtests and exposure experiments |
| `04_portfolio_construction.ipynb` | Composite scores and independent factor sleeves |
| `05_portfolio_optimisation.ipynb` | GMV, MVO and shrinkage experiments |
| `06_portfolio_robustness.ipynb` | Frequency, costs, phases, subperiods and capacity |
| `07_performance_and_short_side_attribution.ipynb` | Performance, risk, contribution and short-side attribution |
| `08_strategy_monitoring_and_diagnostics.ipynb` | Signal, risk, concentration and implementation monitoring |
| `09_final_strategy_assessment.ipynb` | Final evidence synthesis and strategy hierarchy |

---

## Project structure

```text
alpha-research-lab/
├── dashboard/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── methodology.md
│   └── progress_report.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_analysis.ipynb
│   ├── 03_backtest_analysis.ipynb
│   ├── 04_portfolio_construction.ipynb
│   ├── 05_portfolio_optimisation.ipynb
│   ├── 06_portfolio_robustness.ipynb
│   ├── 07_performance_and_short_side_attribution.ipynb
│   ├── 08_strategy_monitoring_and_diagnostics.ipynb
│   ├── 09_final_strategy_assessment.ipynb
│   └── experiments/
├── scripts/
│   ├── download_data.py
│   ├── build_processed_panel.py
│   ├── build_factor_panel.py
│   └── run_factor_backtests.py
├── src/
│   └── alpha_research/
│       ├── config/
│       ├── backtest.py
│       ├── costs.py
│       ├── data_checks.py
│       ├── data_loader.py
│       ├── factors.py
│       ├── metrics.py
│       ├── monitoring.py
│       ├── portfolio.py
│       ├── returns.py
│       ├── risk.py
│       ├── signal_processing.py
│       ├── universe.py
│       ├── validation.py
│       └── visualisation.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

The next engineering stage will move reusable calculations out of notebooks and into the placeholder modules before the dashboard is completed.

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

## Running the data pipeline

```powershell
python scripts/download_data.py
python scripts/build_processed_panel.py
python scripts/build_factor_panel.py
python scripts/run_factor_backtests.py
```

The notebooks currently contain the complete research workflow. Dashboard execution instructions will be added after reusable analytics have been extracted into the package.

---

## Research principles

The project follows several principles:

- economic interpretation should precede optimisation;
- signal validation should precede portfolio selection;
- overlapping forward returns should be treated cautiously;
- turnover and transaction costs should be included;
- long and short books should be analysed separately;
- dollar neutrality should not be confused with beta neutrality;
- nominal position counts should not be confused with risk diversification;
- robustness should be tested across phases, costs, subperiods and rolling windows;
- negative and inconclusive results should be retained;
- complexity should be accepted only when it adds realised value; and
- monitoring warnings should not automatically trigger in-sample strategy changes.

---

## Main limitations

The most important limitations are:

- use of present-day index constituents and resulting survivorship bias;
- no point-in-time sector classifications or delisting returns;
- simplified linear transaction costs;
- incomplete modelling of borrow fees, availability, recalls and financing;
- no nonlinear market-impact model;
- capacity estimates based on historical ADV and fixed participation limits;
- no explicit beta- or sector-neutrality constraint in the final portfolios;
- material dependence on the post-2022 period;
- in-sample research and model-selection effects; and
- no live, paper-trading or genuinely unseen forward evaluation.

See [`docs/methodology.md`](docs/methodology.md) for detailed assumptions.

---

## Next stage

The next stage is an engineering and presentation phase:

1. extract reusable cost, metric, monitoring and visualisation functions from the notebooks;
2. add unit tests and reconciliation tests for those functions;
3. define a stable data-access layer for versioned research exports;
4. build the Streamlit dashboard;
5. update the README with dashboard instructions and screenshots; and
6. add a forward or paper-trading workflow without changing the completed historical specification.

---

## Documentation

- [`docs/methodology.md`](docs/methodology.md)
- [`docs/progress_report.md`](docs/progress_report.md)

---

## Disclaimer

This repository is an educational and research project. It is not investment advice, and its historical results should not be interpreted as evidence of future investment performance.