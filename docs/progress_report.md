# Progress Report

## Overview

The Alpha Research Lab was created to build a complete systematic-equity research workflow rather than a collection of isolated factor notebooks.

The project now connects:

```text
data quality
→ factor construction
→ predictive validation
→ portfolio implementation
→ multi-factor allocation
→ robustness and capacity
→ attribution
→ monitoring
→ final strategy selection
```

The core research phase is complete. The project has selected a primary historical specification, retained transparent benchmarks, documented rejected methods and exported reusable evidence for the next engineering stage.

---

## Milestone 1 — Data foundation

The first milestone created a validated daily equity panel for the present-day S&P 100 universe and SPY.

The work included:

- downloading and standardising price and volume data;
- calculating backward and forward returns;
- joining security and sector metadata;
- checking missing dates and incomplete histories;
- detecting non-positive prices and extreme gaps;
- aligning security and benchmark dates; and
- preserving missing-return diagnostics.

An early lesson was that data availability and historical index membership are different. A security can have a complete price history without having belonged to the index throughout that history.

The current-universe approach was retained as a research simplification, with survivorship bias documented explicitly.

---

## Milestone 2 — Factor construction and validation

The initial factor library contained:

- 12–1 month momentum;
- three-month momentum;
- one-month reversal; and
- 63-day realised volatility.

Each economic definition was separated from subsequent signal processing.

Raw values were:

- winsorised;
- cross-sectionally standardised;
- converted to percentile ranks; and
- optionally standardised within sectors.

Validation included:

- Pearson and Spearman IC;
- IC t-statistics;
- positive-IC frequency;
- quantile returns;
- top-minus-bottom spreads;
- multiple forward horizons;
- non-overlapping samples;
- subperiods; and
- rolling IC.

The two retained factors were:

| Factor | Mean 5-day rank IC | IC t-statistic |
|---|---:|---:|
| 12–1 month momentum | 0.0197 | 3.65 |
| 63-day realised volatility | 0.0253 | 4.61 |

The main lesson was:

> A statistically useful factor is not automatically a useful portfolio.

Momentum showed positive predictive ranking information, but its lowest-ranked securities were weak short candidates. Realised volatility translated more successfully into a symmetric long-short portfolio.

---

## Milestone 3 — Standalone backtesting and exposure diagnosis

A rebalance-based backtester was implemented with:

- persistent holdings;
- configurable frequencies and offsets;
- equal-weight long and short books;
- daily gross and net returns;
- turnover and transaction costs;
- drawdown calculations;
- long- and short-side decomposition; and
- missing-return accounting.

A bug involving the final panel date was identified during this stage. A missing next-day return had initially created an apparent full missing-return exposure. The affected terminal date was removed rather than interpreted as a valid zero return.

Exposure analysis showed that dollar neutrality did not produce beta neutrality.

The raw realised-volatility portfolio combined:

- stock-selection information;
- positive market beta;
- a large long Technology exposure; and
- short exposure to defensive sectors.

Sector- and beta-neutral experiments were useful diagnostics, but beta hedging reduced returns materially.

The lesson was:

> Neutralisation is an experiment that reveals return sources; it is not automatically an improvement.

---

## Milestone 4 — Additional factor experiments

Several additional factors were investigated.

### Idiosyncratic volatility

Idiosyncratic volatility showed positive signal evidence but was highly redundant with realised volatility. It was not retained as a distinct factor.

### Liquidity

The tested liquidity specification had weak or negative recent evidence and was not retained.

### Risk-adjusted momentum

Risk-adjusted momentum did not improve sufficiently on the simpler momentum definition.

### Low beta

Low-beta and betting-against-beta-style experiments revealed unstable leverage and estimation sensitivity. They were not added to the final factor set.

These negative results were retained as part of the research record rather than removed from the project narrative.

---

## Milestone 5 — Multi-factor portfolio construction

The retained momentum and realised-volatility factors were combined using several methods.

### Fixed 50/50 sleeves

Independent factor portfolios were allocated equal capital weights and netted at security level.

The method was simple and transparent. Factor disagreement reduced realised gross exposure and drawdown.

### Composite Score

Processed factor scores were averaged before security selection.

The method allowed factor disagreement to affect rankings rather than cancelling completed sleeve positions. It produced stronger returns and lower turnover, but also greater gross exposure and risk.

### Dynamic and pure inverse-volatility sleeves

Trailing sleeve volatility was used to form risk-based allocations.

Pure inverse-volatility weighting produced the strongest defensive profile among the retained sleeve methods.

The portfolio-construction stage showed that:

> A more sophisticated portfolio is not automatically a better portfolio.

Simple allocation rules remained highly competitive.

---

## Milestone 6 — Portfolio optimisation

Walk-forward global minimum-variance and maximum-Sharpe MVO were tested.

### Global minimum variance

GMV produced moderate sleeve weights and lower estimated variance, but did not establish a decisive improvement over simple risk-based allocations.

### Maximum-Sharpe MVO

The MVO expected-return estimates failed their predictive diagnostics:

- 61.7% of solutions were boundary allocations;
- estimated and realised return spreads were effectively uncorrelated;
- the return-ranking hit rate was 48.3%;
- realised MVO-minus-equal performance averaged −14.3 bps per allocation period; and
- diagnostic annualised return was 4.7%, versus 13.5% for equal weighting.

Increasing MVO intensity progressively reduced return and Sharpe ratio, increased turnover and deepened drawdowns.

Maximum-Sharpe MVO was therefore rejected.

---

## Milestone 7 — Frequency, cost and capacity robustness

The selected portfolio families were tested across:

- daily, 5-day, 10-day and 21-day rebalancing;
- every valid rebalance offset;
- transaction costs of 0–50 basis points;
- three predefined subperiods;
- rolling 252-day windows;
- turnover concentration; and
- security-level ADV capacity.

The final frequencies were:

- Composite Score: 21 trading days;
- Fixed 50/50 Sleeves: 10 trading days; and
- Pure Inverse Volatility: 10 trading days.

### Cost sensitivity

Composite Score retained the highest return and Sharpe ratio at every tested cost.

At 50 basis points, all selected candidates still produced positive phase-averaged returns.

### Subperiod stability

| Subperiod | Composite | Fixed 50/50 | Pure inverse volatility |
|---|---:|---:|---:|
| 2016–2018 | 7.65% | 3.59% | 3.05% |
| 2019–2022 | 2.86% | 1.67% | 5.32% |
| 2023–present | 41.45% | 28.56% | 26.02% |

The full-sample result is materially influenced by the post-2022 period.

### Capacity

All selected portfolios had complete lagged ADV coverage.

At a 1% participation limit, worst-phase fifth-percentile capacity ranged from approximately $24.8 million to $40.2 million.

---

## Milestone 8 — Performance and risk attribution

The attribution notebook reconciled portfolio accounting and analysed:

- gross and net performance;
- factor sleeves;
- long and short books;
- transaction costs;
- market exposure;
- rolling risk;
- drawdowns;
- security contributions;
- sector contributions; and
- concentration.

The analysis confirmed that current portfolio risk is not explained well by nominal position counts alone.

Although the strategies hold many securities, beta and recent return contributions are concentrated among a much smaller group of names and sectors.

The short baskets also provide limited beta offset, particularly for Composite Score.

---

## Milestone 9 — Monitoring and diagnostics

A reusable monitoring framework was designed around four categories:

1. signal health;
2. market risk;
3. concentration; and
4. implementation.

The framework distinguishes:

- structural breaches;
- unavailable diagnostics;
- historically calibrated warnings; and
- passing conditions.

Both retained factors pass their latest signal-health checks.

All implementations pass:

- turnover diagnostics;
- transaction-cost checks;
- liquidity coverage;
- capacity checks; and
- missing-return controls.

All three portfolios nevertheless receive market-risk and concentration warnings.

As of 1 July 2026:

| Portfolio | Holdings-implied beta | 126-day volatility | Largest sector imbalance |
|---|---:|---:|---:|
| Composite Score | 1.83 | 36.7% | 67.9% |
| Fixed 50/50 Sleeves | 1.41 | 30.1% | 44.9% |
| Pure Inverse Volatility | 1.41 | 30.2% | 44.9% |

The warning cluster reflects:

- elevated beta;
- elevated volatility;
- concentrated sector exposure;
- concentrated beta contributions;
- concentrated security contributions; and
- a low effective number of contributing sectors.

The strategy definitions were not changed in response because doing so would create a new in-sample optimisation cycle.

---

## Milestone 10 — Final strategy assessment

All three frozen candidates passed their eligibility gates.

### Final hierarchy

1. **Composite Score — primary implementation**
2. **Pure Inverse Volatility — defensive risk-based alternative**
3. **Fixed 50/50 Sleeves — transparent allocation benchmark**

### Final historical results

| Portfolio | Net ann. return | Ann. volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Composite Score | 16.13% | 21.17% | 0.813 | −29.24% |
| Fixed 50/50 Sleeves | 10.84% | 16.77% | 0.698 | −25.68% |
| Pure Inverse Volatility | 10.96% | 16.25% | 0.722 | −21.77% |
| SPY context | 15.55% | 17.83% | 0.900 | −33.72% |

Composite Score was selected because it combined:

- the highest candidate return and Sharpe ratio;
- strong rebalance-phase robustness;
- the best transaction-cost resilience;
- positive performance in every subperiod; and
- the lowest average turnover.

The decision remains qualified by:

- higher beta and volatility;
- greater sector concentration;
- weaker rolling lower-tail stability than Pure Inverse Volatility; and
- substantial dependence on the post-2022 period.

---

## Milestone 11 — Reusable engineering and artifact refresh

The completed research logic was moved into reusable, tested modules under `src/alpha_research/`.

The engineering layer now provides:

- explicit artifact contracts and validation;
- portfolio, security, attribution, risk, signal, implementation, and liquidity analytics;
- deterministic reconstruction of six attribution datasets and nine monitoring datasets;
- dry-run comparison of reconstructed and stored artifacts;
- optional artifact writing followed by read-back validation;
- dashboard loading with separate structural-readiness and freshness checks.

The complete refresh workflow is exposed through `scripts/refresh_strategy_outputs.py`.

All 15 dashboard artifacts reconcile on columns, keys, values, and row counts against the stored research outputs.

---

## Milestone 12 — Streamlit dashboard and final quality assurance

A six-page Streamlit research dashboard was completed under `dashboard/`.

The application includes:

1. Strategy Overview;
2. Performance & Drawdowns;
3. Factor & Signal Health;
4. Risk & Concentration;
5. Implementation & Liquidity;
6. Attribution.

The interface provides portfolio selection, independent start and end dates, freshness reporting, status diagnostics, summary tables, and Plotly figures.

Page-level notebook reconciliation cells were used to verify:

- performance and drawdown histories;
- factor predictive as-of dates;
- signal and dependence measures;
- beta and concentration measures;
- implementation and liquidity measures;
- portfolio-side and security-level attribution;
- status aggregation and diagnostic figures.

Final quality assurance included:

- the complete pytest suite;
- Ruff linting;
- Python compilation;
- `git diff --check`;
- full 15-dataset dry-run reconstruction;
- Streamlit navigation, filter, empty-state, and invalid-range checks.

All automated checks and artifact reconciliations passed.

---

## Main research lessons

### 1. Signal and portfolio quality are different

Momentum contains predictive information, but its symmetric short side is weak.

### 2. Dollar neutrality is not risk neutrality

Equal long and short capital can retain substantial beta and sector exposure.

### 3. Nominal diversification is not economic diversification

Many holdings can still produce concentrated beta and return contributions.

### 4. Rebalance frequency is an economic parameter

Lower-frequency implementation substantially reduced turnover and improved net performance.

### 5. Complexity requires evidence

Maximum-Sharpe MVO added estimation error and turnover without improving realised results.

### 6. Robustness is multidimensional

A strategy can survive costs and rebalance phases while remaining dependent on market regimes.

### 7. Monitoring should diagnose rather than optimise

Current warnings inform interpretation. They do not justify changing a frozen strategy retrospectively.

### 8. Negative results are valuable

Rejected factors and allocation methods clarify which ideas do not add distinct value.

---

## Current interpretation

The completed research supports the following statement:

> Momentum and realised volatility contain historically useful cross-sectional information in the studied universe, and a simple composite ranking produces the strongest tested implementation after costs and phase robustness.

It also supports an equally important qualification:

> The selected portfolio is not market-neutral, is currently concentrated in market and sector risk, and has benefited substantially from the post-2022 regime.

The project has therefore identified a coherent historical research specification—not a proven production alpha strategy.

---

## Current project status

The core project implementation is complete.

The repository now contains:

- the original research notebooks and empirical record;
- a tested reusable Python package;
- deterministic artifact refresh and validation;
- 15 validated dashboard datasets;
- a complete six-page Streamlit dashboard;
- final automated and manual quality-assurance coverage.

The project is now in its documentation, consistency-review, and portfolio-presentation stage.

---

## Remaining project work

The research implementation, dashboard, quality assurance, and supporting documentation are now complete.

The remaining planned work is:

1. conduct an in-place repository review for consistency and readability;
2. rerun the complete quality-assurance sequence after any cleanup;
3. prepare the final public-repository presentation;
4. update the CV project description and interview narrative.

Possible later extensions include forward or walk-forward evaluation, alternative datasets, scheduled refreshes, and deployment. These are optional extensions rather than requirements for the completed project.
