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

The core research and engineering phases are complete. The project has selected
a primary historical specification, retained transparent benchmarks, documented
rejected methods and exposed the final evidence through reusable analytics and
a Streamlit dashboard. A later controlled single-factor challenge tests and
qualifies that decision without changing the frozen dashboard implementation
set.

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

Among the three monitored multi-factor candidates, Composite Score retained the
highest return and Sharpe ratio at every tested cost.

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

- the highest return and Sharpe ratio among the monitored multi-factor candidates;
- strong rebalance-phase robustness;
- the best transaction-cost resilience among those candidates;
- positive performance in every subperiod; and
- the lowest selected-frequency average turnover among those candidates.

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

All 15 dashboard artifacts reconciled on columns, keys, values, and row counts against the locally generated research outputs.

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

The completed implementation baseline was checked with:

- the complete pytest suite;
- Ruff linting;
- Python compilation;
- `git diff --check`;
- full 15-dataset dry-run reconstruction;
- Streamlit navigation, filter, empty-state, and invalid-range checks.

All baseline checks passed. Following the repository-wide consistency revision,
the complete test suite, static checks, six-page Streamlit smoke validation and
15-dataset dry-run reconciliation were rerun and passed.

---

## Milestone 13 — Documentation and public readiness

The completed repository was reviewed in place for consistency, readability and
public presentation. The review aligned module and script interfaces, clarified
the notebook narrative, added safe command-line help, and updated the operating
documentation without changing the research methodology or reported results.

The public repository is code-only with respect to downloaded and generated
data. It provides scripts to reconstruct those artifacts, documents the
third-party data boundary, and releases the repository code under the MIT
licence.

---

## Milestone 14 — Controlled single-factor benchmark challenge

Notebook 10 reopens the completed hierarchy for a bounded falsification test.
Momentum Only and Realised Volatility Only are compared with all three
multi-factor portfolios on the same 7 January 2016 to 1 July 2026 dates.

The common baseline fixes five-day rebalancing, offset zero and 10 bps costs.
The analysis also covers 1-, 5-, 10- and 21-day frequencies, every valid phase,
0–50 bps costs, the final subperiods, rolling windows, attribution, beta,
concentration, liquidity and capacity.

### Controlled five-day results

| Portfolio | Net ann. return | Ann. volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Momentum Only | 0.95% | 22.19% | 0.154 | −50.96% |
| Realised Volatility Only | 16.08% | 24.86% | 0.724 | −45.85% |
| Composite Score | 13.74% | 21.33% | 0.711 | −30.63% |
| Fixed 50/50 Sleeves | 10.35% | 16.84% | 0.670 | −25.66% |
| Pure Inverse Volatility | 10.41% | 16.29% | 0.690 | −19.57% |

Realised Volatility leads return, most matched phase evidence, cost resilience,
turnover and measured capacity. Its advantage is not explained by missing
returns or weaker liquidity coverage. The same portfolio carries higher beta,
stronger sector tilts, greater volatility, a deeper drawdown and material
dependence on the post-2022 period. Its short leg detracts in every subperiod.

The resulting qualified hierarchy is:

1. **Composite Score — risk-controlled multi-factor primary**
2. **Pure Inverse Volatility — defensive risk-based alternative**
3. **Fixed 50/50 Sleeves — transparent allocation benchmark**
4. **Realised Volatility Only — formal standalone research benchmark**

Momentum Only remains a component benchmark rather than a strategy candidate.
The dashboard continues to monitor the original three frozen implementations;
the formal standalone benchmark extends the research narrative without changing
the 15-artifact bundle.

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

### 9. A hierarchy should remain falsifiable

A controlled standalone benchmark can expose a genuine return advantage while
also showing why a lower-return portfolio remains preferable for a stated risk
objective.

---

## Current interpretation

The completed research supports the following statement:

> Momentum and realised volatility contain historically useful cross-sectional information in the studied universe. Composite is the preferred risk-controlled multi-factor implementation, while Realised Volatility Only is the strongest formal standalone return benchmark.

It also supports an equally important qualification:

> Neither result is unconditional: Composite retains material market and sector risk, while Realised Volatility has higher beta, worse tail risk, long-side dependence and still greater reliance on the post-2022 regime.

The project has therefore identified a coherent historical research specification—not a proven production alpha strategy.

---

## Current project status

The core project implementation is complete.

The repository now contains:

- the original research notebooks and empirical record;
- a controlled standalone-factor challenge that qualifies the frozen hierarchy;
- a tested reusable Python package;
- deterministic artifact refresh and validation;
- contracts and scripts for reconstructing 15 validated dashboard datasets;
- a complete six-page Streamlit dashboard;
- automated and manual quality-assurance coverage; and
- public-facing methodology, provenance and operating documentation.

The documentation, controlled benchmark challenge, repository-wide consistency
review and final release-quality validation are complete. The project is ready
for public release and portfolio presentation.

---

## Release status and optional extensions

The research implementation, controlled benchmark challenge, dashboard,
supporting documentation and post-revision quality assurance are complete. No
further repository work is required for the defined historical research scope.

Possible later extensions include forward or walk-forward evaluation,
alternative datasets, dashboard monitoring of the formal standalone benchmark,
scheduled refreshes and deployment. These are optional extensions rather than
requirements for the completed project.
