# Methodology and Research Assumptions

## 1. Research scope

The project studies systematic equity signals using daily data for 101 present-day S&P 100 securities.

The factor panel covers approximately 2 January 2015 to 2 July 2026. The aligned sample used for the final portfolio comparison runs from 7 January 2016 to 1 July 2026.

SPY is used as:

- the market benchmark;
- the explanatory return in beta estimation; and
- the hedge instrument in diagnostic beta-neutralisation experiments.

The final strategy comparison uses three frozen implementations:

| Portfolio | Frequency | Offset | Cost assumption |
|---|---:|---:|---:|
| Composite Score | 21 trading days | 0 | 10 bps |
| Fixed 50/50 Sleeves | 10 trading days | 0 | 10 bps |
| Pure Inverse Volatility | 10 trading days | 0 | 10 bps |

---

## 2. Universe

The universe is based on present-day S&P 100 constituents rather than historical point-in-time membership.

Historical prices are included subject to each security’s available history.

This design introduces survivorship and membership bias because:

- future survivors are known when the historical universe is formed;
- historical index additions and deletions are not reconstructed;
- delisted securities are not represented fully; and
- sector classifications are treated as historically constant.

The current-universe approach is a research simplification and should not be interpreted as a production-quality universe.

---

## 3. Data conventions

### 3.1 Adjusted prices

Adjusted close is used for return calculations because it incorporates stock splits and distributions.

Backward daily returns are:

$$
r_{i,t}
=
\frac{P_{i,t}}{P_{i,t-1}}-1.
$$

### 3.2 Forward returns

Forward returns are aligned with the signal date:

$$
r^{fwd,h}_{i,t}
=
\frac{P_{i,t+h}}{P_{i,t}}-1.
$$

A signal observed at the close of date $t$ is evaluated against returns beginning after that close.

Forward-return horizons are used for signal validation. One-day returns are used for daily portfolio accounting.

### 3.3 Dollar volume

Dollar volume is calculated as:

$$
\text{Dollar Volume}_{i,t}
=
\text{Close}_{i,t}
\times
\text{Volume}_{i,t}.
$$

Raw close is used because it corresponds more directly to the price at which shares traded.

Capacity calculations use lagged liquidity information so that same-day trading decisions do not use future volume.

### 3.4 Missing returns

Dates without any valid next-day return are excluded from portfolio evaluation.

Missing-return weight is tracked explicitly at both portfolio and security level. A missing return is not silently replaced with a valid observed return.

### 3.5 Data source and snapshot reproducibility

Market prices are retrieved from Yahoo Finance through the independent
open-source [`yfinance`](https://github.com/ranaroussi/yfinance) library. The
current S&P 100 constituent table is retrieved separately from Wikipedia.

The public repository does not distribute these downloaded inputs or generated
Parquet artifacts. An explicit download end date fixes the requested date
boundary but does not pin later vendor revisions or membership changes. Exact
recreation of a historical raw snapshot therefore requires preserving both the
constituent list and downloaded source files.

The repository's MIT licence applies to its code and documentation, not to
third-party market data or dependencies.

---

## 4. Factor definitions

### 4.1 12–1 month momentum

$$
\text{Momentum}^{12-1}_{i,t}
=
\frac{P_{i,t-21}}{P_{i,t-252}}-1.
$$

The most recent month is excluded to reduce contamination from short-term reversal.

### 4.2 Three-month momentum

$$
\text{Momentum}^{3m}_{i,t}
=
\frac{P_{i,t}}{P_{i,t-63}}-1.
$$

### 4.3 One-month reversal

$$
\text{Reversal}^{1m}_{i,t}
=
-\left(
\frac{P_{i,t}}{P_{i,t-21}}-1
\right).
$$

### 4.4 Realised volatility

The 63-day realised-volatility signal is:

$$
\text{Volatility}_{i,t}
=
\operatorname{Std}
\left(
r_{i,t-62},\ldots,r_{i,t}
\right)
\sqrt{252}.
$$

The project tests realised volatility as a cross-sectional characteristic rather than assuming that it is automatically a low-volatility factor.

### 4.5 Retained factors

The final multi-factor portfolios retain:

- 12–1 month momentum; and
- 63-day realised volatility.

Three-month momentum and one-month reversal are not retained because their portfolio evidence is insufficient.

Additional experimental factors are accepted or rejected using the same validation framework. Rejected factors remain documented to reduce outcome selection.

---

## 5. Signal processing

### 5.1 Winsorisation

Raw factor values are clipped to cross-sectional lower and upper quantiles on each date.

The default bounds are the 1st and 99th percentiles.

### 5.2 Cross-sectional standardisation

For factor value $x_{i,t}$, the cross-sectional z-score is:

$$
z_{i,t}
=
\frac{x_{i,t}-\mu_t}{\sigma_t},
$$

where the mean and standard deviation are calculated across eligible securities on date $t$.

### 5.3 Percentile ranks

Percentile ranks provide a scale-free representation of each factor and are used in rank-based validation.

### 5.4 Sector-neutral scores

Diagnostic sector-neutral scores are calculated within each date-sector group:

$$
z^{sector}_{i,t}
=
\frac{x_{i,t}-\mu_{s(i),t}}
{\sigma_{s(i),t}}.
$$

Groups with insufficient observations or zero dispersion receive missing scores.

Sector-neutral scores reduce broad sector differences but do not guarantee exact sector-neutral portfolio weights.

---

## 6. Signal validation

### 6.1 Information coefficient

The information coefficient is the cross-sectional correlation between a factor and forward returns:

$$
IC_t
=
\operatorname{Corr}
\left(
x_{i,t},
r^{fwd,h}_{i,t}
\right).
$$

Spearman rank IC is the primary measure because the portfolio construction is rank-based.

### 6.2 IC t-statistic

The simple IC t-statistic is:

$$
t
=
\frac{\overline{IC}}
{s_{IC}/\sqrt{N}}.
$$

This statistic assumes independent observations. It is interpreted cautiously when forward-return windows overlap or ICs are serially correlated.

### 6.3 Quantile analysis

Eligible securities are divided into cross-sectional quantiles on each signal date.

The analysis considers:

- average return by quantile;
- top-minus-bottom spread;
- monotonicity;
- multiple forward horizons;
- subperiod stability;
- non-overlapping samples; and
- rolling IC.

Overlapping forward-return observations are treated as diagnostics rather than independent tradable returns.

### 6.4 Signal monitoring

The monitoring layer tracks:

- cross-sectional coverage;
- rolling mean rank IC;
- raw-signal dispersion;
- one-day and 21-day rank stability; and
- rolling cross-factor rank correlation.

Signal warnings are separated from structural data failures.

---

## 7. Standalone portfolio construction

The baseline standalone factor portfolio:

- divides eligible securities into five quantiles;
- is long the highest factor quintile;
- is short the lowest factor quintile;
- uses equal weights within each leg;
- targets long exposure of $+1$;
- targets short exposure of $-1$; and
- holds positions between scheduled rebalances.

Before cross-factor netting:

$$
\sum_i w^{long}_{i,t}=1,
\qquad
\sum_i |w^{short}_{i,t}|=1.
$$

The target portfolio is therefore approximately dollar-neutral:

$$
\sum_i w_{i,t}=0.
$$

Dollar neutrality does not imply beta or sector neutrality.

---

## 8. Multi-factor portfolio construction

### 8.1 Composite Score

The Composite Score averages the two processed factor z-scores:

$$
z^{composite}_{i,t}
=
0.5z^{momentum}_{i,t}
+
0.5z^{volatility}_{i,t}.
$$

Securities are ranked once using the combined score. The highest quintile is held long and the lowest quintile short.

Factor disagreement changes the final security ranking rather than cancelling two completed sleeve portfolios.

### 8.2 Fixed 50/50 Sleeves

Momentum and realised volatility are first constructed as independent long-short portfolios.

Their target weights are then combined:

$$
w^{fixed}_{i,t}
=
0.5w^{momentum}_{i,t}
+
0.5w^{volatility}_{i,t}.
$$

Opposing positions in the same security net at the combined portfolio level. Consequently, realised gross exposure may be below the sum of the standalone sleeve exposures.

### 8.3 Pure Inverse Volatility

The Pure Inverse Volatility portfolio uses trailing sleeve-return volatility:

$$
a_{k,t}
=
\frac{1/\widehat{\sigma}_{k,t}}
{\sum_j 1/\widehat{\sigma}_{j,t}},
$$

where $\widehat{\sigma}_{k,t}$ is estimated using trailing, one-day-shifted sleeve returns.

Combined security weights are:

$$
w^{inverse\ vol}_{i,t}
=
\sum_k a_{k,t}w^{k}_{i,t}.
$$

Equal sleeve allocations are used during the initial estimation warm-up.

For two positive-weight sleeves, inverse-volatility allocation equalises component risk contributions under the covariance structure used in the experiment. It does not guarantee equal realised future risk.

---

## 9. Backtest accounting

### 9.1 Rebalance schedule

A strategy is defined by:

- rebalance frequency $F$; and
- rebalance offset $o \in \{0,\ldots,F-1\}$.

Positions persist between rebalance dates.

All valid offsets are evaluated during robustness analysis. Offset zero is used only for the final frozen implementation.

### 9.2 Turnover

Security-level turnover is:

$$
\text{Turnover}_t
=
\sum_i
\left|
w^{target}_{i,t}
-
w^{pre}_{i,t}
\right|,
$$

where $w^{pre}_{i,t}$ is the portfolio weight immediately before trading.

Turnover is reported as:

- daily turnover;
- rebalance-event turnover;
- annualised turnover;
- rolling turnover;
- maximum single-day turnover; and
- concentration among the largest turnover days.

### 9.3 Transaction costs

Transaction-cost drag is:

$$
\text{Cost}_t
=
\text{Turnover}_t
\times
\frac{c}{10{,}000},
$$

where $c$ is the assumed cost in basis points per unit of traded notional.

The baseline assumption is 10 basis points.

The sensitivity grid is:

$$
c \in \{0,5,10,20,50\}.
$$

Net return is:

$$
r^{net}_{p,t}
=
r^{gross}_{p,t}
-
\text{Cost}_t.
$$

The model is linear and does not fully represent spreads, market impact, borrow costs or execution timing.

---

## 10. Performance statistics

For $N$ daily observations, annualised geometric return is:

$$
R_{ann}
=
\left(
\prod_{t=1}^{N}(1+r_t)
\right)^{252/N}
-1.
$$

Annualised volatility is:

$$
\sigma_{ann}
=
\operatorname{Std}(r_t)\sqrt{252}.
$$

The Sharpe ratio is:

$$
\text{Sharpe}
=
\frac{\overline{r}}
{\operatorname{Std}(r)}
\sqrt{252}.
$$

No risk-free-rate adjustment is applied.

Drawdown is measured relative to the running wealth peak:

$$
D_t
=
\frac{W_t}{\max_{s\le t}W_s}-1.
$$

Maximum drawdown is the minimum value of $D_t$.

---

## 11. Portfolio optimisation

### 11.1 Global minimum variance

Walk-forward global minimum-variance allocation estimates the trailing sleeve covariance matrix and chooses non-negative sleeve weights that minimise predicted variance.

The experiment tests whether covariance-based allocation improves on simple fixed or inverse-volatility methods.

### 11.2 Maximum-Sharpe MVO

Maximum-Sharpe MVO estimates both expected sleeve returns and covariance.

Because expected-return estimates are particularly noisy, the research evaluates:

- boundary-solution frequency;
- estimated-versus-realised return-spread correlation;
- sleeve-ranking hit rate;
- realised MVO-minus-equal performance;
- turnover; and
- shrinkage toward equal weighting.

The tested MVO estimator is rejected because its expected-return forecasts contain no reliable realised allocation information.

This conclusion applies to the tested specification, not to every possible optimisation method.

---

## 12. Robustness framework

### 12.1 Rebalance phases

Every offset associated with a selected frequency is evaluated.

Results are summarised using:

- mean and median performance;
- minimum and maximum performance;
- worst drawdown;
- offset ranges; and
- worst-offset Sharpe ratios.

Phase-averaged results remain separate from the offset-zero headline implementation.

### 12.2 Subperiods

The predefined subperiods are:

- 2016–2018;
- 2019–2022; and
- 2023–present.

The periods are fixed before the final assessment and are not selected to maximise contrast.

### 12.3 Rolling stability

Rolling 252-day Sharpe ratios are calculated by rebalance phase.

The summary includes:

- median phase-averaged Sharpe;
- 10th percentile;
- minimum and maximum;
- fraction of positive phase-averaged windows; and
- fraction of windows in which the worst offset remains positive.

### 12.4 Capacity

For a security trade with absolute target-weight change $|\Delta w_{i,t}|$, approximate portfolio capacity under participation limit $p$ is:

$$
\text{Capacity}_{i,t}
=
\frac{
p \times \text{ADV}_{i,t}
}{
|\Delta w_{i,t}|
}.
$$

ADV uses lagged historical dollar volume.

The portfolio trade-event capacity is constrained by the least liquid required trade. Summaries include:

- fifth-percentile capacity;
- median capacity;
- worst-phase capacity; and
- worst historical capacity.

Capacity assumes synchronous execution and is a scenario diagnostic rather than a capital recommendation.

---

## 13. Risk and attribution

### 13.1 Holdings-implied beta

Security beta is estimated using trailing stock and SPY returns:

$$
\widehat{\beta}_{i,t}
=
\frac{
\operatorname{Cov}(r_i,r_m)
}{
\operatorname{Var}(r_m)
}.
$$

Portfolio beta is:

$$
\widehat{\beta}_{p,t}
=
\sum_i w_{i,t}\widehat{\beta}_{i,t}.
$$

Long- and short-side beta contributions are retained separately.

### 13.2 Realised beta

Rolling realised beta is estimated by regressing or covariance-scaling realised portfolio returns against SPY.

Holdings-implied beta is the contemporaneous exposure estimate. Realised beta is backward-looking and may differ because of estimation error and changing holdings.

### 13.3 Sector exposure

For each sector, the framework records:

- long exposure;
- short exposure;
- net exposure; and
- gross exposure.

The largest absolute sector net exposure is used as a directional-concentration diagnostic.

### 13.4 Contribution concentration

Security and sector contributions are accumulated over trailing 63-day windows.

For non-negative contribution magnitudes $x_j$, shares are:

$$
s_j
=
\frac{x_j}{\sum_k x_k}.
$$

The effective contributor count is:

$$
N_{\mathrm{effective}}
=
\frac{1}{\sum_j s_j^2}.
$$

The same concentration concept is applied to:

- position weights;
- sectors;
- absolute beta contributions;
- security return contributions; and
- sector return contributions.

A high position count can coexist with a low effective number of risk or return contributors.

---

## 14. Monitoring framework

Monitoring diagnostics are divided into four categories:

1. signal health;
2. market risk;
3. concentration; and
4. implementation.

Two kinds of controls are used.

### 14.1 Structural controls

Structural diagnostics identify conditions such as:

- missing required inputs;
- failed accounting reconciliation;
- incomplete liquidity coverage;
- invalid transaction-cost rates; or
- missing-return exposure.

These can produce a breach or unavailable status.

### 14.2 Historically calibrated controls

State diagnostics compare the latest value with its own historical distribution.

The principal warning thresholds use:

- the upper 10% historical tail for adverse high values; or
- the lower 10% historical tail for adverse low values.

A warning means that the current state is historically unusual. It does not automatically imply that the strategy definition should change.

### 14.3 Status hierarchy

The severity order is:

```text
PASS
→ WARNING
→ BREACH
→ UNAVAILABLE
```

The overall entity status is the most severe applicable diagnostic status.

---

## 15. Final decision framework

The final strategy decision uses a hierarchy rather than an optimised weighted score.

### Eligibility gates

A portfolio must have:

- a frozen specification;
- aligned and reconciled data;
- acceptable liquidity coverage;
- no active missing-return exposure;
- passing implementation diagnostics; and
- no unresolved structural breach.

### Comparative evidence

Eligible candidates are compared using:

- net return and Sharpe ratio;
- volatility and drawdown;
- phase and cost robustness;
- subperiod and rolling stability;
- turnover and capacity;
- beta and concentration;
- transparency; and
- complexity.

The final hierarchy is:

1. Composite Score — primary implementation;
2. Pure Inverse Volatility — defensive alternative;
3. Fixed 50/50 Sleeves — transparent benchmark.

---

## 16. Risk-control decision

The completed strategies do not include explicit beta or sector constraints.

The latest high-beta and concentrated state was already observed before the final decision. Adding constraints in response would create a new in-sample design cycle.

Beta-targeted and sector-constrained variants are therefore deferred to a separately specified research extension. The current unconstrained strategy remains the reference against which any future controls must be compared.

---

## 17. Reusable analytics, artifact validation, and dashboard

### 17.1 Separation of research and presentation

The research notebooks preserve the chronological modelling process, including exploratory analysis, rejected alternatives, implementation selection, and diagnostic investigation.

The final monitoring and presentation layer does not depend on live notebook state. Reusable calculations are implemented under `src/alpha_research/`, while the Streamlit application under `dashboard/` consumes validated Parquet artifacts.

This separation ensures that the dashboard is a presentation and monitoring layer over the completed methodology rather than an independent source of calculations.

### 17.2 Artifact reconstruction

The reproducible refresh workflow is exposed through:

```text
scripts/refresh_strategy_outputs.py
```

It reconstructs 15 datasets:

* six attribution datasets;
* nine monitoring datasets.

On a clean clone, write mode creates and reads back the validated local artifacts. Once those files exist, dry-run mode compares an in-memory reconstruction with the local reference artifacts without modifying them.

The reconciliation checks cover:

* column sets;
* primary keys;
* row counts;
* date coverage;
* numerical values;
* cross-dataset consistency.

The final validation dry run reconciled all 15 locally generated datasets. Floating-point comparisons use tight numerical tolerances so that material differences cannot be hidden by the audit.

### 17.3 Artifact contracts

Each dashboard artifact has an explicit contract covering its required columns, key structure, data types, and basic validity conditions.

The loader distinguishes between:

* an individually valid artifact;
* a valid group of mutually consistent artifacts;
* structural readiness of the complete dashboard bundle;
* freshness of the observations.

Missing, unreadable, individually invalid, or group-inconsistent artifacts are treated as structural failures. Stale artifacts remain loadable but are reported explicitly.

### 17.4 Temporal and as-of-date semantics

Historical panels use inclusive start and end dates selected by the user.

Latest-state summaries and filtered histories are intentionally distinguished. For example, a factor's latest signal observation can be later than the final date of the selected portfolio history.

Predictive statistics such as the information coefficient can also have an earlier valid date than the latest signal observation because forward-return calculation requires future data. The dashboard therefore retains explicit fields such as:

* `ic_as_of_date`;
* `rolling_mean_ic_252_as_of_date`.

This prevents a predictive statistic from being incorrectly labelled with a later signal date.

### 17.5 Readiness and freshness

Structural readiness answers whether the complete artifact bundle can be used safely by the dashboard.

Freshness answers whether the latest observations are sufficiently recent relative to the current business date.

A stale artifact does not automatically imply a methodological or structural failure. This distinction is important because the repository contains a frozen historical research snapshot rather than a live production feed.

The dashboard can therefore be structurally ready while simultaneously reporting stale data.

### 17.6 Dashboard analytics and figures

Dashboard tables are derived through reusable functions in:

```text
src/alpha_research/dashboard_analytics.py
```

Plotly figures are constructed in:

```text
src/alpha_research/visualisation.py
```

The Streamlit page functions in `dashboard/dashboard_pages.py` are responsible for layout and presentation rather than reproducing the underlying research calculations.

The six dashboard pages cover:

1. strategy overview and diagnostic status;
2. performance and drawdowns;
3. factor and signal health;
4. risk and concentration;
5. implementation and liquidity;
6. portfolio-side and security-level attribution.

### 17.7 Quality assurance

The reusable analytics and dashboard layer were checked through:

* unit tests for valid and malformed inputs;
* duplicate-key rejection;
* missing-column rejection;
* non-numeric-value rejection;
* empty and disjoint filtered states;
* invalid date-range handling;
* page-level notebook reconciliation cells;
* Plotly trace and figure audits;
* full artifact reconstruction;
* artifact read-back validation;
* Ruff linting;
* Python compilation;
* Streamlit manual interface testing.

The final dry-run reconstruction matched every stored artifact on columns, keys, row counts, and values.

The dashboard therefore reports the same research results as the validated artifact pipeline while remaining independent of interactive notebook execution.

---

## 18. Main limitations

1. Present-day constituent survivorship bias.
2. No point-in-time sector classifications.
3. No complete delisting-return treatment.
4. In-sample factor and portfolio selection.
5. Material dependence on the post-2022 period.
6. Linear transaction-cost assumptions.
7. No complete borrow-fee, availability, recall or financing model.
8. No nonlinear market-impact model.
9. Capacity based on historical ADV and fixed participation limits.
10. No explicit beta- or sector-neutrality constraints.
11. Noisy backward-looking beta and covariance estimates.
12. SPY is not exposure-matched to the candidate portfolios.
13. No live, paper-trading or genuinely unseen forward validation.
14. Third-party market data and current constituent membership can be revised after the documented snapshot.

These limitations prevent the results from being interpreted as production-ready or investable evidence.
