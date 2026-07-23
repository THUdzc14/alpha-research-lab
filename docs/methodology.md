# Methodology and Research Assumptions

## Universe

The current research universe is based on the present-day S&P 100 constituents.

Historical price data is used from approximately 2015 onward, subject to each security’s available history.

This creates survivorship bias because the universe is not reconstructed using historical index membership.

The current-universe approach is used as an initial research simplification and should not be interpreted as a production-quality point-in-time universe.

---

## Data conventions

### Adjusted prices

Adjusted close is used for return calculations because it accounts for stock splits and distributions.

### Dollar volume

Dollar volume is calculated using:

$$
\text{close}\times\text{volume}
$$

Raw close is used because it corresponds more closely to the actual market price at which shares traded.

### Return alignment

Backward daily returns are used for historical factor estimation:

$$
r_{i,t}
=
\frac{P_{i,t}}{P_{i,t-1}}-1
$$

Forward returns are aligned with the factor date:

$$
r^{fwd}_{i,t}
=
\frac{P_{i,t+1}}{P_{i,t}}-1
$$

A factor observed at the close of date (t) is therefore evaluated against the return from (t) to (t+1).

The same convention is used for SPY hedge returns.

---

## Factor definitions

### 12–1 month momentum

$$
\text{Momentum}^{12-1}_{i,t}
=
\frac{P_{i,t-21}}{P_{i,t-252}}-1
$$

### 3-month momentum

$$
\text{Momentum}^{3m}_{i,t}
=
\frac{P_{i,t}}{P_{i,t-63}}-1
$$

### 1-month reversal

$$
\text{Reversal}^{1m}_{i,t}
=
-\left(
\frac{P_{i,t}}{P_{i,t-21}}-1
\right)
$$

### Realised volatility

$$
\text{Volatility}_{i,t}
=
\operatorname{Std}
\left(
r_{i,t-62},\ldots,r_{i,t}
\right)
\sqrt{252}
$$

---

## Signal processing

### Winsorisation

Raw factor values are clipped to cross-sectional lower and upper quantiles on each date.

The current default is the 1st and 99th percentile.

### Cross-sectional z-score

$$
z_{i,t}
=
\frac{x_{i,t}-\mu_t}{\sigma_t}
$$

where the mean and standard deviation are calculated across eligible stocks on date (t).

### Sector-neutral z-score

Sector-neutral scores are calculated within each date-sector group:

$$
z^{sector}_{i,t}
=
\frac{
x_{i,t}-\mu_{sector(i),t}
}{
\sigma_{sector(i),t}
}
$$

Groups with insufficient observations or zero dispersion receive missing scores.

---

## Signal validation

### Information coefficient

The information coefficient is the cross-sectional correlation between factor scores and forward returns:

$$
IC_t
=
\operatorname{Corr}
\left(
x_{i,t},
r^{fwd}_{i,t}
\right)
$$

Spearman rank IC is used as the primary measure.

### IC information ratio

$$
ICIR
=
\frac{\overline{IC}}{\operatorname{Std}(IC)}
$$

The current value is not annualised.

### IC t-statistic

The simple statistic is:

$$
t
=
\frac{\overline{IC}}
{s_{IC}/\sqrt{N}}
$$

This assumes independent observations. It should be interpreted cautiously when forward-return windows overlap or IC observations are serially correlated.

### Quantile analysis

Stocks are divided into factor quantiles independently on each date.

The highest factor group is compared with the lowest factor group using equal-weight average forward returns.

Quantile results are initially treated as signal diagnostics rather than tradable wealth series when forward-return periods overlap.

---

## Portfolio construction

The default portfolio:

* rebalances every five trading days;
* is long the highest factor quintile;
* is short the lowest factor quintile;
* applies equal weights within each leg;
* has long gross exposure of 1;
* has short gross exposure of 1.

Therefore:

$$
\sum_i w_i^{long}=1,
\qquad
\sum_i |w_i^{short}|=1
$$

and the initial stock portfolio is approximately dollar neutral:

$$
\sum_i w_i=0
$$

Positions are held constant between rebalances.

---

## Turnover and transaction costs

Stock turnover is defined as:

$$
\text{Turnover}_t
=
\sum_i
\left|
w^{target}_{i,t}
-
w^{previous}_{i,t}
\right|
$$

The default stock transaction cost is 10 basis points per unit of traded notional.

Benchmark-hedge turnover is calculated separately, with a default cost of 1 basis point.

The cost model is simplified and does not currently include:

* bid-ask spread variation;
* market impact;
* liquidity-dependent costs;
* borrow fees;
* short-sale constraints.

---

## Beta estimation and hedging

Rolling stock beta is estimated using historical stock and SPY returns:

$$
\widehat\beta_{i,t}
=
\frac{
\operatorname{Cov}(r_i,r_m)
}{
\operatorname{Var}(r_m)
}
$$

The current default uses a 126-day window with a minimum of 63 observations.

The estimated stock-portfolio beta is:

$$
\widehat\beta_{p,t}
=
\sum_i w_{i,t}\widehat\beta_{i,t}
$$

The SPY hedge weight is:

$$
w_{\text{SPY},t}
=
-\widehat\beta_{p,t}
$$

The hedge is updated on portfolio rebalance dates and held between rebalances.

Beta neutrality is ex ante and approximate because estimated stock betas may differ from subsequently realised betas.

---

## Sector exposure

For each sector, the framework records:

* long weight;
* short weight;
* net weight;
* gross weight.

Sector-neutral factor scores substantially reduce sector exposure but do not mathematically guarantee zero sector portfolio weights, because the final long and short quintiles are selected globally.

Exact sector neutrality would require explicit portfolio constraints or sector-balanced selection.

---

## Performance statistics

The framework currently reports:

* total return;
* annualised return;
* annualised volatility;
* Sharpe ratio;
* maximum drawdown;
* positive-day fraction;
* average daily turnover;
* average rebalance turnover;
* total transaction costs.

No risk-free-rate adjustment is currently applied to the Sharpe ratio.

---

## Interpretation standards

A factor is not considered promising based only on a positive full-sample IC or return.

The research process considers:

* sign and magnitude of IC;
* quantile monotonicity;
* horizon consistency;
* subperiod stability;
* non-overlapping observations;
* rebalance-offset robustness;
* long- and short-leg contributions;
* turnover and transaction costs;
* market beta;
* sector concentration;
* performance after neutralisation.

---

## Current limitations

The largest limitations are:

1. current-constituent survivorship bias;
2. no point-in-time industry classifications;
3. no delisting returns;
4. simplified transaction costs;
5. no borrow-cost model;
6. no market-impact model;
7. in-sample factor selection;
8. limited factor library;
9. single-benchmark beta model;
10. no walk-forward or live evaluation.

These should be addressed before treating the framework as production-ready or the results as investable evidence.
