# Progress Report

## Overview

This project began as an attempt to build a complete systematic alpha-research workflow rather than a collection of isolated factor notebooks.

The central aim has been to connect each research stage:

```text
data quality
→ factor definition
→ predictive validation
→ portfolio implementation
→ risk attribution
```

The project is still in development, but it has reached a useful intermediate milestone: an initial factor can now be traced from raw market data through to a cost-adjusted and exposure-controlled backtest.

---

## Milestone 1: Data foundation

The first milestone was the construction of a reliable equity panel.

The work included:

* downloading the current S&P 100 universe;
* retrieving daily stock and SPY data;
* standardising prices and volumes;
* constructing backward and forward returns;
* joining sector and company metadata;
* identifying incomplete stock histories;
* checking missingness, gaps, extreme returns, and invalid prices.

One early lesson was that data availability and universe membership are different concepts. A stock may have historical price data long before entering an index. Using present-day constituents historically therefore introduces survivorship bias even when the price panel appears complete.

The current-universe approach was retained for the minimum viable framework, but the limitation is recorded explicitly.

---

## Milestone 2: Factor construction

The initial factor library was intentionally small and interpretable:

* 12–1 month momentum;
* 3-month momentum;
* 1-month reversal;
* realised volatility.

Each factor was implemented as a separately testable function.

The factor layer was kept distinct from signal processing. Raw factor values were then:

* winsorised;
* cross-sectionally standardised;
* converted to percentile ranks.

This separation made it easier to distinguish an economic definition from later portfolio-processing choices.

---

## Milestone 3: Signal validation

The first validation layer examined whether factor rankings predicted subsequent cross-sectional returns.

The analysis included:

* daily Pearson and Spearman IC;
* average IC and IC variability;
* positive-IC frequency;
* quantile returns;
* top-minus-bottom spreads;
* horizon comparisons;
* subperiod analysis;
* non-overlapping samples;
* rolling IC.

This stage produced an important lesson:

> A statistically positive factor relationship does not automatically produce an attractive portfolio.

The 12–1 momentum factor had a positive IC, but the portfolio result was weak because the bottom-ranked stocks were poor short candidates.

Realised volatility showed a clearer quantile relationship and translated more successfully into a long-short strategy.

---

## Milestone 4: Portfolio backtesting

The next milestone was a rebalance-based backtester.

The backtester introduced:

* persistent holdings;
* configurable rebalance frequency;
* multiple rebalance offsets;
* equal-weight long and short books;
* daily P&L;
* turnover;
* transaction costs;
* drawdowns;
* leg decomposition;
* subperiod reporting.

A practical bug was identified during this stage: the final panel date had no next-day return. Treating this as a zero return created a full missing-return exposure. The backtester was corrected to remove dates without any valid forward return.

This reinforced the importance of reporting diagnostics alongside headline performance.

---

## Milestone 5: Exposure diagnosis

The raw realised-volatility strategy initially appeared strong, with an annualised return of approximately 15.8% and a Sharpe ratio of approximately 0.73.

However, the long and short books revealed an asymmetric structure:

* the long high-volatility book had high market beta;
* the short low-volatility book had substantially lower opposing beta;
* the combined portfolio retained a market beta of approximately 0.82.

The strategy also had large sector tilts, particularly:

* long Information Technology;
* short Consumer Staples;
* short Health Care and Utilities.

This was a central research lesson:

> Dollar neutrality does not imply market or sector neutrality.

A portfolio can hold equal long and short capital while retaining substantial systematic risk.

---

## Milestone 6: Neutralisation

Two neutralisation mechanisms were added.

### Sector-neutral scores

Stocks were standardised within date-sector groups. This changed the question from:

> Which stocks have the highest volatility across the full market?

to:

> Which stocks have the highest volatility relative to their sector peers?

### Beta hedge

Rolling stock betas were estimated using historical daily returns.

At each rebalance, the stock portfolio’s expected beta was calculated and offset using a SPY position.

The four resulting strategy variants were:

* raw;
* sector neutral;
* beta neutral;
* sector and beta neutral.

---

## Current results

| Strategy              | Ann. return | Ann. volatility | Sharpe | Max drawdown | Realised beta |
| --------------------- | ----------: | --------------: | -----: | -----------: | ------------: |
| Raw                   |       15.8% |           24.0% |   0.73 |       −45.3% |          0.82 |
| Sector neutral        |       11.8% |           15.4% |   0.80 |       −27.7% |          0.48 |
| Beta neutral          |        2.7% |           18.6% |   0.24 |       −38.1% |         −0.06 |
| Sector + beta neutral |        3.6% |           12.5% |   0.34 |       −26.2% |         −0.06 |

The most practically promising variant so far is the sector-neutral realised-volatility strategy.

It gives up some raw return but achieves:

* lower volatility;
* a higher Sharpe ratio;
* a substantially smaller drawdown;
* much smaller sector imbalances.

The fully neutralised result remains positive, suggesting that some stock-selection information survives, but its economic magnitude is modest.

---

## What has been learned

### 1. Research conclusions change as realism increases

The apparent strength of a factor can fall materially after:

* implementing actual holding periods;
* introducing transaction costs;
* separating the long and short legs;
* measuring market beta;
* controlling sector exposure.

This is not a failure of the research process. It is the purpose of the process.

### 2. Signal quality and portfolio quality are different

Momentum showed predictive ranking information, but its short book did not work well.

A useful signal may need a portfolio design that differs from a symmetric long-short construction.

### 3. Exposure attribution is essential

The raw realised-volatility result combined:

* stock-selection information;
* market exposure;
* sector allocation.

Without attribution, these components could easily be mistaken for a single source of alpha.

### 4. Neutralisation is an experiment, not automatically an improvement

Sector neutralisation improved risk-adjusted performance.

Beta neutralisation reduced return sharply.

Both were useful because they clarified where performance came from, even though only one improved the headline Sharpe ratio.

### 5. Negative results are useful

Three-month momentum and one-month reversal were not compelling in the current framework.

Retaining these findings reduces the temptation to report only successful factors and provides useful comparisons for future work.

---

## Current interpretation

The evidence currently supports the following statement:

> Realised volatility contains meaningful cross-sectional information in this universe, particularly within sectors, but its strongest raw performance is partly explained by market beta and sector tilts.

This is a more defensible conclusion than describing the raw backtest as standalone alpha.

---

## Next research stage

The next stage will expand the factor library with economically distinct signals:

* liquidity or illiquidity;
* rolling market beta;
* idiosyncratic volatility;
* risk-adjusted momentum.

The main goal will not be to maximise the number of factors. It will be to determine whether complementary signals can improve:

* robustness across regimes;
* portfolio diversification;
* drawdown control;
* exposure-adjusted performance.

The same validation and neutralisation workflow will be applied to each new signal.
