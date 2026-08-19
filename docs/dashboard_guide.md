# Dashboard and Reproducibility Guide

This guide explains how to rebuild the project artifacts, validate them, and run the Streamlit research dashboard.

The dashboard consumes validated Parquet artifacts. It does not depend on notebook state or require the research notebooks to be executed interactively.

## System overview

The dashboard workflow is:

1. Load the processed research inputs.
2. Reconstruct the attribution and monitoring outputs.
3. validate each output against its artifact contract.
4. Compare reconstructed outputs with the stored artifacts.
5. Optionally write and read back the refreshed artifacts.
6. Load the validated artifacts into the dashboard data layer.
7. Derive dashboard tables and figures through reusable analytics modules.
8. Render the six Streamlit pages.

The main implementation paths are:

- `scripts/refresh_strategy_outputs.py`
- `src/alpha_research/artifacts.py`
- `src/alpha_research/dashboard_data.py`
- `src/alpha_research/dashboard_analytics.py`
- `src/alpha_research/visualisation.py`
- `dashboard/dashboard_pages.py`
- `dashboard/streamlit_app.py`

## Dashboard artifacts

The refresh workflow produces and validates 15 dashboard datasets.

### Attribution artifacts

Stored under `data/processed/`:

| Dataset | File |
|---|---|
| Selected implementations | `attribution_selected_implementations.parquet` |
| Portfolio daily attribution | `attribution_portfolio_daily.parquet` |
| Security holdings | `attribution_security_holdings.parquet` |
| Target weights | `attribution_target_weights.parquet` |
| Benchmark daily attribution | `attribution_benchmark_daily.parquet` |
| Security daily attribution | `attribution_security_daily.parquet` |

### Monitoring artifacts

Stored under `data/processed/monitoring/`:

| Dataset | File |
|---|---|
| Signal health | `strategy_monitoring_signal_health_daily.parquet` |
| Factor dependence | `strategy_monitoring_factor_dependence_daily.parquet` |
| Performance and risk | `strategy_monitoring_performance_risk_daily.parquet` |
| Beta monitoring | `strategy_monitoring_beta_daily.parquet` |
| Concentration monitoring | `strategy_monitoring_concentration_daily.parquet` |
| Implementation monitoring | `strategy_monitoring_implementation_daily.parquet` |
| Liquidity coverage | `strategy_monitoring_liquidity_coverage_daily.parquet` |
| Diagnostic flags | `strategy_monitoring_diagnostic_flags.parquet` |
| Latest overview | `strategy_monitoring_latest_overview.parquet` |

## Rebuilding and validating the artifacts

Run commands from the repository root.

### Dry-run reconciliation

```powershell
.venv\Scripts\python scripts\refresh_strategy_outputs.py
```

The dry run:

- reconstructs all 15 datasets;
- validates their schemas and keys;
- compares their columns, keys, and values with the stored artifacts;
- reports row counts and reconciliation results;
- does not write any artifacts.

A successful run ends with:

```text
All refresh reconciliations pass: True
Dry run only: no artifacts were written.
```

### Write refreshed artifacts

```powershell
.venv\Scripts\python scripts\refresh_strategy_outputs.py --write
```

Write mode:

1. performs the same reconstruction and validation;
2. writes the refreshed Parquet files;
3. reads the files back from disk;
4. validates the read-back results;
5. confirms that the persisted data match the reconstructed outputs.

Use write mode only when the source data or research configuration have intentionally changed.

## Dashboard readiness and freshness

Dashboard readiness and data freshness are separate concepts.

Structural readiness checks whether every required artifact:

- exists;
- can be loaded;
- satisfies its dataset contract;
- satisfies the relevant cross-dataset group contract.

Structural failures prevent the affected dashboard data from being treated as ready.

Freshness checks compare each artifact's latest observation date with the current business date. The default stale-data tolerance is five business days.

Possible statuses include:

- `READY`: structurally valid and within the freshness tolerance;
- `STALE`: structurally valid but older than the freshness tolerance;
- `UNDATED`: structurally valid but without a natural observation date;
- `MISSING`: the required file does not exist;
- `READ_ERROR`: the artifact cannot be read;
- `INVALID`: the individual artifact contract fails;
- `INVALID_GROUP`: a cross-artifact consistency check fails.

Stale data do not make the dashboard structurally invalid. They produce an explicit warning so that a historical frozen research snapshot remains inspectable.

Where appropriate, undated summary artifacts use a related dated artifact as their freshness reference. `selected_implementations` remains naturally undated.

## Running the Streamlit dashboard

Install the project and development dependencies if necessary:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
```

Start Streamlit from the repository root:

```powershell
.venv\Scripts\python -m streamlit run dashboard\streamlit_app.py
```

The application provides six pages:

| Page | Purpose |
|---|---|
| Strategy Overview | Latest portfolio status, implementation roles, diagnostics, and health summaries |
| Performance & Drawdowns | Performance statistics, cumulative wealth, drawdowns, rolling Sharpe ratios, and volatility |
| Factor & Signal Health | Coverage, dispersion, predictive information coefficients, rank stability, and factor dependence |
| Risk & Concentration | Holdings-implied beta, realised beta, beta gaps, position concentration, and contribution concentration |
| Implementation & Liquidity | Turnover, trade size, capacity, missing-return exposure, and liquidity coverage |
| Attribution | Long/short/cost attribution and security-level contribution analysis |

## Global controls

The sidebar provides:

- page navigation;
- portfolio selection;
- a separate start-date selector;
- a separate end-date selector;
- artifact-readiness and freshness information.

The start and end dates are separate controls so that either boundary can be changed independently.

The controls are applied where they are analytically meaningful:

- performance histories include SPY automatically as the benchmark;
- factor pages apply the date range to the historical panels;
- the factor snapshot retains the latest available predictive observation and reports its as-of date explicitly;
- the Strategy Overview represents the latest available monitoring state;
- Attribution provides an additional portfolio selector for security-level analysis.

## Refreshing the Streamlit view

Streamlit caches the dashboard bundle for a limited period. After writing refreshed artifacts:

1. rerun the Streamlit application; or
2. clear the Streamlit cache and rerun the page.

This ensures the interface loads the new files rather than a cached bundle.

## Quality-assurance commands

Run the following commands from the repository root.

### Test suite

```powershell
.venv\Scripts\python -m pytest -q
```

### Ruff

```powershell
.venv\Scripts\python -m ruff check dashboard scripts src tests
```

### Python compilation

```powershell
.venv\Scripts\python -m compileall -q dashboard scripts src tests
```

### Whitespace and patch validation

```powershell
git diff --check
```

### Full artifact reconciliation

```powershell
.venv\Scripts\python scripts\refresh_strategy_outputs.py
```

Before a release or portfolio milestone, also inspect every Streamlit page manually and confirm that:

- page navigation works;
- portfolio and date filters work;
- invalid date ranges are handled;
- freshness warnings are visible;
- summary tables are populated;
- figures render with the expected traces;
- empty filtered states do not cause an exception.

## Reproducibility boundaries

The research notebooks remain the narrative record of the modelling process and empirical investigation.

The production-style dashboard path is separated from notebook state:

- reusable calculations live under `src/alpha_research/`;
- artifacts are regenerated through `scripts/refresh_strategy_outputs.py`;
- the dashboard loads artifacts through `dashboard_data.py`;
- page-level analytics are implemented in `dashboard_analytics.py`;
- figure construction is implemented in `visualisation.py`;
- Streamlit rendering is isolated under `dashboard/`.

The current artifacts represent a frozen research snapshot rather than a live production feed. A stale-data warning is therefore expected when the dashboard is run sufficiently long after the final observation date.

## Dashboard examples

### Strategy overview

![Strategy overview dashboard](images/dashboard_strategy_overview.png)

### Performance and drawdowns

![Performance and drawdowns dashboard](images/dashboard_performance.png)

### Attribution

![Attribution dashboard](images/dashboard_attribution.png)

## Current limitations

The project does not currently provide:

- live market-data ingestion;
- scheduled production refreshes;
- authentication or user-level access controls;
- persistent application deployment;
- execution or order-management integration;
- production monitoring or alert delivery.

These boundaries are intentional. The project is a reproducible research and portfolio-monitoring demonstration rather than a live investment system.
