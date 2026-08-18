"""Reusable dashboard tables and chart-data preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from alpha_research.config.research import TRADING_DAYS_PER_YEAR
from alpha_research.metrics import summarise_returns
from alpha_research.monitoring import (
    DIAGNOSTIC_FLAG_EXPORT_COLUMNS,
    MONITORING_CATEGORIES,
    STATUS_SEVERITY,
    prepare_diagnostic_flags,
)

PERFORMANCE_HISTORY_COLUMNS = (
    "date",
    "portfolio",
    "return",
    "wealth",
    "indexed_wealth",
    "drawdown",
    "trailing_return_252",
    "rolling_sharpe_252",
    "annualised_volatility_126",
    "maximum_drawdown_252",
)

BETA_HISTORY_COLUMNS = (
    "date",
    "portfolio",
    "beta_coverage",
    "holdings_market_beta",
    "realised_gross_beta_126",
    "beta_measurement_gap",
)

CONCENTRATION_HISTORY_COLUMNS = (
    "date",
    "portfolio",
    "effective_position_count",
    "largest_absolute_sector_net_exposure",
    "top_five_absolute_beta_contribution_share",
    "effective_contributor_count_63",
    "top_five_contributor_share_63",
    "effective_contribution_sector_count_63",
)

IMPLEMENTATION_SOURCE_COLUMNS = (
    "date",
    "portfolio",
    "turnover",
    "transaction_cost",
    "trade_count",
    "annualised_turnover_63",
    "largest_trade_weight_63",
    "minimum_trade_capacity_1pct_usd_63",
    "maximum_missing_return_weight_63",
)

IMPLEMENTATION_HISTORY_COLUMNS = (
    *IMPLEMENTATION_SOURCE_COLUMNS,
    "minimum_trade_capacity_1pct_usd_millions_63",
)

LIQUIDITY_COVERAGE_HISTORY_COLUMNS = (
    "date",
    "portfolio",
    "turnover",
    "liquidity_covered_turnover",
    "liquidity_coverage",
)

DIAGNOSTIC_TABLE_COLUMNS = tuple(DIAGNOSTIC_FLAG_EXPORT_COLUMNS)

MONITORING_OVERVIEW_COLUMNS = (
    "entity_type",
    "entity",
    "diagnostics",
    "passes",
    "warnings",
    "breaches",
    "unavailable",
    "overall_status",
    "signal_status",
    "market_risk_status",
    "concentration_status",
    "implementation_status",
)

_MONITORING_COUNT_COLUMNS = (
    "diagnostics",
    "passes",
    "warnings",
    "breaches",
    "unavailable",
)

_MONITORING_CATEGORY_STATUS_COLUMNS = (
    "signal_status",
    "market_risk_status",
    "concentration_status",
    "implementation_status",
)

LATEST_PORTFOLIO_SNAPSHOT_COLUMNS = (
    "portfolio",
    "implementation_role",
    "rebalance_frequency",
    "rebalance_offset",
    "latest_date",
    "overall_status",
    "market_risk_status",
    "concentration_status",
    "implementation_status",
    "drawdown",
    "rolling_sharpe_252",
    "annualised_volatility_126",
    "holdings_market_beta",
    "realised_gross_beta_126",
    "beta_measurement_gap",
    "effective_position_count",
    "largest_absolute_sector_net_exposure",
    "top_five_absolute_beta_contribution_share",
    "effective_contribution_sector_count_63",
    "top_five_contributor_share_63",
    "annualised_turnover_63",
    "largest_trade_weight_63",
    "minimum_trade_capacity_1pct_usd_63",
    "maximum_missing_return_weight_63",
    "liquidity_coverage",
)


def _require_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(f"{name} is missing columns: {sorted(missing_columns)}")


def _normalise_date_bound(value: Any | None, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None

    result = pd.to_datetime(value, errors="coerce")

    if pd.isna(result):
        raise ValueError(f"{name} must be a valid date.")

    result = pd.Timestamp(result)

    if result.tzinfo is not None:
        result = result.tz_localize(None)

    return result.normalize()


def _resolve_portfolios(
    data: pd.DataFrame,
    portfolios: Sequence[str] | None,
    *,
    name: str,
) -> list[str]:
    available = list(pd.unique(data["portfolio"].dropna()))

    if portfolios is None:
        return available

    if isinstance(portfolios, str):
        raise TypeError("portfolios must be a sequence of portfolio names.")

    selected = list(portfolios)

    if not selected:
        raise ValueError("portfolios must not be empty.")

    if len(selected) != len(set(selected)):
        raise ValueError("portfolios must contain unique names.")

    missing = sorted(set(selected) - set(available))

    if missing:
        raise ValueError(f"{name} is missing portfolios: {missing}")

    return selected


def _sort_portfolios(
    data: pd.DataFrame,
    portfolios: Sequence[str],
) -> pd.DataFrame:
    positions = {portfolio: position for position, portfolio in enumerate(portfolios)}
    result = data.assign(_portfolio_order=data["portfolio"].map(positions))

    return (
        result.sort_values(["_portfolio_order", "date"], kind="stable")
        .drop(columns="_portfolio_order")
        .reset_index(drop=True)
    )


def prepare_performance_history(
    performance_risk: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare validated, filtered performance data for dashboard charts."""
    required_columns = set(PERFORMANCE_HISTORY_COLUMNS) - {"indexed_wealth"}
    _require_columns(
        performance_risk,
        required_columns,
        name="performance_risk",
    )
    selected_portfolios = _resolve_portfolios(
        performance_risk,
        portfolios,
        name="performance_risk",
    )
    prepared = performance_risk.loc[
        performance_risk["portfolio"].isin(selected_portfolios),
        list(required_columns),
    ].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError("performance_risk.date contains invalid dates.")

    if prepared.duplicated(["portfolio", "date"]).any():
        raise ValueError("performance_risk contains duplicate portfolio-date rows.")

    normalised_start = _normalise_date_bound(start_date, name="start_date")
    normalised_end = _normalise_date_bound(end_date, name="end_date")

    if (
        normalised_start is not None
        and normalised_end is not None
        and normalised_start > normalised_end
    ):
        raise ValueError("start_date must not be after end_date.")

    if normalised_start is not None:
        prepared = prepared.loc[prepared["date"].ge(normalised_start)]

    if normalised_end is not None:
        prepared = prepared.loc[prepared["date"].le(normalised_end)]

    remaining_portfolios = set(prepared["portfolio"])
    missing_after_filter = [
        portfolio
        for portfolio in selected_portfolios
        if portfolio not in remaining_portfolios
    ]

    if missing_after_filter:
        raise ValueError(
            "No performance observations remain for portfolios: "
            f"{missing_after_filter}"
        )

    prepared = _sort_portfolios(prepared, selected_portfolios)
    first_wealth = prepared.groupby("portfolio", sort=False)["wealth"].transform(
        "first"
    )

    if first_wealth.isna().any() or first_wealth.eq(0.0).any():
        raise ValueError("performance_risk contains an invalid initial wealth value.")

    prepared["indexed_wealth"] = prepared["wealth"] / first_wealth

    return prepared.loc[:, PERFORMANCE_HISTORY_COLUMNS]


def build_performance_summary(
    performance_risk: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Summarise filtered artifact returns with the shared metric functions."""
    history = prepare_performance_history(
        performance_risk,
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    portfolio_order = list(pd.unique(history["portfolio"]))
    rows = []

    for portfolio in portfolio_order:
        portfolio_data = history.loc[history["portfolio"].eq(portfolio)]
        metrics = summarise_returns(
            portfolio_data["return"],
            periods_per_year=periods_per_year,
        )
        rows.append(
            {
                "portfolio": portfolio,
                "start_date": portfolio_data["date"].min(),
                "end_date": portfolio_data["date"].max(),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def _latest_portfolio_rows(
    data: pd.DataFrame,
    *,
    name: str,
    portfolios: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    required_columns = {"portfolio", "date", *metric_columns}
    _require_columns(data, required_columns, name=name)
    prepared = data.loc[
        data["portfolio"].isin(portfolios),
        ["portfolio", "date", *metric_columns],
    ].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError(f"{name}.date contains invalid dates.")

    if prepared.duplicated(["portfolio", "date"]).any():
        raise ValueError(f"{name} contains duplicate portfolio-date rows.")

    missing_portfolios = [
        portfolio
        for portfolio in portfolios
        if portfolio not in set(prepared["portfolio"])
    ]

    if missing_portfolios:
        raise ValueError(f"{name} is missing portfolios: {missing_portfolios}")

    return (
        prepared.sort_values(["portfolio", "date"], kind="stable")
        .groupby("portfolio", sort=False, as_index=False)
        .tail(1)
        .rename(columns={"date": f"{name}_date"})
        .reset_index(drop=True)
    )


def build_latest_portfolio_snapshot(
    selected_implementations: pd.DataFrame,
    latest_overview: pd.DataFrame,
    performance_risk: pd.DataFrame,
    beta: pd.DataFrame,
    concentration: pd.DataFrame,
    implementation: pd.DataFrame,
    liquidity_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact latest-state table used by the dashboard overview."""
    _require_columns(
        selected_implementations,
        {"portfolio", "rebalance_frequency", "rebalance_offset", "role"},
        name="selected_implementations",
    )

    if selected_implementations["portfolio"].duplicated().any():
        raise ValueError("selected_implementations contains duplicate portfolios.")

    portfolios = selected_implementations["portfolio"].tolist()
    snapshot = selected_implementations[
        ["portfolio", "role", "rebalance_frequency", "rebalance_offset"]
    ].rename(columns={"role": "implementation_role"})

    _require_columns(
        latest_overview,
        {
            "entity_type",
            "entity",
            "overall_status",
            "market_risk_status",
            "concentration_status",
            "implementation_status",
        },
        name="latest_overview",
    )
    overview = latest_overview.loc[
        latest_overview["entity_type"].astype(str).str.casefold().eq("portfolio"),
        [
            "entity",
            "overall_status",
            "market_risk_status",
            "concentration_status",
            "implementation_status",
        ],
    ].rename(columns={"entity": "portfolio"})

    if overview["portfolio"].duplicated().any():
        raise ValueError("latest_overview contains duplicate portfolio rows.")

    missing_overviews = [
        portfolio
        for portfolio in portfolios
        if portfolio not in set(overview["portfolio"])
    ]

    if missing_overviews:
        raise ValueError(f"latest_overview is missing portfolios: {missing_overviews}")

    snapshot = snapshot.merge(
        overview, on="portfolio", how="left", validate="one_to_one"
    )
    state_inputs = {
        "performance_risk": (
            performance_risk,
            (
                "drawdown",
                "rolling_sharpe_252",
                "annualised_volatility_126",
            ),
        ),
        "beta": (
            beta,
            (
                "holdings_market_beta",
                "realised_gross_beta_126",
                "beta_measurement_gap",
            ),
        ),
        "concentration": (
            concentration,
            (
                "effective_position_count",
                "largest_absolute_sector_net_exposure",
                "top_five_absolute_beta_contribution_share",
                "effective_contribution_sector_count_63",
                "top_five_contributor_share_63",
            ),
        ),
        "implementation": (
            implementation,
            (
                "annualised_turnover_63",
                "largest_trade_weight_63",
                "minimum_trade_capacity_1pct_usd_63",
                "maximum_missing_return_weight_63",
            ),
        ),
        "liquidity_coverage": (
            liquidity_coverage,
            ("liquidity_coverage",),
        ),
    }

    for name, (data, metric_columns) in state_inputs.items():
        latest = _latest_portfolio_rows(
            data,
            name=name,
            portfolios=portfolios,
            metric_columns=metric_columns,
        )
        snapshot = snapshot.merge(
            latest,
            on="portfolio",
            how="left",
            validate="one_to_one",
        )

    latest_date_columns = [f"{name}_date" for name in state_inputs]
    aligned_dates = snapshot[latest_date_columns].nunique(axis=1, dropna=False).eq(1)

    if not aligned_dates.all():
        misaligned_portfolios = snapshot.loc[~aligned_dates, "portfolio"].tolist()
        raise ValueError(
            "Latest portfolio-state dates do not align for: " f"{misaligned_portfolios}"
        )

    snapshot["latest_date"] = snapshot["performance_risk_date"]

    return snapshot.loc[:, LATEST_PORTFOLIO_SNAPSHOT_COLUMNS].reset_index(drop=True)


SIGNAL_HEALTH_HISTORY_COLUMNS = (
    "date",
    "factor",
    "signal_coverage",
    "raw_iqr",
    "ic",
    "rolling_mean_ic_252",
    "rank_stability_1d",
    "rank_stability_21d",
)

FACTOR_DEPENDENCE_HISTORY_COLUMNS = (
    "date",
    "factor_rank_correlation",
    "observations",
    "rolling_factor_rank_correlation_252",
)

LATEST_FACTOR_SNAPSHOT_COLUMNS = (
    "factor",
    "latest_date",
    "overall_status",
    "signal_status",
    "signal_coverage",
    "raw_iqr",
    "ic_as_of_date",
    "ic",
    "rolling_mean_ic_252_as_of_date",
    "rolling_mean_ic_252",
    "rank_stability_1d",
    "rank_stability_21d",
)

_PREDICTIVE_SIGNAL_METRICS = (
    "ic",
    "rolling_mean_ic_252",
)


def _resolve_factors(
    data: pd.DataFrame,
    factors: Sequence[str] | None,
    *,
    name: str,
) -> list[str]:
    available = list(pd.unique(data["factor"].dropna()))

    if factors is None:
        return available

    if isinstance(factors, str):
        raise TypeError("factors must be a sequence of factor names.")

    selected = list(factors)

    if not selected:
        raise ValueError("factors must not be empty.")

    if len(selected) != len(set(selected)):
        raise ValueError("factors must contain unique names.")

    missing = sorted(set(selected) - set(available))

    if missing:
        raise ValueError(f"{name} is missing factors: {missing}")

    return selected


def _filter_dashboard_dates(
    data: pd.DataFrame,
    *,
    name: str,
    start_date: Any | None,
    end_date: Any | None,
) -> pd.DataFrame:
    normalised_start = _normalise_date_bound(start_date, name="start_date")
    normalised_end = _normalise_date_bound(end_date, name="end_date")

    if (
        normalised_start is not None
        and normalised_end is not None
        and normalised_start > normalised_end
    ):
        raise ValueError("start_date must not be after end_date.")

    result = data

    if normalised_start is not None:
        result = result.loc[result["date"].ge(normalised_start)]

    if normalised_end is not None:
        result = result.loc[result["date"].le(normalised_end)]

    if result.empty:
        raise ValueError(f"No {name} observations remain after date filtering.")

    return result.copy()


def _prepare_portfolio_monitoring_history(
    data: pd.DataFrame,
    *,
    columns: Sequence[str],
    name: str,
    portfolios: Sequence[str] | None,
    start_date: Any | None,
    end_date: Any | None,
) -> pd.DataFrame:
    _require_columns(data, set(columns), name=name)
    portfolio_order = _resolve_portfolios(
        data,
        portfolios,
        name=name,
    )

    prepared = data.loc[
        data["portfolio"].isin(portfolio_order),
        list(columns),
    ].copy()
    prepared["date"] = pd.to_datetime(
        prepared["date"],
        errors="coerce",
    )

    if prepared["date"].isna().any():
        raise ValueError(f"{name}.date contains invalid dates.")

    if prepared.duplicated(["portfolio", "date"]).any():
        raise ValueError(f"{name} contains duplicate portfolio-date rows.")

    metric_columns = [
        column for column in columns if column not in {"date", "portfolio"}
    ]

    for column in metric_columns:
        numeric_values = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )
        invalid_values = prepared[column].notna() & numeric_values.isna()

        if invalid_values.any():
            raise ValueError(f"{name}.{column} contains non-numeric values.")

        prepared[column] = numeric_values

    prepared = _filter_dashboard_dates(
        prepared,
        name=name.replace("_", "-"),
        start_date=start_date,
        end_date=end_date,
    )

    remaining_portfolios = set(prepared["portfolio"])
    missing_after_filter = [
        portfolio
        for portfolio in portfolio_order
        if portfolio not in remaining_portfolios
    ]

    if missing_after_filter:
        raise ValueError(
            f"No {name.replace('_', '-')} observations remain "
            f"for portfolios: {missing_after_filter}"
        )

    return _sort_portfolios(prepared, portfolio_order).loc[:, columns]


def prepare_beta_history(
    beta: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare validated beta-monitoring history for dashboard display."""
    return _prepare_portfolio_monitoring_history(
        beta,
        columns=BETA_HISTORY_COLUMNS,
        name="beta",
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )


def prepare_concentration_history(
    concentration: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare validated concentration history for dashboard display."""
    return _prepare_portfolio_monitoring_history(
        concentration,
        columns=CONCENTRATION_HISTORY_COLUMNS,
        name="concentration",
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )


def prepare_implementation_history(
    implementation: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare implementation history and a display-scale capacity field."""
    history = _prepare_portfolio_monitoring_history(
        implementation,
        columns=IMPLEMENTATION_SOURCE_COLUMNS,
        name="implementation",
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )

    history["minimum_trade_capacity_1pct_usd_millions_63"] = (
        history["minimum_trade_capacity_1pct_usd_63"] / 1_000_000.0
    )

    return history.loc[:, IMPLEMENTATION_HISTORY_COLUMNS]


def prepare_liquidity_coverage_history(
    liquidity_coverage: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare traded-weight liquidity coverage for dashboard display."""
    return _prepare_portfolio_monitoring_history(
        liquidity_coverage,
        columns=LIQUIDITY_COVERAGE_HISTORY_COLUMNS,
        name="liquidity_coverage",
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )


def _normalise_label_filter(
    values: Sequence[str] | None,
    *,
    name: str,
) -> list[str] | None:
    if values is None:
        return None

    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of labels.")

    selected = list(values)

    if not selected:
        raise ValueError(f"{name} must not be empty.")

    if len(selected) != len(set(selected)):
        raise ValueError(f"{name} must contain unique labels.")

    return selected


def _filter_dashboard_labels(
    data: pd.DataFrame,
    *,
    column: str,
    values: Sequence[str] | None,
    name: str,
    available_labels: set[str] | None = None,
) -> pd.DataFrame:
    selected = _normalise_label_filter(
        values,
        name=name,
    )

    if selected is None:
        return data

    available = (
        set(data[column].dropna()) if available_labels is None else available_labels
    )
    missing = sorted(set(selected) - available)

    if missing:
        raise ValueError(f"{name} contains unknown labels: {missing}")

    return data.loc[data[column].isin(selected)].copy()


def prepare_diagnostic_table(
    diagnostic_flags: pd.DataFrame,
    *,
    entity_types: Sequence[str] | None = None,
    entities: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Prepare detailed dashboard diagnostics with optional filters."""
    prepared = prepare_diagnostic_flags(diagnostic_flags)

    available_labels = {
        column: set(prepared[column].dropna()) for column in ("entity_type", "entity")
    }
    available_labels["category"] = set(MONITORING_CATEGORIES)
    available_labels["status"] = set(STATUS_SEVERITY)

    filters = (
        ("entity_type", entity_types, "entity_types"),
        ("entity", entities, "entities"),
        ("category", categories, "categories"),
        ("status", statuses, "statuses"),
    )

    for column, values, name in filters:
        prepared = _filter_dashboard_labels(
            prepared,
            column=column,
            values=values,
            name=name,
            available_labels=available_labels[column],
        )

    return prepared.loc[:, DIAGNOSTIC_TABLE_COLUMNS].reset_index(drop=True)


def prepare_monitoring_overview(
    latest_overview: pd.DataFrame,
    *,
    entity_types: Sequence[str] | None = None,
    entities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate and filter the materialised status overview."""
    _require_columns(
        latest_overview,
        set(MONITORING_OVERVIEW_COLUMNS),
        name="latest_overview",
    )

    prepared = latest_overview.loc[
        :,
        MONITORING_OVERVIEW_COLUMNS,
    ].copy()

    if prepared.empty:
        raise ValueError("latest_overview must not be empty.")

    if prepared.duplicated(["entity_type", "entity"]).any():
        raise ValueError("latest_overview contains duplicate entity rows.")

    for column in _MONITORING_COUNT_COLUMNS:
        numeric_values = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )
        invalid_values = prepared[column].notna() & numeric_values.isna()

        if invalid_values.any() or numeric_values.isna().any():
            raise ValueError(f"latest_overview.{column} " "contains invalid counts.")

        if numeric_values.lt(0).any() or numeric_values.mod(1).ne(0).any():
            raise ValueError(f"latest_overview.{column} " "contains invalid counts.")

        prepared[column] = numeric_values.astype(int)

    component_counts = prepared[
        [
            "passes",
            "warnings",
            "breaches",
            "unavailable",
        ]
    ].sum(axis=1)

    if not component_counts.eq(prepared["diagnostics"]).all():
        raise ValueError("latest_overview diagnostic counts " "do not reconcile.")

    allowed_statuses = set(STATUS_SEVERITY)
    unknown_overall = sorted(
        set(prepared["overall_status"].dropna()) - allowed_statuses
    )

    if unknown_overall or prepared["overall_status"].isna().any():
        raise ValueError(
            "latest_overview contains unknown " f"overall statuses: {unknown_overall}"
        )

    expected_overall = pd.Series(
        "PASS",
        index=prepared.index,
        dtype="object",
    )
    expected_overall.loc[prepared["warnings"].gt(0)] = "WARNING"
    expected_overall.loc[prepared["unavailable"].gt(0)] = "UNAVAILABLE"
    expected_overall.loc[prepared["breaches"].gt(0)] = "BREACH"

    if not prepared["overall_status"].eq(expected_overall).all():
        raise ValueError("latest_overview overall statuses " "do not reconcile.")

    allowed_category_statuses = {
        *allowed_statuses,
        "N/A",
    }

    for column in _MONITORING_CATEGORY_STATUS_COLUMNS:
        unknown = sorted(set(prepared[column].dropna()) - allowed_category_statuses)

        if unknown or prepared[column].isna().any():
            raise ValueError(
                f"latest_overview.{column} contains " f"unknown statuses: {unknown}"
            )

    available_labels = {
        "entity_type": set(prepared["entity_type"].dropna()),
        "entity": set(prepared["entity"].dropna()),
    }

    prepared = _filter_dashboard_labels(
        prepared,
        column="entity_type",
        values=entity_types,
        name="entity_types",
        available_labels=available_labels["entity_type"],
    )
    prepared = _filter_dashboard_labels(
        prepared,
        column="entity",
        values=entities,
        name="entities",
        available_labels=available_labels["entity"],
    )

    return prepared.loc[:, MONITORING_OVERVIEW_COLUMNS].reset_index(drop=True)


def prepare_signal_health_history(
    signal_health: pd.DataFrame,
    *,
    factors: Sequence[str] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare retained-signal histories for dashboard tables and figures."""
    _require_columns(
        signal_health,
        set(SIGNAL_HEALTH_HISTORY_COLUMNS),
        name="signal_health",
    )
    factor_order = _resolve_factors(
        signal_health,
        factors,
        name="signal_health",
    )
    prepared = signal_health.loc[
        signal_health["factor"].isin(factor_order),
        SIGNAL_HEALTH_HISTORY_COLUMNS,
    ].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError("signal_health.date contains invalid dates.")

    if prepared.duplicated(["factor", "date"]).any():
        raise ValueError("signal_health contains duplicate factor-date rows.")

    prepared = _filter_dashboard_dates(
        prepared,
        name="signal-health",
        start_date=start_date,
        end_date=end_date,
    )
    remaining_factors = set(prepared["factor"])
    missing_after_filter = [
        factor for factor in factor_order if factor not in remaining_factors
    ]

    if missing_after_filter:
        raise ValueError(
            "No signal-health observations remain for factors: "
            f"{missing_after_filter}"
        )

    positions = {factor: position for position, factor in enumerate(factor_order)}
    prepared["_factor_order"] = prepared["factor"].map(positions)

    return (
        prepared.sort_values(["_factor_order", "date"], kind="stable")
        .drop(columns="_factor_order")
        .reset_index(drop=True)
        .loc[:, SIGNAL_HEALTH_HISTORY_COLUMNS]
    )


def prepare_factor_dependence_history(
    factor_dependence: pd.DataFrame,
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Prepare retained-factor dependence history for dashboard display."""
    _require_columns(
        factor_dependence,
        set(FACTOR_DEPENDENCE_HISTORY_COLUMNS),
        name="factor_dependence",
    )
    prepared = factor_dependence.loc[:, FACTOR_DEPENDENCE_HISTORY_COLUMNS].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError("factor_dependence.date contains invalid dates.")

    if prepared["date"].duplicated().any():
        raise ValueError("factor_dependence contains duplicate dates.")

    prepared = _filter_dashboard_dates(
        prepared,
        name="factor-dependence",
        start_date=start_date,
        end_date=end_date,
    )

    return prepared.sort_values("date", kind="stable").reset_index(drop=True)


def build_latest_factor_snapshot(
    latest_overview: pd.DataFrame,
    signal_health: pd.DataFrame,
    *,
    factors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the compact latest retained-factor monitoring table.

    Predictive metrics use the latest non-missing observation because their
    forward-return labels are unavailable at the right edge of the signal
    history. Their metric-specific ``*_as_of_date`` columns make that lag
    explicit instead of presenting a stale value as a current-date value.
    """
    history = prepare_signal_health_history(
        signal_health,
        factors=factors,
    )
    factor_order = list(pd.unique(history["factor"]))

    latest = (
        history.sort_values(["factor", "date"], kind="stable")
        .groupby("factor", sort=False, as_index=False)
        .tail(1)
        .rename(columns={"date": "latest_date"})
        .drop(columns=list(_PREDICTIVE_SIGNAL_METRICS))
    )

    for metric in _PREDICTIVE_SIGNAL_METRICS:
        available = history.dropna(subset=[metric])

        latest_available = (
            available.sort_values(["factor", "date"], kind="stable")
            .groupby("factor", sort=False, as_index=False)
            .tail(1)
            .loc[:, ["factor", "date", metric]]
            .rename(columns={"date": f"{metric}_as_of_date"})
        )

        latest = latest.merge(
            latest_available,
            on="factor",
            how="left",
            validate="one_to_one",
        )

    _require_columns(
        latest_overview,
        {
            "entity_type",
            "entity",
            "overall_status",
            "signal_status",
        },
        name="latest_overview",
    )

    overview = latest_overview.loc[
        latest_overview["entity_type"].astype(str).str.casefold().eq("factor"),
        ["entity", "overall_status", "signal_status"],
    ].rename(columns={"entity": "factor"})

    if overview["factor"].duplicated().any():
        raise ValueError("latest_overview contains duplicate factor rows.")

    missing_overviews = [
        factor for factor in factor_order if factor not in set(overview["factor"])
    ]

    if missing_overviews:
        raise ValueError(f"latest_overview is missing factors: {missing_overviews}")

    snapshot = latest.merge(
        overview,
        on="factor",
        how="left",
        validate="one_to_one",
    )

    positions = {factor: position for position, factor in enumerate(factor_order)}
    snapshot["_factor_order"] = snapshot["factor"].map(positions)

    return (
        snapshot.sort_values("_factor_order", kind="stable")
        .drop(columns="_factor_order")
        .reset_index(drop=True)
        .loc[:, LATEST_FACTOR_SNAPSHOT_COLUMNS]
    )
