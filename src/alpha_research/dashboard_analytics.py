"""Reusable dashboard tables and chart-data preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from alpha_research.config.research import TRADING_DAYS_PER_YEAR
from alpha_research.metrics import summarise_returns

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
