"""Pure presentation helpers shared by the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DASHBOARD_PAGES = (
    "Strategy Overview",
    "Performance & Drawdowns",
    "Factor & Signal Health",
    "Risk & Concentration",
    "Implementation & Liquidity",
    "Attribution",
)

FRESHNESS_TABLE_COLUMNS = (
    "group",
    "dataset",
    "latest_observation_date",
    "freshness_reference_date",
    "age_business_days",
    "is_stale",
    "status",
    "error",
)

_FRESHNESS_STATUS_ORDER = {
    "MISSING": 0,
    "READ_ERROR": 1,
    "INVALID": 2,
    "INVALID_GROUP": 3,
    "NOT_LOADED": 4,
    "STALE": 5,
    "READY": 6,
    "UNDATED": 7,
}


@dataclass(frozen=True)
class DashboardFilterOptions:
    """Validated portfolio choices and their common evaluation window."""

    portfolios: tuple[str, ...]
    minimum_date: pd.Timestamp
    maximum_date: pd.Timestamp


def _require_frame_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(f"{name} is missing columns: " f"{sorted(missing_columns)}")


def build_dashboard_filter_options(
    selected_implementations: pd.DataFrame,
    performance_risk: pd.DataFrame,
) -> DashboardFilterOptions:
    """Build ordered portfolios and their common dated coverage."""
    _require_frame_columns(
        selected_implementations,
        {"portfolio"},
        name="selected_implementations",
    )
    _require_frame_columns(
        performance_risk,
        {"portfolio", "date"},
        name="performance_risk",
    )

    if selected_implementations.empty:
        raise ValueError("selected_implementations must not be empty.")

    if selected_implementations["portfolio"].isna().any():
        raise ValueError("selected_implementations contains missing portfolios.")

    if selected_implementations["portfolio"].duplicated().any():
        raise ValueError("selected_implementations contains duplicate portfolios.")

    portfolios = tuple(selected_implementations["portfolio"].astype(str))
    performance = performance_risk.loc[
        performance_risk["portfolio"].isin(portfolios),
        ["portfolio", "date"],
    ].copy()
    performance["date"] = pd.to_datetime(
        performance["date"],
        errors="coerce",
    )

    if performance["date"].isna().any():
        raise ValueError("performance_risk.date contains invalid dates.")

    if performance.duplicated(["portfolio", "date"]).any():
        raise ValueError("performance_risk contains duplicate portfolio-date rows.")

    missing_portfolios = [
        portfolio
        for portfolio in portfolios
        if portfolio not in set(performance["portfolio"])
    ]

    if missing_portfolios:
        raise ValueError(
            "performance_risk is missing portfolios: " f"{missing_portfolios}"
        )

    coverage = performance.groupby(
        "portfolio",
        sort=False,
    )["date"].agg(
        start_date="min",
        end_date="max",
    )
    minimum_date = pd.Timestamp(coverage["start_date"].max()).normalize()
    maximum_date = pd.Timestamp(coverage["end_date"].min()).normalize()

    if minimum_date > maximum_date:
        raise ValueError("Selected portfolios have no common dated coverage.")

    return DashboardFilterOptions(
        portfolios=portfolios,
        minimum_date=minimum_date,
        maximum_date=maximum_date,
    )


def prepare_dashboard_freshness_table(
    metadata: pd.DataFrame,
    *,
    stale_only: bool = False,
) -> pd.DataFrame:
    """Prepare ordered artifact freshness metadata."""
    _require_frame_columns(
        metadata,
        set(FRESHNESS_TABLE_COLUMNS),
        name="metadata",
    )

    if not isinstance(stale_only, bool):
        raise TypeError("stale_only must be Boolean.")

    prepared = metadata.loc[
        :,
        FRESHNESS_TABLE_COLUMNS,
    ].copy()
    prepared["latest_observation_date"] = pd.to_datetime(
        prepared["latest_observation_date"],
        errors="coerce",
    )
    prepared["freshness_reference_date"] = pd.to_datetime(
        prepared["freshness_reference_date"],
        errors="coerce",
    )

    if stale_only:
        stale_mask = prepared["is_stale"].astype("boolean").fillna(False)
        prepared = prepared.loc[stale_mask]

    prepared["_status_order"] = (
        prepared["status"].map(_FRESHNESS_STATUS_ORDER).fillna(-1)
    )

    return (
        prepared.sort_values(
            [
                "_status_order",
                "group",
                "dataset",
            ],
            kind="stable",
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )
