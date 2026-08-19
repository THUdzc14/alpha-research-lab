"""Diagnostic calibration and monitoring-status aggregation.

Measurement modules such as :mod:`alpha_research.validation` calculate the
underlying histories. This module interprets those histories using the
structural and historically calibrated rules frozen in Notebook 08.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from alpha_research.config.research import (
    FACTOR_COLUMNS,
    MONITORING_SPECIFICATION,
    TRADING_DAYS_PER_YEAR,
)
from alpha_research.costs import (
    calculate_security_trade_capacity,
    prepare_lagged_dollar_volume,
)
from alpha_research.metrics import calculate_rolling_return_state

STATUS_SEVERITY = {
    "PASS": 0,
    "WARNING": 1,
    "UNAVAILABLE": 2,
    "BREACH": 3,
}
SEVERITY_STATUS = {severity: status for status, severity in STATUS_SEVERITY.items()}

MONITORING_CATEGORIES = (
    "Signal",
    "Market risk",
    "Concentration",
    "Implementation",
)
MONITORING_STATUS_COLUMNS = (
    "signal_status",
    "market_risk_status",
    "concentration_status",
    "implementation_status",
)

DIAGNOSTIC_KEY_COLUMNS = (
    "entity_type",
    "entity",
    "category",
    "diagnostic",
)
DIAGNOSTIC_FLAG_COLUMNS = (
    *DIAGNOSTIC_KEY_COLUMNS,
    "latest_date",
    "latest_value",
    "adverse_direction",
    "calibration",
    "threshold_value",
    "historical_percentile",
    "status",
    "notes",
)
DIAGNOSTIC_FLAG_EXPORT_COLUMNS = (
    *DIAGNOSTIC_FLAG_COLUMNS[:-1],
    "status_severity",
    "notes",
)
ACTIVE_DIAGNOSTIC_COLUMNS = DIAGNOSTIC_FLAG_COLUMNS


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a scalar monitoring input is missing."""
    missing = pd.isna(value)

    if not isinstance(missing, (bool, np.bool_)):
        raise TypeError("Monitoring values must be scalars.")

    return bool(missing)


def _validate_flag_labels(
    *,
    entity_type: str,
    entity: str,
    category: str,
    diagnostic: str,
) -> None:
    labels = {
        "entity_type": entity_type,
        "entity": entity,
        "category": category,
        "diagnostic": diagnostic,
    }
    invalid = [
        name
        for name, value in labels.items()
        if not isinstance(value, str) or not value
    ]

    if invalid:
        raise ValueError(f"Diagnostic labels must be non-empty strings: {invalid}")

    if category not in MONITORING_CATEGORIES:
        raise ValueError(
            f"category must be one of {MONITORING_CATEGORIES}; got {category!r}."
        )


def create_structural_flag(
    *,
    entity_type: str,
    entity: str,
    category: str,
    diagnostic: str,
    latest_date: Any,
    latest_value: Any,
    passes: bool | None,
    calibration: str,
    threshold_value: float = np.nan,
    notes: str = "",
) -> dict[str, Any]:
    """Create one PASS/BREACH structural diagnostic.

    Missing measurements or an indeterminate pass condition produce
    ``UNAVAILABLE`` rather than a breach.
    """
    _validate_flag_labels(
        entity_type=entity_type,
        entity=entity,
        category=category,
        diagnostic=diagnostic,
    )

    if not isinstance(calibration, str) or not calibration:
        raise ValueError("calibration must be a non-empty string.")

    if not isinstance(notes, str):
        raise TypeError("notes must be a string.")

    if _is_missing_scalar(latest_value) or passes is None or _is_missing_scalar(passes):
        status = "UNAVAILABLE"
    elif isinstance(passes, (bool, np.bool_)):
        status = "PASS" if bool(passes) else "BREACH"
    else:
        raise TypeError("passes must be a Boolean or None.")

    return {
        "entity_type": entity_type,
        "entity": entity,
        "category": category,
        "diagnostic": diagnostic,
        "latest_date": pd.to_datetime(latest_date),
        "latest_value": latest_value,
        "adverse_direction": "Structural failure",
        "calibration": calibration,
        "threshold_value": threshold_value,
        "historical_percentile": np.nan,
        "status": status,
        "notes": notes,
    }


def create_historical_flag(
    *,
    entity_type: str,
    entity: str,
    category: str,
    diagnostic: str,
    latest_date: Any,
    latest_value: Any,
    history: Iterable[float] | pd.Series,
    adverse_direction: str,
    lower_tail: float = MONITORING_SPECIFICATION.historical_lower_tail,
    upper_tail: float = MONITORING_SPECIFICATION.historical_upper_tail,
    notes: str = "",
) -> dict[str, Any]:
    """Create a historically calibrated PASS/WARNING diagnostic.

    Calibration uses the supplied history without date substitution.  A
    missing latest value or unusable history produces ``UNAVAILABLE``; adverse
    values must cross the selected tail threshold strictly to produce a warning.
    """
    _validate_flag_labels(
        entity_type=entity_type,
        entity=entity,
        category=category,
        diagnostic=diagnostic,
    )

    if adverse_direction not in {"Higher", "Lower"}:
        raise ValueError("adverse_direction must be 'Higher' or 'Lower'.")

    if not 0.0 < lower_tail < upper_tail < 1.0:
        raise ValueError("Historical tails must satisfy 0 < lower < upper < 1.")

    if not isinstance(notes, str):
        raise TypeError("notes must be a string.")

    valid_history = pd.to_numeric(
        pd.Series(history, copy=True),
        errors="coerce",
    ).dropna()

    history_is_available = not valid_history.empty and valid_history.nunique() > 1
    latest_is_available = not _is_missing_scalar(latest_value)

    if not history_is_available:
        threshold_value = np.nan
        historical_percentile = np.nan
        calibration = "Historical calibration unavailable"
        status = "UNAVAILABLE"
    else:
        if adverse_direction == "Higher":
            threshold_value = float(valid_history.quantile(upper_tail))
            calibration = f"Historical upper {1.0 - upper_tail:.0%} tail"
            warning = latest_value > threshold_value if latest_is_available else False
        else:
            threshold_value = float(valid_history.quantile(lower_tail))
            calibration = f"Historical lower {lower_tail:.0%} tail"
            warning = latest_value < threshold_value if latest_is_available else False

        if latest_is_available:
            historical_percentile = float(valid_history.le(latest_value).mean())
            status = "WARNING" if warning else "PASS"
        else:
            historical_percentile = np.nan
            status = "UNAVAILABLE"

    return {
        "entity_type": entity_type,
        "entity": entity,
        "category": category,
        "diagnostic": diagnostic,
        "latest_date": pd.to_datetime(latest_date),
        "latest_value": latest_value,
        "adverse_direction": adverse_direction,
        "calibration": calibration,
        "threshold_value": threshold_value,
        "historical_percentile": historical_percentile,
        "status": status,
        "notes": notes,
    }


def prepare_diagnostic_flags(
    flags: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    entity_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate, rank, and deterministically sort diagnostic flags."""
    prepared = pd.DataFrame(flags).copy()
    missing_columns = set(DIAGNOSTIC_FLAG_COLUMNS) - set(prepared.columns)

    if missing_columns:
        raise KeyError(
            f"Diagnostic flags are missing columns: {sorted(missing_columns)}"
        )

    prepared = prepared.loc[:, DIAGNOSTIC_FLAG_COLUMNS].copy()

    if prepared.empty:
        result = prepared.copy()
        result["status_severity"] = pd.Series(dtype="int64")
        return result.loc[:, DIAGNOSTIC_FLAG_EXPORT_COLUMNS]

    if prepared.duplicated(list(DIAGNOSTIC_KEY_COLUMNS)).any():
        raise ValueError("Duplicate diagnostic flags found.")

    unknown_statuses = sorted(set(prepared["status"].dropna()) - set(STATUS_SEVERITY))

    if unknown_statuses or prepared["status"].isna().any():
        raise ValueError(f"Unknown diagnostic statuses: {unknown_statuses}")

    unknown_categories = sorted(
        set(prepared["category"].dropna()) - set(MONITORING_CATEGORIES)
    )

    if unknown_categories or prepared["category"].isna().any():
        raise ValueError(f"Unknown monitoring categories: {unknown_categories}")

    prepared["latest_date"] = pd.to_datetime(prepared["latest_date"])
    prepared["status_severity"] = prepared["status"].map(STATUS_SEVERITY).astype(int)

    if entity_order is None:
        ordered_entities = list(dict.fromkeys(prepared["entity"]))
    else:
        ordered_entities = list(entity_order)

        if len(set(ordered_entities)) != len(ordered_entities):
            raise ValueError("entity_order must contain unique values.")

        missing_entities = sorted(set(prepared["entity"]) - set(ordered_entities))

        if missing_entities:
            raise ValueError(
                f"entity_order is missing monitored entities: {missing_entities}"
            )

    entity_positions = {
        entity: position for position, entity in enumerate(ordered_entities)
    }
    category_positions = {
        category: position for position, category in enumerate(MONITORING_CATEGORIES)
    }

    prepared["_entity_order"] = prepared["entity"].map(entity_positions)
    prepared["_category_order"] = prepared["category"].map(category_positions)

    return (
        prepared.sort_values(
            ["_entity_order", "_category_order", "diagnostic"],
            kind="stable",
        )
        .drop(columns=["_entity_order", "_category_order"])
        .reset_index(drop=True)
        .loc[:, DIAGNOSTIC_FLAG_EXPORT_COLUMNS]
    )


def build_signal_diagnostic_flags(
    signal_health: pd.DataFrame,
    *,
    factors: Sequence[str] = tuple(FACTOR_COLUMNS),
    min_observations: int = (
        MONITORING_SPECIFICATION.minimum_cross_sectional_observations
    ),
) -> pd.DataFrame:
    """Build Notebook 08's structural and predictive flags for each factor.

    Structural coverage uses the latest signal date.  Predictive diagnostics use
    the latest date with an observable rolling IC, preserving their distinct
    as-of-date semantics.
    """
    if min_observations <= 0:
        raise ValueError("min_observations must be positive.")

    factor_order = list(factors)

    if not factor_order or len(set(factor_order)) != len(factor_order):
        raise ValueError("factors must contain unique factor names.")

    required_columns = {
        "factor",
        "date",
        "universe_observations",
        "signal_observations",
        "signal_coverage",
        "rolling_mean_ic_252",
    }
    missing_columns = required_columns - set(signal_health.columns)

    if missing_columns:
        raise KeyError(f"signal_health is missing columns: {sorted(missing_columns)}")

    prepared = signal_health.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])

    if prepared.duplicated(["factor", "date"]).any():
        raise ValueError("signal_health contains duplicate factor-date rows.")

    unknown_factors = sorted(set(factor_order) - set(prepared["factor"]))

    if unknown_factors:
        raise ValueError(f"signal_health is missing factors: {unknown_factors}")

    flag_rows: list[dict[str, Any]] = []

    for factor_name in factor_order:
        factor_data = (
            prepared.loc[prepared["factor"].eq(factor_name)].sort_values("date").copy()
        )
        latest_row = factor_data.iloc[-1]
        universe_observations = latest_row["universe_observations"]
        signal_observations = latest_row["signal_observations"]

        if pd.notna(universe_observations) and universe_observations > 0:
            required_coverage = min_observations / universe_observations
        else:
            required_coverage = np.nan

        if pd.notna(signal_observations) and pd.notna(universe_observations):
            coverage_notes = (
                f"{int(signal_observations)} of "
                f"{int(universe_observations)} observations available"
            )
            coverage_passes: bool | None = bool(signal_observations >= min_observations)
        else:
            coverage_notes = "Signal observation counts unavailable"
            coverage_passes = None

        flag_rows.append(
            create_structural_flag(
                entity_type="Factor",
                entity=factor_name,
                category="Signal",
                diagnostic="Signal coverage",
                latest_date=latest_row["date"],
                latest_value=latest_row["signal_coverage"],
                passes=coverage_passes,
                calibration=(
                    f"At least {min_observations} cross-sectional observations"
                ),
                threshold_value=required_coverage,
                notes=coverage_notes,
            )
        )

        predictive_history = factor_data.dropna(subset=["rolling_mean_ic_252"])

        if predictive_history.empty:
            latest_predictive_date = factor_data["date"].max()
            latest_predictive_value = np.nan
        else:
            latest_predictive_row = predictive_history.iloc[-1]
            latest_predictive_date = latest_predictive_row["date"]
            latest_predictive_value = latest_predictive_row["rolling_mean_ic_252"]

        flag_rows.append(
            create_historical_flag(
                entity_type="Factor",
                entity=factor_name,
                category="Signal",
                diagnostic="252-observation rolling mean IC",
                latest_date=latest_predictive_date,
                latest_value=latest_predictive_value,
                history=predictive_history["rolling_mean_ic_252"],
                adverse_direction="Lower",
                notes=("Uses the latest date with observable five-day forward returns"),
            )
        )

    return prepare_diagnostic_flags(flag_rows, entity_order=factor_order)


def summarise_diagnostic_flags(diagnostic_flags: pd.DataFrame) -> pd.DataFrame:
    """Summarise diagnostic counts and worst status by monitored entity."""
    prepared = prepare_diagnostic_flags(diagnostic_flags)

    if prepared.empty:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []], names=["entity_type", "entity"]
        )
        return pd.DataFrame(
            columns=[
                "diagnostics",
                "passes",
                "warnings",
                "breaches",
                "unavailable",
                "overall_status",
            ],
            index=empty_index,
        )

    summary = prepared.groupby(["entity_type", "entity"], sort=False).agg(
        diagnostics=("status", "size"),
        passes=("status", lambda values: values.eq("PASS").sum()),
        warnings=("status", lambda values: values.eq("WARNING").sum()),
        breaches=("status", lambda values: values.eq("BREACH").sum()),
        unavailable=(
            "status",
            lambda values: values.eq("UNAVAILABLE").sum(),
        ),
        maximum_status_severity=("status_severity", "max"),
    )
    summary["overall_status"] = summary["maximum_status_severity"].map(SEVERITY_STATUS)

    return summary.drop(columns="maximum_status_severity")


def build_monitoring_flag_matrix(diagnostic_flags: pd.DataFrame) -> pd.DataFrame:
    """Return each entity's worst status within every monitoring category."""
    prepared = prepare_diagnostic_flags(diagnostic_flags)

    if prepared.empty:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []], names=["entity_type", "entity"]
        )
        return pd.DataFrame(columns=MONITORING_CATEGORIES, index=empty_index)

    category_status = (
        prepared.groupby(
            ["entity_type", "entity", "category"],
            sort=False,
        )["status_severity"]
        .max()
        .map(SEVERITY_STATUS)
        .rename("status")
        .reset_index()
    )

    return category_status.pivot(
        index=["entity_type", "entity"],
        columns="category",
        values="status",
    ).reindex(columns=MONITORING_CATEGORIES)


def select_active_diagnostic_flags(
    diagnostic_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Return non-passing flags ordered from most to least severe."""
    prepared = prepare_diagnostic_flags(diagnostic_flags)

    return (
        prepared.loc[prepared["status"].ne("PASS")]
        .sort_values(
            [
                "status_severity",
                "entity_type",
                "entity",
                "category",
                "diagnostic",
            ],
            ascending=[False, True, True, True, True],
            kind="stable",
        )
        .loc[:, ACTIVE_DIAGNOSTIC_COLUMNS]
        .reset_index(drop=True)
    )


def build_latest_monitoring_overview(
    diagnostic_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Build the dashboard-ready latest monitoring overview."""
    summary = summarise_diagnostic_flags(diagnostic_flags)
    flag_matrix = build_monitoring_flag_matrix(diagnostic_flags)

    overview = (
        summary.join(flag_matrix, how="left")
        .rename(
            columns={
                "Signal": "signal_status",
                "Market risk": "market_risk_status",
                "Concentration": "concentration_status",
                "Implementation": "implementation_status",
            }
        )
        .reset_index()
    )
    overview.columns.name = None
    status_columns = list(MONITORING_STATUS_COLUMNS)
    overview[status_columns] = overview[status_columns].astype("object").fillna("N/A")

    return overview


PORTFOLIO_CONCENTRATION_FLAG_SPECIFICATIONS = {
    "largest_absolute_sector_net_exposure": (
        "Largest absolute sector net exposure",
        "Higher",
    ),
    "top_five_absolute_beta_contribution_share": (
        "Top-five absolute beta-contribution share",
        "Higher",
    ),
    "top_five_contributor_share_63": (
        "Top-five 63-day security-contribution share",
        "Higher",
    ),
    "effective_contribution_sector_count_63": (
        "Effective 63-day contribution sectors",
        "Lower",
    ),
}


PORTFOLIO_IMPLEMENTATION_FLAG_SPECIFICATIONS = {
    "annualised_turnover_63": (
        "Annualised trailing turnover",
        "Higher",
    ),
    "largest_trade_weight_63": (
        "Largest trailing trade weight",
        "Higher",
    ),
    "minimum_trade_capacity_1pct_usd_63": (
        "Minimum trailing capacity at 1%",
        "Lower",
    ),
}


def build_portfolio_diagnostic_flags(
    performance_risk: pd.DataFrame,
    beta_state: pd.DataFrame,
    concentration_state: pd.DataFrame,
    implementation_state: pd.DataFrame,
    liquidity_coverage: pd.DataFrame,
    *,
    portfolios: Sequence[str],
    structural_coverage_tolerance: float = (
        MONITORING_SPECIFICATION.structural_coverage_tolerance
    ),
) -> pd.DataFrame:
    """Build Notebook 08's 13 diagnostics for each selected portfolio."""
    portfolio_order = list(portfolios)

    if not portfolio_order or len(portfolio_order) != len(set(portfolio_order)):
        raise ValueError("portfolios must contain unique portfolio names.")

    required_inputs = {
        "performance_risk": (
            performance_risk,
            {
                "portfolio",
                "date",
                "annualised_volatility_126",
                "drawdown",
            },
        ),
        "beta_state": (
            beta_state,
            {
                "portfolio",
                "date",
                "beta_coverage",
                "holdings_market_beta",
            },
        ),
        "concentration_state": (
            concentration_state,
            {
                "portfolio",
                "date",
                *PORTFOLIO_CONCENTRATION_FLAG_SPECIFICATIONS,
            },
        ),
        "implementation_state": (
            implementation_state,
            {
                "portfolio",
                "date",
                "maximum_missing_return_weight_63",
                *PORTFOLIO_IMPLEMENTATION_FLAG_SPECIFICATIONS,
            },
        ),
        "liquidity_coverage": (
            liquidity_coverage,
            {
                "portfolio",
                "date",
                "liquidity_coverage",
            },
        ),
    }

    prepared_inputs = {}

    for name, (data, required_columns) in required_inputs.items():
        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            raise KeyError(f"{name} is missing columns: " f"{sorted(missing_columns)}")

        prepared = data.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])

        if prepared.duplicated(
            [
                "portfolio",
                "date",
            ]
        ).any():
            raise ValueError(f"{name} contains duplicate portfolio-date rows.")

        missing_portfolios = sorted(set(portfolio_order) - set(prepared["portfolio"]))

        if missing_portfolios:
            raise ValueError(f"{name} is missing portfolios: " f"{missing_portfolios}")

        prepared_inputs[name] = prepared

    flag_rows = []

    for portfolio_name in portfolio_order:
        portfolio_risk = (
            prepared_inputs["performance_risk"]
            .loc[lambda data: data["portfolio"].eq(portfolio_name)]
            .sort_values("date")
        )

        portfolio_beta = (
            prepared_inputs["beta_state"]
            .loc[lambda data: data["portfolio"].eq(portfolio_name)]
            .sort_values("date")
        )

        portfolio_concentration = (
            prepared_inputs["concentration_state"]
            .loc[lambda data: data["portfolio"].eq(portfolio_name)]
            .sort_values("date")
        )

        portfolio_implementation = (
            prepared_inputs["implementation_state"]
            .loc[lambda data: data["portfolio"].eq(portfolio_name)]
            .sort_values("date")
        )

        portfolio_liquidity = (
            prepared_inputs["liquidity_coverage"]
            .loc[lambda data: data["portfolio"].eq(portfolio_name)]
            .sort_values("date")
        )

        latest_risk = portfolio_risk.iloc[-1]
        latest_beta = portfolio_beta.iloc[-1]
        latest_concentration = portfolio_concentration.iloc[-1]
        latest_implementation = portfolio_implementation.iloc[-1]
        latest_liquidity = portfolio_liquidity.iloc[-1]

        # Structural diagnostics
        flag_rows.append(
            create_structural_flag(
                entity_type="Portfolio",
                entity=portfolio_name,
                category="Market risk",
                diagnostic="Holdings beta coverage",
                latest_date=latest_beta["date"],
                latest_value=latest_beta["beta_coverage"],
                passes=(
                    latest_beta["beta_coverage"] >= 1.0 - structural_coverage_tolerance
                ),
                calibration="Complete beta coverage",
                threshold_value=1.0,
            )
        )

        flag_rows.append(
            create_structural_flag(
                entity_type="Portfolio",
                entity=portfolio_name,
                category="Implementation",
                diagnostic="Liquidity coverage",
                latest_date=latest_liquidity["date"],
                latest_value=latest_liquidity["liquidity_coverage"],
                passes=(
                    latest_liquidity["liquidity_coverage"]
                    >= 1.0 - structural_coverage_tolerance
                ),
                calibration="Complete traded-weight coverage",
                threshold_value=1.0,
            )
        )

        flag_rows.append(
            create_structural_flag(
                entity_type="Portfolio",
                entity=portfolio_name,
                category="Implementation",
                diagnostic="Trailing missing-return weight",
                latest_date=latest_implementation["date"],
                latest_value=latest_implementation["maximum_missing_return_weight_63"],
                passes=(
                    latest_implementation["maximum_missing_return_weight_63"]
                    <= structural_coverage_tolerance
                ),
                calibration="No missing-return exposure",
                threshold_value=0.0,
            )
        )

        # Market-risk diagnostics
        flag_rows.append(
            create_historical_flag(
                entity_type="Portfolio",
                entity=portfolio_name,
                category="Market risk",
                diagnostic="Annualised volatility (126 days)",
                latest_date=latest_risk["date"],
                latest_value=latest_risk["annualised_volatility_126"],
                history=portfolio_risk["annualised_volatility_126"],
                adverse_direction="Higher",
            )
        )

        flag_rows.append(
            create_historical_flag(
                entity_type="Portfolio",
                entity=portfolio_name,
                category="Market risk",
                diagnostic="Current drawdown severity",
                latest_date=latest_risk["date"],
                latest_value=-latest_risk["drawdown"],
                history=-portfolio_risk["drawdown"],
                adverse_direction="Higher",
            )
        )

        flag_rows.append(
            create_historical_flag(
                entity_type="Portfolio",
                entity=portfolio_name,
                category="Market risk",
                diagnostic="Absolute holdings-implied beta",
                latest_date=latest_beta["date"],
                latest_value=abs(latest_beta["holdings_market_beta"]),
                history=portfolio_beta["holdings_market_beta"].abs(),
                adverse_direction="Higher",
                notes=(
                    "Holdings-implied beta is used as the "
                    "contemporaneous exposure measure"
                ),
            )
        )

        # Concentration diagnostics
        for (
            metric,
            (
                diagnostic,
                adverse_direction,
            ),
        ) in PORTFOLIO_CONCENTRATION_FLAG_SPECIFICATIONS.items():
            flag_rows.append(
                create_historical_flag(
                    entity_type="Portfolio",
                    entity=portfolio_name,
                    category="Concentration",
                    diagnostic=diagnostic,
                    latest_date=latest_concentration["date"],
                    latest_value=latest_concentration[metric],
                    history=portfolio_concentration[metric],
                    adverse_direction=adverse_direction,
                )
            )

        # Implementation diagnostics
        for (
            metric,
            (
                diagnostic,
                adverse_direction,
            ),
        ) in PORTFOLIO_IMPLEMENTATION_FLAG_SPECIFICATIONS.items():
            flag_rows.append(
                create_historical_flag(
                    entity_type="Portfolio",
                    entity=portfolio_name,
                    category="Implementation",
                    diagnostic=diagnostic,
                    latest_date=latest_implementation["date"],
                    latest_value=latest_implementation[metric],
                    history=portfolio_implementation[metric],
                    adverse_direction=adverse_direction,
                )
            )

    return prepare_diagnostic_flags(
        flag_rows,
        entity_order=portfolio_order,
    )


def build_strategy_diagnostic_flags(
    signal_health: pd.DataFrame,
    performance_risk: pd.DataFrame,
    beta_state: pd.DataFrame,
    concentration_state: pd.DataFrame,
    implementation_state: pd.DataFrame,
    liquidity_coverage: pd.DataFrame,
    *,
    factors: Sequence[str] = tuple(FACTOR_COLUMNS),
    portfolios: Sequence[str],
) -> pd.DataFrame:
    """Compose all factor and portfolio diagnostics."""
    factor_order = list(factors)
    portfolio_order = list(portfolios)

    signal_flags = build_signal_diagnostic_flags(
        signal_health,
        factors=factor_order,
    )

    portfolio_flags = build_portfolio_diagnostic_flags(
        performance_risk,
        beta_state,
        concentration_state,
        implementation_state,
        liquidity_coverage,
        portfolios=portfolio_order,
    )

    return prepare_diagnostic_flags(
        pd.concat(
            [
                signal_flags,
                portfolio_flags,
            ],
            ignore_index=True,
        ),
        entity_order=[
            *factor_order,
            *portfolio_order,
        ],
    )


def _require_monitoring_columns(
    data: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    missing_columns = required - set(data.columns)

    if missing_columns:
        raise KeyError(f"{name} is missing columns: " f"{sorted(missing_columns)}")


def _select_monitoring_portfolios(
    data: pd.DataFrame,
    portfolios: Sequence[str] | None,
    *,
    name: str,
) -> list[str]:
    available_portfolios = list(pd.unique(data["portfolio"].dropna()))

    if portfolios is None:
        return available_portfolios

    selected_portfolios = list(portfolios)

    if not selected_portfolios or len(selected_portfolios) != len(
        set(selected_portfolios)
    ):
        raise ValueError("portfolios must contain unique portfolio names.")

    missing_portfolios = sorted(set(selected_portfolios) - set(available_portfolios))

    if missing_portfolios:
        raise ValueError(f"{name} is missing portfolios: " f"{missing_portfolios}")

    return selected_portfolios


def calculate_performance_risk_state(
    portfolio_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    portfolio_return_column: str = "net_return",
    benchmark_return_column: str = "benchmark_return",
    benchmark_name: str = "SPY",
    performance_window: int = (MONITORING_SPECIFICATION.performance_window),
    risk_window: int = MONITORING_SPECIFICATION.risk_window,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build rolling portfolio and benchmark risk histories.

    Each trailing statistic includes returns through its reported date.  The
    benchmark is processed through the same calculation path as the selected
    portfolios so output definitions remain comparable.
    """
    _require_monitoring_columns(
        portfolio_daily,
        {
            "portfolio",
            "date",
            portfolio_return_column,
        },
        name="portfolio_daily",
    )

    _require_monitoring_columns(
        benchmark_daily,
        {
            "date",
            benchmark_return_column,
        },
        name="benchmark_daily",
    )

    portfolio_order = _select_monitoring_portfolios(
        portfolio_daily,
        portfolios,
        name="portfolio_daily",
    )

    if not isinstance(benchmark_name, str) or not benchmark_name:
        raise ValueError("benchmark_name must be a non-empty string.")

    if benchmark_name in portfolio_order:
        raise ValueError("benchmark_name must differ from portfolio names.")

    selected_portfolios = portfolio_daily.loc[
        portfolio_daily["portfolio"].isin(portfolio_order),
        [
            "date",
            "portfolio",
            portfolio_return_column,
        ],
    ].copy()

    selected_portfolios["date"] = pd.to_datetime(selected_portfolios["date"])

    if selected_portfolios.duplicated(
        [
            "portfolio",
            "date",
        ]
    ).any():
        raise ValueError("portfolio_daily contains duplicate portfolio-date rows.")

    selected_portfolios = selected_portfolios.rename(
        columns={
            portfolio_return_column: "return",
        }
    )

    benchmark = benchmark_daily[
        [
            "date",
            benchmark_return_column,
        ]
    ].copy()

    benchmark["date"] = pd.to_datetime(benchmark["date"])

    if benchmark["date"].duplicated().any():
        raise ValueError("benchmark_daily contains duplicate dates.")

    benchmark = benchmark.rename(
        columns={
            benchmark_return_column: "return",
        }
    ).assign(portfolio=benchmark_name)

    combined_returns = pd.concat(
        [
            selected_portfolios,
            benchmark,
        ],
        ignore_index=True,
    )

    return_states = []

    for portfolio_name in [
        *portfolio_order,
        benchmark_name,
    ]:
        portfolio_returns = combined_returns.loc[
            combined_returns["portfolio"].eq(portfolio_name)
        ]

        return_states.append(
            calculate_rolling_return_state(
                portfolio_returns,
                return_column="return",
                date_column="date",
                performance_window=performance_window,
                risk_window=risk_window,
                periods_per_year=periods_per_year,
            )
        )

    return (
        pd.concat(
            return_states,
            ignore_index=True,
        )
        .sort_values(
            [
                "portfolio",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_implementation_monitoring_state(
    security_daily: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    implementation_window: int = (MONITORING_SPECIFICATION.implementation_window),
    liquidity_window: int = (MONITORING_SPECIFICATION.monitoring_liquidity_window),
    liquidity_min_periods: int = (
        MONITORING_SPECIFICATION.monitoring_liquidity_min_periods
    ),
    participation_rate: float = (MONITORING_SPECIFICATION.capacity_participation_rate),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    tolerance: float = (MONITORING_SPECIFICATION.numerical_tolerance),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build implementation history and liquidity coverage.

    Returns
    -------
    implementation_state
        Daily and rolling turnover, cost, trade-size, capacity and
        missing-return measurements.
    liquidity_coverage
        Daily traded-weight coverage by valid lagged liquidity data.
    """
    if implementation_window <= 0:
        raise ValueError("implementation_window must be positive.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    _require_monitoring_columns(
        security_daily,
        {
            "portfolio",
            "date",
            "ticker",
            "trade",
            "absolute_trade_weight",
            "transaction_cost_contribution",
            "return_record_missing",
            "missing_return_weight_contribution",
        },
        name="security_daily",
    )

    _require_monitoring_columns(
        market_data,
        {
            "date",
            "ticker",
            "dollar_volume",
        },
        name="market_data",
    )

    portfolio_order = _select_monitoring_portfolios(
        security_daily,
        portfolios,
        name="security_daily",
    )

    security = security_daily.loc[
        security_daily["portfolio"].isin(portfolio_order)
    ].copy()

    security["date"] = pd.to_datetime(security["date"])

    if security.duplicated(
        [
            "portfolio",
            "date",
            "ticker",
        ]
    ).any():
        raise ValueError(
            "security_daily contains duplicate portfolio-date-ticker rows."
        )

    numeric_columns = [
        "trade",
        "absolute_trade_weight",
        "transaction_cost_contribution",
        "missing_return_weight_contribution",
    ]

    for column in numeric_columns:
        security[column] = pd.to_numeric(
            security[column],
            errors="raise",
        )

    if security[numeric_columns].isna().any().any():
        raise ValueError("Security implementation columns contain missing values.")

    trade_weight_difference = (
        security["absolute_trade_weight"] - security["trade"].abs()
    ).abs()

    if trade_weight_difference.gt(tolerance).any():
        raise ValueError("absolute_trade_weight does not reconcile with abs(trade).")

    if security["absolute_trade_weight"].lt(-tolerance).any():
        raise ValueError("absolute_trade_weight must be non-negative.")

    if security["transaction_cost_contribution"].lt(-tolerance).any():
        raise ValueError("transaction_cost_contribution must be non-negative.")

    if security["missing_return_weight_contribution"].lt(-tolerance).any():
        raise ValueError("missing_return_weight_contribution must be non-negative.")

    liquidity_column = f"lagged_median_dollar_volume_{liquidity_window}"

    liquidity_panel = prepare_lagged_dollar_volume(
        market_data[
            [
                "date",
                "ticker",
                "dollar_volume",
            ]
        ],
        window=liquidity_window,
        min_periods=liquidity_min_periods,
        aggregation="median",
        output_column=liquidity_column,
    )

    implementation_security = security.merge(
        liquidity_panel[
            [
                "date",
                "ticker",
                liquidity_column,
            ]
        ],
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="many_to_one",
    )

    implementation_security = calculate_security_trade_capacity(
        implementation_security,
        participation_rate=participation_rate,
        trade_weight_column="absolute_trade_weight",
        liquidity_column=liquidity_column,
        output_column="trade_capacity_usd",
        tolerance=tolerance,
    )

    implementation_security["trade_indicator"] = implementation_security[
        "absolute_trade_weight"
    ].gt(tolerance)

    valid_liquidity = implementation_security[liquidity_column].gt(0.0)

    implementation_security["liquidity_covered_trade_weight"] = np.where(
        valid_liquidity,
        implementation_security["absolute_trade_weight"],
        0.0,
    )

    liquidity_coverage = (
        implementation_security.groupby(
            [
                "portfolio",
                "date",
            ],
            sort=False,
        )
        .agg(
            turnover=(
                "absolute_trade_weight",
                "sum",
            ),
            liquidity_covered_turnover=(
                "liquidity_covered_trade_weight",
                "sum",
            ),
        )
        .reset_index()
    )

    liquidity_coverage["liquidity_coverage"] = np.where(
        liquidity_coverage["turnover"].gt(tolerance),
        (
            liquidity_coverage["liquidity_covered_turnover"]
            / liquidity_coverage["turnover"]
        ),
        1.0,
    )

    implementation_state = (
        implementation_security.groupby(
            [
                "portfolio",
                "date",
            ],
            sort=False,
        )
        .agg(
            turnover=(
                "absolute_trade_weight",
                "sum",
            ),
            transaction_cost=(
                "transaction_cost_contribution",
                "sum",
            ),
            trade_count=(
                "trade_indicator",
                "sum",
            ),
            largest_trade_weight=(
                "absolute_trade_weight",
                "max",
            ),
            missing_return_weight=(
                "missing_return_weight_contribution",
                "sum",
            ),
            bottleneck_capacity=(
                "trade_capacity_usd",
                "min",
            ),
            fifth_percentile_capacity=(
                "trade_capacity_usd",
                lambda values: (
                    values.dropna().quantile(0.05) if values.notna().any() else np.nan
                ),
            ),
            median_capacity=(
                "trade_capacity_usd",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "portfolio",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    participation_label = f"{participation_rate:.0%}".replace("%", "pct")

    bottleneck_capacity_column = f"bottleneck_capacity_{participation_label}_usd"

    implementation_state = implementation_state.rename(
        columns={
            "bottleneck_capacity": (bottleneck_capacity_column),
            "fifth_percentile_capacity": (
                "fifth_percentile_capacity_" f"{participation_label}_usd"
            ),
            "median_capacity": (f"median_capacity_{participation_label}_usd"),
        }
    )

    implementation_state["transaction_cost_bps_per_turnover"] = np.where(
        implementation_state["turnover"].gt(tolerance),
        (
            implementation_state["transaction_cost"]
            / implementation_state["turnover"]
            * 10_000
        ),
        np.nan,
    )

    implementation_state["trade_day"] = implementation_state["turnover"].gt(tolerance)

    implementation_groups = implementation_state.groupby(
        "portfolio",
        sort=False,
    )

    turnover_window_column = f"turnover_{implementation_window}"

    cost_window_column = f"transaction_cost_{implementation_window}"

    implementation_state[turnover_window_column] = implementation_groups[
        "turnover"
    ].transform(
        lambda values: values.rolling(
            implementation_window,
            min_periods=implementation_window,
        ).sum()
    )

    implementation_state[cost_window_column] = implementation_groups[
        "transaction_cost"
    ].transform(
        lambda values: values.rolling(
            implementation_window,
            min_periods=implementation_window,
        ).sum()
    )

    implementation_state[f"annualised_turnover_{implementation_window}"] = (
        implementation_state[turnover_window_column]
        * periods_per_year
        / implementation_window
    )

    implementation_state[f"annualised_transaction_cost_{implementation_window}"] = (
        implementation_state[cost_window_column]
        * periods_per_year
        / implementation_window
    )

    implementation_state[
        "transaction_cost_bps_per_turnover_" f"{implementation_window}"
    ] = np.where(
        implementation_state[turnover_window_column].gt(tolerance),
        (
            implementation_state[cost_window_column]
            / implementation_state[turnover_window_column]
            * 10_000
        ),
        np.nan,
    )

    implementation_state[f"trade_days_{implementation_window}"] = implementation_groups[
        "trade_day"
    ].transform(
        lambda values: values.rolling(
            implementation_window,
            min_periods=1,
        ).sum()
    )

    implementation_state[
        f"largest_trade_weight_{implementation_window}"
    ] = implementation_groups["largest_trade_weight"].transform(
        lambda values: values.rolling(
            implementation_window,
            min_periods=1,
        ).max()
    )

    implementation_state[
        "minimum_trade_capacity_"
        f"{participation_label}_usd_"
        f"{implementation_window}"
    ] = implementation_groups[bottleneck_capacity_column].transform(
        lambda values: values.rolling(
            implementation_window,
            min_periods=1,
        ).min()
    )

    implementation_state[
        "maximum_missing_return_weight_" f"{implementation_window}"
    ] = implementation_groups["missing_return_weight"].transform(
        lambda values: values.rolling(
            implementation_window,
            min_periods=1,
        ).max()
    )

    return (
        implementation_state.reset_index(drop=True),
        liquidity_coverage.sort_values(
            [
                "portfolio",
                "date",
            ]
        ).reset_index(drop=True),
    )
