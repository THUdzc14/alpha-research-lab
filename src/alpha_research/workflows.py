"""High-level, in-memory research workflows.

Workflow functions compose reusable analytical modules. They do not perform
file I/O and do not redefine financial calculations owned by those modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from alpha_research.config.research import (
    FACTOR_COLUMNS,
    FACTOR_SPECIFICATIONS,
    SIGNAL_VALIDATION_RETURN_COLUMN,
    selected_implementations_frame,
)
from alpha_research.monitoring import (
    build_latest_monitoring_overview,
    build_strategy_diagnostic_flags,
    calculate_implementation_monitoring_state,
    calculate_performance_risk_state,
)
from alpha_research.risk import (
    calculate_beta_state,
    calculate_concentration_state,
    prepare_holdings_beta_detail,
)
from alpha_research.validation import (
    calculate_factor_dependence,
    calculate_signal_health,
)

MONITORING_DATASET_NAMES = (
    "signal_health",
    "factor_dependence",
    "performance_risk",
    "beta",
    "concentration",
    "implementation",
    "liquidity_coverage",
    "diagnostic_flags",
    "latest_overview",
)


def validate_frozen_implementations(
    selected_implementations: pd.DataFrame,
) -> tuple[str, ...]:
    """Validate inputs against the frozen Notebook 06 implementation set."""
    expected = selected_implementations_frame()
    required_columns = set(expected.columns)
    missing_columns = required_columns - set(selected_implementations.columns)

    if missing_columns:
        raise KeyError(
            "selected_implementations is missing columns: " f"{sorted(missing_columns)}"
        )

    actual = selected_implementations.loc[:, expected.columns].copy()

    if actual.duplicated("portfolio").any():
        raise ValueError("selected_implementations contains duplicate portfolios.")

    expected = expected.sort_values("portfolio").reset_index(drop=True)
    actual = actual.sort_values("portfolio").reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(
            "selected_implementations does not match the frozen research "
            "specification."
        ) from exc

    return tuple(selected_implementations_frame()["portfolio"])


def build_strategy_monitoring_datasets(
    factor_panel: pd.DataFrame,
    selected_implementations: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    security_holdings: pd.DataFrame,
    security_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all dashboard-ready strategy-monitoring datasets in memory."""
    portfolios = validate_frozen_implementations(selected_implementations)
    raw_factor_columns: Mapping[str, str] = {
        specification.name: specification.raw_column
        for specification in FACTOR_SPECIFICATIONS
    }

    signal_health = calculate_signal_health(
        factor_panel,
        factor_columns=FACTOR_COLUMNS,
        raw_factor_columns=raw_factor_columns,
        forward_return_column=SIGNAL_VALIDATION_RETURN_COLUMN,
    )
    factor_dependence = calculate_factor_dependence(
        factor_panel,
        factor_columns=tuple(FACTOR_COLUMNS.values()),
    )
    performance_risk = calculate_performance_risk_state(
        portfolio_daily,
        benchmark_daily,
        portfolios=portfolios,
    )
    holdings_beta_detail = prepare_holdings_beta_detail(
        security_holdings,
        factor_panel,
        portfolios=portfolios,
    )
    beta_state = calculate_beta_state(
        portfolio_daily,
        benchmark_daily,
        security_holdings,
        factor_panel,
        portfolios=portfolios,
    )
    concentration_state = calculate_concentration_state(
        security_daily,
        factor_panel,
        holdings_beta_detail,
        portfolios=portfolios,
    )
    implementation_state, liquidity_coverage = (
        calculate_implementation_monitoring_state(
            security_daily,
            factor_panel,
            portfolios=portfolios,
        )
    )
    diagnostic_flags = build_strategy_diagnostic_flags(
        signal_health,
        performance_risk,
        beta_state,
        concentration_state,
        implementation_state,
        liquidity_coverage,
        factors=tuple(FACTOR_COLUMNS),
        portfolios=portfolios,
    )
    latest_overview = build_latest_monitoring_overview(diagnostic_flags)

    result = {
        "signal_health": signal_health,
        "factor_dependence": factor_dependence,
        "performance_risk": performance_risk,
        "beta": beta_state,
        "concentration": concentration_state,
        "implementation": implementation_state,
        "liquidity_coverage": liquidity_coverage,
        "diagnostic_flags": diagnostic_flags,
        "latest_overview": latest_overview,
    }

    if tuple(result) != MONITORING_DATASET_NAMES:
        raise RuntimeError("Monitoring workflow returned an invalid dataset set.")

    return result
