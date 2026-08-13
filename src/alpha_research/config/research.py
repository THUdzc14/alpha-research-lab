"""Frozen research specifications for the retained alpha strategies.

This module contains methodological choices that should remain stable when
market data are refreshed.  New observations may update measurements, but
must not silently change factor definitions, strategy frequencies, costs, or
monitoring windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_NUMERICAL_TOLERANCE = 1e-12

BACKTEST_RETURN_COLUMN = "forward_ret_1d"
SIGNAL_VALIDATION_RETURN_COLUMN = "forward_ret_5d"
COMPOSITE_SCORE_COLUMN = "mom_vol_composite_z"

MOMENTUM_LONG_LAG = 252
MOMENTUM_SKIP_LAG = 21
REALISED_VOLATILITY_WINDOW = 63
FACTOR_WINSOR_LOWER_QUANTILE = 0.01
FACTOR_WINSOR_UPPER_QUANTILE = 0.99

PORTFOLIO_QUANTILES = 5
PORTFOLIO_LONG_QUANTILE = 5
PORTFOLIO_SHORT_QUANTILE = 1
PORTFOLIO_LONG_GROSS = 1.0
PORTFOLIO_SHORT_GROSS = 1.0
PORTFOLIO_MINIMUM_OBSERVATIONS = 30

FIXED_SLEEVE_ALLOCATIONS = {
    "Momentum": 0.5,
    "Realised Volatility": 0.5,
}

INVERSE_VOLATILITY_LOOKBACK = 63
INVERSE_VOLATILITY_MIN_PERIODS = 42

BASELINE_TRANSACTION_COST_BPS = 10.0
ROBUSTNESS_REBALANCE_FREQUENCIES = (1, 5, 10, 21)
ROBUSTNESS_TRANSACTION_COST_GRID_BPS = (0.0, 5.0, 10.0, 20.0, 50.0)
CAPACITY_PARTICIPATION_LIMITS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class FactorSpecification:
    """Frozen processed-factor definition used by retained strategies."""

    name: str
    raw_column: str
    score_column: str
    composite_weight: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Factor name must not be empty.")

        if not self.raw_column or not self.score_column:
            raise ValueError("Factor columns must not be empty.")

        if not math.isfinite(self.composite_weight) or self.composite_weight < 0.0:
            raise ValueError("composite_weight must be finite and non-negative.")


@dataclass(frozen=True)
class StrategySpecification:
    """Frozen implementation and final-assessment labels for one strategy."""

    portfolio: str
    rebalance_frequency: int
    rebalance_offset: int
    transaction_cost_bps: float
    implementation_role: str
    final_role: str
    implementation_order: int
    decision_order: int

    def __post_init__(self) -> None:
        if not self.portfolio:
            raise ValueError("Portfolio name must not be empty.")

        if self.rebalance_frequency <= 0:
            raise ValueError("rebalance_frequency must be positive.")

        if not 0 <= self.rebalance_offset < self.rebalance_frequency:
            raise ValueError(
                "rebalance_offset must satisfy "
                "0 <= rebalance_offset < rebalance_frequency."
            )

        if (
            not math.isfinite(self.transaction_cost_bps)
            or self.transaction_cost_bps < 0.0
        ):
            raise ValueError("transaction_cost_bps must be finite and non-negative.")

        if not self.implementation_role or not self.final_role:
            raise ValueError("Strategy roles must not be empty.")

        if self.implementation_order <= 0 or self.decision_order <= 0:
            raise ValueError("Strategy order values must be positive.")


@dataclass(frozen=True)
class MonitoringSpecification:
    """Frozen rolling windows and liquidity assumptions for monitoring."""

    signal_window: int = 252
    signal_min_periods: int = 126
    signal_stability_lags: tuple[int, ...] = (1, 21)
    performance_window: int = 252
    risk_window: int = 126
    concentration_window: int = 63
    implementation_window: int = 63
    minimum_cross_sectional_observations: int = 30
    capacity_adv_window: int = 21
    capacity_adv_min_periods: int = 21
    monitoring_liquidity_window: int = 21
    monitoring_liquidity_min_periods: int = 10
    capacity_participation_rate: float = 0.01
    historical_lower_tail: float = 0.10
    historical_upper_tail: float = 0.90
    structural_coverage_tolerance: float = 1e-8
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE

    def __post_init__(self) -> None:
        integer_parameters = {
            "signal_window": self.signal_window,
            "signal_min_periods": self.signal_min_periods,
            "performance_window": self.performance_window,
            "risk_window": self.risk_window,
            "concentration_window": self.concentration_window,
            "implementation_window": self.implementation_window,
            "minimum_cross_sectional_observations": (
                self.minimum_cross_sectional_observations
            ),
            "capacity_adv_window": self.capacity_adv_window,
            "capacity_adv_min_periods": self.capacity_adv_min_periods,
            "monitoring_liquidity_window": self.monitoring_liquidity_window,
            "monitoring_liquidity_min_periods": (self.monitoring_liquidity_min_periods),
        }

        invalid_parameters = [
            name for name, value in integer_parameters.items() if value <= 0
        ]

        if invalid_parameters:
            raise ValueError(
                "Monitoring integer parameters must be positive: "
                f"{invalid_parameters}"
            )

        if self.capacity_adv_min_periods > self.capacity_adv_window:
            raise ValueError(
                "capacity_adv_min_periods cannot exceed capacity_adv_window."
            )

        if self.monitoring_liquidity_min_periods > self.monitoring_liquidity_window:
            raise ValueError(
                "monitoring_liquidity_min_periods cannot exceed "
                "monitoring_liquidity_window."
            )

        if not 0.0 < self.capacity_participation_rate <= 1.0:
            raise ValueError("capacity_participation_rate must be in (0, 1].")

        if not (
            math.isfinite(self.historical_lower_tail)
            and math.isfinite(self.historical_upper_tail)
            and 0.0 < self.historical_lower_tail < self.historical_upper_tail < 1.0
        ):
            raise ValueError(
                "Historical tails must be finite and satisfy "
                "0 < historical_lower_tail < historical_upper_tail < 1."
            )

        if (
            not math.isfinite(self.structural_coverage_tolerance)
            or self.structural_coverage_tolerance < 0.0
        ):
            raise ValueError(
                "structural_coverage_tolerance must be finite and non-negative."
            )

        if (
            not math.isfinite(self.numerical_tolerance)
            or self.numerical_tolerance < 0.0
        ):
            raise ValueError("numerical_tolerance must be finite and non-negative.")

        if self.signal_min_periods > self.signal_window:
            raise ValueError("signal_min_periods must not exceed signal_window.")

        if not self.signal_stability_lags:
            raise ValueError("signal_stability_lags must not be empty.")

        if len(set(self.signal_stability_lags)) != len(self.signal_stability_lags):
            raise ValueError("signal_stability_lags must contain unique values.")

        if any(lag <= 0 for lag in self.signal_stability_lags):
            raise ValueError("signal_stability_lags must be positive.")


FACTOR_SPECIFICATIONS = (
    FactorSpecification(
        name="Momentum",
        raw_column="mom_12_1m_raw",
        score_column="mom_12_1m_z",
        composite_weight=0.5,
    ),
    FactorSpecification(
        name="Realised Volatility",
        raw_column="realised_vol_63_raw",
        score_column="realised_vol_63_z",
        composite_weight=0.5,
    ),
)


STRATEGY_SPECIFICATIONS = (
    StrategySpecification(
        portfolio="Composite Score",
        rebalance_frequency=21,
        rebalance_offset=0,
        transaction_cost_bps=BASELINE_TRANSACTION_COST_BPS,
        implementation_role="Primary specification",
        final_role="Primary implementation",
        implementation_order=1,
        decision_order=1,
    ),
    StrategySpecification(
        portfolio="Fixed 50/50 Sleeves",
        rebalance_frequency=10,
        rebalance_offset=0,
        transaction_cost_bps=BASELINE_TRANSACTION_COST_BPS,
        implementation_role="Transparent sleeve benchmark",
        final_role="Transparent allocation benchmark",
        implementation_order=2,
        decision_order=3,
    ),
    StrategySpecification(
        portfolio="Pure Inverse Volatility",
        rebalance_frequency=10,
        rebalance_offset=0,
        transaction_cost_bps=BASELINE_TRANSACTION_COST_BPS,
        implementation_role="Risk-based sleeve benchmark",
        final_role="Defensive risk-based alternative",
        implementation_order=3,
        decision_order=2,
    ),
)


MONITORING_SPECIFICATION = MonitoringSpecification()


FACTOR_COLUMNS = {
    specification.name: specification.score_column
    for specification in FACTOR_SPECIFICATIONS
}


COMPOSITE_FACTOR_WEIGHTS = {
    specification.score_column: specification.composite_weight
    for specification in FACTOR_SPECIFICATIONS
}


def selected_implementations_frame() -> pd.DataFrame:
    """Return the exact specification table exported by Notebook 06."""
    rows = [
        {
            "portfolio": specification.portfolio,
            "rebalance_frequency": specification.rebalance_frequency,
            "rebalance_offset": specification.rebalance_offset,
            "role": specification.implementation_role,
        }
        for specification in sorted(
            STRATEGY_SPECIFICATIONS,
            key=lambda item: item.implementation_order,
        )
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "portfolio",
            "rebalance_frequency",
            "rebalance_offset",
            "role",
        ],
    )


def final_strategy_order() -> tuple[str, ...]:
    """Return portfolio names in final decision order."""
    return tuple(
        specification.portfolio
        for specification in sorted(
            STRATEGY_SPECIFICATIONS,
            key=lambda item: item.decision_order,
        )
    )


if not math.isclose(
    sum(COMPOSITE_FACTOR_WEIGHTS.values()),
    1.0,
    rel_tol=0.0,
    abs_tol=DEFAULT_NUMERICAL_TOLERANCE,
):
    raise ValueError("Composite factor weights must sum to one.")


if not math.isclose(
    sum(FIXED_SLEEVE_ALLOCATIONS.values()),
    1.0,
    rel_tol=0.0,
    abs_tol=DEFAULT_NUMERICAL_TOLERANCE,
):
    raise ValueError("Fixed sleeve allocations must sum to one.")


if len({item.portfolio for item in STRATEGY_SPECIFICATIONS}) != len(
    STRATEGY_SPECIFICATIONS
):
    raise ValueError("Strategy portfolio names must be unique.")
