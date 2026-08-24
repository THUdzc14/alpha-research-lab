"""High-level, in-memory research workflows.

Workflow functions compose reusable analytical modules. They do not perform
file I/O and do not redefine financial calculations owned by those modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from alpha_research.attribution import (
    prepare_security_attribution,
    reconcile_security_attribution,
)
from alpha_research.backtest import (
    BacktestConfig,
    run_target_weight_backtest,
)
from alpha_research.config.research import (
    BACKTEST_RETURN_COLUMN,
    BASELINE_TRANSACTION_COST_BPS,
    COMPOSITE_FACTOR_WEIGHTS,
    COMPOSITE_SCORE_COLUMN,
    DEFAULT_NUMERICAL_TOLERANCE,
    FACTOR_COLUMNS,
    FACTOR_SPECIFICATIONS,
    FIXED_SLEEVE_ALLOCATIONS,
    INVERSE_VOLATILITY_LOOKBACK,
    INVERSE_VOLATILITY_MIN_PERIODS,
    PORTFOLIO_LONG_GROSS,
    PORTFOLIO_LONG_QUANTILE,
    PORTFOLIO_MINIMUM_OBSERVATIONS,
    PORTFOLIO_QUANTILES,
    PORTFOLIO_SHORT_GROSS,
    PORTFOLIO_SHORT_QUANTILE,
    SIGNAL_VALIDATION_RETURN_COLUMN,
    STRATEGY_SPECIFICATIONS,
    selected_implementations_frame,
)
from alpha_research.monitoring import (
    build_latest_monitoring_overview,
    build_strategy_diagnostic_flags,
    calculate_implementation_monitoring_state,
    calculate_performance_risk_state,
)
from alpha_research.portfolio import (
    build_factor_target_weights,
    calculate_rebalance_inverse_volatility_allocations,
    combine_dynamic_sleeve_target_weights,
    combine_factor_scores,
    combine_sleeve_target_weights,
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

ATTRIBUTION_DATASET_NAMES = (
    "selected_implementations",
    "portfolio_daily",
    "security_holdings",
    "target_weights",
    "benchmark_daily",
    "security_daily",
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


def _frozen_backtest_config(
    portfolio: str,
) -> BacktestConfig:
    specification_lookup = {
        specification.portfolio: specification
        for specification in STRATEGY_SPECIFICATIONS
    }
    specification = specification_lookup[portfolio]

    return BacktestConfig(
        rebalance_frequency=(specification.rebalance_frequency),
        quantiles=PORTFOLIO_QUANTILES,
        long_quantile=PORTFOLIO_LONG_QUANTILE,
        short_quantile=PORTFOLIO_SHORT_QUANTILE,
        long_gross=PORTFOLIO_LONG_GROSS,
        short_gross=PORTFOLIO_SHORT_GROSS,
        transaction_cost_bps=(specification.transaction_cost_bps),
        min_observations=(PORTFOLIO_MINIMUM_OBSERVATIONS),
        rebalance_offset=(specification.rebalance_offset),
    )


def _extract_active_gross_returns(
    daily: pd.DataFrame,
) -> pd.Series:
    indexed = (
        daily.copy()
        .assign(date=lambda data: pd.to_datetime(data["date"]))
        .set_index("date")
        .sort_index()
    )

    return indexed["gross_return"].where(
        indexed["gross_exposure"].gt(DEFAULT_NUMERICAL_TOLERANCE)
    )


def build_common_strategy_backtests(
    factor_panel: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Build retained factors and candidate strategies on one common schedule.

    The two standalone factors and three multi-factor candidates are constructed
    with the same rebalance frequency, offset, exposure budgets, and transaction
    cost. Component-factor backtests are reused when estimating the lagged pure
    inverse-volatility sleeve allocations.

    Returns
    -------
    daily_results
        Drift-aware portfolio-level backtest results in stable comparison order.
    holdings_results
        Security-level holdings, pre-trade weights, and trades from the same runs.
    target_weights
        Dated target weights supplied to the backtest engine.
    """
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be a BacktestConfig instance.")

    if config.beta_neutral:
        raise ValueError("Common strategy backtests do not support beta-neutral hedges.")

    required_columns = {
        "date",
        "ticker",
        BACKTEST_RETURN_COLUMN,
        *FACTOR_COLUMNS.values(),
    }
    missing_columns = required_columns - set(factor_panel.columns)

    if missing_columns:
        raise KeyError("factor_panel is missing columns: " f"{sorted(missing_columns)}")

    if factor_panel[["date", "ticker"]].duplicated().any():
        raise ValueError("factor_panel contains duplicate date-ticker rows.")

    panel = factor_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel["ticker"] = panel["ticker"].astype(str)

    return_panel = panel[
        [
            "date",
            "ticker",
            BACKTEST_RETURN_COLUMN,
        ]
    ].copy()

    sleeve_targets = {
        sleeve_name: build_factor_target_weights(
            panel,
            factor_column=factor_column,
            return_column=BACKTEST_RETURN_COLUMN,
            config=config,
        )
        for sleeve_name, factor_column in FACTOR_COLUMNS.items()
    }

    target_dates = {
        sleeve_name: pd.DatetimeIndex(targets["date"].unique()).sort_values()
        for sleeve_name, targets in sleeve_targets.items()
    }
    reference_dates = target_dates[next(iter(FACTOR_COLUMNS))]

    for sleeve_name, dates in target_dates.items():
        if not dates.equals(reference_dates):
            raise ValueError(
                f"{sleeve_name} target dates do not match the common rebalance schedule."
            )

    sleeve_daily: dict[str, pd.DataFrame] = {}
    sleeve_holdings: dict[str, pd.DataFrame] = {}

    for sleeve_name, targets in sleeve_targets.items():
        daily, holdings = run_target_weight_backtest(
            return_panel,
            targets,
            return_column=BACKTEST_RETURN_COLUMN,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        sleeve_daily[sleeve_name] = daily
        sleeve_holdings[sleeve_name] = holdings

    sleeve_return_frame = pd.concat(
        {
            sleeve_name: _extract_active_gross_returns(daily)
            for sleeve_name, daily in sleeve_daily.items()
        },
        axis=1,
    ).sort_index()

    inverse_volatility_allocations = (
        calculate_rebalance_inverse_volatility_allocations(
            sleeve_return_frame,
            rebalance_dates=reference_dates,
            lookback=INVERSE_VOLATILITY_LOOKBACK,
            min_periods=INVERSE_VOLATILITY_MIN_PERIODS,
        )
    )

    composite_panel = panel.copy()
    composite_panel[COMPOSITE_SCORE_COLUMN] = combine_factor_scores(
        composite_panel,
        factor_weights=COMPOSITE_FACTOR_WEIGHTS,
    )

    target_weights = {
        "Momentum Only": sleeve_targets["Momentum"],
        "Realised Volatility Only": sleeve_targets["Realised Volatility"],
        "Composite Score": build_factor_target_weights(
            composite_panel,
            factor_column=COMPOSITE_SCORE_COLUMN,
            return_column=BACKTEST_RETURN_COLUMN,
            config=config,
        ),
        "Fixed 50/50 Sleeves": combine_sleeve_target_weights(
            sleeve_targets,
            sleeve_allocations=FIXED_SLEEVE_ALLOCATIONS,
        ),
        "Pure Inverse Volatility": combine_dynamic_sleeve_target_weights(
            sleeve_targets,
            sleeve_allocations=inverse_volatility_allocations,
        ),
    }

    daily_results = {
        "Momentum Only": sleeve_daily["Momentum"],
        "Realised Volatility Only": sleeve_daily["Realised Volatility"],
    }
    holdings_results = {
        "Momentum Only": sleeve_holdings["Momentum"],
        "Realised Volatility Only": sleeve_holdings["Realised Volatility"],
    }

    for portfolio in (
        "Composite Score",
        "Fixed 50/50 Sleeves",
        "Pure Inverse Volatility",
    ):
        daily, holdings = run_target_weight_backtest(
            return_panel,
            target_weights[portfolio],
            return_column=BACKTEST_RETURN_COLUMN,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        daily_results[portfolio] = daily
        holdings_results[portfolio] = holdings

    if not (
        tuple(daily_results) == tuple(holdings_results) == tuple(target_weights)
    ):
        raise RuntimeError("Common strategy backtests returned an invalid portfolio set.")

    return daily_results, holdings_results, target_weights


def build_frozen_strategy_target_weights(
    factor_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Construct targets for the three frozen strategies."""
    portfolios = tuple(selected_implementations_frame()["portfolio"])

    composite_name = "Composite Score"
    fixed_name = "Fixed 50/50 Sleeves"
    inverse_volatility_name = "Pure Inverse Volatility"

    fixed_config = _frozen_backtest_config(fixed_name)
    inverse_volatility_config = _frozen_backtest_config(inverse_volatility_name)

    if (
        fixed_config.rebalance_frequency
        != inverse_volatility_config.rebalance_frequency
        or fixed_config.rebalance_offset != inverse_volatility_config.rebalance_offset
    ):
        raise ValueError(
            "The retained sleeve strategies must share a rebalance schedule."
        )

    required_columns = {
        "date",
        "ticker",
        BACKTEST_RETURN_COLUMN,
        *FACTOR_COLUMNS.values(),
    }

    missing_columns = required_columns - set(factor_panel.columns)

    if missing_columns:
        raise KeyError("factor_panel is missing columns: " f"{sorted(missing_columns)}")

    if factor_panel[["date", "ticker"]].duplicated().any():
        raise ValueError("factor_panel contains duplicate date-ticker rows.")

    composite_panel = factor_panel.copy()

    composite_panel[COMPOSITE_SCORE_COLUMN] = combine_factor_scores(
        composite_panel,
        factor_weights=COMPOSITE_FACTOR_WEIGHTS,
    )

    composite_targets = build_factor_target_weights(
        composite_panel,
        factor_column=COMPOSITE_SCORE_COLUMN,
        return_column=BACKTEST_RETURN_COLUMN,
        config=_frozen_backtest_config(composite_name),
    )

    sleeve_targets = {
        sleeve_name: build_factor_target_weights(
            factor_panel,
            factor_column=factor_column,
            return_column=BACKTEST_RETURN_COLUMN,
            config=fixed_config,
        )
        for sleeve_name, factor_column in FACTOR_COLUMNS.items()
    }

    target_dates = {
        sleeve_name: pd.DatetimeIndex(targets["date"].unique()).sort_values()
        for sleeve_name, targets in sleeve_targets.items()
    }

    reference_dates = target_dates[next(iter(FACTOR_COLUMNS))]

    for sleeve_name, dates in target_dates.items():
        if not dates.equals(reference_dates):
            raise ValueError(
                f"{sleeve_name} target dates do "
                "not match the retained rebalance "
                "schedule."
            )

    return_panel = factor_panel[
        [
            "date",
            "ticker",
            BACKTEST_RETURN_COLUMN,
        ]
    ].copy()

    sleeve_daily = {
        sleeve_name: run_target_weight_backtest(
            return_panel,
            targets,
            return_column=BACKTEST_RETURN_COLUMN,
            transaction_cost_bps=0.0,
        )[0]
        for sleeve_name, targets in sleeve_targets.items()
    }

    sleeve_return_frame = pd.concat(
        {
            sleeve_name: (_extract_active_gross_returns(daily))
            for sleeve_name, daily in sleeve_daily.items()
        },
        axis=1,
    ).sort_index()

    inverse_volatility_allocations = calculate_rebalance_inverse_volatility_allocations(
        sleeve_return_frame,
        rebalance_dates=reference_dates,
        lookback=(INVERSE_VOLATILITY_LOOKBACK),
        min_periods=(INVERSE_VOLATILITY_MIN_PERIODS),
    )

    fixed_targets = combine_sleeve_target_weights(
        sleeve_targets,
        sleeve_allocations=(FIXED_SLEEVE_ALLOCATIONS),
    )

    inverse_volatility_targets = combine_dynamic_sleeve_target_weights(
        sleeve_targets,
        sleeve_allocations=(inverse_volatility_allocations),
    )

    result = {
        composite_name: composite_targets,
        fixed_name: fixed_targets,
        inverse_volatility_name: (inverse_volatility_targets),
    }

    if tuple(result) != portfolios:
        raise RuntimeError("Frozen strategy targets have an invalid portfolio set.")

    return result


def build_selected_attribution_datasets(
    return_panel: pd.DataFrame,
    target_weights_by_portfolio: Mapping[
        str,
        pd.DataFrame,
    ],
    benchmark_daily: pd.DataFrame,
    analysis_dates: pd.Index,
    selected_implementations: pd.DataFrame | None = None,
    return_column: str = BACKTEST_RETURN_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Replay frozen strategies and build attribution data."""
    if selected_implementations is None:
        selected_implementations = selected_implementations_frame()

    portfolios = validate_frozen_implementations(selected_implementations)

    if set(target_weights_by_portfolio) != set(portfolios):
        raise ValueError(
            "target_weights_by_portfolio does not match the frozen portfolio set."
        )

    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(analysis_dates).dropna()))

    if dates.empty:
        raise ValueError("analysis_dates is empty.")

    if dates.duplicated().any():
        raise ValueError("analysis_dates contains duplicates.")

    dates = dates.sort_values()

    required_benchmark_columns = {
        "date",
        "benchmark",
        "benchmark_return",
    }

    missing_benchmark_columns = required_benchmark_columns - set(
        benchmark_daily.columns
    )

    if missing_benchmark_columns:
        raise KeyError(
            "benchmark_daily is missing columns: "
            f"{sorted(missing_benchmark_columns)}"
        )

    prepared_benchmark = benchmark_daily[
        [
            "date",
            "benchmark",
            "benchmark_return",
        ]
    ].copy()

    prepared_benchmark["date"] = pd.to_datetime(
        prepared_benchmark["date"],
        errors="raise",
    )

    prepared_benchmark = (
        prepared_benchmark.loc[prepared_benchmark["date"].isin(dates)]
        .sort_values(["benchmark", "date"])
        .reset_index(drop=True)
    )

    if prepared_benchmark[["benchmark", "date"]].duplicated().any():
        raise ValueError("benchmark_daily contains duplicate keys.")

    benchmark_names = prepared_benchmark["benchmark"].drop_duplicates()

    if len(benchmark_names) != 1:
        raise ValueError("benchmark_daily must contain exactly one benchmark.")

    benchmark_dates = pd.DatetimeIndex(prepared_benchmark["date"]).sort_values()

    if not benchmark_dates.equals(dates):
        raise ValueError("benchmark_daily dates do not match analysis_dates.")

    implementation_lookup = {
        specification.portfolio: specification
        for specification in STRATEGY_SPECIFICATIONS
    }

    implementation_rows = selected_implementations.set_index("portfolio")

    portfolio_daily_parts = []
    security_holding_parts = []
    target_weight_parts = []

    for portfolio in portfolios:
        implementation = implementation_rows.loc[portfolio]
        specification = implementation_lookup[portfolio]
        targets = target_weights_by_portfolio[portfolio].copy()

        daily, holdings = run_target_weight_backtest(
            return_panel=return_panel,
            target_weights=targets,
            return_column=return_column,
            transaction_cost_bps=(specification.transaction_cost_bps),
        )

        for data in (
            daily,
            holdings,
            targets,
        ):
            data["date"] = pd.to_datetime(
                data["date"],
                errors="raise",
            )

        daily = (
            daily.loc[daily["date"].isin(dates)]
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Re-anchor cumulative wealth to the beginning
        # of the exported analysis window.
        daily["gross_cumulative_return"] = (1.0 + daily["gross_return"]).cumprod()

        daily["net_cumulative_return"] = (1.0 + daily["net_return"]).cumprod()

        holdings = (
            holdings.loc[holdings["date"].isin(dates)]
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        targets = (
            targets.loc[targets["date"].isin(dates)]
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        portfolio_dates = pd.DatetimeIndex(daily["date"])

        if not portfolio_dates.equals(dates):
            raise ValueError(f"{portfolio} backtest dates do not match analysis_dates.")

        metadata = {
            "portfolio": portfolio,
            "rebalance_frequency": int(implementation["rebalance_frequency"]),
            "rebalance_offset": int(implementation["rebalance_offset"]),
        }

        daily = daily.assign(
            **metadata,
            transaction_cost_bps=(specification.transaction_cost_bps),
            role=implementation["role"],
        )

        holdings = holdings.assign(
            **metadata,
            role=implementation["role"],
        )

        targets = targets.assign(
            **metadata,
            role=implementation["role"],
        )

        portfolio_daily_parts.append(daily)
        security_holding_parts.append(holdings)
        target_weight_parts.append(targets)

    portfolio_daily = (
        pd.concat(
            portfolio_daily_parts,
            ignore_index=True,
        )
        .sort_values(["portfolio", "date"])
        .reset_index(drop=True)
    )

    security_holdings = (
        pd.concat(
            security_holding_parts,
            ignore_index=True,
        )
        .sort_values(["portfolio", "date", "ticker"])
        .reset_index(drop=True)
    )

    target_weights = (
        pd.concat(
            target_weight_parts,
            ignore_index=True,
        )
        .sort_values(["portfolio", "date", "ticker"])
        .reset_index(drop=True)
    )

    security_daily = prepare_security_attribution(
        security_holdings,
        return_panel,
        return_column=return_column,
        transaction_cost_bps=(BASELINE_TRANSACTION_COST_BPS),
    )

    attribution_audit = reconcile_security_attribution(
        portfolio_daily,
        security_daily,
    )

    if not attribution_audit["audit_passes"].all():
        raise ValueError("Security-level attribution reconciliation failed.")

    result = {
        "selected_implementations": (selected_implementations.copy()),
        "portfolio_daily": portfolio_daily,
        "security_holdings": security_holdings,
        "target_weights": target_weights,
        "benchmark_daily": prepared_benchmark,
        "security_daily": security_daily,
    }

    if tuple(result) != ATTRIBUTION_DATASET_NAMES:
        raise RuntimeError("Attribution workflow returned an invalid dataset set.")

    return result


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
