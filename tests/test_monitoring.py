from inspect import signature

import numpy as np
import pandas as pd
import pytest

from alpha_research.config.research import (
    FACTOR_COLUMNS,
    MONITORING_SPECIFICATION,
    TRADING_DAYS_PER_YEAR,
)
from alpha_research.monitoring import (
    STATUS_SEVERITY,
    build_latest_monitoring_overview,
    build_monitoring_flag_matrix,
    build_portfolio_diagnostic_flags,
    build_signal_diagnostic_flags,
    build_strategy_diagnostic_flags,
    calculate_implementation_monitoring_state,
    calculate_performance_risk_state,
    create_historical_flag,
    create_structural_flag,
    prepare_diagnostic_flags,
    select_active_diagnostic_flags,
    summarise_diagnostic_flags,
)


def test_public_monitoring_defaults_match_frozen_research_configuration():
    historical_parameters = signature(create_historical_flag).parameters
    assert (
        historical_parameters["lower_tail"].default
        == MONITORING_SPECIFICATION.historical_lower_tail
    )
    assert (
        historical_parameters["upper_tail"].default
        == MONITORING_SPECIFICATION.historical_upper_tail
    )

    signal_parameters = signature(build_signal_diagnostic_flags).parameters
    assert signal_parameters["factors"].default == tuple(FACTOR_COLUMNS)
    assert (
        signal_parameters["min_observations"].default
        == MONITORING_SPECIFICATION.minimum_cross_sectional_observations
    )

    portfolio_parameters = signature(build_portfolio_diagnostic_flags).parameters
    assert (
        portfolio_parameters["structural_coverage_tolerance"].default
        == MONITORING_SPECIFICATION.structural_coverage_tolerance
    )

    performance_parameters = signature(calculate_performance_risk_state).parameters
    assert (
        performance_parameters["performance_window"].default
        == MONITORING_SPECIFICATION.performance_window
    )
    assert (
        performance_parameters["risk_window"].default
        == MONITORING_SPECIFICATION.risk_window
    )
    assert (
        performance_parameters["periods_per_year"].default
        == TRADING_DAYS_PER_YEAR
    )

    implementation_parameters = signature(
        calculate_implementation_monitoring_state
    ).parameters
    assert (
        implementation_parameters["implementation_window"].default
        == MONITORING_SPECIFICATION.implementation_window
    )
    assert (
        implementation_parameters["liquidity_window"].default
        == MONITORING_SPECIFICATION.monitoring_liquidity_window
    )
    assert (
        implementation_parameters["liquidity_min_periods"].default
        == MONITORING_SPECIFICATION.monitoring_liquidity_min_periods
    )
    assert (
        implementation_parameters["participation_rate"].default
        == MONITORING_SPECIFICATION.capacity_participation_rate
    )
    assert (
        implementation_parameters["periods_per_year"].default
        == TRADING_DAYS_PER_YEAR
    )
    assert (
        implementation_parameters["tolerance"].default
        == MONITORING_SPECIFICATION.numerical_tolerance
    )


def structural_flag(*, diagnostic="Coverage", value=1.0, passes=True):
    return create_structural_flag(
        entity_type="Portfolio",
        entity="Strategy",
        category="Implementation",
        diagnostic=diagnostic,
        latest_date="2026-01-05",
        latest_value=value,
        passes=passes,
        calibration="Complete coverage",
        threshold_value=1.0,
    )


def historical_flag(
    *,
    diagnostic="Volatility",
    value=4.0,
    history=(1.0, 2.0, 3.0, 4.0),
    direction="Higher",
):
    return create_historical_flag(
        entity_type="Portfolio",
        entity="Strategy",
        category="Market risk",
        diagnostic=diagnostic,
        latest_date="2026-01-05",
        latest_value=value,
        history=history,
        adverse_direction=direction,
    )


def test_structural_flag_assigns_pass_breach_and_unavailable():
    assert structural_flag(passes=True)["status"] == "PASS"
    assert structural_flag(passes=False)["status"] == "BREACH"
    assert structural_flag(value=np.nan, passes=True)["status"] == "UNAVAILABLE"
    assert structural_flag(passes=None)["status"] == "UNAVAILABLE"


def test_structural_flag_rejects_non_boolean_pass_condition():
    with pytest.raises(TypeError, match="Boolean"):
        structural_flag(passes=1)


def test_historical_higher_tail_flag_matches_notebook_calibration():
    result = historical_flag()

    assert result["threshold_value"] == pytest.approx(3.7)
    assert result["historical_percentile"] == pytest.approx(1.0)
    assert result["calibration"] == "Historical upper 10% tail"
    assert result["status"] == "WARNING"


def test_historical_lower_tail_and_boundary_use_strict_warning_rule():
    warning = historical_flag(
        value=1.0,
        direction="Lower",
    )
    boundary = historical_flag(
        value=1.3,
        direction="Lower",
    )

    assert warning["threshold_value"] == pytest.approx(1.3)
    assert warning["status"] == "WARNING"
    assert boundary["status"] == "PASS"


def test_historical_flag_is_unavailable_for_constant_or_missing_state():
    constant = historical_flag(
        value=1.0,
        history=(1.0, 1.0, 1.0),
    )
    missing = historical_flag(value=np.nan)

    assert constant["status"] == "UNAVAILABLE"
    assert constant["calibration"] == "Historical calibration unavailable"
    assert np.isnan(constant["threshold_value"])
    assert missing["status"] == "UNAVAILABLE"
    assert missing["threshold_value"] == pytest.approx(3.7)


def test_historical_flag_rejects_unknown_adverse_direction():
    with pytest.raises(ValueError, match="adverse_direction"):
        historical_flag(direction="Both")


def make_signal_health() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=5)
    rows = []

    for factor, rolling_ic in {
        "Momentum": [np.nan, 0.01, 0.02, 0.03, np.nan],
        "Realised Volatility": [np.nan, 0.02, 0.03, 0.04, np.nan],
    }.items():
        for date, ic in zip(dates, rolling_ic, strict=True):
            rows.append(
                {
                    "factor": factor,
                    "date": date,
                    "universe_observations": 100,
                    "signal_observations": 100,
                    "signal_coverage": 1.0,
                    "rolling_mean_ic_252": ic,
                }
            )

    return pd.DataFrame(rows)


def test_signal_flags_use_latest_signal_and_latest_evaluated_dates():
    result = build_signal_diagnostic_flags(make_signal_health())

    assert len(result) == 4
    assert result["status"].eq("PASS").all()

    momentum = result.loc[result["entity"].eq("Momentum")]
    coverage = momentum.loc[momentum["diagnostic"].eq("Signal coverage")].iloc[0]
    predictive = momentum.loc[
        momentum["diagnostic"].eq("252-observation rolling mean IC")
    ].iloc[0]

    assert coverage["latest_date"] == pd.Timestamp("2025-01-05")
    assert coverage["threshold_value"] == pytest.approx(0.30)
    assert coverage["notes"] == "100 of 100 observations available"
    assert predictive["latest_date"] == pd.Timestamp("2025-01-04")
    assert predictive["latest_value"] == pytest.approx(0.03)


def test_signal_flags_reject_duplicate_factor_dates():
    signal_health = make_signal_health()
    duplicated = pd.concat([signal_health, signal_health.iloc[[0]]])

    with pytest.raises(ValueError, match="duplicate factor-date"):
        build_signal_diagnostic_flags(duplicated)


def test_preparation_rejects_duplicate_flags_and_unknown_statuses():
    flag = structural_flag()

    with pytest.raises(ValueError, match="Duplicate"):
        prepare_diagnostic_flags([flag, flag])

    invalid = flag | {"status": "ALERT"}

    with pytest.raises(ValueError, match="Unknown diagnostic statuses"):
        prepare_diagnostic_flags([invalid])


def test_status_aggregation_and_overview_match_notebook_hierarchy():
    flags = [
        structural_flag(diagnostic="Coverage", passes=True),
        historical_flag(diagnostic="Volatility", value=4.0),
        create_structural_flag(
            entity_type="Portfolio",
            entity="Strategy",
            category="Concentration",
            diagnostic="Metadata",
            latest_date="2026-01-05",
            latest_value=np.nan,
            passes=None,
            calibration="Complete metadata",
        ),
        structural_flag(diagnostic="Missing returns", passes=False),
    ]
    prepared = prepare_diagnostic_flags(flags)
    summary = summarise_diagnostic_flags(prepared)
    matrix = build_monitoring_flag_matrix(prepared)
    overview = build_latest_monitoring_overview(prepared)

    entity_summary = summary.loc[("Portfolio", "Strategy")]

    assert entity_summary["diagnostics"] == 4
    assert entity_summary["passes"] == 1
    assert entity_summary["warnings"] == 1
    assert entity_summary["unavailable"] == 1
    assert entity_summary["breaches"] == 1
    assert entity_summary["overall_status"] == "BREACH"
    assert matrix.loc[("Portfolio", "Strategy"), "Implementation"] == "BREACH"
    assert overview.loc[0, "signal_status"] == "N/A"
    assert overview.loc[0, "market_risk_status"] == "WARNING"
    assert overview.loc[0, "concentration_status"] == "UNAVAILABLE"
    assert overview.loc[0, "implementation_status"] == "BREACH"
    assert overview.loc[0, "overall_status"] == "BREACH"


def test_active_flags_are_sorted_by_severity_and_exclude_passes():
    flags = [
        structural_flag(diagnostic="Pass", passes=True),
        historical_flag(diagnostic="Warning", value=4.0),
        structural_flag(diagnostic="Breach", passes=False),
    ]

    result = select_active_diagnostic_flags(pd.DataFrame(flags))

    assert result["diagnostic"].tolist() == ["Breach", "Warning"]
    assert result["status"].tolist() == ["BREACH", "WARNING"]
    assert "status_severity" not in result
    assert STATUS_SEVERITY["BREACH"] > STATUS_SEVERITY["WARNING"]


def make_portfolio_monitoring_inputs():
    dates = pd.bdate_range(
        "2025-01-01",
        periods=4,
    )

    performance_risk = pd.DataFrame(
        {
            "portfolio": "Strategy",
            "date": dates,
            "annualised_volatility_126": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "drawdown": [
                0.0,
                -0.1,
                -0.2,
                -0.4,
            ],
        }
    )

    beta_state = pd.DataFrame(
        {
            "portfolio": "Strategy",
            "date": dates,
            "beta_coverage": 1.0,
            "holdings_market_beta": [
                0.5,
                0.6,
                0.7,
                1.0,
            ],
        }
    )

    concentration_state = pd.DataFrame(
        {
            "portfolio": "Strategy",
            "date": dates,
            "largest_absolute_sector_net_exposure": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "top_five_absolute_beta_contribution_share": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "top_five_contributor_share_63": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "effective_contribution_sector_count_63": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    implementation_state = pd.DataFrame(
        {
            "portfolio": "Strategy",
            "date": dates,
            "maximum_missing_return_weight_63": 0.0,
            "annualised_turnover_63": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "largest_trade_weight_63": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "minimum_trade_capacity_1pct_usd_63": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    liquidity_coverage = pd.DataFrame(
        {
            "portfolio": "Strategy",
            "date": dates,
            "liquidity_coverage": 1.0,
        }
    )

    return (
        performance_risk,
        beta_state,
        concentration_state,
        implementation_state,
        liquidity_coverage,
    )


def test_portfolio_flags_reproduce_thirteen_diagnostic_contract():
    result = build_portfolio_diagnostic_flags(
        *make_portfolio_monitoring_inputs(),
        portfolios=("Strategy",),
    )

    assert len(result) == 13

    assert result["category"].value_counts().to_dict() == {
        "Implementation": 5,
        "Market risk": 4,
        "Concentration": 4,
    }

    assert result["status"].value_counts().to_dict() == {
        "WARNING": 10,
        "PASS": 3,
    }


def test_complete_strategy_flags_feed_latest_overview():
    dates = pd.bdate_range(
        "2025-01-01",
        periods=4,
    )

    signal_health = pd.DataFrame(
        {
            "factor": "Factor A",
            "date": dates,
            "universe_observations": 100,
            "signal_observations": 100,
            "signal_coverage": 1.0,
            "rolling_mean_ic_252": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )

    result = build_strategy_diagnostic_flags(
        signal_health,
        *make_portfolio_monitoring_inputs(),
        factors=("Factor A",),
        portfolios=("Strategy",),
    )

    overview = build_latest_monitoring_overview(result)

    assert len(result) == 15
    assert set(overview["entity"]) == {
        "Factor A",
        "Strategy",
    }

    strategy = overview.loc[overview["entity"].eq("Strategy")].iloc[0]

    assert strategy["overall_status"] == "WARNING"
    assert strategy["implementation_status"] == "WARNING"
