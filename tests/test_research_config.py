import pandas as pd

from alpha_research.config.research import (
    BASELINE_TRANSACTION_COST_BPS,
    CAPACITY_PARTICIPATION_LIMITS,
    COMPOSITE_FACTOR_WEIGHTS,
    FACTOR_COLUMNS,
    FACTOR_WINSOR_LOWER_QUANTILE,
    FACTOR_WINSOR_UPPER_QUANTILE,
    FIXED_SLEEVE_ALLOCATIONS,
    INVERSE_VOLATILITY_LOOKBACK,
    INVERSE_VOLATILITY_MIN_PERIODS,
    MOMENTUM_LONG_LAG,
    MOMENTUM_SKIP_LAG,
    MONITORING_SPECIFICATION,
    PORTFOLIO_LONG_GROSS,
    PORTFOLIO_LONG_QUANTILE,
    PORTFOLIO_MINIMUM_OBSERVATIONS,
    PORTFOLIO_QUANTILES,
    PORTFOLIO_SHORT_GROSS,
    PORTFOLIO_SHORT_QUANTILE,
    REALISED_VOLATILITY_WINDOW,
    ROBUSTNESS_REBALANCE_FREQUENCIES,
    ROBUSTNESS_TRANSACTION_COST_GRID_BPS,
    STRATEGY_SPECIFICATIONS,
    STRATEGY_EVALUATION_START_DATE,
    final_strategy_order,
    selected_implementations_frame,
)


def test_selected_implementations_frame_matches_frozen_export():
    expected = pd.DataFrame(
        [
            {
                "portfolio": "Composite Score",
                "rebalance_frequency": 21,
                "rebalance_offset": 0,
                "role": "Primary specification",
            },
            {
                "portfolio": "Fixed 50/50 Sleeves",
                "rebalance_frequency": 10,
                "rebalance_offset": 0,
                "role": "Transparent sleeve benchmark",
            },
            {
                "portfolio": "Pure Inverse Volatility",
                "rebalance_frequency": 10,
                "rebalance_offset": 0,
                "role": "Risk-based sleeve benchmark",
            },
        ]
    )

    pd.testing.assert_frame_equal(
        selected_implementations_frame(),
        expected,
    )


def test_retained_factor_configuration():
    assert FACTOR_COLUMNS == {
        "Momentum": "mom_12_1m_z",
        "Realised Volatility": "realised_vol_63_z",
    }

    assert COMPOSITE_FACTOR_WEIGHTS == {
        "mom_12_1m_z": 0.5,
        "realised_vol_63_z": 0.5,
    }

    assert sum(COMPOSITE_FACTOR_WEIGHTS.values()) == 1.0

    assert MOMENTUM_LONG_LAG == 252
    assert MOMENTUM_SKIP_LAG == 21
    assert REALISED_VOLATILITY_WINDOW == 63
    assert FACTOR_WINSOR_LOWER_QUANTILE == 0.01
    assert FACTOR_WINSOR_UPPER_QUANTILE == 0.99


def test_portfolio_construction_configuration():
    assert PORTFOLIO_QUANTILES == 5
    assert PORTFOLIO_LONG_QUANTILE == 5
    assert PORTFOLIO_SHORT_QUANTILE == 1
    assert PORTFOLIO_LONG_GROSS == 1.0
    assert PORTFOLIO_SHORT_GROSS == 1.0
    assert PORTFOLIO_MINIMUM_OBSERVATIONS == 30

    assert FIXED_SLEEVE_ALLOCATIONS == {
        "Momentum": 0.5,
        "Realised Volatility": 0.5,
    }
    assert INVERSE_VOLATILITY_LOOKBACK == 63
    assert INVERSE_VOLATILITY_MIN_PERIODS == 42


def test_frozen_strategy_parameters_are_valid_and_unique():
    portfolios = [item.portfolio for item in STRATEGY_SPECIFICATIONS]

    assert len(portfolios) == len(set(portfolios))

    for specification in STRATEGY_SPECIFICATIONS:
        assert specification.rebalance_frequency > 0
        assert 0 <= specification.rebalance_offset < specification.rebalance_frequency
        assert specification.transaction_cost_bps == BASELINE_TRANSACTION_COST_BPS


def test_final_strategy_order_matches_notebook_09():
    assert final_strategy_order() == (
        "Composite Score",
        "Pure Inverse Volatility",
        "Fixed 50/50 Sleeves",
    )

    final_roles = {
        specification.portfolio: specification.final_role
        for specification in STRATEGY_SPECIFICATIONS
    }

    assert final_roles == {
        "Composite Score": "Primary implementation",
        "Fixed 50/50 Sleeves": "Transparent allocation benchmark",
        "Pure Inverse Volatility": "Defensive risk-based alternative",
    }


def test_robustness_grids_match_notebook_06():
    assert STRATEGY_EVALUATION_START_DATE == pd.Timestamp("2016-01-07")
    assert ROBUSTNESS_REBALANCE_FREQUENCIES == (1, 5, 10, 21)
    assert ROBUSTNESS_TRANSACTION_COST_GRID_BPS == (
        0.0,
        5.0,
        10.0,
        20.0,
        50.0,
    )
    assert CAPACITY_PARTICIPATION_LIMITS == (0.01, 0.05, 0.10)


def test_monitoring_windows_match_notebook_definitions():
    assert MONITORING_SPECIFICATION.signal_window == 252
    assert MONITORING_SPECIFICATION.signal_min_periods == 126
    assert MONITORING_SPECIFICATION.signal_stability_lags == (1, 21)
    assert MONITORING_SPECIFICATION.performance_window == 252
    assert MONITORING_SPECIFICATION.risk_window == 126
    assert MONITORING_SPECIFICATION.concentration_window == 63
    assert MONITORING_SPECIFICATION.implementation_window == 63
    assert MONITORING_SPECIFICATION.capacity_adv_window == 21
    assert MONITORING_SPECIFICATION.capacity_adv_min_periods == 21
    assert MONITORING_SPECIFICATION.monitoring_liquidity_min_periods == 10
    assert MONITORING_SPECIFICATION.capacity_participation_rate == 0.01
    assert MONITORING_SPECIFICATION.historical_lower_tail == 0.10
    assert MONITORING_SPECIFICATION.historical_upper_tail == 0.90
    assert MONITORING_SPECIFICATION.structural_coverage_tolerance == 1e-8
