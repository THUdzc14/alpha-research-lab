import math

import numpy as np
import pandas as pd
import pytest

from alpha_research.metrics import (
    annualised_downside_deviation,
    annualised_geometric_return,
    annualised_sharpe_ratio,
    annualised_volatility,
    calculate_drawdown_duration,
    calculate_drawdown_from_returns,
    calculate_rolling_return_state,
    calculate_wealth_index,
    compound_return,
    maximum_drawdown,
    summarise_concentration,
    summarise_returns,
)


def test_compound_return_and_wealth_index():
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.date_range("2025-01-01", periods=3),
    )

    wealth = calculate_wealth_index(returns)
    expected_wealth = pd.Series(
        [1.10, 1.045, 1.0659],
        index=returns.index,
        name="wealth",
    )

    pd.testing.assert_series_equal(wealth, expected_wealth)
    assert compound_return(returns) == pytest.approx(0.0659)


def test_drawdown_anchors_initial_wealth_as_a_peak():
    returns = pd.Series([-0.10, 0.20, -0.25, 1.0 / 3.0])

    drawdown = calculate_drawdown_from_returns(returns)

    expected = pd.Series(
        [-0.10, 0.0, -0.25, 0.0],
        name="drawdown",
    )

    pd.testing.assert_series_equal(drawdown, expected)
    assert maximum_drawdown(returns) == pytest.approx(-0.25)


def test_drawdown_duration_counts_consecutive_underwater_periods():
    drawdown = pd.Series([0.0, -0.10, -0.05, 0.0, -0.02, -0.03])

    duration = calculate_drawdown_duration(drawdown)

    expected = pd.Series(
        [0, 1, 2, 0, 1, 2],
        name="drawdown_duration",
    )

    pd.testing.assert_series_equal(duration, expected)


def test_drawdown_duration_treats_tiny_negative_values_as_underwater():
    drawdown = pd.Series([0.0, -1e-16, -0.01, 0.0])

    duration = calculate_drawdown_duration(drawdown)

    expected = pd.Series(
        [0, 1, 2, 0],
        name="drawdown_duration",
    )

    pd.testing.assert_series_equal(duration, expected)


def test_annualised_geometric_return_uses_valid_observations():
    returns = pd.Series([0.10, np.nan, -0.05])
    expected = (1.10 * 0.95) ** (12 / 2) - 1.0

    result = annualised_geometric_return(
        returns,
        periods_per_year=12,
    )

    assert result == pytest.approx(expected)


def test_annualised_volatility_and_sharpe_match_manual_formulas():
    returns = pd.Series([0.01, -0.02, 0.03, 0.00])
    expected_volatility = returns.std(ddof=1) * math.sqrt(12)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(12)

    assert annualised_volatility(returns, 12) == pytest.approx(expected_volatility)
    assert annualised_sharpe_ratio(returns, 12) == pytest.approx(expected_sharpe)


def test_annualised_downside_deviation_includes_zero_for_positive_returns():
    returns = pd.Series([0.02, -0.03, -0.04, 0.01])
    expected = math.sqrt((0.0 + 0.03**2 + 0.04**2 + 0.0) / 4 * 12)

    result = annualised_downside_deviation(
        returns,
        periods_per_year=12,
    )

    assert result == pytest.approx(expected)


def test_summarise_returns_matches_component_metrics():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0])

    summary = summarise_returns(
        returns,
        periods_per_year=12,
    )

    assert summary["observations"] == 4
    assert summary["total_return"] == pytest.approx(compound_return(returns))
    assert summary["annualised_return"] == pytest.approx(
        annualised_geometric_return(returns, 12)
    )
    assert summary["annualised_volatility"] == pytest.approx(
        annualised_volatility(returns, 12)
    )
    assert summary["sharpe_ratio"] == pytest.approx(
        annualised_sharpe_ratio(returns, 12)
    )
    assert summary["maximum_drawdown"] == pytest.approx(maximum_drawdown(returns))
    assert summary["positive_day_fraction"] == pytest.approx(0.5)


def test_empty_and_constant_return_edge_cases():
    empty_summary = summarise_returns(pd.Series(dtype=float))

    assert empty_summary["observations"] == 0
    assert np.isnan(empty_summary["total_return"])
    assert np.isnan(empty_summary["annualised_return"])
    assert np.isnan(empty_summary["annualised_volatility"])
    assert np.isnan(empty_summary["sharpe_ratio"])
    assert np.isnan(empty_summary["maximum_drawdown"])
    assert np.isnan(empty_summary["positive_day_fraction"])

    assert np.isnan(annualised_sharpe_ratio([0.01, 0.01, 0.01]))


def test_invalid_return_and_parameter_values_raise():
    with pytest.raises(ValueError, match="less than -1.0"):
        compound_return([0.01, -1.01])

    with pytest.raises(ValueError, match="infinite"):
        annualised_volatility([0.01, np.inf])

    with pytest.raises(ValueError, match="periods_per_year"):
        annualised_geometric_return([0.01, 0.02], periods_per_year=0)

    with pytest.raises(ValueError, match="initial_wealth"):
        calculate_wealth_index([0.01], initial_wealth=0.0)


def test_calculate_rolling_return_state_matches_manual_last_window():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "return": [0.10, -0.05, 0.02, -0.01, 0.03],
        }
    )

    result = calculate_rolling_return_state(
        data,
        performance_window=3,
        risk_window=2,
        periods_per_year=12,
    )

    final_three = data["return"].iloc[-3:]
    final_two = data["return"].iloc[-2:]

    assert result["trailing_return_3"].iloc[-1] == pytest.approx(
        compound_return(final_three)
    )
    assert result["rolling_sharpe_3"].iloc[-1] == pytest.approx(
        annualised_sharpe_ratio(final_three, 12)
    )
    assert result["annualised_volatility_2"].iloc[-1] == pytest.approx(
        annualised_volatility(final_two, 12)
    )
    assert result["annualised_downside_deviation_2"].iloc[-1] == pytest.approx(
        annualised_downside_deviation(final_two, 12)
    )
    assert result["maximum_drawdown_3"].iloc[-1] == pytest.approx(
        maximum_drawdown(final_three)
    )

    assert result["trailing_return_3"].iloc[:2].isna().all()
    assert result["annualised_volatility_2"].iloc[:1].isna().all()


def test_calculate_rolling_return_state_sorts_dates_and_rejects_duplicates():
    data = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-01", "2025-01-03"],
            "return": [0.01, 0.02, -0.01],
        }
    )

    result = calculate_rolling_return_state(
        data,
        performance_window=2,
        risk_window=2,
    )

    assert result["date"].is_monotonic_increasing

    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate dates"):
        calculate_rolling_return_state(
            duplicated,
            performance_window=2,
            risk_window=2,
        )


def test_summarise_concentration():
    result = summarise_concentration([1.0, 1.0, 2.0])

    assert result["count"] == 3
    assert result["effective_count"] == pytest.approx(8.0 / 3.0)
    assert result["largest_share"] == pytest.approx(0.5)
    assert result["top_three_share"] == pytest.approx(1.0)
    assert result["top_five_share"] == pytest.approx(1.0)


def test_summarise_concentration_handles_inactive_and_invalid_values():
    inactive = summarise_concentration([0.0, 0.0, np.nan])

    assert inactive["count"] == 0
    assert np.isnan(inactive["effective_count"])
    assert np.isnan(inactive["largest_share"])

    with pytest.raises(ValueError, match="non-negative"):
        summarise_concentration([0.5, -0.1])
