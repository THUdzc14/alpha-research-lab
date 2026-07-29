import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest import BacktestConfig
from alpha_research.portfolio import (
    build_factor_target_weights,
    combine_sleeve_target_weights,
    combine_factor_scores,
    rescale_target_weights_to_gross,
    estimate_trailing_sleeve_volatility,
    calculate_inverse_volatility_allocations,
    combine_dynamic_sleeve_target_weights,
)


def make_portfolio_panel() -> pd.DataFrame:
    rows = []

    for date in pd.bdate_range("2024-01-01", periods=6):
        for stock_number in range(10):
            rows.append(
                {
                    "date": date,
                    "ticker": f"S{stock_number:02d}",
                    "factor": float(stock_number),
                    "forward_ret_1d": 0.0,
                }
            )

    return pd.DataFrame(rows)


def test_factor_targets_use_only_rebalance_dates():
    panel = make_portfolio_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    expected_dates = [
        panel["date"].drop_duplicates().iloc[0],
        panel["date"].drop_duplicates().iloc[5],
    ]

    assert targets["date"].drop_duplicates().tolist() == expected_dates


def test_factor_targets_have_expected_exposures_and_direction():
    panel = make_portfolio_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    for _, cross_section in targets.groupby("date"):
        long_gross = cross_section["weight"].clip(lower=0.0).sum()
        short_gross = -cross_section["weight"].clip(upper=0.0).sum()

        assert long_gross == pytest.approx(1.0)
        assert short_gross == pytest.approx(1.0)
        assert cross_section["weight"].sum() == pytest.approx(0.0)

        weights = cross_section.set_index("ticker")["weight"]

        assert weights["S09"] > 0
        assert weights["S08"] > 0
        assert weights["S00"] < 0
        assert weights["S01"] < 0


def test_factor_targets_exclude_dates_without_usable_returns():
    panel = make_portfolio_panel()

    last_date = panel["date"].max()
    panel.loc[
        panel["date"] == last_date,
        "forward_ret_1d",
    ] = np.nan

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    assert last_date not in targets["date"].unique()


def test_factor_targets_are_zero_when_coverage_is_insufficient():
    panel = make_portfolio_panel()

    first_date = panel["date"].min()
    first_date_rows = panel["date"] == first_date

    panel.loc[
        panel.index[first_date_rows][:2],
        "factor",
    ] = np.nan

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    first_targets = targets.loc[
        targets["date"] == first_date,
        "weight",
    ]

    assert np.all(first_targets == 0.0)


def make_sleeve_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=2)

    momentum = pd.DataFrame(
        {
            "date": [dates[0]] * 4 + [dates[1]] * 4,
            "ticker": ["A", "B", "C", "D"] * 2,
            "weight": [0.5, 0.5, -0.5, -0.5] * 2,
        }
    )

    volatility = pd.DataFrame(
        {
            "date": [dates[0]] * 4 + [dates[1]] * 4,
            "ticker": ["A", "B", "C", "D"] * 2,
            "weight": [-0.5, 0.5, 0.5, -0.5] * 2,
        }
    )

    return momentum, volatility


def test_combine_sleeves_applies_allocations_and_nets_positions():
    momentum, volatility = make_sleeve_targets()

    combined = combine_sleeve_target_weights(
        sleeve_targets={
            "momentum": momentum,
            "volatility": volatility,
        },
        sleeve_allocations={
            "momentum": 0.5,
            "volatility": 0.5,
        },
    )

    first_date = combined["date"].min()
    weights = combined.loc[combined["date"] == first_date].set_index("ticker")["weight"]

    assert weights["A"] == pytest.approx(0.0)
    assert weights["B"] == pytest.approx(0.5)
    assert weights["C"] == pytest.approx(0.0)
    assert weights["D"] == pytest.approx(-0.5)


def test_combine_sleeves_preserves_dollar_neutrality():
    momentum, volatility = make_sleeve_targets()

    combined = combine_sleeve_target_weights(
        sleeve_targets={
            "momentum": momentum,
            "volatility": volatility,
        },
        sleeve_allocations={
            "momentum": 0.5,
            "volatility": 0.5,
        },
    )

    net_exposure = combined.groupby("date")["weight"].sum()

    assert np.allclose(net_exposure, 0.0)


def test_combine_sleeves_requires_allocations_to_sum_to_one():
    momentum, volatility = make_sleeve_targets()

    with pytest.raises(ValueError, match="sum to one"):
        combine_sleeve_target_weights(
            sleeve_targets={
                "momentum": momentum,
                "volatility": volatility,
            },
            sleeve_allocations={
                "momentum": 0.6,
                "volatility": 0.6,
            },
        )


def test_combine_sleeves_requires_matching_rebalance_dates():
    momentum, volatility = make_sleeve_targets()

    volatility = volatility.loc[volatility["date"] != volatility["date"].max()]

    with pytest.raises(ValueError, match="identical rebalance dates"):
        combine_sleeve_target_weights(
            sleeve_targets={
                "momentum": momentum,
                "volatility": volatility,
            },
            sleeve_allocations={
                "momentum": 0.5,
                "volatility": 0.5,
            },
        )


def test_combine_factor_scores_calculates_weighted_score():
    panel = pd.DataFrame(
        {
            "momentum": [1.0, -1.0, 0.5],
            "volatility": [0.0, 2.0, -0.5],
        }
    )

    result = combine_factor_scores(
        panel=panel,
        factor_weights={
            "momentum": 0.5,
            "volatility": 0.5,
        },
    )

    expected = pd.Series(
        [0.5, 0.5, 0.0],
        name="composite_factor_score",
    )

    pd.testing.assert_series_equal(result, expected)


def test_combine_factor_scores_requires_all_components():
    panel = pd.DataFrame(
        {
            "momentum": [1.0, np.nan],
            "volatility": [2.0, 3.0],
        }
    )

    result = combine_factor_scores(
        panel=panel,
        factor_weights={
            "momentum": 0.5,
            "volatility": 0.5,
        },
    )

    assert result.iloc[0] == pytest.approx(1.5)
    assert np.isnan(result.iloc[1])


def test_combine_factor_scores_requires_weights_to_sum_to_one():
    panel = pd.DataFrame(
        {
            "momentum": [1.0],
            "volatility": [2.0],
        }
    )

    with pytest.raises(ValueError, match="sum to one"):
        combine_factor_scores(
            panel=panel,
            factor_weights={
                "momentum": 0.7,
                "volatility": 0.7,
            },
        )


def test_rescale_target_weights_reaches_requested_gross():
    dates = pd.bdate_range("2024-01-01", periods=2)

    targets = pd.DataFrame(
        {
            "date": [dates[0]] * 4 + [dates[1]] * 4,
            "ticker": ["A", "B", "C", "D"] * 2,
            "weight": [
                0.30,
                0.20,
                -0.25,
                -0.25,
                0.20,
                0.30,
                -0.30,
                -0.20,
            ],
        }
    )

    result = rescale_target_weights_to_gross(
        target_weights=targets,
        target_gross=2.0,
    )

    gross_exposure = result.groupby("date")["weight"].agg(
        lambda weights: weights.abs().sum()
    )

    assert np.allclose(gross_exposure, 2.0)


def test_rescale_target_weights_preserves_dollar_neutrality():
    date = pd.Timestamp("2024-01-01")

    targets = pd.DataFrame(
        {
            "date": [date] * 4,
            "ticker": ["A", "B", "C", "D"],
            "weight": [0.30, 0.20, -0.25, -0.25],
        }
    )

    result = rescale_target_weights_to_gross(
        target_weights=targets,
        target_gross=2.0,
    )

    assert result["weight"].sum() == pytest.approx(0.0)


def test_rescale_target_weights_accepts_date_specific_gross():
    dates = pd.bdate_range("2024-01-01", periods=2)

    targets = pd.DataFrame(
        {
            "date": [dates[0]] * 2 + [dates[1]] * 2,
            "ticker": ["A", "B"] * 2,
            "weight": [0.5, -0.5, 0.5, -0.5],
        }
    )

    gross_schedule = pd.Series(
        [2.0, 0.0],
        index=dates,
    )

    result = rescale_target_weights_to_gross(
        target_weights=targets,
        target_gross=gross_schedule,
    )

    gross_exposure = result.groupby("date")["weight"].agg(
        lambda weights: weights.abs().sum()
    )

    assert gross_exposure.loc[dates[0]] == pytest.approx(2.0)
    assert gross_exposure.loc[dates[1]] == pytest.approx(0.0)


def test_rescale_target_weights_rejects_impossible_scaling():
    date = pd.Timestamp("2024-01-01")

    targets = pd.DataFrame(
        {
            "date": [date, date],
            "ticker": ["A", "B"],
            "weight": [0.0, 0.0],
        }
    )

    gross_schedule = pd.Series(
        [2.0],
        index=[date],
    )

    with pytest.raises(
        ValueError,
        match="zero-weight portfolio",
    ):
        rescale_target_weights_to_gross(
            target_weights=targets,
            target_gross=gross_schedule,
        )


def test_trailing_volatility_uses_only_prior_returns():
    dates = pd.bdate_range("2024-01-01", periods=5)

    sleeve_returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, 0.50, 0.01],
            "B": [0.02, -0.02, 0.04, 0.01, 0.02],
        },
        index=dates,
    )

    result = estimate_trailing_sleeve_volatility(
        sleeve_returns=sleeve_returns,
        lookback=3,
        min_periods=3,
        periods_per_year=1,
    )

    expected_a = sleeve_returns["A"].iloc[:3].std(ddof=1)
    expected_b = sleeve_returns["B"].iloc[:3].std(ddof=1)

    assert np.isnan(result.loc[dates[2], "A"])
    assert result.loc[dates[3], "A"] == pytest.approx(expected_a)
    assert result.loc[dates[3], "B"] == pytest.approx(expected_b)


def test_inverse_volatility_favours_lower_risk_sleeve():
    date = pd.Timestamp("2024-01-01")

    volatility = pd.DataFrame(
        {
            "Low Risk": [0.10],
            "High Risk": [0.20],
        },
        index=[date],
    )

    result = calculate_inverse_volatility_allocations(
        sleeve_volatility=volatility,
        allocation_floor=0.0,
    )

    assert result.loc[date, "Low Risk"] == pytest.approx(2.0 / 3.0)
    assert result.loc[date, "High Risk"] == pytest.approx(1.0 / 3.0)
    assert result.loc[date].sum() == pytest.approx(1.0)


def test_inverse_volatility_uses_equal_weight_during_warmup():
    date = pd.Timestamp("2024-01-01")

    volatility = pd.DataFrame(
        {
            "A": [np.nan],
            "B": [np.nan],
        },
        index=[date],
    )

    result = calculate_inverse_volatility_allocations(
        sleeve_volatility=volatility,
        allocation_floor=0.20,
    )

    assert result.loc[date, "A"] == pytest.approx(0.5)
    assert result.loc[date, "B"] == pytest.approx(0.5)


def test_inverse_volatility_respects_allocation_floor():
    date = pd.Timestamp("2024-01-01")

    volatility = pd.DataFrame(
        {
            "Low Risk": [0.01],
            "High Risk": [1.00],
        },
        index=[date],
    )

    result = calculate_inverse_volatility_allocations(
        sleeve_volatility=volatility,
        allocation_floor=0.20,
    )

    assert result.loc[date].min() >= 0.20
    assert result.loc[date].max() <= 0.80
    assert result.loc[date].sum() == pytest.approx(1.0)


def test_combine_dynamic_sleeves_applies_date_allocations():
    momentum, volatility = make_sleeve_targets()
    dates = pd.DatetimeIndex(momentum["date"].unique()).sort_values()

    allocations = pd.DataFrame(
        {
            "momentum": [0.75, 0.25],
            "volatility": [0.25, 0.75],
        },
        index=dates,
    )

    result = combine_dynamic_sleeve_target_weights(
        sleeve_targets={
            "momentum": momentum,
            "volatility": volatility,
        },
        sleeve_allocations=allocations,
    )

    first_weights = result.loc[result["date"] == dates[0]].set_index("ticker")["weight"]

    second_weights = result.loc[result["date"] == dates[1]].set_index("ticker")[
        "weight"
    ]

    assert first_weights["A"] == pytest.approx(0.25)
    assert first_weights["B"] == pytest.approx(0.50)
    assert first_weights["C"] == pytest.approx(-0.25)
    assert first_weights["D"] == pytest.approx(-0.50)

    assert second_weights["A"] == pytest.approx(-0.25)
    assert second_weights["B"] == pytest.approx(0.50)
    assert second_weights["C"] == pytest.approx(0.25)
    assert second_weights["D"] == pytest.approx(-0.50)
