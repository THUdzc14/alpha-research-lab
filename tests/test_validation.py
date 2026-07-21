import numpy as np
import pandas as pd
import pytest

from alpha_research.validation import (
    assign_quantiles,
    calculate_daily_ic,
    calculate_long_short_spread,
    calculate_quantile_returns,
    summarise_ic,
    calculate_ic_by_horizon,
    calculate_non_overlapping_ic,
    calculate_rolling_ic,
    calculate_subperiod_ic,
    select_non_overlapping_dates,
)


def make_validation_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01"] * 10 + ["2024-01-02"] * 10)

    tickers = [f"S{i:02d}" for i in range(10)] * 2

    factor = list(range(10)) + list(range(10))

    # Perfect positive relation on the first date and negative on the second.
    forward_returns = list(range(10)) + list(reversed(range(10)))

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "factor": factor,
            "forward_return": forward_returns,
        }
    )


def test_daily_spearman_ic():
    panel = make_validation_panel()

    result = calculate_daily_ic(
        panel,
        factor_column="factor",
        forward_return_column="forward_return",
        method="spearman",
        min_observations=5,
    )

    assert result.loc[0, "ic"] == pytest.approx(1.0)
    assert result.loc[1, "ic"] == pytest.approx(-1.0)


def test_daily_pearson_ic():
    panel = make_validation_panel()

    result = calculate_daily_ic(
        panel,
        factor_column="factor",
        forward_return_column="forward_return",
        method="pearson",
        min_observations=5,
    )

    assert result.loc[0, "ic"] == pytest.approx(1.0)
    assert result.loc[1, "ic"] == pytest.approx(-1.0)


def test_insufficient_observations_returns_nan():
    panel = make_validation_panel().iloc[:4]

    result = calculate_daily_ic(
        panel,
        factor_column="factor",
        forward_return_column="forward_return",
        min_observations=5,
    )

    assert np.isnan(result.loc[0, "ic"])
    assert result.loc[0, "observations"] == 4


def test_ic_summary():
    ic_series = pd.DataFrame(
        {
            "ic": [0.10, 0.20, 0.30, np.nan],
        }
    )

    result = summarise_ic(ic_series)

    assert result.loc[0, "count"] == 3
    assert result.loc[0, "mean_ic"] == pytest.approx(0.20)
    assert result.loc[0, "positive_fraction"] == pytest.approx(1.0)


def test_assign_quantiles_orders_factor():
    panel = make_validation_panel().iloc[:10].copy()

    quantiles = assign_quantiles(
        panel,
        factor_column="factor",
        quantiles=5,
        min_observations=10,
    )

    assert quantiles.min() == 1
    assert quantiles.max() == 5

    lowest_factor_quantile = quantiles.loc[panel["factor"].idxmin()]
    highest_factor_quantile = quantiles.loc[panel["factor"].idxmax()]

    assert lowest_factor_quantile == 1
    assert highest_factor_quantile == 5


def test_calculate_quantile_returns():
    panel = make_validation_panel().iloc[:10].copy()

    result = calculate_quantile_returns(
        panel,
        factor_column="factor",
        forward_return_column="forward_return",
        quantiles=5,
        min_observations=10,
    )

    assert set(result["quantile"]) == {1, 2, 3, 4, 5}

    bottom_return = result.loc[
        result["quantile"] == 1,
        "mean_forward_return",
    ].iloc[0]

    top_return = result.loc[
        result["quantile"] == 5,
        "mean_forward_return",
    ].iloc[0]

    assert top_return > bottom_return


def test_long_short_spread():
    quantile_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "quantile": [1, 5, 1, 5],
            "mean_forward_return": [0.01, 0.05, -0.02, 0.03],
        }
    )

    result = calculate_long_short_spread(
        quantile_returns,
        top_quantile=5,
        bottom_quantile=1,
    )

    assert result.loc[0, "long_short_return"] == pytest.approx(0.04)
    assert result.loc[1, "long_short_return"] == pytest.approx(0.05)


def make_multi_year_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=20)
    rows = []

    for date_number, date in enumerate(dates):
        for stock_number in range(10):
            rows.append(
                {
                    "date": date,
                    "ticker": f"S{stock_number:02d}",
                    "factor": stock_number,
                    "forward_ret_1d": stock_number * 0.01,
                    "forward_ret_5d": stock_number * 0.02,
                }
            )

    return pd.DataFrame(rows)


def test_select_non_overlapping_dates():
    panel = make_multi_year_panel()

    result = select_non_overlapping_dates(
        panel,
        step=5,
        offset=0,
    )

    assert result["date"].nunique() == 4


def test_non_overlapping_offsets_are_all_returned():
    panel = make_multi_year_panel()

    result = calculate_non_overlapping_ic(
        panel,
        factor_column="factor",
        forward_return_column="forward_ret_5d",
        horizon=5,
        min_observations=5,
    )

    assert set(result["offset"]) == {0, 1, 2, 3, 4}
    assert np.allclose(result["mean_ic"], 1.0)


def test_calculate_ic_by_horizon():
    panel = make_multi_year_panel()

    result = calculate_ic_by_horizon(
        panel,
        factor_column="factor",
        forward_return_columns=[
            "forward_ret_1d",
            "forward_ret_5d",
        ],
        min_observations=5,
    )

    assert set(result["forward_return_column"]) == {
        "forward_ret_1d",
        "forward_ret_5d",
    }

    assert np.allclose(result["mean_ic"], 1.0)


def test_calculate_subperiod_ic():
    panel = make_multi_year_panel()

    periods = {
        "first_half": ("2020-01-01", "2020-01-14"),
        "second_half": ("2020-01-15", "2020-02-01"),
    }

    result = calculate_subperiod_ic(
        panel,
        factor_column="factor",
        forward_return_column="forward_ret_1d",
        periods=periods,
        min_observations=5,
    )

    assert set(result["period"]) == {
        "first_half",
        "second_half",
    }

    assert np.allclose(result["mean_ic"], 1.0)


def test_calculate_rolling_ic():
    ic_series = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "ic": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    result = calculate_rolling_ic(
        ic_series,
        window=3,
        min_periods=3,
    )

    assert np.isnan(result.loc[1, "rolling_ic_3"])
    assert result.loc[2, "rolling_ic_3"] == pytest.approx(0.2)
    assert result.loc[4, "rolling_ic_3"] == pytest.approx(0.4)
