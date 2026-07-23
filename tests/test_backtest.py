import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest import (
    BacktestConfig,
    calculate_turnover,
    construct_long_short_weights,
    get_rebalance_dates,
    run_long_short_backtest,
    summarise_backtest,
    run_rebalance_offset_backtests,
    summarise_backtest_legs,
    summarise_backtest_subperiods,
)


def make_cross_section() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"S{i:02d}" for i in range(10)],
            "factor": np.arange(10, dtype=float),
        }
    )


def make_backtest_panel(
    dates: int = 10,
    stocks: int = 10,
) -> pd.DataFrame:
    rows = []

    for date in pd.bdate_range("2024-01-01", periods=dates):
        for stock_number in range(stocks):
            rows.append(
                {
                    "date": date,
                    "ticker": f"S{stock_number:02d}",
                    "factor": float(stock_number),
                    "forward_ret_1d": stock_number * 0.001,
                }
            )

    return pd.DataFrame(rows)


def test_construct_weights_has_expected_exposure():
    cross_section = make_cross_section()

    weights = construct_long_short_weights(
        cross_section,
        factor_column="factor",
        quantiles=5,
        long_quantile=5,
        short_quantile=1,
        long_gross=1.0,
        short_gross=1.0,
        min_observations=10,
    )

    assert weights[weights > 0].sum() == pytest.approx(1.0)
    assert -weights[weights < 0].sum() == pytest.approx(1.0)
    assert weights.sum() == pytest.approx(0.0)

    assert (weights > 0).sum() == 2
    assert (weights < 0).sum() == 2


def test_highest_factor_stocks_are_long():
    cross_section = make_cross_section()

    weights = construct_long_short_weights(
        cross_section,
        factor_column="factor",
        min_observations=10,
    )

    assert weights.loc["S09"] > 0
    assert weights.loc["S08"] > 0

    assert weights.loc["S00"] < 0
    assert weights.loc["S01"] < 0


def test_rebalance_dates():
    dates = pd.bdate_range("2024-01-01", periods=12)

    result = get_rebalance_dates(
        dates,
        frequency=5,
        offset=0,
    )

    assert list(result) == [
        dates[0],
        dates[5],
        dates[10],
    ]


def test_turnover_from_zero_portfolio():
    previous = pd.Series(dtype="float64")

    target = pd.Series(
        {
            "AAA": 0.5,
            "BBB": 0.5,
            "CCC": -0.5,
            "DDD": -0.5,
        }
    )

    turnover = calculate_turnover(previous, target)

    assert turnover == pytest.approx(2.0)


def test_backtest_produces_positive_return_when_factor_predicts_returns():
    panel = make_backtest_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        transaction_cost_bps=0.0,
        min_observations=10,
    )

    daily, holdings = run_long_short_backtest(
        panel,
        factor_column="factor",
        config=config,
    )

    assert (daily["gross_return"] > 0).all()
    assert daily["net_cumulative_return"].iloc[-1] > 1.0
    assert not holdings.empty


def test_transaction_cost_is_deducted_on_rebalance():
    panel = make_backtest_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        transaction_cost_bps=10.0,
        min_observations=10,
    )

    daily, _ = run_long_short_backtest(
        panel,
        factor_column="factor",
        config=config,
    )

    first_day = daily.iloc[0]

    expected_cost = 2.0 * 10.0 / 10_000.0

    assert first_day["turnover"] == pytest.approx(2.0)
    assert first_day["transaction_cost"] == pytest.approx(expected_cost)

    assert first_day["net_return"] == pytest.approx(
        first_day["gross_return"] - expected_cost
    )


def test_weights_remain_constant_between_rebalances():
    panel = make_backtest_panel(dates=6)

    config = BacktestConfig(
        rebalance_frequency=5,
        transaction_cost_bps=0.0,
        min_observations=10,
    )

    _, holdings = run_long_short_backtest(
        panel,
        factor_column="factor",
        config=config,
    )

    day_1 = holdings.loc[holdings["date"] == holdings["date"].unique()[0]].set_index(
        "ticker"
    )["weight"]

    day_2 = holdings.loc[holdings["date"] == holdings["date"].unique()[1]].set_index(
        "ticker"
    )["weight"]

    pd.testing.assert_series_equal(
        day_1,
        day_2,
        check_names=False,
    )


def test_backtest_summary():
    daily_results = pd.DataFrame(
        {
            "net_return": [0.01, -0.005, 0.02, 0.0],
            "turnover": [2.0, 0.0, 1.0, 0.0],
            "transaction_cost": [0.002, 0.0, 0.001, 0.0],
            "is_rebalance": [True, False, True, False],
            "missing_return_weight": [0.0, 0.0, 0.0, 0.0],
        }
    )

    summary = summarise_backtest(daily_results)

    assert summary["observations"].iloc[0] == 4
    assert summary["total_return"].iloc[0] > 0
    assert summary["max_drawdown"].iloc[0] <= 0


def test_backtest_contains_leg_returns():
    panel = make_backtest_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        transaction_cost_bps=0.0,
        min_observations=10,
    )

    daily, _ = run_long_short_backtest(
        panel,
        factor_column="factor",
        config=config,
    )

    assert {
        "long_return",
        "short_return",
    }.issubset(daily.columns)

    assert np.allclose(
        daily["gross_return"],
        daily["long_return"] + daily["short_return"],
    )


def test_summarise_backtest_legs():
    panel = make_backtest_panel()

    config = BacktestConfig(
        transaction_cost_bps=0.0,
        min_observations=10,
    )

    daily, _ = run_long_short_backtest(
        panel,
        factor_column="factor",
        config=config,
    )

    summary = summarise_backtest_legs(daily)

    assert set(summary["portfolio"]) == {
        "long_leg",
        "short_leg",
        "stock_long_short",
        "benchmark_hedge",
        "gross_portfolio",
        "net_portfolio",
    }


def test_all_rebalance_offsets_are_returned():
    panel = make_backtest_panel(dates=20)

    config = BacktestConfig(
        rebalance_frequency=5,
        transaction_cost_bps=0.0,
        min_observations=10,
    )

    result = run_rebalance_offset_backtests(
        panel,
        factor_column="factor",
        base_config=config,
    )

    assert set(result["offset"]) == {0, 1, 2, 3, 4}


def test_summarise_backtest_subperiods():
    panel = make_backtest_panel(dates=20)

    config = BacktestConfig(
        transaction_cost_bps=0.0,
        min_observations=10,
    )

    daily, _ = run_long_short_backtest(
        panel,
        factor_column="factor",
        config=config,
    )

    periods = {
        "first": ("2024-01-01", "2024-01-12"),
        "second": ("2024-01-15", "2024-02-01"),
    }

    result = summarise_backtest_subperiods(
        daily,
        periods=periods,
    )

    assert set(result["period"]) == {"first", "second"}
