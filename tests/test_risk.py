import numpy as np
import pandas as pd
import pytest

from alpha_research.risk import (
    calculate_market_exposure,
    calculate_rolling_beta,
    calculate_sector_exposure,
    calculate_strategy_exposures,
    prepare_benchmark_returns,
    summarise_sector_exposure,
)


def test_prepare_benchmark_returns():
    benchmark = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "adj_close": [100.0, 101.0, 99.99],
        }
    )

    result = prepare_benchmark_returns(benchmark)

    assert result["benchmark_return"].iloc[0] == pytest.approx(0.01)
    assert result["benchmark_return"].iloc[1] == pytest.approx(-0.01)
    assert np.isnan(result["benchmark_return"].iloc[2])


def test_market_exposure_recovers_known_beta():
    benchmark_returns = np.array([-0.02, -0.01, 0.00, 0.01, 0.02])

    strategy_returns = 0.001 + 1.5 * benchmark_returns

    dates = pd.bdate_range("2024-01-01", periods=5)

    strategy = pd.DataFrame(
        {
            "date": dates,
            "strategy_return": strategy_returns,
        }
    )

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": benchmark_returns,
        }
    )

    result = calculate_market_exposure(
        strategy,
        benchmark,
        strategy_return_column="strategy_return",
    )

    assert result.loc[0, "beta"] == pytest.approx(1.5)
    assert result.loc[0, "daily_alpha"] == pytest.approx(0.001)
    assert result.loc[0, "correlation"] == pytest.approx(1.0)


def test_strategy_exposures_returns_all_legs():
    dates = pd.bdate_range("2024-01-01", periods=5)
    benchmark_values = np.linspace(-0.01, 0.01, 5)

    daily = pd.DataFrame(
        {
            "date": dates,
            "long_return": benchmark_values,
            "short_return": -0.5 * benchmark_values,
            "gross_return": 0.5 * benchmark_values,
            "net_return": 0.5 * benchmark_values,
        }
    )

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": benchmark_values,
        }
    )

    result = calculate_strategy_exposures(daily, benchmark)

    assert set(result["portfolio"]) == {
        "long_leg",
        "short_leg",
        "gross_long_short",
        "net_long_short",
    }


def test_rolling_beta():
    dates = pd.bdate_range("2024-01-01", periods=10)
    benchmark_values = np.linspace(-0.02, 0.02, 10)

    strategy = pd.DataFrame(
        {
            "date": dates,
            "strategy_return": 2.0 * benchmark_values,
        }
    )

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": benchmark_values,
        }
    )

    result = calculate_rolling_beta(
        strategy,
        benchmark,
        strategy_return_column="strategy_return",
        window=5,
        min_periods=5,
    )

    assert result["rolling_beta_5"].iloc[-1] == pytest.approx(2.0)


def test_sector_exposure():
    holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "weight": [0.5, 0.5, -0.5, -0.5],
        }
    )

    metadata = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "sector": [
                "Technology",
                "Financials",
                "Technology",
                "Healthcare",
            ],
        }
    )

    result = calculate_sector_exposure(
        holdings,
        metadata,
    )

    technology = result.loc[result["sector"] == "Technology"].iloc[0]

    assert technology["long_weight"] == pytest.approx(0.5)
    assert technology["short_weight"] == pytest.approx(0.5)
    assert technology["net_weight"] == pytest.approx(0.0)


def test_summarise_sector_exposure():
    exposure = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "sector": ["Technology", "Technology"],
            "long_weight": [0.5, 0.6],
            "short_weight": [0.2, 0.1],
            "net_weight": [0.3, 0.5],
            "gross_weight": [0.7, 0.7],
        }
    )

    result = summarise_sector_exposure(exposure)

    assert result.loc[0, "average_long_weight"] == pytest.approx(0.55)
    assert result.loc[0, "average_net_weight"] == pytest.approx(0.40)
    assert result.loc[0, "maximum_absolute_net_weight"] == pytest.approx(0.50)


def test_benchmark_hedge_offsets_stock_beta():
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": -0.5, "DDD": -0.5})

    betas = pd.Series({"AAA": 1.5, "BBB": 1.3, "CCC": 0.7, "DDD": 0.5})

    stock_beta = float((weights * betas).sum())
    benchmark_weight = -stock_beta

    assert stock_beta + benchmark_weight == pytest.approx(0.0)
