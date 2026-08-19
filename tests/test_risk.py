from inspect import signature

import numpy as np
import pandas as pd
import pytest

from alpha_research.config.research import (
    DEFAULT_NUMERICAL_TOLERANCE,
    MONITORING_SPECIFICATION,
    TRADING_DAYS_PER_YEAR,
)
from alpha_research.risk import (
    calculate_beta_state,
    calculate_concentration_state,
    calculate_contribution_concentration_state,
    calculate_market_exposure,
    calculate_rolling_beta,
    calculate_rolling_contribution_detail,
    calculate_rolling_market_model,
    calculate_realised_beta_state,
    calculate_sector_exposure,
    calculate_strategy_exposures,
    prepare_benchmark_returns,
    summarise_sector_exposure,
)


def test_public_risk_defaults_match_frozen_research_configuration():
    market_model_parameters = signature(calculate_rolling_market_model).parameters
    assert (
        market_model_parameters["annualisation_factor"].default
        == TRADING_DAYS_PER_YEAR
    )

    realised_beta_parameters = signature(calculate_realised_beta_state).parameters
    assert (
        realised_beta_parameters["window"].default
        == MONITORING_SPECIFICATION.risk_window
    )

    beta_state_parameters = signature(calculate_beta_state).parameters
    assert (
        beta_state_parameters["window"].default
        == MONITORING_SPECIFICATION.risk_window
    )
    assert beta_state_parameters["tolerance"].default == DEFAULT_NUMERICAL_TOLERANCE

    for concentration_function in (
        calculate_rolling_contribution_detail,
        calculate_contribution_concentration_state,
        calculate_concentration_state,
    ):
        assert (
            signature(concentration_function).parameters["window"].default
            == MONITORING_SPECIFICATION.concentration_window
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


def make_market_model_data(
    observations: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create aligned synthetic stock and market data."""
    dates = pd.bdate_range(
        "2024-01-01",
        periods=observations + 1,
    )

    market_returns = np.resize(
        np.array(
            [0.01, -0.02, 0.015, 0.005, -0.01, 0.02],
            dtype=float,
        ),
        observations,
    )

    noise = np.resize(
        np.array(
            [0.001, -0.002, 0.003, -0.001, 0.002, -0.003],
            dtype=float,
        ),
        observations,
    )

    benchmark_prices = [100.0]
    stock_prices = [50.0]

    for market_return, residual in zip(market_returns, noise):
        stock_return = 0.0005 + 1.4 * market_return + residual

        benchmark_prices.append(benchmark_prices[-1] * (1.0 + market_return))
        stock_prices.append(stock_prices[-1] * (1.0 + stock_return))

    equity = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "adj_close": stock_prices,
        }
    )
    equity["ret_1d"] = equity["adj_close"].pct_change()

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "adj_close": benchmark_prices,
        }
    )

    return equity, benchmark


def test_rolling_market_model_matches_window_ols():
    equity, benchmark = make_market_model_data()

    result = calculate_rolling_market_model(
        equity,
        benchmark,
        window=6,
        min_periods=6,
        annualisation_factor=252,
        output_prefix="model",
    )

    stock_window = equity["ret_1d"].iloc[-6:].to_numpy()
    market_window = benchmark["adj_close"].pct_change().iloc[-6:].to_numpy()

    design = np.column_stack([np.ones(6), market_window])

    alpha, beta = np.linalg.lstsq(
        design,
        stock_window,
        rcond=None,
    )[0]

    residuals = stock_window - (alpha + beta * market_window)

    expected_idio_vol = residuals.std(ddof=1) * np.sqrt(252)

    assert result["model_alpha"].iloc[-1] == pytest.approx(alpha)
    assert result["model_beta"].iloc[-1] == pytest.approx(beta)
    assert result["model_residual"].iloc[-1] == pytest.approx(residuals[-1])
    assert result["model_idio_vol"].iloc[-1] == pytest.approx(expected_idio_vol)


def test_rolling_market_model_uses_no_future_returns():
    equity, benchmark = make_market_model_data()

    baseline = calculate_rolling_market_model(
        equity,
        benchmark,
        window=6,
        min_periods=6,
        output_prefix="model",
    )

    changed_equity = equity.copy()
    changed_equity.loc[
        changed_equity.index[-1],
        "ret_1d",
    ] = 0.75

    changed = calculate_rolling_market_model(
        changed_equity,
        benchmark,
        window=6,
        min_periods=6,
        output_prefix="model",
    )

    columns = [
        "model_alpha",
        "model_beta",
        "model_idio_vol",
    ]

    assert np.allclose(
        baseline[columns].iloc[-2],
        changed[columns].iloc[-2],
    )


def test_rolling_market_model_separates_tickers():
    equity, benchmark = make_market_model_data()

    aaa_only = calculate_rolling_market_model(
        equity,
        benchmark,
        window=6,
        min_periods=6,
        output_prefix="model",
    )

    second = equity.copy()
    second["ticker"] = "BBB"
    second["ret_1d"] = 0.001 + 0.5 * benchmark["adj_close"].pct_change()

    panel = pd.concat(
        [second, equity],
        ignore_index=True,
    )

    combined = calculate_rolling_market_model(
        panel,
        benchmark,
        window=6,
        min_periods=6,
        output_prefix="model",
    )

    columns = [
        "model_alpha",
        "model_beta",
        "model_idio_vol",
    ]

    combined_aaa = combined.loc[combined["ticker"] == "AAA", columns].reset_index(
        drop=True
    )

    assert np.allclose(
        aaa_only[columns],
        combined_aaa,
        equal_nan=True,
    )

    final_betas = combined.groupby("ticker")["model_beta"].last()
    assert final_betas["BBB"] == pytest.approx(0.5)


def test_rolling_market_model_requires_sufficient_history():
    equity, benchmark = make_market_model_data(
        observations=6,
    )

    result = calculate_rolling_market_model(
        equity,
        benchmark,
        window=6,
        min_periods=6,
        output_prefix="model",
    )

    assert result["model_idio_vol"].iloc[:6].isna().all()
    assert pd.notna(result["model_idio_vol"].iloc[6])


def test_rolling_market_model_counts_only_aligned_returns():
    equity, benchmark = make_market_model_data(
        observations=10,
    )

    # Remove one benchmark date from the first usable window.
    benchmark = benchmark.drop(index=3).reset_index(drop=True)

    result = calculate_rolling_market_model(
        equity,
        benchmark,
        window=6,
        min_periods=6,
        output_prefix="model",
    )

    assert pd.isna(result["model_idio_vol"].iloc[6])
    assert pd.notna(result["model_idio_vol"].iloc[-1])


def test_rolling_market_model_handles_zero_market_variance():
    dates = pd.bdate_range(
        "2024-01-01",
        periods=8,
    )

    equity = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "ret_1d": np.linspace(-0.01, 0.01, len(dates)),
        }
    )

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "adj_close": 100.0,
        }
    )

    result = calculate_rolling_market_model(
        equity,
        benchmark,
        window=5,
        min_periods=5,
        output_prefix="model",
    )

    output_columns = [
        "model_alpha",
        "model_beta",
        "model_residual",
        "model_idio_vol",
    ]

    assert result[output_columns].isna().all().all()
