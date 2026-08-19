import numpy as np
import pandas as pd
import pytest

from alpha_research.config.research import (
    STRATEGY_EVALUATION_START_DATE,
)
from alpha_research.artifacts import (
    validate_attribution_artifacts,
    validate_monitoring_artifacts,
)
from alpha_research.refresh import (
    REFRESH_DATASET_GROUPS,
    build_complete_research_refresh,
    key_values_match,
)
from alpha_research.workflows import (
    ATTRIBUTION_DATASET_NAMES,
    MONITORING_DATASET_NAMES,
    build_strategy_monitoring_datasets,
)


@pytest.fixture(scope="module")
def refresh_inputs():
    # dates = pd.bdate_range("2024-01-02", periods=130)
    dates = pd.bdate_range(
        "2015-12-15",
        periods=130,
    )
    tickers = [f"S{number:02d}" for number in range(30)]
    market_returns = np.resize(
        np.array([-0.01, -0.005, 0.005, 0.01]),
        len(dates) - 1,
    )
    benchmark_prices = np.concatenate(
        [
            np.array([100.0]),
            100.0 * np.cumprod(1.0 + market_returns),
        ]
    )
    factor_rows = []

    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            score = float(ticker_number)
            asset_return = (
                market_returns[date_number] + (ticker_number - 14.5) / 10_000.0
                if date_number < len(market_returns)
                else np.nan
            )
            factor_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "mom_12_1m_z": score,
                    "mom_12_1m_raw": score * 10.0,
                    "realised_vol_63_z": score,
                    "realised_vol_63_raw": score / 10.0,
                    "forward_ret_1d": asset_return,
                    "forward_ret_5d": score / 1_000.0,
                    "beta_126": 0.8 + ticker_number / 100.0,
                    "sector": ("Technology" if ticker_number % 2 == 0 else "Financials"),
                    "dollar_volume": (1_000_000.0 + ticker_number * 10_000.0),
                }
            )

    return {
        "factor_panel": pd.DataFrame(factor_rows),
        "benchmark_prices": pd.DataFrame(
            {
                "date": dates,
                "ticker": "SPY",
                "adj_close": benchmark_prices,
            }
        ),
    }


@pytest.fixture(scope="module")
def refreshed_datasets(refresh_inputs):
    return build_complete_research_refresh(**refresh_inputs)


def test_complete_refresh_builds_both_dataset_groups(refreshed_datasets):
    assert tuple(refreshed_datasets) == REFRESH_DATASET_GROUPS
    assert tuple(refreshed_datasets["attribution"]) == (ATTRIBUTION_DATASET_NAMES)
    assert tuple(refreshed_datasets["monitoring"]) == MONITORING_DATASET_NAMES


def test_complete_refresh_aligns_active_strategies_and_benchmark(
    refresh_inputs,
    refreshed_datasets,
):
    attribution = refreshed_datasets["attribution"]

    benchmark_dates = pd.DatetimeIndex(attribution["benchmark_daily"]["date"])

    expected_dates = pd.DatetimeIndex(refresh_inputs["benchmark_prices"]["date"].iloc[:-1])

    expected_dates = expected_dates[expected_dates >= STRATEGY_EVALUATION_START_DATE]

    assert benchmark_dates.equals(expected_dates)

    assert benchmark_dates.min() == STRATEGY_EVALUATION_START_DATE

    for _, portfolio_daily in attribution["portfolio_daily"].groupby("portfolio"):
        portfolio_dates = pd.DatetimeIndex(portfolio_daily["date"])

        assert portfolio_dates.equals(benchmark_dates)


def test_complete_refresh_outputs_pass_artifact_validation(
    refreshed_datasets,
):
    validate_attribution_artifacts(refreshed_datasets["attribution"])
    validate_monitoring_artifacts(refreshed_datasets["monitoring"])


def test_complete_refresh_monitoring_matches_standalone_workflow(
    refresh_inputs,
    refreshed_datasets,
):
    attribution = refreshed_datasets["attribution"]
    standalone = build_strategy_monitoring_datasets(
        factor_panel=refresh_inputs["factor_panel"],
        selected_implementations=attribution["selected_implementations"],
        portfolio_daily=attribution["portfolio_daily"],
        security_holdings=attribution["security_holdings"],
        security_daily=attribution["security_daily"],
        benchmark_daily=attribution["benchmark_daily"],
    )

    assert tuple(standalone) == MONITORING_DATASET_NAMES

    for name in MONITORING_DATASET_NAMES:
        pd.testing.assert_frame_equal(
            standalone[name],
            refreshed_datasets["monitoring"][name],
            check_exact=True,
        )


def test_complete_refresh_rejects_duplicate_benchmark_dates(refresh_inputs):
    invalid_inputs = dict(refresh_inputs)
    invalid_inputs["benchmark_prices"] = pd.concat(
        [
            refresh_inputs["benchmark_prices"],
            refresh_inputs["benchmark_prices"].iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        build_complete_research_refresh(**invalid_inputs)


def test_refresh_key_comparison_ignores_equivalent_storage_dtypes():
    reference = pd.DataFrame(
        {
            "benchmark": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "date": pd.Series(
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                dtype="datetime64[ms]",
            ),
        }
    )
    reconstructed = pd.DataFrame(
        {
            "benchmark": ["SPY", "SPY"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )

    assert key_values_match(
        reference,
        reconstructed,
        ["benchmark", "date"],
    )

    reconstructed.loc[1, "benchmark"] = "QQQ"

    assert not key_values_match(
        reference,
        reconstructed,
        ["benchmark", "date"],
    )
