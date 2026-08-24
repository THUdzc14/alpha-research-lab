import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest import BacktestConfig, get_rebalance_dates
from alpha_research.portfolio import (
    calculate_rebalance_inverse_volatility_allocations,
)
from alpha_research.workflows import (
    build_common_strategy_backtests,
    build_frozen_strategy_target_weights,
)


def make_common_strategy_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=60)
    tickers = [f"S{number:02d}" for number in range(30)]
    rows = []

    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "mom_12_1m_z": float(ticker_number),
                    "realised_vol_63_z": float((ticker_number * 7) % len(tickers)),
                    "forward_ret_1d": (
                        (ticker_number % 5 - 2) / 1_000.0
                        + (date_number % 3 - 1) / 10_000.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_rebalance_inverse_volatility_uses_only_prior_joint_history():
    dates = pd.bdate_range("2025-01-02", periods=4)
    sleeve_returns = pd.DataFrame(
        {
            "Momentum": [0.01, -0.01, 0.01, -0.01],
            "Realised Volatility": [0.02, -0.02, 0.02, -0.02],
        },
        index=dates,
    )

    result = calculate_rebalance_inverse_volatility_allocations(
        sleeve_returns,
        rebalance_dates=pd.DatetimeIndex([dates[1], dates[3]]),
        lookback=3,
        min_periods=2,
        periods_per_year=252,
    )

    assert result.loc[dates[1], "Momentum"] == pytest.approx(0.5)
    assert result.loc[dates[1], "Realised Volatility"] == pytest.approx(0.5)
    assert result.loc[dates[3], "Momentum"] == pytest.approx(2.0 / 3.0)
    assert result.loc[dates[3], "Realised Volatility"] == pytest.approx(1.0 / 3.0)


def test_common_strategy_workflow_builds_all_portfolios_on_one_schedule():
    panel = make_common_strategy_panel()
    original_panel = panel.copy(deep=True)
    config = BacktestConfig(
        rebalance_frequency=5,
        rebalance_offset=2,
        transaction_cost_bps=10.0,
    )

    daily_results, holdings_results, target_weights = (
        build_common_strategy_backtests(panel, config)
    )

    expected_portfolios = (
        "Momentum Only",
        "Realised Volatility Only",
        "Composite Score",
        "Fixed 50/50 Sleeves",
        "Pure Inverse Volatility",
    )
    expected_rebalance_dates = get_rebalance_dates(
        panel["date"],
        frequency=config.rebalance_frequency,
        offset=config.rebalance_offset,
    )

    assert tuple(daily_results) == expected_portfolios
    assert tuple(holdings_results) == expected_portfolios
    assert tuple(target_weights) == expected_portfolios

    for portfolio in expected_portfolios:
        daily = daily_results[portfolio]
        holdings = holdings_results[portfolio]
        targets = target_weights[portfolio]

        assert pd.DatetimeIndex(targets["date"].unique()).equals(
            expected_rebalance_dates
        )
        assert not targets.duplicated(["date", "ticker"]).any()
        assert not holdings.duplicated(["date", "ticker"]).any()
        assert np.allclose(
            daily["gross_return"] - daily["transaction_cost"],
            daily["net_return"],
        )
        assert np.allclose(
            daily["turnover"] * config.transaction_cost_bps / 10_000.0,
            daily["transaction_cost"],
        )

        target_net = targets.groupby("date")["weight"].sum()
        target_gross = targets.groupby("date")["weight"].agg(
            lambda weights: weights.abs().sum()
        )

        assert np.allclose(target_net, 0.0, atol=1e-12)
        assert target_gross.le(2.0 + 1e-12).all()

    for portfolio in (
        "Momentum Only",
        "Realised Volatility Only",
        "Composite Score",
    ):
        gross = target_weights[portfolio].groupby("date")["weight"].agg(
            lambda weights: weights.abs().sum()
        )
        assert np.allclose(gross, 2.0)

    pd.testing.assert_frame_equal(panel, original_panel)


def test_common_strategy_workflow_rejects_invalid_inputs():
    panel = make_common_strategy_panel()
    duplicated_panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date-ticker"):
        build_common_strategy_backtests(
            duplicated_panel,
            BacktestConfig(),
        )

    with pytest.raises(ValueError, match="beta-neutral"):
        build_common_strategy_backtests(
            panel,
            BacktestConfig(beta_neutral=True),
        )


def test_frozen_strategy_workflow_builds_all_selected_targets():
    dates = pd.bdate_range("2025-01-02", periods=60)
    tickers = [f"S{number:02d}" for number in range(30)]
    rows = []

    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            score = float(ticker_number)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "mom_12_1m_z": score,
                    "realised_vol_63_z": score,
                    "forward_ret_1d": (
                        (ticker_number % 5 - 2) / 1_000.0
                        + (date_number % 3 - 1) / 10_000.0
                    ),
                }
            )

    result = build_frozen_strategy_target_weights(pd.DataFrame(rows))

    assert tuple(result) == (
        "Composite Score",
        "Fixed 50/50 Sleeves",
        "Pure Inverse Volatility",
    )
    assert result["Composite Score"]["date"].nunique() == 3
    assert result["Fixed 50/50 Sleeves"]["date"].nunique() == 6
    assert result["Pure Inverse Volatility"]["date"].nunique() == 6

    for targets in result.values():
        gross_exposure = targets.groupby("date")["weight"].agg(
            lambda weights: weights.abs().sum()
        )

        assert np.allclose(gross_exposure, 2.0)


def test_frozen_strategy_workflow_rejects_duplicate_panel_keys():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ticker": ["AAA", "AAA"],
            "mom_12_1m_z": [1.0, 1.0],
            "realised_vol_63_z": [1.0, 1.0],
            "forward_ret_1d": [0.01, 0.01],
        }
    )

    with pytest.raises(ValueError, match="duplicate date-ticker"):
        build_frozen_strategy_target_weights(panel)
