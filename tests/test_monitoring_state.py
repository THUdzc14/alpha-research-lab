import pandas as pd
import pytest

from alpha_research.monitoring import (
    calculate_implementation_monitoring_state,
    calculate_performance_risk_state,
)


def make_implementation_inputs():
    dates = pd.bdate_range(
        "2025-01-01",
        periods=4,
    )

    absolute_trades = [
        0.0,
        0.1,
        0.0,
        0.2,
    ]

    security_daily = pd.DataFrame(
        {
            "portfolio": "Strategy",
            "date": dates,
            "ticker": "AAA",
            "trade": absolute_trades,
            "absolute_trade_weight": absolute_trades,
            "transaction_cost_contribution": [
                value * 0.001 for value in absolute_trades
            ],
            "return_record_missing": False,
            "missing_return_weight_contribution": 0.0,
        }
    )

    market_data = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "dollar_volume": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
        }
    )

    return security_daily, market_data


def test_performance_risk_state_includes_portfolio_and_benchmark():
    dates = pd.bdate_range(
        "2025-01-01",
        periods=4,
    )

    portfolio_daily = pd.DataFrame(
        {
            "date": dates,
            "portfolio": "Strategy",
            "net_return": [
                0.01,
                -0.02,
                0.03,
                0.01,
            ],
        }
    )

    benchmark_daily = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": [
                0.005,
                -0.01,
                0.02,
                0.005,
            ],
        }
    )

    result = calculate_performance_risk_state(
        portfolio_daily,
        benchmark_daily,
        performance_window=2,
        risk_window=2,
    )

    assert len(result) == 8
    assert set(result["portfolio"]) == {
        "Strategy",
        "SPY",
    }

    strategy = result.loc[result["portfolio"].eq("Strategy")]

    assert pd.isna(strategy["trailing_return_2"].iloc[0])

    assert strategy["trailing_return_2"].iloc[1] == pytest.approx(1.01 * 0.98 - 1.0)

    assert strategy["drawdown"].iloc[1] < 0.0


def test_implementation_state_uses_lagged_liquidity():
    security_daily, market_data = make_implementation_inputs()

    state, coverage = calculate_implementation_monitoring_state(
        security_daily,
        market_data,
        implementation_window=2,
        liquidity_window=2,
        liquidity_min_periods=1,
        participation_rate=0.01,
    )

    assert coverage["liquidity_coverage"].eq(1.0).all()

    assert state["bottleneck_capacity_1pct_usd"].iloc[1] == pytest.approx(10.0)

    assert state["bottleneck_capacity_1pct_usd"].iloc[3] == pytest.approx(12.5)

    assert state["annualised_turnover_2"].iloc[3] == pytest.approx(25.2)

    assert state["minimum_trade_capacity_1pct_usd_2"].iloc[3] == pytest.approx(12.5)


def test_implementation_state_reports_partial_liquidity_coverage():
    dates = pd.bdate_range(
        "2025-01-01",
        periods=3,
    )

    rows = []

    for date_number, date in enumerate(dates):
        for ticker in [
            "AAA",
            "BBB",
        ]:
            trade = 0.1 if date_number == 1 else 0.0

            rows.append(
                {
                    "portfolio": "Strategy",
                    "date": date,
                    "ticker": ticker,
                    "trade": trade,
                    "absolute_trade_weight": abs(trade),
                    "transaction_cost_contribution": (abs(trade) * 0.001),
                    "return_record_missing": False,
                    "missing_return_weight_contribution": 0.0,
                }
            )

    security_daily = pd.DataFrame(rows)

    # BBB deliberately has no liquidity history.
    market_data = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "dollar_volume": [
                100.0,
                200.0,
                300.0,
            ],
        }
    )

    _, coverage = calculate_implementation_monitoring_state(
        security_daily,
        market_data,
        implementation_window=2,
        liquidity_window=2,
        liquidity_min_periods=1,
    )

    trade_date = coverage.loc[coverage["date"].eq(dates[1])].iloc[0]

    assert trade_date["liquidity_coverage"] == pytest.approx(0.5)


def test_implementation_state_rejects_trade_weight_mismatch():
    security_daily, market_data = make_implementation_inputs()

    security_daily.loc[
        0,
        "absolute_trade_weight",
    ] = 0.1

    with pytest.raises(
        ValueError,
        match="does not reconcile",
    ):
        calculate_implementation_monitoring_state(
            security_daily,
            market_data,
            implementation_window=2,
            liquidity_window=2,
            liquidity_min_periods=1,
        )
