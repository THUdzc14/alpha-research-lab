import numpy as np
import pandas as pd
import pytest

from alpha_research.costs import (
    apply_linear_transaction_costs,
    calculate_daily_trade_capacity,
    calculate_linear_transaction_cost,
    calculate_security_trade_capacity,
    calculate_top_n_turnover_share,
    prepare_lagged_dollar_volume,
    summarise_capacity_across_offsets,
    summarise_capacity_by_offset,
    summarise_turnover,
)


def test_calculate_linear_transaction_cost_scalar():
    result = calculate_linear_transaction_cost(
        turnover=2.0,
        transaction_cost_bps=10.0,
    )

    assert result == pytest.approx(0.002)


def test_calculate_linear_transaction_cost_series_preserves_index():
    turnover = pd.Series(
        [1.0, 0.0, 0.5],
        index=pd.date_range("2025-01-01", periods=3),
    )

    result = calculate_linear_transaction_cost(
        turnover,
        transaction_cost_bps=20.0,
    )

    expected = pd.Series(
        [0.002, 0.0, 0.001],
        index=turnover.index,
        name="transaction_cost",
    )

    pd.testing.assert_series_equal(result, expected)


def test_calculate_linear_transaction_cost_rejects_invalid_values():
    with pytest.raises(ValueError, match="turnover"):
        calculate_linear_transaction_cost(-0.1, 10.0)

    with pytest.raises(ValueError, match="transaction_cost_bps"):
        calculate_linear_transaction_cost(1.0, -10.0)


def test_apply_linear_transaction_costs_returns_copy():
    data = pd.DataFrame(
        {
            "gross_return": [0.01, -0.02],
            "turnover": [1.0, 0.5],
        }
    )

    result = apply_linear_transaction_costs(
        data,
        transaction_cost_bps=10.0,
    )

    assert "transaction_cost" not in data.columns
    assert result["transaction_cost"].tolist() == pytest.approx([0.001, 0.0005])
    assert result["net_return"].tolist() == pytest.approx([0.009, -0.0205])


def test_top_n_turnover_share():
    turnover = pd.Series([0.0, 1.0, 2.0, 3.0])

    assert calculate_top_n_turnover_share(turnover, 1) == pytest.approx(0.5)
    assert calculate_top_n_turnover_share(turnover, 2) == pytest.approx(5.0 / 6.0)
    assert np.isnan(calculate_top_n_turnover_share(pd.Series([0.0, 0.0]), 1))


def test_summarise_turnover_matches_notebook_06_formulas():
    turnover = pd.Series([0.0, 1.0, 0.0, 2.0])

    result = summarise_turnover(
        turnover,
        periods_per_year=12,
        concentration_window=2,
    )

    active = pd.Series([1.0, 2.0])

    assert result["observations"] == 4
    assert result["trading_days"] == 2
    assert result["trading_day_fraction"] == pytest.approx(0.5)
    assert result["mean_daily_turnover"] == pytest.approx(0.75)
    assert result["annualised_turnover"] == pytest.approx(9.0)
    assert result["mean_rebalance_turnover"] == pytest.approx(active.mean())
    assert result["median_rebalance_turnover"] == pytest.approx(active.median())
    assert result["p95_rebalance_turnover"] == pytest.approx(active.quantile(0.95))
    assert result["maximum_daily_turnover"] == pytest.approx(2.0)
    assert result["maximum_2_day_turnover"] == pytest.approx(2.0)
    assert result["top_1_day_turnover_share"] == pytest.approx(2.0 / 3.0)
    assert result["top_5_day_turnover_share"] == pytest.approx(1.0)


def make_market_data() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=4)

    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "close": [1.0] * 8,
            "volume": [10.0, 20.0, 30.0, 400.0, 100.0, 200.0, 300.0, 4000.0],
        }
    )


def test_prepare_lagged_dollar_volume_uses_only_prior_observations():
    result = prepare_lagged_dollar_volume(
        make_market_data(),
        window=2,
        min_periods=2,
        aggregation="mean",
        output_column="lagged_adv_2",
    )

    aaa = result.loc[result["ticker"].eq("AAA")]
    bbb = result.loc[result["ticker"].eq("BBB")]

    assert aaa["lagged_adv_2"].tolist()[:2] == pytest.approx(
        [np.nan, np.nan],
        nan_ok=True,
    )
    assert aaa["lagged_adv_2"].tolist()[2:] == pytest.approx([15.0, 25.0])
    assert bbb["lagged_adv_2"].tolist()[2:] == pytest.approx([150.0, 250.0])


def test_prepare_lagged_median_dollar_volume():
    market_data = make_market_data()
    market_data["dollar_volume"] = market_data["close"] * market_data["volume"]

    result = prepare_lagged_dollar_volume(
        market_data,
        window=3,
        min_periods=2,
        aggregation="median",
        output_column="lagged_median_3",
    )

    aaa = result.loc[result["ticker"].eq("AAA")]

    assert aaa["lagged_median_3"].iloc[2] == pytest.approx(15.0)
    assert aaa["lagged_median_3"].iloc[3] == pytest.approx(20.0)


def test_prepare_lagged_dollar_volume_rejects_duplicate_keys():
    market_data = make_market_data()
    duplicated = pd.concat([market_data, market_data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate date/ticker"):
        prepare_lagged_dollar_volume(duplicated)


def test_calculate_security_trade_capacity():
    trades = pd.DataFrame(
        {
            "absolute_trade_weight": [0.20, 0.10, 0.0],
            "lagged_adv_21": [10_000_000.0, 4_000_000.0, 5_000_000.0],
        }
    )

    result = calculate_security_trade_capacity(
        trades,
        participation_rate=0.01,
    )

    assert result["trade_capacity_usd"].iloc[0] == pytest.approx(500_000.0)
    assert result["trade_capacity_usd"].iloc[1] == pytest.approx(400_000.0)
    assert np.isnan(result["trade_capacity_usd"].iloc[2])


def make_trade_data() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=2)

    return pd.DataFrame(
        {
            "portfolio": ["Test"] * 4,
            "rebalance_frequency": [5] * 4,
            "rebalance_offset": [0] * 4,
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "absolute_trade_weight": [0.20, 0.10, 0.10, 0.10],
            "lagged_adv_21": [10_000_000.0, 4_000_000.0, 8_000_000.0, 12_000_000.0],
        }
    )


def test_calculate_daily_trade_capacity_matches_notebook_06():
    result = calculate_daily_trade_capacity(
        make_trade_data(),
        participation_limits=(0.01, 0.05),
    )

    first_day = result.loc[
        result["date"].eq(result["date"].min()) & result["participation_limit"].eq(0.01)
    ].iloc[0]

    assert len(result) == 4
    assert first_day["traded_security_count"] == 2
    assert bool(first_day["fully_covered"])
    assert first_day["largest_security_trade_share"] == pytest.approx(2.0 / 3.0)
    assert first_day["top_5_security_trade_share"] == pytest.approx(1.0)
    assert first_day["effective_traded_security_count"] == pytest.approx(1.8)
    assert first_day["capacity_usd"] == pytest.approx(400_000.0)
    assert first_day["bottleneck_ticker"] == "BBB"


def test_daily_trade_capacity_requires_full_liquidity_coverage():
    trades = make_trade_data()
    trades.loc[0, "lagged_adv_21"] = np.nan

    result = calculate_daily_trade_capacity(
        trades,
        participation_limits=(0.01,),
    )

    first_day = result.loc[result["date"].eq(result["date"].min())].iloc[0]

    assert not bool(first_day["fully_covered"])
    assert np.isnan(first_day["capacity_usd"])
    assert pd.isna(first_day["bottleneck_ticker"])


def test_capacity_summaries_match_notebook_06_aggregations():
    capacity_daily = calculate_daily_trade_capacity(
        make_trade_data(),
        participation_limits=(0.01,),
    )

    by_offset = summarise_capacity_by_offset(capacity_daily)
    row = by_offset.iloc[0]

    assert row["rebalance_days"] == 2
    assert row["fully_covered_fraction"] == pytest.approx(1.0)
    assert row["minimum_capacity_usd"] == pytest.approx(400_000.0)
    assert row["fifth_percentile_capacity_usd"] == pytest.approx(420_000.0)
    assert row["tenth_percentile_capacity_usd"] == pytest.approx(440_000.0)
    assert row["median_capacity_usd"] == pytest.approx(600_000.0)

    second_offset = by_offset.copy()
    second_offset["rebalance_offset"] = 1
    second_offset["fully_covered_fraction"] = 0.9
    second_offset["minimum_capacity_usd"] = 300_000.0
    second_offset["fifth_percentile_capacity_usd"] = 350_000.0
    second_offset["median_capacity_usd"] = 500_000.0

    phase = summarise_capacity_across_offsets(
        pd.concat([by_offset, second_offset], ignore_index=True)
    ).iloc[0]

    assert phase["offset_count"] == 2
    assert phase["minimum_adv_coverage"] == pytest.approx(0.9)
    assert phase["median_fifth_percentile_capacity_usd"] == pytest.approx(385_000.0)
    assert phase["worst_phase_fifth_percentile_capacity_usd"] == pytest.approx(
        350_000.0
    )
    assert phase["median_capacity_usd"] == pytest.approx(550_000.0)
    assert phase["worst_historical_capacity_usd"] == pytest.approx(300_000.0)
