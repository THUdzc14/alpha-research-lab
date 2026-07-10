import pandas as pd

from alpha_research.data_checks import (
    basic_panel_summary,
    benchmark_alignment,
    extreme_returns,
    missing_dates_by_ticker,
    non_positive_prices,
    prepare_ohlcv,
)


def _sample_prices():
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "open": [100.0, 101.0, 50.0, 55.0],
            "high": [101.0, 102.0, 51.0, 56.0],
            "low": [99.0, 100.0, 49.0, 54.0],
            "close": [100.0, 101.0, 50.0, 55.0],
            "adj_close": [100.0, 101.0, 50.0, 55.0],
            "volume": [1000, 1100, 2000, 2100],
        }
    )


def test_prepare_ohlcv_adds_returns():
    df = prepare_ohlcv(_sample_prices())

    assert "ret_1d" in df.columns
    assert "open_to_prev_close" in df.columns


def test_basic_panel_summary():
    summary = basic_panel_summary(_sample_prices())

    assert summary.loc[0, "rows"] == 4
    assert summary.loc[0, "tickers"] == 2


def test_non_positive_prices_empty_for_valid_data():
    result = non_positive_prices(_sample_prices())

    assert result.empty


def test_extreme_returns_finds_large_move():
    result = extreme_returns(_sample_prices(), threshold=0.05)

    assert not result.empty
    assert "BBB" in set(result["ticker"])


def test_missing_dates_by_ticker_accepts_reference_dates():
    reference_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    result = missing_dates_by_ticker(_sample_prices(), reference_dates)

    assert result["missing_days"].max() == 1


def test_benchmark_alignment():
    prices = _sample_prices()
    benchmark = prices.query("ticker == 'AAA'").copy()

    result = benchmark_alignment(prices, benchmark)

    assert result.loc[0, "common_dates"] == 2
