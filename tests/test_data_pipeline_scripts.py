from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import build_factor_panel as factor_panel_script
from scripts import build_processed_panel as processed_panel_script
from scripts import download_data


def make_yfinance_output() -> pd.DataFrame:
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03"], name="Date")
    columns = pd.MultiIndex.from_product(
        [download_data.YFINANCE_FIELDS, ["AAA", "BBB"]],
        names=["Price", "Ticker"],
    )
    values = np.arange(1.0, 1.0 + len(dates) * len(columns)).reshape(
        len(dates),
        len(columns),
    )

    return pd.DataFrame(values, index=dates, columns=columns)


def test_normalise_yfinance_output_accepts_both_multi_index_orientations():
    field_first = make_yfinance_output()
    ticker_first = field_first.swaplevel(0, 1, axis="columns").sort_index(axis="columns")

    expected = download_data._normalise_yfinance_output(field_first, ["AAA", "BBB"])
    actual = download_data._normalise_yfinance_output(ticker_first, ["AAA", "BBB"])

    assert tuple(actual.columns) == download_data.RAW_PRICE_COLUMNS
    assert list(zip(actual["ticker"], actual["date"], strict=True)) == [
        ("AAA", date(2025, 1, 2)),
        ("AAA", date(2025, 1, 3)),
        ("BBB", date(2025, 1, 2)),
        ("BBB", date(2025, 1, 3)),
    ]
    pd.testing.assert_frame_equal(actual, expected)


def test_download_prices_uses_the_frozen_yfinance_request(monkeypatch):
    raw = make_yfinance_output()
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return raw

    monkeypatch.setattr(download_data.yf, "download", fake_download)

    result = download_data.download_prices(
        ["AAA", "BBB"],
        start="2025-01-01",
        end="2025-02-01",
    )

    assert captured == {
        "tickers": ["AAA", "BBB"],
        "start": "2025-01-01",
        "end": "2025-02-01",
        "auto_adjust": False,
        "group_by": "column",
        "progress": True,
        "threads": True,
    }
    assert tuple(result.columns) == download_data.RAW_PRICE_COLUMNS


def make_raw_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 2),
                date(2025, 1, 3),
            ],
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "open": [100.0, 110.0, 50.0, 45.0],
            "high": [101.0, 111.0, 51.0, 46.0],
            "low": [99.0, 109.0, 49.0, 44.0],
            "close": [100.0, 110.0, 50.0, 45.0],
            "adj_close": [100.0, 110.0, 50.0, 45.0],
            "volume": [2_000.0, 2_100.0, 1_000.0, 1_100.0],
        }
    )


def make_universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA.O", "BBB.O"],
            "yf_ticker": ["AAA", "BBB"],
            "name": ["Alpha", "Beta"],
            "sector": ["Technology", "Financials"],
        }
    )


def test_build_processed_panel_adds_returns_and_metadata_without_mutation():
    prices = make_raw_prices()
    universe = make_universe()
    original_prices = prices.copy(deep=True)
    original_universe = universe.copy(deep=True)

    result = processed_panel_script.build_processed_panel(prices, universe)

    assert result["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert result["original_ticker"].tolist() == ["AAA.O", "AAA.O", "BBB.O", "BBB.O"]
    assert result.groupby("ticker", sort=False).head(1)["ret_1d"].isna().all()
    assert result.loc[result["ticker"].eq("AAA"), "ret_1d"].iloc[1] == pytest.approx(0.10)
    assert result.loc[result["ticker"].eq("BBB"), "ret_1d"].iloc[1] == pytest.approx(-0.10)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(universe, original_universe)


def test_build_processed_panel_accepts_ticker_only_universe_metadata():
    universe = make_universe().drop(columns="yf_ticker").assign(ticker=["AAA", "BBB"])

    result = processed_panel_script.build_processed_panel(make_raw_prices(), universe)

    assert result["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert result["original_ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]


def test_build_factor_panel_preserves_the_published_feature_schema():
    equity_panel = processed_panel_script.build_processed_panel(
        make_raw_prices(),
        make_universe(),
    )
    benchmark = make_raw_prices().loc[lambda data: data["ticker"].eq("AAA")].copy()
    benchmark["ticker"] = "SPY"
    original_panel = equity_panel.copy(deep=True)
    original_benchmark = benchmark.copy(deep=True)

    result = factor_panel_script.build_factor_panel(equity_panel, benchmark)

    expected_columns = {
        *factor_panel_script.FACTOR_MAP,
        *(f"{prefix}_winsorised" for prefix in factor_panel_script.FACTOR_MAP.values()),
        *(f"{prefix}_z" for prefix in factor_panel_script.FACTOR_MAP.values()),
        *(f"{prefix}_rank" for prefix in factor_panel_script.FACTOR_MAP.values()),
        *(f"{prefix}_sector_neutral_z" for prefix in factor_panel_script.FACTOR_MAP.values()),
        factor_panel_script.STOCK_BETA_COLUMN,
    }
    assert expected_columns.issubset(result.columns)
    assert not any(column.startswith("market_model_") for column in result.columns)
    pd.testing.assert_frame_equal(equity_panel, original_panel)
    pd.testing.assert_frame_equal(benchmark, original_benchmark)
