from datetime import date

import pandas as pd
import pytest

from scripts import build_return_panel as return_panel_script


def make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2025-01-03",
                "2025-01-02",
                "2025-01-03",
                "2025-01-02",
            ],
            "ticker": ["BBB", "AAA", "AAA", "BBB"],
            "adj_close": [45.0, 100.0, 110.0, 50.0],
            "close": [45.0, 100.0, 110.0, 50.0],
            "volume": [1_000.0, 2_000.0, 2_100.0, 1_100.0],
        }
    )


def test_build_return_panel_preserves_schema_sorting_and_ticker_boundaries():
    prices = make_prices()
    original = prices.copy(deep=True)

    result = return_panel_script.build_return_panel(prices)

    assert tuple(result.columns) == return_panel_script.RETURN_PANEL_COLUMNS
    assert list(zip(result["ticker"], result["date"], strict=True)) == [
        ("AAA", date(2025, 1, 2)),
        ("AAA", date(2025, 1, 3)),
        ("BBB", date(2025, 1, 2)),
        ("BBB", date(2025, 1, 3)),
    ]
    first_ticker_rows = result.groupby("ticker", sort=False).head(1)
    assert first_ticker_rows["ret_1d"].isna().all()
    assert result.loc[result["ticker"].eq("AAA"), "ret_1d"].iloc[1] == pytest.approx(0.10)
    assert result.loc[result["ticker"].eq("BBB"), "ret_1d"].iloc[1] == pytest.approx(-0.10)
    pd.testing.assert_frame_equal(prices, original)


def test_main_writes_the_frozen_return_panel_path(tmp_path, monkeypatch, capsys):
    raw_directory = tmp_path / "raw"
    processed_directory = tmp_path / "processed"
    raw_directory.mkdir()
    make_prices().to_parquet(raw_directory / "sp100_prices.parquet", index=False)

    monkeypatch.setattr(return_panel_script, "RAW_DATA_DIR", raw_directory)
    monkeypatch.setattr(return_panel_script, "PROCESSED_DATA_DIR", processed_directory)

    return_panel_script.main()

    output_path = processed_directory / "sp100_returns.parquet"
    expected = return_panel_script.build_return_panel(make_prices())
    actual = pd.read_parquet(output_path)

    assert output_path.exists()
    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == "Saved return panel with 4 rows.\n"
