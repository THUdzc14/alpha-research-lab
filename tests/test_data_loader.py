import pandas as pd
import pytest

from alpha_research.data_loader import validate_ohlcv_panel


def test_validate_ohlcv_panel_accepts_minimal_valid_panel():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "adj_close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
        }
    )

    validate_ohlcv_panel(df)


def test_validate_ohlcv_panel_rejects_duplicate_date_ticker():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "adj_close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        validate_ohlcv_panel(df)
