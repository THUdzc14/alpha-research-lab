"""Small persistence and structural-validation helpers for tabular inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file without applying analytical transformations."""
    return pd.read_parquet(path)


def validate_ohlcv_panel(df: pd.DataFrame) -> None:
    """Validate the minimal OHLCV schema and its key price/volume constraints."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows found.")

    if (df["adj_close"] <= 0).any():
        raise ValueError("Non-positive adjusted prices found.")

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume found.")
