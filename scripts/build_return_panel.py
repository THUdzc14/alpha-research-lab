"""Build the historical one-day return panel from raw market prices."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from alpha_research.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.returns import add_return_features

RETURN_PANEL_COLUMNS = ("date", "ticker", "ret_1d")


def build_return_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen date/ticker/one-day-return artifact schema.

    Return calculation is delegated to the package implementation.  Dates are
    converted back to date-only values to preserve the established Parquet
    representation produced from the raw Yahoo Finance panel.
    """
    return_panel = add_return_features(prices).loc[:, RETURN_PANEL_COLUMNS].copy()
    return_panel["date"] = return_panel["date"].dt.date

    return return_panel


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(() if argv is None else argv)

    prices = load_parquet(RAW_DATA_DIR / "sp100_prices.parquet")
    return_panel = build_return_panel(prices)
    save_parquet(return_panel, PROCESSED_DATA_DIR / "sp100_returns.parquet")
    print(f"Saved return panel with {len(return_panel):,} rows.")


if __name__ == "__main__":
    main(sys.argv[1:])
