"""Build the metadata-enriched equity panel from raw market data."""

from __future__ import annotations

import pandas as pd

from alpha_research.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.returns import add_return_features


def build_processed_panel(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Add return features and merge the frozen universe metadata."""
    panel = add_return_features(prices)

    if "yf_ticker" in universe.columns:
        metadata = universe[["yf_ticker", "ticker", "name", "sector"]].rename(
            columns={"yf_ticker": "price_ticker"}
        )
    else:
        metadata = universe[["ticker", "name", "sector"]].copy()
        metadata.insert(0, "price_ticker", metadata["ticker"])

    panel = panel.merge(
        metadata,
        left_on="ticker",
        right_on="price_ticker",
        how="left",
        validate="many_to_one",
    )

    panel = panel.drop(columns=["price_ticker"])
    panel = panel.rename(columns={"ticker_x": "ticker", "ticker_y": "original_ticker"})

    return panel


def main() -> None:
    prices = load_parquet(RAW_DATA_DIR / "sp100_prices.parquet")
    universe = pd.read_csv(RAW_DATA_DIR / "sp100_universe.csv")
    panel = build_processed_panel(prices, universe)

    output_path = PROCESSED_DATA_DIR / "equity_panel.parquet"
    save_parquet(panel, output_path)

    print(f"Saved processed panel: {output_path}")
    print(f"Rows: {len(panel):,}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Dates: {panel['date'].nunique()}")
    print(f"Period: {panel['date'].min()} to {panel['date'].max()}")


if __name__ == "__main__":
    main()
