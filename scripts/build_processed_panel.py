from __future__ import annotations

import pandas as pd

from alpha_research.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.returns import add_return_features


def main() -> None:
    prices = load_parquet(RAW_DATA_DIR / "sp100_prices.parquet")

    panel = add_return_features(prices)

    universe = pd.read_csv(RAW_DATA_DIR / "sp100_universe.csv")

    ticker_col = "yf_ticker" if "yf_ticker" in universe.columns else "ticker"

    metadata = universe[[ticker_col, "ticker", "name", "sector"]].rename(
        columns={ticker_col: "price_ticker"}
    )

    panel = panel.merge(
        metadata,
        left_on="ticker",
        right_on="price_ticker",
        how="left",
        validate="many_to_one",
    )

    panel = panel.drop(columns=["price_ticker"])
    panel = panel.rename(columns={"ticker_x": "ticker", "ticker_y": "original_ticker"})

    output_path = PROCESSED_DATA_DIR / "equity_panel.parquet"
    save_parquet(panel, output_path)

    print(f"Saved processed panel: {output_path}")
    print(f"Rows: {len(panel):,}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Dates: {panel['date'].nunique()}")
    print(f"Period: {panel['date'].min()} to {panel['date'].max()}")


if __name__ == "__main__":
    main()
