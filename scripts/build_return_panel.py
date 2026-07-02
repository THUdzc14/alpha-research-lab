from __future__ import annotations

from alpha_research.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.factors import adjusted_returns


def main() -> None:
    prices = load_parquet(RAW_DATA_DIR / "sp100_prices.parquet")
    returns = adjusted_returns(prices)
    save_parquet(returns, PROCESSED_DATA_DIR / "sp100_returns.parquet")
    print(f"Saved return panel with {len(returns):,} rows.")


if __name__ == "__main__":
    main()
