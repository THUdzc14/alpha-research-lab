from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from alpha_research.config.paths import RAW_DATA_DIR
from alpha_research.data_loader import save_parquet, validate_ohlcv_panel
from alpha_research.universe import get_universe


def _normalise_yfinance_output(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convert yfinance's wide multi-index output into a tidy OHLCV panel."""
    if data.empty:
        raise ValueError("No data returned by yfinance.")

    # yfinance can return either [field, ticker] or [ticker, field] multi-index columns.
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(map(str, data.columns.get_level_values(0)))
        expected_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if expected_fields.intersection(level0):
            data = data.stack(level=1, future_stack=True).reset_index()
            ticker_col = "Ticker"
        else:
            data = data.stack(level=0, future_stack=True).reset_index()
            ticker_col = "Ticker"
    else:
        # Single ticker case.
        data = data.reset_index()
        data["Ticker"] = tickers[0]
        ticker_col = "Ticker"

    rename_map = {
        "Date": "date",
        ticker_col: "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename_map)

    required = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns after normalisation: {sorted(missing)}")

    data = data[required].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    data["ticker"] = data["ticker"].astype(str)
    data = data.dropna(subset=["adj_close"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    return data


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=True,
        threads=True,
    )
    return _normalise_yfinance_output(raw, tickers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="sp100", choices=["sp100"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe = get_universe(args.universe)
    tickers = universe["yf_ticker"].tolist()

    universe_path = RAW_DATA_DIR / f"{args.universe}_universe.csv"
    universe.to_csv(universe_path, index=False)

    prices = download_prices(tickers, start=args.start, end=args.end)
    validate_ohlcv_panel(prices)

    prices_path = RAW_DATA_DIR / f"{args.universe}_prices.parquet"
    save_parquet(prices, prices_path)

    benchmark = download_prices(["SPY"], start=args.start, end=args.end)
    benchmark_path = RAW_DATA_DIR / "spy_benchmark.parquet"
    save_parquet(benchmark, benchmark_path)

    print(f"Saved universe:  {universe_path}")
    print(f"Saved prices:    {prices_path}")
    print(f"Saved benchmark: {benchmark_path}")
    print(f"Rows: {len(prices):,}; tickers: {prices['ticker'].nunique()}")


if __name__ == "__main__":
    main()
