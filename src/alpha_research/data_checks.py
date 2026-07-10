from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
REQUIRED_OHLCV_COLUMNS = ["date", "ticker", *PRICE_COLUMNS, "volume"]


@dataclass(frozen=True)
class DataQualityConfig:
    extreme_return_threshold: float = 0.25
    extreme_gap_threshold: float = 0.15
    min_history_days: int = 252
    min_coverage_ratio: float = 0.95


def prepare_ohlcv(prices: pd.DataFrame) -> pd.DataFrame:
    """Standardise date/ticker sorting and add basic return/gap fields."""
    df = prices.copy()

    missing = set(REQUIRED_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["open_to_prev_close"] = df["open"] / df["prev_close"] - 1.0

    return df


def basic_panel_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """High-level summary of the OHLCV panel."""
    df = prepare_ohlcv(prices)

    summary = {
        "rows": len(df),
        "tickers": df["ticker"].nunique(),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "trading_dates": df["date"].nunique(),
        "duplicate_date_ticker_rows": df[["date", "ticker"]].duplicated().sum(),
        "missing_adj_close": df["adj_close"].isna().sum(),
        "missing_volume": df["volume"].isna().sum(),
    }

    return pd.DataFrame([summary])


def non_positive_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Find rows with zero or negative OHLC/adjusted prices."""
    df = prepare_ohlcv(prices)

    mask = False
    for col in PRICE_COLUMNS:
        mask = mask | (df[col] <= 0)

    cols = ["date", "ticker", *PRICE_COLUMNS, "volume"]
    return df.loc[mask, cols].sort_values(["date", "ticker"])


def missing_dates_by_ticker(
    prices: pd.DataFrame,
    reference_dates: pd.Series | pd.Index | None = None,
) -> pd.DataFrame:
    """Count missing trading dates for each ticker.

    If reference_dates is provided, it should usually be benchmark dates, e.g. SPY dates.
    Otherwise, the union of dates in the price panel is used.
    """
    df = prepare_ohlcv(prices)

    if reference_dates is None:
        all_dates = pd.Index(sorted(df["date"].unique()))
    else:
        all_dates = pd.to_datetime(pd.Index(reference_dates)).sort_values().unique()

    out = []

    for ticker, group in df.groupby("ticker"):
        ticker_dates = pd.Index(group["date"].unique())
        missing = pd.Index(all_dates).difference(ticker_dates)

        out.append(
            {
                "ticker": ticker,
                "available_days": len(ticker_dates),
                "reference_days": len(all_dates),
                "missing_days": len(missing),
                "coverage_ratio": (
                    len(ticker_dates) / len(all_dates) if len(all_dates) else np.nan
                ),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
            }
        )

    return (
        pd.DataFrame(out)
        .sort_values(["coverage_ratio", "available_days"], ascending=[True, True])
        .reset_index(drop=True)
    )


def missing_tickers_by_date(prices: pd.DataFrame) -> pd.DataFrame:
    """Show how many tickers are available on each date."""
    df = prepare_ohlcv(prices)
    total_tickers = df["ticker"].nunique()

    out = (
        df.groupby("date")["ticker"].nunique().rename("available_tickers").reset_index()
    )
    out["total_tickers"] = total_tickers
    out["missing_tickers"] = out["total_tickers"] - out["available_tickers"]
    out["coverage_ratio"] = out["available_tickers"] / out["total_tickers"]

    return out.sort_values("date").reset_index(drop=True)


def available_history_by_ticker(prices: pd.DataFrame) -> pd.DataFrame:
    """Summarise first/last date and observation count by ticker."""
    df = prepare_ohlcv(prices)

    return (
        df.groupby("ticker")
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            observations=("date", "count"),
            missing_adj_close=("adj_close", lambda x: x.isna().sum()),
            missing_volume=("volume", lambda x: x.isna().sum()),
        )
        .reset_index()
        .sort_values(["observations", "ticker"])
        .reset_index(drop=True)
    )


def extreme_returns(
    prices: pd.DataFrame,
    threshold: float = 0.25,
) -> pd.DataFrame:
    """Find large close-to-close adjusted returns."""
    df = prepare_ohlcv(prices)

    out = df.loc[df["ret_1d"].abs() >= threshold].copy()
    cols = ["date", "ticker", "adj_close", "ret_1d", "volume"]

    return out[cols].sort_values("ret_1d", key=lambda x: x.abs(), ascending=False)


def price_gaps(
    prices: pd.DataFrame,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Find large open-to-previous-close gaps."""
    df = prepare_ohlcv(prices)

    out = df.loc[df["open_to_prev_close"].abs() >= threshold].copy()
    cols = ["date", "ticker", "open", "prev_close", "open_to_prev_close", "volume"]

    return out[cols].sort_values(
        "open_to_prev_close",
        key=lambda x: x.abs(),
        ascending=False,
    )


def sector_coverage(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Summarise available tickers by sector."""
    df = prepare_ohlcv(prices)

    available = set(df["ticker"].unique())

    uni = universe.copy()
    ticker_col = "yf_ticker" if "yf_ticker" in uni.columns else "ticker"

    uni["in_price_panel"] = uni[ticker_col].isin(available)

    return (
        uni.groupby("sector")
        .agg(
            universe_tickers=(ticker_col, "count"),
            available_tickers=("in_price_panel", "sum"),
        )
        .reset_index()
        .assign(
            missing_tickers=lambda x: x["universe_tickers"] - x["available_tickers"],
            coverage_ratio=lambda x: x["available_tickers"] / x["universe_tickers"],
        )
        .sort_values("coverage_ratio")
        .reset_index(drop=True)
    )


def benchmark_alignment(prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compare price-panel dates with benchmark dates."""
    prices_df = prepare_ohlcv(prices)
    bench_df = prepare_ohlcv(benchmark)

    price_dates = pd.Index(prices_df["date"].unique()).sort_values()
    benchmark_dates = pd.Index(bench_df["date"].unique()).sort_values()

    common_dates = price_dates.intersection(benchmark_dates)
    price_only = price_dates.difference(benchmark_dates)
    benchmark_only = benchmark_dates.difference(price_dates)

    summary = {
        "price_panel_start": price_dates.min(),
        "price_panel_end": price_dates.max(),
        "benchmark_start": benchmark_dates.min(),
        "benchmark_end": benchmark_dates.max(),
        "price_panel_dates": len(price_dates),
        "benchmark_dates": len(benchmark_dates),
        "common_dates": len(common_dates),
        "price_only_dates": len(price_only),
        "benchmark_only_dates": len(benchmark_only),
        "alignment_ratio_vs_benchmark": (
            len(common_dates) / len(benchmark_dates) if len(benchmark_dates) else np.nan
        ),
    }

    return pd.DataFrame([summary])


def data_quality_report(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    universe: pd.DataFrame,
    config: DataQualityConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the standard Week 1 data-quality checks."""
    if config is None:
        config = DataQualityConfig()

    benchmark_dates = pd.to_datetime(benchmark["date"])

    return {
        "basic_summary": basic_panel_summary(prices),
        "non_positive_prices": non_positive_prices(prices),
        "missing_dates_by_ticker": missing_dates_by_ticker(prices, benchmark_dates),
        "missing_tickers_by_date": missing_tickers_by_date(prices),
        "available_history_by_ticker": available_history_by_ticker(prices),
        "extreme_returns": extreme_returns(
            prices,
            threshold=config.extreme_return_threshold,
        ),
        "price_gaps": price_gaps(
            prices,
            threshold=config.extreme_gap_threshold,
        ),
        "sector_coverage": sector_coverage(prices, universe),
        "benchmark_alignment": benchmark_alignment(prices, benchmark),
    }
