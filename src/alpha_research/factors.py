from __future__ import annotations

import pandas as pd


def adjusted_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate close-to-close adjusted returns by ticker."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    return df[["date", "ticker", "ret_1d"]]


def momentum_12_1m(returns: pd.DataFrame, trading_days_per_month: int = 21) -> pd.DataFrame:
    """12-1 month momentum using daily returns.

    Uses cumulative return from approximately t-252 to t-21.
    This function produces same-date factor values. You must shift signals later
    before applying future returns in validation/backtesting.
    """
    df = returns.sort_values(["ticker", "date"]).copy()
    short_skip = trading_days_per_month
    long_window = 12 * trading_days_per_month

    gross = 1.0 + df["ret_1d"]
    df["mom_12_1m"] = (
        gross.groupby(df["ticker"])
        .rolling(long_window, min_periods=int(0.8 * long_window))
        .apply(lambda x: x[:-short_skip].prod() - 1.0, raw=False)
        .reset_index(level=0, drop=True)
    )
    return df[["date", "ticker", "mom_12_1m"]]
