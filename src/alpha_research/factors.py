from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the input panel.

    Required columns:
        date, ticker, adj_close

    If ret_1d is absent, it is calculated from adjusted close.
    """
    required = {"date", "ticker", "adj_close"}
    missing = required - set(panel.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows found.")

    if "ret_1d" not in df.columns:
        df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()

    return df


def momentum_12_1m(
    panel: pd.DataFrame,
    long_lag: int = 252,
    skip_lag: int = 21,
) -> pd.Series:
    """Calculate 12-1 month momentum.

    Formula at date t:

        adj_close[t - skip_lag] / adj_close[t - long_lag] - 1

    With the defaults, this measures the return from approximately
    12 months ago to 1 month ago, deliberately excluding the most
    recent month.
    """
    if long_lag <= skip_lag:
        raise ValueError("long_lag must be greater than skip_lag.")

    df = _prepare_panel(panel)
    prices = df.groupby("ticker")["adj_close"]

    start_price = prices.shift(long_lag)
    end_price = prices.shift(skip_lag)

    factor = end_price / start_price - 1.0
    factor.name = "mom_12_1m_raw"

    return factor


def momentum_3m(
    panel: pd.DataFrame,
    lookback: int = 63,
) -> pd.Series:
    """Calculate trailing 3-month momentum.

    Formula at date t:

        adj_close[t] / adj_close[t - lookback] - 1
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    df = _prepare_panel(panel)
    prices = df.groupby("ticker")["adj_close"]

    factor = df["adj_close"] / prices.shift(lookback) - 1.0
    factor.name = "mom_3m_raw"

    return factor


def reversal_1m(
    panel: pd.DataFrame,
    lookback: int = 21,
) -> pd.Series:
    """Calculate the 1-month reversal score.

    The raw one-month return is:

        adj_close[t] / adj_close[t - lookback] - 1

    The reversal score is its negative, so recent losers receive
    higher factor values:

        reversal = -one_month_return
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    df = _prepare_panel(panel)
    prices = df.groupby("ticker")["adj_close"]

    one_month_return = df["adj_close"] / prices.shift(lookback) - 1.0

    factor = -one_month_return
    factor.name = "reversal_1m_raw"

    return factor


def realised_volatility(
    panel: pd.DataFrame,
    window: int = 63,
    annualisation_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate annualised realised volatility from daily returns.

    Formula:

        rolling_std(ret_1d, window) * sqrt(annualisation_factor)

    Pandas uses the sample standard deviation by default, with ddof=1.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    if annualisation_factor <= 0:
        raise ValueError("annualisation_factor must be positive.")

    df = _prepare_panel(panel)

    rolling_vol = (
        df.groupby("ticker")["ret_1d"]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )

    factor = rolling_vol * np.sqrt(annualisation_factor)
    factor.name = f"realised_vol_{window}_raw"

    return factor


def add_raw_factors(
    panel: pd.DataFrame,
    momentum_12m_lag: int = 252,
    momentum_skip_lag: int = 21,
    momentum_3m_lookback: int = 63,
    reversal_lookback: int = 21,
    volatility_window: int = 63,
) -> pd.DataFrame:
    """Add the initial raw factor library to the processed equity panel.

    Added columns:
        mom_12_1m_raw
        mom_3m_raw
        reversal_1m_raw
        realised_vol_63_raw

    Missing values are intentionally preserved when insufficient history
    is available.
    """
    df = _prepare_panel(panel)

    df["mom_12_1m_raw"] = momentum_12_1m(
        df,
        long_lag=momentum_12m_lag,
        skip_lag=momentum_skip_lag,
    )

    df["mom_3m_raw"] = momentum_3m(
        df,
        lookback=momentum_3m_lookback,
    )

    df["reversal_1m_raw"] = reversal_1m(
        df,
        lookback=reversal_lookback,
    )

    volatility_column = f"realised_vol_{volatility_window}_raw"
    df[volatility_column] = realised_volatility(
        df,
        window=volatility_window,
    )

    return df
