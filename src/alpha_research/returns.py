from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create backward- and forward-looking return fields.

    Backward-looking returns may be used as factor inputs.
    Forward returns are labels used only for signal validation.

    No signal shifting is performed here.
    """
    df = prices.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped_prices = df.groupby("ticker")["adj_close"]

    df["ret_1d"] = grouped_prices.pct_change(1)
    df["ret_5d"] = grouped_prices.pct_change(5)
    df["ret_21d"] = grouped_prices.pct_change(21)

    df["forward_ret_1d"] = grouped_prices.shift(-1) / df["adj_close"] - 1.0
    df["forward_ret_5d"] = grouped_prices.shift(-5) / df["adj_close"] - 1.0

    df["dollar_volume"] = df["close"] * df["volume"]
    df["log_dollar_volume"] = np.log1p(df["dollar_volume"])

    return df
