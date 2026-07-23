from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def prepare_benchmark_returns(
    benchmark: pd.DataFrame,
    date_column: str = "date",
    price_column: str = "adj_close",
    output_column: str = "benchmark_return",
) -> pd.DataFrame:
    """Create daily benchmark returns from an adjusted-price series."""
    required = {date_column, price_column}
    missing = required - set(benchmark.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = benchmark[[date_column, price_column]].copy()
    df[date_column] = pd.to_datetime(df[date_column])

    df = (
        df.sort_values(date_column)
        .drop_duplicates(subset=[date_column])
        .reset_index(drop=True)
    )

    df[output_column] = df[price_column].shift(-1) / df[price_column] - 1.0

    return df[[date_column, output_column]]


def calculate_market_exposure(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    strategy_return_column: str,
    benchmark_return_column: str = "benchmark_return",
    date_column: str = "date",
) -> pd.DataFrame:
    """Estimate alpha, beta and correlation against a benchmark.

    Regression:

        strategy_return = alpha_daily + beta * benchmark_return + residual
    """
    strategy_required = {date_column, strategy_return_column}
    benchmark_required = {date_column, benchmark_return_column}

    strategy_missing = strategy_required - set(strategy_returns.columns)
    benchmark_missing = benchmark_required - set(benchmark_returns.columns)

    if strategy_missing:
        raise ValueError(f"Missing strategy columns: {sorted(strategy_missing)}")

    if benchmark_missing:
        raise ValueError(f"Missing benchmark columns: {sorted(benchmark_missing)}")

    merged = strategy_returns[[date_column, strategy_return_column]].merge(
        benchmark_returns[[date_column, benchmark_return_column]],
        on=date_column,
        how="inner",
    )

    merged = merged.dropna()

    if len(merged) < 2:
        raise ValueError("Insufficient aligned observations.")

    strategy = merged[strategy_return_column]
    benchmark = merged[benchmark_return_column]

    benchmark_variance = benchmark.var(ddof=1)

    if benchmark_variance <= 0:
        beta = np.nan
        alpha_daily = np.nan
    else:
        beta = strategy.cov(benchmark) / benchmark_variance
        alpha_daily = strategy.mean() - beta * benchmark.mean()

    residual = strategy - (alpha_daily + beta * benchmark)

    annualised_alpha = (
        (1.0 + alpha_daily) ** TRADING_DAYS_PER_YEAR - 1.0
        if pd.notna(alpha_daily) and alpha_daily > -1
        else np.nan
    )

    residual_volatility = residual.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return pd.DataFrame(
        [
            {
                "observations": len(merged),
                "beta": beta,
                "daily_alpha": alpha_daily,
                "annualised_alpha": annualised_alpha,
                "correlation": strategy.corr(benchmark),
                "annualised_residual_volatility": residual_volatility,
                "r_squared": strategy.corr(benchmark) ** 2,
            }
        ]
    )


def calculate_strategy_exposures(
    daily_results: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate benchmark exposure for each strategy return stream."""
    return_columns = {
        "long_leg": "long_return",
        "short_leg": "short_return",
        "gross_long_short": "gross_return",
        "net_long_short": "net_return",
    }

    rows = []

    for portfolio_name, return_column in return_columns.items():
        if return_column not in daily_results.columns:
            continue

        result = (
            calculate_market_exposure(
                strategy_returns=daily_results,
                benchmark_returns=benchmark_returns,
                strategy_return_column=return_column,
            )
            .iloc[0]
            .to_dict()
        )

        result["portfolio"] = portfolio_name
        rows.append(result)

    return pd.DataFrame(rows)


def calculate_rolling_beta(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    strategy_return_column: str,
    window: int = 126,
    min_periods: int | None = None,
    benchmark_return_column: str = "benchmark_return",
) -> pd.DataFrame:
    """Calculate rolling realised beta against the benchmark."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    if min_periods is None:
        min_periods = window // 2

    merged = strategy_returns[["date", strategy_return_column]].merge(
        benchmark_returns[["date", benchmark_return_column]],
        on="date",
        how="inner",
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    rolling_covariance = (
        merged[strategy_return_column]
        .rolling(window, min_periods=min_periods)
        .cov(merged[benchmark_return_column])
    )

    rolling_variance = (
        merged[benchmark_return_column].rolling(window, min_periods=min_periods).var()
    )

    merged[f"rolling_beta_{window}"] = rolling_covariance / rolling_variance

    return merged


def calculate_sector_exposure(
    holdings: pd.DataFrame,
    metadata: pd.DataFrame,
    rebalance_dates: pd.Series | pd.Index | None = None,
) -> pd.DataFrame:
    """Calculate long, short and net portfolio weights by date and sector."""
    holdings_required = {"date", "ticker", "weight"}
    metadata_required = {"ticker", "sector"}

    holdings_missing = holdings_required - set(holdings.columns)
    metadata_missing = metadata_required - set(metadata.columns)

    if holdings_missing:
        raise ValueError(f"Missing holdings columns: {sorted(holdings_missing)}")

    if metadata_missing:
        raise ValueError(f"Missing metadata columns: {sorted(metadata_missing)}")

    df = holdings.copy()
    df["date"] = pd.to_datetime(df["date"])

    if rebalance_dates is not None:
        selected_dates = pd.DatetimeIndex(
            pd.to_datetime(pd.Series(rebalance_dates).dropna().unique())
        )
        df = df.loc[df["date"].isin(selected_dates)]

    metadata_clean = (
        metadata[["ticker", "sector"]].drop_duplicates(subset=["ticker"]).copy()
    )

    df = df.merge(
        metadata_clean,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    df["sector"] = df["sector"].fillna("Unknown")
    df["long_weight"] = df["weight"].clip(lower=0.0)
    df["short_weight"] = -df["weight"].clip(upper=0.0)

    exposure = (
        df.groupby(["date", "sector"])
        .agg(
            long_weight=("long_weight", "sum"),
            short_weight=("short_weight", "sum"),
            net_weight=("weight", "sum"),
        )
        .reset_index()
    )

    exposure["gross_weight"] = exposure["long_weight"] + exposure["short_weight"]

    return exposure


def summarise_sector_exposure(
    sector_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise average and maximum sector exposures through time."""
    required = {
        "sector",
        "long_weight",
        "short_weight",
        "net_weight",
        "gross_weight",
    }

    missing = required - set(sector_exposure.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        sector_exposure.groupby("sector")
        .agg(
            average_long_weight=("long_weight", "mean"),
            average_short_weight=("short_weight", "mean"),
            average_net_weight=("net_weight", "mean"),
            average_gross_weight=("gross_weight", "mean"),
            maximum_absolute_net_weight=(
                "net_weight",
                lambda x: x.abs().max(),
            ),
        )
        .sort_values(
            "average_gross_weight",
            ascending=False,
        )
        .reset_index()
    )


def calculate_rolling_stock_beta(
    equity_panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    stock_return_column: str = "ret_1d",
    benchmark_price_column: str = "adj_close",
    window: int = 126,
    min_periods: int = 63,
    output_column: str = "beta_126",
) -> pd.DataFrame:
    """Calculate rolling stock betas against the benchmark.

    Beta at date t uses historical close-to-close returns through date t.
    It is therefore available when constructing positions at the close of t.
    """
    required = {
        "date",
        "ticker",
        stock_return_column,
    }

    missing = required - set(equity_panel.columns)

    if missing:
        raise ValueError(f"Missing equity columns: {sorted(missing)}")

    benchmark_required = {"date", benchmark_price_column}
    benchmark_missing = benchmark_required - set(benchmark.columns)

    if benchmark_missing:
        raise ValueError(f"Missing benchmark columns: {sorted(benchmark_missing)}")

    stocks = equity_panel[["date", "ticker", stock_return_column]].copy()

    stocks["date"] = pd.to_datetime(stocks["date"])

    benchmark_returns = benchmark[["date", benchmark_price_column]].copy()

    benchmark_returns["date"] = pd.to_datetime(benchmark_returns["date"])

    benchmark_returns = benchmark_returns.sort_values("date").drop_duplicates("date")

    # Historical return from t-1 to t, used for beta estimation.
    benchmark_returns["market_ret_1d"] = benchmark_returns[
        benchmark_price_column
    ].pct_change()

    merged = stocks.merge(
        benchmark_returns[["date", "market_ret_1d"]],
        on="date",
        how="left",
        validate="many_to_one",
    )

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = merged.groupby("ticker", group_keys=False)

    def _stock_beta(group: pd.DataFrame) -> pd.Series:
        covariance = (
            group[stock_return_column]
            .rolling(window, min_periods=min_periods)
            .cov(group["market_ret_1d"])
        )

        variance = group["market_ret_1d"].rolling(window, min_periods=min_periods).var()

        return covariance / variance

    # Restore original row alignment after groupby-rolling calculations.
    merged[output_column] = (
        merged.groupby("ticker", group_keys=False)
        .apply(_stock_beta)
        .reset_index(level=0, drop=True)
    )

    return merged[["date", "ticker", output_column]]
