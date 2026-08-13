from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from alpha_research.config.research import (
    DEFAULT_NUMERICAL_TOLERANCE,
    MONITORING_SPECIFICATION,
)
from alpha_research.metrics import summarise_concentration

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


def calculate_rolling_market_model(
    equity: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
    annualisation_factor: int = 252,
    output_prefix: str = "market_model",
) -> pd.DataFrame:
    """Estimate a rolling single-factor market model for each ticker.

    The model is:

        stock_return = alpha + beta * market_return + residual

    Rolling estimates use only observations through the current date.
    Idiosyncratic volatility is the annualised sample standard deviation
    of the in-window residuals.

    Parameters
    ----------
    equity
        Equity panel containing ``date``, ``ticker``, and ``ret_1d``.
    benchmark
        Benchmark data containing ``date`` and ``adj_close``.
    window
        Maximum rolling-window length.
    min_periods
        Minimum number of aligned observations required. Defaults to
        ``window``.
    annualisation_factor
        Number of trading periods per year.
    output_prefix
        Prefix used for the four output columns.

    Returns
    -------
    pd.DataFrame
        A copy of ``equity`` with rolling alpha, beta, current residual,
        and annualised idiosyncratic volatility appended.
    """
    if min_periods is None:
        min_periods = window

    if window < 2:
        raise ValueError("window must be at least 2.")

    if not 2 <= min_periods <= window:
        raise ValueError("min_periods must be between 2 and window.")

    if annualisation_factor <= 0:
        raise ValueError("annualisation_factor must be positive.")

    required_equity_columns = {
        "date",
        "ticker",
        "ret_1d",
    }
    required_benchmark_columns = {
        "date",
        "adj_close",
    }

    missing_equity_columns = required_equity_columns - set(equity.columns)
    missing_benchmark_columns = required_benchmark_columns - set(benchmark.columns)

    if missing_equity_columns:
        raise ValueError(
            "Equity data is missing required columns: "
            f"{sorted(missing_equity_columns)}"
        )

    if missing_benchmark_columns:
        raise ValueError(
            "Benchmark data is missing required columns: "
            f"{sorted(missing_benchmark_columns)}"
        )

    result = equity.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["_original_order"] = np.arange(len(result))

    market = benchmark[["date", "adj_close"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date")

    if market["date"].duplicated().any():
        raise ValueError("Benchmark data contains duplicate dates.")

    market["_market_ret_1d"] = market["adj_close"].pct_change()

    result = result.merge(
        market[["date", "_market_ret_1d"]],
        on="date",
        how="left",
        validate="many_to_one",
    )

    alpha_column = f"{output_prefix}_alpha"
    beta_column = f"{output_prefix}_beta"
    residual_column = f"{output_prefix}_residual"
    idio_vol_column = f"{output_prefix}_idio_vol"

    def calculate_for_ticker(
        group: pd.DataFrame,
    ) -> pd.DataFrame:
        group = group.sort_values("date").copy()

        stock_return = group["ret_1d"]
        market_return = group["_market_ret_1d"]

        # Both rolling series must use exactly the same aligned dates.
        paired_stock = stock_return.where(market_return.notna())
        paired_market = market_return.where(stock_return.notna())

        rolling_stock = paired_stock.rolling(
            window=window,
            min_periods=min_periods,
        )
        rolling_market = paired_market.rolling(
            window=window,
            min_periods=min_periods,
        )

        stock_mean = rolling_stock.mean()
        market_mean = rolling_market.mean()

        stock_variance = rolling_stock.var(ddof=1)
        market_variance = rolling_market.var(ddof=1)
        covariance = rolling_stock.cov(paired_market)

        valid_market_variance = market_variance.notna() & (market_variance > 0.0)

        beta = (covariance / market_variance).where(valid_market_variance)

        alpha = (stock_mean - beta * market_mean).where(valid_market_variance)

        residual_variance = (stock_variance - covariance.pow(2) / market_variance).clip(
            lower=0.0
        )

        group[alpha_column] = alpha
        group[beta_column] = beta

        # This is the residual of the current observation under the
        # rolling model estimated through the current date.
        group[residual_column] = (paired_stock - alpha - beta * paired_market).where(
            valid_market_variance
        )

        group[idio_vol_column] = (
            np.sqrt(residual_variance) * np.sqrt(annualisation_factor)
        ).where(valid_market_variance)

        return group

    result = pd.concat(
        [
            calculate_for_ticker(group)
            for _, group in result.groupby(
                "ticker",
                sort=False,
            )
        ],
        ignore_index=True,
    )

    result = (
        result.sort_values("_original_order")
        .drop(
            columns=[
                "_market_ret_1d",
                "_original_order",
            ]
        )
        .reset_index(drop=True)
    )

    return result


REALISED_RETURN_STREAMS = {
    "gross": "gross_return",
    "net": "net_return",
    "long": "long_return",
    "short": "short_return",
}

BETA_STATE_EXPORT_COLUMNS = (
    "date",
    "portfolio",
    "beta_coverage",
    "holdings_market_beta",
    "realised_gross_beta_126",
    "realised_net_beta_126",
    "beta_measurement_gap",
    "holdings_long_beta_contribution",
    "holdings_short_beta_contribution",
    "realised_long_beta_126",
    "realised_short_beta_126",
    "long_basket_beta",
    "short_basket_beta",
)


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required - set(data.columns)

    if missing:
        raise KeyError(f"{name} is missing columns: {sorted(missing)}")


def _validate_unique_keys(
    data: pd.DataFrame,
    keys: list[str],
    *,
    name: str,
) -> None:
    if data.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate {keys} rows.")


def _ordered_portfolios(
    data: pd.DataFrame,
    portfolios: Sequence[str] | None,
) -> list[str]:
    available = list(pd.unique(data["portfolio"].dropna()))

    if portfolios is None:
        return available

    requested = list(portfolios)

    if not requested or len(set(requested)) != len(requested):
        raise ValueError("portfolios must contain unique portfolio names.")

    missing = sorted(set(requested) - set(available))

    if missing:
        raise ValueError(f"Input data are missing portfolios: {missing}")

    return requested


def calculate_realised_beta_state(
    portfolio_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    return_streams: Mapping[str, str] = REALISED_RETURN_STREAMS,
    benchmark_return_column: str = "benchmark_return",
    window: int = MONITORING_SPECIFICATION.risk_window,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Calculate rolling realised beta and correlation by return stream."""
    if min_periods is None:
        min_periods = window

    if window < 2 or not 2 <= min_periods <= window:
        raise ValueError(
            "window and min_periods must satisfy 2 <= min_periods <= window."
        )

    if not return_streams:
        raise ValueError("return_streams must not be empty.")

    required_portfolio_columns = {
        "date",
        "portfolio",
        *return_streams.values(),
    }
    _require_columns(
        portfolio_daily,
        required_portfolio_columns,
        name="portfolio_daily",
    )
    _require_columns(
        benchmark_daily,
        {"date", benchmark_return_column},
        name="benchmark_daily",
    )

    portfolios_to_use = _ordered_portfolios(portfolio_daily, portfolios)
    portfolio_data = portfolio_daily.loc[
        portfolio_daily["portfolio"].isin(portfolios_to_use),
        list(required_portfolio_columns),
    ].copy()
    benchmark_data = benchmark_daily[["date", benchmark_return_column]].copy()
    portfolio_data["date"] = pd.to_datetime(portfolio_data["date"])
    benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])

    _validate_unique_keys(
        portfolio_data,
        ["portfolio", "date"],
        name="portfolio_daily",
    )
    _validate_unique_keys(
        benchmark_data,
        ["date"],
        name="benchmark_daily",
    )

    portfolio_market_daily = portfolio_data.merge(
        benchmark_data,
        on="date",
        how="left",
        validate="many_to_one",
    )
    result_parts = []

    for portfolio_name in portfolios_to_use:
        group = (
            portfolio_market_daily.loc[
                portfolio_market_daily["portfolio"].eq(portfolio_name)
            ]
            .sort_values("date")
            .copy()
        )
        market_returns = group[benchmark_return_column]
        rolling_market_variance = market_returns.rolling(
            window,
            min_periods=min_periods,
        ).var()

        for stream_name, return_column in return_streams.items():
            stream_returns = group[return_column]
            rolling_covariance = stream_returns.rolling(
                window,
                min_periods=min_periods,
            ).cov(market_returns)
            rolling_correlation = stream_returns.rolling(
                window,
                min_periods=min_periods,
            ).corr(market_returns)

            result_parts.append(
                pd.DataFrame(
                    {
                        "date": group["date"].to_numpy(),
                        "portfolio": portfolio_name,
                        "return_stream": stream_name,
                        f"realised_beta_{window}": (
                            rolling_covariance / rolling_market_variance
                        ).to_numpy(),
                        f"market_correlation_{window}": (
                            rolling_correlation.to_numpy()
                        ),
                    }
                )
            )

    return (
        pd.concat(result_parts, ignore_index=True)
        .sort_values(["portfolio", "return_stream", "date"])
        .reset_index(drop=True)
    )


def prepare_holdings_beta_detail(
    security_holdings: pd.DataFrame,
    security_betas: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    beta_column: str = "beta_126",
) -> pd.DataFrame:
    """Attach security betas and calculate holdings-level beta contributions."""
    _require_columns(
        security_holdings,
        {"date", "ticker", "portfolio", "weight"},
        name="security_holdings",
    )
    _require_columns(
        security_betas,
        {"date", "ticker", beta_column},
        name="security_betas",
    )

    portfolios_to_use = _ordered_portfolios(security_holdings, portfolios)
    holdings = security_holdings.loc[
        security_holdings["portfolio"].isin(portfolios_to_use)
    ].copy()
    betas = security_betas[["date", "ticker", beta_column]].copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    betas["date"] = pd.to_datetime(betas["date"])

    _validate_unique_keys(
        holdings,
        ["portfolio", "date", "ticker"],
        name="security_holdings",
    )
    _validate_unique_keys(
        betas,
        ["date", "ticker"],
        name="security_betas",
    )

    result = holdings.merge(
        betas,
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    result["long_weight"] = result["weight"].clip(lower=0.0)
    result["short_weight"] = -result["weight"].clip(upper=0.0)
    result["absolute_weight"] = result["weight"].abs()
    result["beta_covered_gross_weight"] = result["absolute_weight"].where(
        result[beta_column].notna(),
        0.0,
    )
    result["long_beta_contribution"] = result["long_weight"] * result[beta_column]
    result["short_beta_contribution"] = -result["short_weight"] * result[beta_column]
    result["market_beta_contribution"] = result["weight"] * result[beta_column]

    return result.sort_values(["portfolio", "date", "ticker"]).reset_index(drop=True)


def calculate_holdings_beta_state(
    holdings_beta_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate holdings-implied beta, basket beta, and beta coverage."""
    _require_columns(
        holdings_beta_detail,
        {
            "portfolio",
            "date",
            "ticker",
            "weight",
            "long_weight",
            "short_weight",
            "absolute_weight",
            "beta_covered_gross_weight",
            "long_beta_contribution",
            "short_beta_contribution",
            "market_beta_contribution",
        },
        name="holdings_beta_detail",
    )
    _validate_unique_keys(
        holdings_beta_detail,
        ["portfolio", "date", "ticker"],
        name="holdings_beta_detail",
    )

    result = (
        holdings_beta_detail.groupby(["portfolio", "date"], sort=False)
        .agg(
            holdings_long_exposure=("long_weight", "sum"),
            holdings_short_exposure=("short_weight", "sum"),
            holdings_gross_exposure=("absolute_weight", "sum"),
            holdings_net_exposure=("weight", "sum"),
            beta_covered_gross_weight=("beta_covered_gross_weight", "sum"),
            holdings_long_beta_contribution=(
                "long_beta_contribution",
                lambda values: values.sum(min_count=1),
            ),
            holdings_short_beta_contribution=(
                "short_beta_contribution",
                lambda values: values.sum(min_count=1),
            ),
            holdings_market_beta=(
                "market_beta_contribution",
                lambda values: values.sum(min_count=1),
            ),
        )
        .reset_index()
    )
    result["long_basket_beta"] = (
        result["holdings_long_beta_contribution"] / result["holdings_long_exposure"]
    )
    result["short_basket_beta"] = (
        -result["holdings_short_beta_contribution"] / result["holdings_short_exposure"]
    )
    result["beta_coverage"] = (
        result["beta_covered_gross_weight"] / result["holdings_gross_exposure"]
    ).where(result["holdings_gross_exposure"].gt(0.0))

    return result.sort_values(["portfolio", "date"]).reset_index(drop=True)


def calculate_beta_state(
    portfolio_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    security_holdings: pd.DataFrame,
    security_betas: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    beta_column: str = "beta_126",
    benchmark_return_column: str = "benchmark_return",
    window: int = MONITORING_SPECIFICATION.risk_window,
    min_periods: int | None = None,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Build Notebook 08's combined realised and holdings beta state."""
    if window != 126:
        raise ValueError(
            "The exported beta-state schema currently requires window=126."
        )

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    portfolios_to_use = _ordered_portfolios(portfolio_daily, portfolios)
    holdings_detail = prepare_holdings_beta_detail(
        security_holdings,
        security_betas,
        portfolios=portfolios_to_use,
        beta_column=beta_column,
    )
    holdings_state = calculate_holdings_beta_state(holdings_detail)
    realised_state = calculate_realised_beta_state(
        portfolio_daily,
        benchmark_daily,
        portfolios=portfolios_to_use,
        benchmark_return_column=benchmark_return_column,
        window=window,
        min_periods=min_periods,
    )

    realised_wide = (
        realised_state.pivot(
            index=["date", "portfolio"],
            columns="return_stream",
            values=f"realised_beta_{window}",
        )
        .rename(columns=lambda name: f"realised_{name}_beta_{window}")
        .reset_index()
    )
    realised_wide.columns.name = None

    result = (
        holdings_state.merge(
            realised_wide,
            on=["date", "portfolio"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["portfolio", "date"])
        .reset_index(drop=True)
    )
    result["beta_measurement_gap"] = (
        result["holdings_market_beta"] - result["realised_gross_beta_126"]
    )

    exposure_columns = {
        "long": ("holdings_long_exposure", "long_exposure"),
        "short": ("holdings_short_exposure", "short_exposure"),
        "net": ("holdings_net_exposure", "net_exposure"),
        "gross": ("holdings_gross_exposure", "gross_exposure"),
    }
    _require_columns(
        portfolio_daily,
        {"portfolio", "date", *(column for _, column in exposure_columns.values())},
        name="portfolio_daily",
    )
    portfolio_exposures = portfolio_daily.loc[
        portfolio_daily["portfolio"].isin(portfolios_to_use),
        [
            "portfolio",
            "date",
            *(column for _, column in exposure_columns.values()),
        ],
    ].copy()
    portfolio_exposures["date"] = pd.to_datetime(portfolio_exposures["date"])
    exposure_audit = holdings_state.merge(
        portfolio_exposures,
        on=["portfolio", "date"],
        how="left",
        validate="one_to_one",
    )
    exposure_differences = pd.concat(
        [
            (exposure_audit[holdings_column] - exposure_audit[portfolio_column]).abs()
            for holdings_column, portfolio_column in exposure_columns.values()
        ],
        axis=1,
    )
    maximum_exposure_difference = exposure_differences.max().max()

    if (
        exposure_differences.isna().any().any()
        or maximum_exposure_difference > tolerance
    ):
        raise ValueError(
            "Security holdings do not reconcile with portfolio-level exposures."
        )

    holdings_difference = (
        result["holdings_long_beta_contribution"]
        + result["holdings_short_beta_contribution"]
        - result["holdings_market_beta"]
    ).abs()
    realised_difference = (
        result["realised_long_beta_126"]
        + result["realised_short_beta_126"]
        - result["realised_gross_beta_126"]
    ).abs()
    maximum_decomposition_difference = pd.concat(
        [holdings_difference, realised_difference]
    ).max()

    if maximum_decomposition_difference > tolerance:
        raise ValueError("Long- and short-side beta contributions do not reconcile.")

    return result.loc[:, BETA_STATE_EXPORT_COLUMNS]


def prepare_concentration_security_detail(
    security_daily: pd.DataFrame,
    sector_panel: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Prepare security weights, contributions, and dated sector metadata."""
    _require_columns(
        security_daily,
        {"portfolio", "date", "ticker", "weight", "gross_contribution"},
        name="security_daily",
    )
    _require_columns(
        sector_panel,
        {"date", "ticker", "sector"},
        name="sector_panel",
    )

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    portfolios_to_use = _ordered_portfolios(security_daily, portfolios)
    security = security_daily.loc[
        security_daily["portfolio"].isin(portfolios_to_use)
    ].copy()
    sectors = sector_panel[["date", "ticker", "sector"]].copy()
    security["date"] = pd.to_datetime(security["date"])
    sectors["date"] = pd.to_datetime(sectors["date"])

    _validate_unique_keys(
        security,
        ["portfolio", "date", "ticker"],
        name="security_daily",
    )

    sector_key_counts = sectors.groupby(
        ["date", "ticker"],
        sort=False,
    )[
        "sector"
    ].nunique(dropna=False)

    if sector_key_counts.gt(1).any():
        raise ValueError(
            "sector_panel contains conflicting sector values for a date-ticker key."
        )

    sectors = sectors.drop_duplicates(["date", "ticker"])
    result = security.merge(
        sectors,
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    result["absolute_weight"] = result["weight"].abs()
    result["long_weight"] = result["weight"].clip(lower=0.0)
    result["short_weight"] = -result["weight"].clip(upper=0.0)
    result["absolute_gross_contribution"] = result["gross_contribution"].abs()

    held = result.loc[result["absolute_weight"].gt(tolerance)].copy()

    if not held.empty:
        held["sector_covered_weight"] = held["absolute_weight"].where(
            held["sector"].notna(),
            0.0,
        )
        coverage = held.groupby(["portfolio", "date"], sort=False).agg(
            held_gross_weight=("absolute_weight", "sum"),
            sector_covered_weight=("sector_covered_weight", "sum"),
        )
        sector_coverage = (
            coverage["sector_covered_weight"] / coverage["held_gross_weight"]
        )

        if sector_coverage.lt(1.0 - tolerance).any():
            raise ValueError("Some held portfolio weight lacks sector metadata.")

    return result.sort_values(["portfolio", "ticker", "date"]).reset_index(drop=True)


def calculate_position_concentration_state(
    concentration_security_detail: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Calculate daily gross-, long-, and short-book concentration."""
    _require_columns(
        concentration_security_detail,
        {
            "portfolio",
            "date",
            "ticker",
            "absolute_weight",
            "long_weight",
            "short_weight",
        },
        name="concentration_security_detail",
    )
    rows = []

    for (portfolio_name, date), group in concentration_security_detail.groupby(
        ["portfolio", "date"],
        sort=False,
    ):
        gross_statistics = summarise_concentration(
            group["absolute_weight"],
            tolerance=tolerance,
        )
        long_statistics = summarise_concentration(
            group["long_weight"],
            tolerance=tolerance,
        )
        short_statistics = summarise_concentration(
            group["short_weight"],
            tolerance=tolerance,
        )
        rows.append(
            {
                "portfolio": portfolio_name,
                "date": date,
                "gross_exposure": group["absolute_weight"].sum(),
                "held_position_count": gross_statistics["count"],
                "effective_position_count": gross_statistics["effective_count"],
                "largest_absolute_weight": group["absolute_weight"].max(),
                "largest_position_gross_share": gross_statistics["largest_share"],
                "top_five_position_gross_share": gross_statistics["top_five_share"],
                "effective_long_position_count": long_statistics["effective_count"],
                "effective_short_position_count": short_statistics["effective_count"],
                "largest_long_book_share": long_statistics["largest_share"],
                "largest_short_book_share": short_statistics["largest_share"],
            }
        )

    return pd.DataFrame(rows).sort_values(["portfolio", "date"]).reset_index(drop=True)


def calculate_sector_concentration_state(
    concentration_security_detail: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Calculate daily sector exposures and sector concentration."""
    _require_columns(
        concentration_security_detail,
        {"portfolio", "date", "sector", "long_weight", "short_weight"},
        name="concentration_security_detail",
    )
    sector_exposure = (
        concentration_security_detail.groupby(
            ["portfolio", "date", "sector"],
            sort=False,
            observed=True,
        )
        .agg(
            long_exposure=("long_weight", "sum"),
            short_exposure=("short_weight", "sum"),
        )
        .reset_index()
    )
    sector_exposure["gross_exposure"] = (
        sector_exposure["long_exposure"] + sector_exposure["short_exposure"]
    )
    sector_exposure["net_exposure"] = (
        sector_exposure["long_exposure"] - sector_exposure["short_exposure"]
    )
    rows = []

    for (portfolio_name, date), group in sector_exposure.groupby(
        ["portfolio", "date"],
        sort=False,
    ):
        statistics = summarise_concentration(
            group["gross_exposure"],
            tolerance=tolerance,
        )
        rows.append(
            {
                "portfolio": portfolio_name,
                "date": date,
                "held_sector_count": statistics["count"],
                "effective_sector_count": statistics["effective_count"],
                "largest_sector_gross_share": statistics["largest_share"],
                "top_three_sector_gross_share": statistics["top_three_share"],
                "largest_absolute_sector_net_exposure": (
                    group["net_exposure"].abs().max()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["portfolio", "date"]).reset_index(drop=True)


def calculate_beta_concentration_state(
    holdings_beta_detail: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Calculate concentration of absolute holdings beta contributions."""
    _require_columns(
        holdings_beta_detail,
        {"portfolio", "date", "ticker", "market_beta_contribution"},
        name="holdings_beta_detail",
    )
    rows = []

    for (portfolio_name, date), group in holdings_beta_detail.groupby(
        ["portfolio", "date"],
        sort=False,
    ):
        statistics = summarise_concentration(
            group["market_beta_contribution"].abs(),
            tolerance=tolerance,
        )
        rows.append(
            {
                "portfolio": portfolio_name,
                "date": date,
                "effective_beta_contributor_count": statistics["effective_count"],
                "largest_absolute_beta_contribution_share": (
                    statistics["largest_share"]
                ),
                "top_five_absolute_beta_contribution_share": (
                    statistics["top_five_share"]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["portfolio", "date"]).reset_index(drop=True)


def _rolling_contribution_matrix_to_long(
    matrix: pd.DataFrame,
    *,
    entity_name: str,
    value_name: str,
) -> pd.DataFrame:
    """Convert a date-by-entity rolling matrix to a tidy table."""
    return (
        matrix.rename_axis(
            index="date",
            columns=entity_name,
        )
        .reset_index()
        .melt(
            id_vars="date",
            var_name=entity_name,
            value_name=value_name,
        )
    )


def calculate_rolling_contribution_detail(
    concentration_security_detail: pd.DataFrame,
    *,
    window: int = MONITORING_SPECIFICATION.concentration_window,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fixed-date-window security and sector contributions.

    Security attribution is sparse because zero-weight securities are absent
    from many portfolio dates. Reindexing each portfolio to its full calendar
    and filling absent contributors with zero makes a 63-day window mean 63
    portfolio dates rather than 63 non-zero observations for each contributor.

    Returns
    -------
    security_result
        Rolling signed and absolute contributions by portfolio, date and ticker.
    sector_result
        Rolling signed and absolute contributions by portfolio, date and sector.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    _require_columns(
        concentration_security_detail,
        {
            "portfolio",
            "date",
            "ticker",
            "sector",
            "gross_contribution",
            "absolute_gross_contribution",
        },
        name="concentration_security_detail",
    )

    detail = concentration_security_detail.copy()
    detail["date"] = pd.to_datetime(detail["date"])

    _validate_unique_keys(
        detail,
        [
            "portfolio",
            "date",
            "ticker",
        ],
        name="concentration_security_detail",
    )

    if detail.empty:
        security_columns = [
            "portfolio",
            "date",
            "ticker",
            f"rolling_gross_contribution_{window}",
            f"rolling_absolute_gross_contribution_{window}",
        ]
        sector_columns = [
            "portfolio",
            "date",
            "sector",
            f"rolling_gross_contribution_{window}",
            f"rolling_absolute_gross_contribution_{window}",
        ]

        return (
            pd.DataFrame(columns=security_columns),
            pd.DataFrame(columns=sector_columns),
        )

    sector_daily = (
        detail.dropna(subset=["sector"])
        .groupby(
            [
                "portfolio",
                "date",
                "sector",
            ],
            sort=False,
            observed=True,
            as_index=False,
        )
        .agg(
            gross_contribution=(
                "gross_contribution",
                "sum",
            ),
            absolute_gross_contribution=(
                "absolute_gross_contribution",
                "sum",
            ),
        )
    )

    security_frames = []
    sector_frames = []

    for portfolio, portfolio_detail in detail.groupby(
        "portfolio",
        sort=False,
    ):
        portfolio_dates = pd.DatetimeIndex(
            portfolio_detail["date"].drop_duplicates().sort_values()
        )

        # Security-level rolling contributions
        security_signed = (
            portfolio_detail.pivot(
                index="date",
                columns="ticker",
                values="gross_contribution",
            )
            .reindex(portfolio_dates)
            .fillna(0.0)
        )

        security_absolute = (
            portfolio_detail.pivot(
                index="date",
                columns="ticker",
                values="absolute_gross_contribution",
            )
            .reindex(portfolio_dates)
            .fillna(0.0)
        )

        rolling_security_signed = security_signed.rolling(
            window=window,
            min_periods=window,
        ).sum()

        rolling_security_absolute = security_absolute.rolling(
            window=window,
            min_periods=window,
        ).sum()

        security_long = _rolling_contribution_matrix_to_long(
            rolling_security_signed,
            entity_name="ticker",
            value_name=f"rolling_gross_contribution_{window}",
        ).merge(
            _rolling_contribution_matrix_to_long(
                rolling_security_absolute,
                entity_name="ticker",
                value_name=(f"rolling_absolute_gross_contribution_{window}"),
            ),
            on=[
                "date",
                "ticker",
            ],
            how="inner",
            validate="one_to_one",
        )

        security_long.insert(
            0,
            "portfolio",
            portfolio,
        )
        security_frames.append(security_long)

        # Sector-level rolling contributions
        portfolio_sector = sector_daily.loc[sector_daily["portfolio"].eq(portfolio)]

        if portfolio_sector.empty:
            continue

        sector_signed = (
            portfolio_sector.pivot(
                index="date",
                columns="sector",
                values="gross_contribution",
            )
            .reindex(portfolio_dates)
            .fillna(0.0)
        )

        sector_absolute = (
            portfolio_sector.pivot(
                index="date",
                columns="sector",
                values="absolute_gross_contribution",
            )
            .reindex(portfolio_dates)
            .fillna(0.0)
        )

        rolling_sector_signed = sector_signed.rolling(
            window=window,
            min_periods=window,
        ).sum()

        rolling_sector_absolute = sector_absolute.rolling(
            window=window,
            min_periods=window,
        ).sum()

        sector_long = _rolling_contribution_matrix_to_long(
            rolling_sector_signed,
            entity_name="sector",
            value_name=f"rolling_gross_contribution_{window}",
        ).merge(
            _rolling_contribution_matrix_to_long(
                rolling_sector_absolute,
                entity_name="sector",
                value_name=(f"rolling_absolute_gross_contribution_{window}"),
            ),
            on=[
                "date",
                "sector",
            ],
            how="inner",
            validate="one_to_one",
        )

        sector_long.insert(
            0,
            "portfolio",
            portfolio,
        )
        sector_frames.append(sector_long)

    security_result = (
        pd.concat(
            security_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "portfolio",
                "date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    if sector_frames:
        sector_result = (
            pd.concat(
                sector_frames,
                ignore_index=True,
            )
            .sort_values(
                [
                    "portfolio",
                    "date",
                    "sector",
                ]
            )
            .reset_index(drop=True)
        )
    else:
        sector_result = pd.DataFrame(
            columns=[
                "portfolio",
                "date",
                "sector",
                f"rolling_gross_contribution_{window}",
                f"rolling_absolute_gross_contribution_{window}",
            ]
        )

    return security_result, sector_result


def calculate_contribution_concentration_state(
    concentration_security_detail: pd.DataFrame,
    *,
    window: int = MONITORING_SPECIFICATION.concentration_window,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate contribution concentration over fixed portfolio-date windows.

    An absent security row represents zero contribution on that portfolio
    date. Rolling contribution details are therefore constructed on the full
    portfolio-date calendar before concentration statistics are calculated.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    (
        security_rolling,
        sector_rolling,
    ) = calculate_rolling_contribution_detail(
        concentration_security_detail,
        window=window,
    )

    rolling_absolute_column = f"rolling_absolute_gross_contribution_{window}"

    security_rows = []

    for (
        portfolio_name,
        date,
    ), group in security_rolling.groupby(
        [
            "portfolio",
            "date",
        ],
        sort=False,
    ):
        contribution_statistics = summarise_concentration(
            group[rolling_absolute_column],
            tolerance=tolerance,
        )

        active_contributors = group.loc[group[rolling_absolute_column].gt(tolerance)]

        if active_contributors.empty:
            largest_contributor = pd.NA
        else:
            largest_contributor = active_contributors.loc[
                active_contributors[rolling_absolute_column].idxmax(),
                "ticker",
            ]

        security_rows.append(
            {
                "portfolio": portfolio_name,
                "date": date,
                f"effective_contributor_count_{window}": (
                    contribution_statistics["effective_count"]
                ),
                f"largest_contributor_share_{window}": (
                    contribution_statistics["largest_share"]
                ),
                f"top_five_contributor_share_{window}": (
                    contribution_statistics["top_five_share"]
                ),
                f"largest_contributor_{window}": (largest_contributor),
            }
        )

    security_state = (
        pd.DataFrame(security_rows)
        .sort_values(
            [
                "portfolio",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    sector_rows = []

    for (
        portfolio_name,
        date,
    ), group in sector_rolling.groupby(
        [
            "portfolio",
            "date",
        ],
        sort=False,
    ):
        contribution_statistics = summarise_concentration(
            group[rolling_absolute_column],
            tolerance=tolerance,
        )

        active_sectors = group.loc[group[rolling_absolute_column].gt(tolerance)]

        if active_sectors.empty:
            largest_sector = pd.NA
        else:
            largest_sector = active_sectors.loc[
                active_sectors[rolling_absolute_column].idxmax(),
                "sector",
            ]

        sector_rows.append(
            {
                "portfolio": portfolio_name,
                "date": date,
                (
                    f"effective_contribution_sector_count_{window}"
                ): contribution_statistics["effective_count"],
                (
                    f"largest_contribution_sector_share_{window}"
                ): contribution_statistics["largest_share"],
                f"largest_contribution_sector_{window}": (largest_sector),
            }
        )

    sector_state = (
        pd.DataFrame(sector_rows)
        .sort_values(
            [
                "portfolio",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    return security_state, sector_state


def calculate_concentration_state(
    security_daily: pd.DataFrame,
    sector_panel: pd.DataFrame,
    holdings_beta_detail: pd.DataFrame,
    *,
    portfolios: Sequence[str] | None = None,
    window: int = MONITORING_SPECIFICATION.concentration_window,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Build Notebook 08's complete daily concentration state."""
    if window != 63:
        raise ValueError(
            "The exported concentration-state schema currently requires window=63."
        )

    detail = prepare_concentration_security_detail(
        security_daily,
        sector_panel,
        portfolios=portfolios,
        tolerance=tolerance,
    )
    portfolios_to_use = _ordered_portfolios(detail, portfolios)
    _ordered_portfolios(holdings_beta_detail, portfolios_to_use)
    holdings_detail = holdings_beta_detail.loc[
        holdings_beta_detail["portfolio"].isin(portfolios_to_use)
    ].copy()
    position_state = calculate_position_concentration_state(
        detail,
        tolerance=tolerance,
    )
    sector_state = calculate_sector_concentration_state(
        detail,
        tolerance=tolerance,
    )
    beta_state = calculate_beta_concentration_state(
        holdings_detail,
        tolerance=tolerance,
    )
    security_contribution_state, sector_contribution_state = (
        calculate_contribution_concentration_state(
            detail,
            window=window,
            tolerance=tolerance,
        )
    )

    return (
        position_state.merge(
            sector_state,
            on=["portfolio", "date"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            beta_state,
            on=["portfolio", "date"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            security_contribution_state,
            on=["portfolio", "date"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            sector_contribution_state,
            on=["portfolio", "date"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["portfolio", "date"])
        .reset_index(drop=True)
    )
