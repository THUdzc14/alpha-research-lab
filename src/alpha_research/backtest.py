from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a cross-sectional long-short backtest."""

    rebalance_frequency: int = 5
    quantiles: int = 5
    long_quantile: int = 5
    short_quantile: int = 1
    long_gross: float = 1.0
    short_gross: float = 1.0
    transaction_cost_bps: float = 10.0
    min_observations: int = 30
    rebalance_offset: int = 0


def prepare_backtest_panel(
    panel: pd.DataFrame,
    factor_column: str,
    return_column: str = "forward_ret_1d",
) -> pd.DataFrame:
    """Validate and prepare a panel for backtesting."""
    required = {
        "date",
        "ticker",
        factor_column,
        return_column,
    }

    missing = required - set(panel.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = panel[["date", "ticker", factor_column, return_column]].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows found.")

    return df


def get_rebalance_dates(
    dates: pd.Series | pd.Index | np.ndarray,
    frequency: int,
    offset: int = 0,
) -> pd.DatetimeIndex:
    """Select every `frequency`-th trading date."""
    if frequency <= 0:
        raise ValueError("frequency must be positive.")

    if not 0 <= offset < frequency:
        raise ValueError("offset must satisfy 0 <= offset < frequency.")

    unique_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(dates).dropna().unique())
    ).sort_values()

    return unique_dates[offset::frequency]


def construct_long_short_weights(
    cross_section: pd.DataFrame,
    factor_column: str,
    quantiles: int = 5,
    long_quantile: int = 5,
    short_quantile: int = 1,
    long_gross: float = 1.0,
    short_gross: float = 1.0,
    min_observations: int = 30,
) -> pd.Series:
    """Construct equal-weight long-short weights for one date.

    The highest factor quantile is held long and the lowest factor
    quantile is held short.

    The resulting portfolio has:

        sum(long weights) = long_gross
        sum(abs(short weights)) = short_gross
    """
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")

    if long_quantile == short_quantile:
        raise ValueError("Long and short quantiles must differ.")

    if long_quantile < 1 or long_quantile > quantiles:
        raise ValueError("long_quantile is outside the valid range.")

    if short_quantile < 1 or short_quantile > quantiles:
        raise ValueError("short_quantile is outside the valid range.")

    if long_gross < 0 or short_gross < 0:
        raise ValueError("Gross exposures must be non-negative.")

    if factor_column not in cross_section.columns:
        raise ValueError(f"Column not found: {factor_column}")

    valid = cross_section.dropna(subset=[factor_column]).copy()

    weights = pd.Series(
        0.0,
        index=cross_section["ticker"],
        dtype="float64",
        name="weight",
    )

    if len(valid) < min_observations:
        return weights

    # Ranking first avoids qcut errors caused by tied factor values.
    ranks = valid[factor_column].rank(method="first")

    valid["quantile"] = (
        pd.qcut(
            ranks,
            q=quantiles,
            labels=False,
        )
        + 1
    )

    long_tickers = valid.loc[
        valid["quantile"] == long_quantile,
        "ticker",
    ]

    short_tickers = valid.loc[
        valid["quantile"] == short_quantile,
        "ticker",
    ]

    if long_tickers.empty or short_tickers.empty:
        return weights

    weights.loc[long_tickers] = long_gross / len(long_tickers)
    weights.loc[short_tickers] = -short_gross / len(short_tickers)

    return weights


def calculate_turnover(
    previous_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """Calculate total traded notional as a fraction of portfolio capital.

    Turnover is defined as:

        sum(abs(target weight - previous weight))

    For a new dollar-neutral portfolio with long gross 1 and short gross 1,
    initial turnover is therefore 2.
    """
    tickers = previous_weights.index.union(target_weights.index)

    previous = previous_weights.reindex(tickers, fill_value=0.0)
    target = target_weights.reindex(tickers, fill_value=0.0)

    return float((target - previous).abs().sum())


def run_long_short_backtest(
    panel: pd.DataFrame,
    factor_column: str,
    return_column: str = "forward_ret_1d",
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an equal-weight quantile long-short backtest.

    Returns:
        daily_results:
            One row per date containing gross return, turnover,
            costs, net return and cumulative return.

        holdings:
            One row per date/ticker containing the portfolio weight.
    """
    if config is None:
        config = BacktestConfig()

    df = prepare_backtest_panel(
        panel,
        factor_column=factor_column,
        return_column=return_column,
    )

    valid_return_dates = df.groupby("date")[return_column].apply(
        lambda x: x.notna().any()
    )

    valid_return_dates = valid_return_dates[valid_return_dates].index

    df = df.loc[df["date"].isin(valid_return_dates)].copy()

    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()

    rebalance_dates = set(
        get_rebalance_dates(
            all_dates,
            frequency=config.rebalance_frequency,
            offset=config.rebalance_offset,
        )
    )

    current_weights = pd.Series(dtype="float64")
    daily_rows: list[dict[str, object]] = []
    holdings_rows: list[pd.DataFrame] = []

    for date in all_dates:
        cross_section = df.loc[df["date"] == date].copy()

        turnover = 0.0
        is_rebalance = date in rebalance_dates

        if is_rebalance:
            target_weights = construct_long_short_weights(
                cross_section=cross_section,
                factor_column=factor_column,
                quantiles=config.quantiles,
                long_quantile=config.long_quantile,
                short_quantile=config.short_quantile,
                long_gross=config.long_gross,
                short_gross=config.short_gross,
                min_observations=config.min_observations,
            )

            turnover = calculate_turnover(
                previous_weights=current_weights,
                target_weights=target_weights,
            )

            current_weights = target_weights

        date_weights = pd.DataFrame(
            {
                "date": date,
                "ticker": cross_section["ticker"],
            }
        )

        date_weights["weight"] = date_weights["ticker"].map(current_weights).fillna(0.0)

        holdings_rows.append(date_weights)

        return_by_ticker = cross_section.set_index("ticker")[return_column]

        aligned_returns = return_by_ticker.reindex(current_weights.index)

        missing_return_weight = float(
            current_weights.loc[aligned_returns.isna()].abs().sum()
        )

        # Missing held returns are treated as zero for this MVP.
        # The missing exposure is reported for diagnostics.
        aligned_returns = aligned_returns.fillna(0.0)

        long_weights = current_weights.clip(lower=0.0)
        short_weights = current_weights.clip(upper=0.0)

        long_return = float(
            (
                long_weights.reindex(
                    aligned_returns.index,
                    fill_value=0.0,
                )
                * aligned_returns
            ).sum()
        )

        # Short weights are negative, so this is the P&L contribution
        # from the short book.
        short_return = float(
            (
                short_weights.reindex(
                    aligned_returns.index,
                    fill_value=0.0,
                )
                * aligned_returns
            ).sum()
        )

        gross_return = long_return + short_return

        transaction_cost = turnover * config.transaction_cost_bps / 10_000.0

        net_return = gross_return - transaction_cost

        long_exposure = float(current_weights[current_weights > 0].sum())

        short_exposure = float(-current_weights[current_weights < 0].sum())

        daily_rows.append(
            {
                "date": date,
                "is_rebalance": is_rebalance,
                "long_return": long_return,
                "short_return": short_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "net_exposure": long_exposure - short_exposure,
                "gross_exposure": long_exposure + short_exposure,
                "missing_return_weight": missing_return_weight,
            }
        )

    daily_results = pd.DataFrame(daily_rows)

    daily_results["gross_cumulative_return"] = (
        1.0 + daily_results["gross_return"]
    ).cumprod()

    daily_results["net_cumulative_return"] = (
        1.0 + daily_results["net_return"]
    ).cumprod()

    holdings = pd.concat(
        holdings_rows,
        ignore_index=True,
    )

    return daily_results, holdings


def calculate_drawdown(
    cumulative_return: pd.Series,
) -> pd.Series:
    """Calculate drawdown from a cumulative wealth series."""
    running_max = cumulative_return.cummax()

    return cumulative_return / running_max - 1.0


def summarise_backtest(
    daily_results: pd.DataFrame,
    return_column: str = "net_return",
    annualisation_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate standard performance statistics."""
    if return_column not in daily_results.columns:
        raise ValueError(f"Column not found: {return_column}")

    returns = daily_results[return_column].dropna()
    observations = len(returns)

    if observations == 0:
        raise ValueError("No valid returns available.")

    total_return = (1.0 + returns).prod() - 1.0

    annualised_return = (1.0 + total_return) ** (
        annualisation_factor / observations
    ) - 1.0

    annualised_volatility = returns.std(ddof=1) * np.sqrt(annualisation_factor)

    if annualised_volatility > 0:
        sharpe_ratio = (
            returns.mean() / returns.std(ddof=1) * np.sqrt(annualisation_factor)
        )
    else:
        sharpe_ratio = np.nan

    cumulative = (1.0 + returns).cumprod()
    drawdown = calculate_drawdown(cumulative)

    rebalance_rows = daily_results.loc[daily_results["is_rebalance"]]

    return pd.DataFrame(
        [
            {
                "observations": observations,
                "total_return": total_return,
                "annualised_return": annualised_return,
                "annualised_volatility": annualised_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": drawdown.min(),
                "positive_day_fraction": (returns > 0).mean(),
                "average_daily_turnover": daily_results["turnover"].mean(),
                "average_rebalance_turnover": rebalance_rows["turnover"].mean(),
                "total_transaction_cost": daily_results["transaction_cost"].sum(),
                "maximum_missing_return_weight": daily_results[
                    "missing_return_weight"
                ].max(),
            }
        ]
    )


def summarise_backtest_legs(
    daily_results: pd.DataFrame,
    annualisation_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Summarise long, short and combined gross return streams."""

    required = {
        "long_return",
        "short_return",
        "gross_return",
        "net_return",
    }

    missing = required - set(daily_results.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []

    for name, column in {
        "long_leg": "long_return",
        "short_leg": "short_return",
        "gross_long_short": "gross_return",
        "net_long_short": "net_return",
    }.items():
        returns = daily_results[column].dropna()
        observations = len(returns)

        total_return = (1.0 + returns).prod() - 1.0

        annualised_return = (1.0 + total_return) ** (
            annualisation_factor / observations
        ) - 1.0

        annualised_volatility = returns.std(ddof=1) * np.sqrt(annualisation_factor)

        if returns.std(ddof=1) > 0:
            sharpe_ratio = (
                returns.mean() / returns.std(ddof=1) * np.sqrt(annualisation_factor)
            )
        else:
            sharpe_ratio = np.nan

        cumulative = (1.0 + returns).cumprod()
        drawdown = calculate_drawdown(cumulative)

        rows.append(
            {
                "portfolio": name,
                "total_return": total_return,
                "annualised_return": annualised_return,
                "annualised_volatility": annualised_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": drawdown.min(),
                "positive_day_fraction": (returns > 0).mean(),
            }
        )

    return pd.DataFrame(rows)


def run_rebalance_offset_backtests(
    panel: pd.DataFrame,
    factor_column: str,
    return_column: str = "forward_ret_1d",
    base_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run the backtest for every possible rebalance offset."""

    if base_config is None:
        base_config = BacktestConfig()

    summaries = []

    for offset in range(base_config.rebalance_frequency):
        config = BacktestConfig(
            rebalance_frequency=base_config.rebalance_frequency,
            quantiles=base_config.quantiles,
            long_quantile=base_config.long_quantile,
            short_quantile=base_config.short_quantile,
            long_gross=base_config.long_gross,
            short_gross=base_config.short_gross,
            transaction_cost_bps=base_config.transaction_cost_bps,
            min_observations=base_config.min_observations,
            rebalance_offset=offset,
        )

        daily, _ = run_long_short_backtest(
            panel=panel,
            factor_column=factor_column,
            return_column=return_column,
            config=config,
        )

        summary = summarise_backtest(daily).iloc[0].to_dict()
        summary["offset"] = offset

        summaries.append(summary)

    return pd.DataFrame(summaries)


def summarise_backtest_subperiods(
    daily_results: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    return_column: str = "net_return",
) -> pd.DataFrame:
    """Calculate backtest statistics over specified date ranges."""

    if "date" not in daily_results.columns:
        raise ValueError("Input must contain a 'date' column.")

    df = daily_results.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []

    for period_name, (start_date, end_date) in periods.items():
        subset = df.loc[
            df["date"].between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
            )
        ].copy()

        if subset.empty:
            continue

        summary = (
            summarise_backtest(
                subset,
                return_column=return_column,
            )
            .iloc[0]
            .to_dict()
        )

        summary["period"] = period_name
        summary["start_date"] = pd.Timestamp(start_date)
        summary["end_date"] = pd.Timestamp(end_date)

        rows.append(summary)

    return pd.DataFrame(rows)
