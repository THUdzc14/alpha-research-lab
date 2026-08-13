"""Reusable transaction-cost, turnover, liquidity, and capacity analytics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from alpha_research.config.research import (
    CAPACITY_PARTICIPATION_LIMITS,
    DEFAULT_NUMERICAL_TOLERANCE,
    TRADING_DAYS_PER_YEAR,
)


def _validate_non_negative_rate(value: float, name: str) -> float:
    result = float(value)

    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return result


def _as_numeric_series(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool,
) -> pd.Series:
    try:
        result = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values.") from exc

    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} contains missing values.")

    if not np.isfinite(result.dropna().to_numpy()).all():
        raise ValueError(f"{name} contains infinite values.")

    return result


def calculate_linear_transaction_cost(
    turnover: float | pd.Series,
    transaction_cost_bps: float,
) -> float | pd.Series:
    """Apply a linear cost rate to full-L1 traded notional.

    A turnover value of 1.0 charged at 10 basis points produces a cost drag
    of 0.001 relative to portfolio capital.
    """
    cost_bps = _validate_non_negative_rate(
        transaction_cost_bps,
        "transaction_cost_bps",
    )

    if isinstance(turnover, pd.Series):
        numeric_turnover = _as_numeric_series(
            turnover,
            name="turnover",
            allow_missing=True,
        )

        if numeric_turnover.dropna().lt(0.0).any():
            raise ValueError("turnover must be non-negative.")

        return (numeric_turnover * cost_bps / 10_000.0).rename("transaction_cost")

    numeric_turnover = float(turnover)

    if not math.isfinite(numeric_turnover) or numeric_turnover < 0.0:
        raise ValueError("turnover must be finite and non-negative.")

    return numeric_turnover * cost_bps / 10_000.0


def apply_linear_transaction_costs(
    data: pd.DataFrame,
    transaction_cost_bps: float,
    turnover_column: str = "turnover",
    gross_return_column: str = "gross_return",
    cost_column: str = "transaction_cost",
    net_return_column: str = "net_return",
) -> pd.DataFrame:
    """Return a copy with linear transaction costs and net returns added."""
    required_columns = {turnover_column, gross_return_column}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = data.copy()
    result[gross_return_column] = _as_numeric_series(
        result[gross_return_column],
        name=gross_return_column,
        allow_missing=True,
    )

    result[cost_column] = calculate_linear_transaction_cost(
        result[turnover_column],
        transaction_cost_bps=transaction_cost_bps,
    )
    result[net_return_column] = result[gross_return_column] - result[cost_column]

    return result


def calculate_top_n_turnover_share(
    turnover: pd.Series,
    number_of_days: int,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> float:
    """Calculate the fraction of turnover occurring on the largest N days."""
    if number_of_days <= 0:
        raise ValueError("number_of_days must be positive.")

    tolerance = _validate_non_negative_rate(tolerance, "tolerance")
    numeric_turnover = _as_numeric_series(
        turnover,
        name="turnover",
        allow_missing=False,
    )

    if numeric_turnover.lt(-tolerance).any():
        raise ValueError("turnover cannot contain materially negative values.")

    numeric_turnover = numeric_turnover.clip(lower=0.0)
    total_turnover = float(numeric_turnover.sum())

    if total_turnover <= tolerance:
        return np.nan

    return float(numeric_turnover.nlargest(number_of_days).sum() / total_turnover)


def summarise_turnover(
    turnover: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    concentration_window: int = 21,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> dict[str, int | float]:
    """Reproduce the per-offset turnover statistics from Notebook 06."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if concentration_window <= 0:
        raise ValueError("concentration_window must be positive.")

    tolerance = _validate_non_negative_rate(tolerance, "tolerance")
    numeric_turnover = _as_numeric_series(
        turnover,
        name="turnover",
        allow_missing=False,
    )

    if numeric_turnover.empty:
        raise ValueError("turnover is empty.")

    if numeric_turnover.lt(-tolerance).any():
        raise ValueError("turnover cannot contain materially negative values.")

    numeric_turnover = numeric_turnover.clip(lower=0.0)
    active_turnover = numeric_turnover.loc[numeric_turnover.gt(tolerance)]
    observations = len(numeric_turnover)

    return {
        "observations": observations,
        "trading_days": len(active_turnover),
        "trading_day_fraction": len(active_turnover) / observations,
        "mean_daily_turnover": float(numeric_turnover.mean()),
        "annualised_turnover": float(
            numeric_turnover.sum() * periods_per_year / observations
        ),
        "mean_rebalance_turnover": float(active_turnover.mean()),
        "median_rebalance_turnover": float(active_turnover.median()),
        "p90_rebalance_turnover": float(active_turnover.quantile(0.90)),
        "p95_rebalance_turnover": float(active_turnover.quantile(0.95)),
        "p99_rebalance_turnover": float(active_turnover.quantile(0.99)),
        "maximum_daily_turnover": float(numeric_turnover.max()),
        f"maximum_{concentration_window}_day_turnover": float(
            numeric_turnover.rolling(
                concentration_window,
                min_periods=1,
            )
            .sum()
            .max()
        ),
        "top_1_day_turnover_share": calculate_top_n_turnover_share(
            numeric_turnover,
            1,
            tolerance=tolerance,
        ),
        "top_5_day_turnover_share": calculate_top_n_turnover_share(
            numeric_turnover,
            5,
            tolerance=tolerance,
        ),
        "top_10_day_turnover_share": calculate_top_n_turnover_share(
            numeric_turnover,
            10,
            tolerance=tolerance,
        ),
    }


def prepare_lagged_dollar_volume(
    market_data: pd.DataFrame,
    window: int = 21,
    min_periods: int | None = None,
    aggregation: Literal["mean", "median"] = "mean",
    date_column: str = "date",
    ticker_column: str = "ticker",
    price_column: str = "close",
    volume_column: str = "volume",
    dollar_volume_column: str = "dollar_volume",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Add a one-period-lagged rolling dollar-volume estimate.

    If ``dollar_volume_column`` is absent, dollar volume is constructed from
    price times volume. The final shift prevents same-day liquidity from
    affecting the capacity assigned to a trade.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    if min_periods is None:
        min_periods = window

    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must be between 1 and window.")

    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'.")

    required_columns = {date_column, ticker_column}

    if dollar_volume_column not in market_data.columns:
        required_columns.update({price_column, volume_column})

    missing_columns = required_columns - set(market_data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = market_data.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    if result[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")

    result[ticker_column] = result[ticker_column].astype(str)

    if result[[date_column, ticker_column]].duplicated().any():
        raise ValueError("Duplicate date/ticker liquidity rows found.")

    if dollar_volume_column not in result.columns:
        price = _as_numeric_series(
            result[price_column],
            name=price_column,
            allow_missing=True,
        )
        volume = _as_numeric_series(
            result[volume_column],
            name=volume_column,
            allow_missing=True,
        )

        if price.dropna().lt(0.0).any() or volume.dropna().lt(0.0).any():
            raise ValueError("Price and volume must be non-negative.")

        result[dollar_volume_column] = price * volume
    else:
        result[dollar_volume_column] = _as_numeric_series(
            result[dollar_volume_column],
            name=dollar_volume_column,
            allow_missing=True,
        )

        if result[dollar_volume_column].dropna().lt(0.0).any():
            raise ValueError("Dollar volume must be non-negative.")

    result = result.sort_values([ticker_column, date_column]).reset_index(drop=True)

    if output_column is None:
        output_column = f"lagged_{aggregation}_dollar_volume_{window}"

    grouped_dollar_volume = result.groupby(
        ticker_column,
        sort=False,
    )[dollar_volume_column]

    if aggregation == "mean":
        result[output_column] = grouped_dollar_volume.transform(
            lambda values: values.rolling(
                window,
                min_periods=min_periods,
            )
            .mean()
            .shift(1)
        )
    else:
        result[output_column] = grouped_dollar_volume.transform(
            lambda values: values.rolling(
                window,
                min_periods=min_periods,
            )
            .median()
            .shift(1)
        )

    return result


def calculate_security_trade_capacity(
    trades: pd.DataFrame,
    participation_rate: float,
    trade_weight_column: str = "absolute_trade_weight",
    liquidity_column: str = "lagged_adv_21",
    output_column: str = "trade_capacity_usd",
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Add implied capital capacity for each security-level trade."""
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("participation_rate must be in (0, 1].")

    tolerance = _validate_non_negative_rate(tolerance, "tolerance")
    required_columns = {trade_weight_column, liquidity_column}
    missing_columns = required_columns - set(trades.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = trades.copy()
    result[trade_weight_column] = _as_numeric_series(
        result[trade_weight_column],
        name=trade_weight_column,
        allow_missing=False,
    )
    result[liquidity_column] = _as_numeric_series(
        result[liquidity_column],
        name=liquidity_column,
        allow_missing=True,
    )

    if result[trade_weight_column].lt(-tolerance).any():
        raise ValueError("Trade weights must be non-negative.")

    if result[liquidity_column].dropna().lt(0.0).any():
        raise ValueError("Liquidity must be non-negative.")

    active_trade = result[trade_weight_column].gt(tolerance)
    valid_liquidity = result[liquidity_column].gt(0.0)

    result[output_column] = np.where(
        active_trade & valid_liquidity,
        (participation_rate * result[liquidity_column] / result[trade_weight_column]),
        np.nan,
    )

    return result


def calculate_daily_trade_capacity(
    trades: pd.DataFrame,
    participation_limits: Sequence[float] = CAPACITY_PARTICIPATION_LIMITS,
    group_columns: Sequence[str] = (
        "portfolio",
        "rebalance_frequency",
        "rebalance_offset",
        "date",
    ),
    ticker_column: str = "ticker",
    trade_weight_column: str = "absolute_trade_weight",
    liquidity_column: str = "lagged_adv_21",
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Calculate strict full-coverage daily capacity as in Notebook 06."""
    tolerance = _validate_non_negative_rate(tolerance, "tolerance")
    participation_limits = tuple(float(value) for value in participation_limits)

    if not participation_limits:
        raise ValueError("participation_limits is empty.")

    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in participation_limits
    ):
        raise ValueError("Participation limits must be finite and in (0, 1].")

    if len(set(participation_limits)) != len(participation_limits):
        raise ValueError("participation_limits contains duplicates.")

    required_columns = {
        *group_columns,
        ticker_column,
        trade_weight_column,
        liquidity_column,
    }
    missing_columns = required_columns - set(trades.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = trades.copy()
    prepared[trade_weight_column] = _as_numeric_series(
        prepared[trade_weight_column],
        name=trade_weight_column,
        allow_missing=False,
    )
    prepared[liquidity_column] = _as_numeric_series(
        prepared[liquidity_column],
        name=liquidity_column,
        allow_missing=True,
    )

    if prepared[trade_weight_column].lt(-tolerance).any():
        raise ValueError("Trade weights must be non-negative.")

    if prepared[liquidity_column].dropna().lt(0.0).any():
        raise ValueError("Liquidity must be non-negative.")

    duplicate_key = [*group_columns, ticker_column]

    if prepared[duplicate_key].duplicated().any():
        raise ValueError("Duplicate group/ticker trade rows found.")

    prepared = prepared.loc[prepared[trade_weight_column].gt(tolerance)].copy()

    output_columns = [
        *group_columns,
        "traded_security_count",
        "fully_covered",
        "largest_security_trade_share",
        "top_5_security_trade_share",
        "effective_traded_security_count",
        "participation_limit",
        "capacity_usd",
        "bottleneck_ticker",
    ]

    if prepared.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, object]] = []

    for group_values, group in prepared.groupby(
        list(group_columns),
        sort=True,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_metadata = dict(zip(group_columns, group_values, strict=True))
        valid_liquidity = group[liquidity_column].notna() & group[liquidity_column].gt(
            0.0
        )
        fully_covered = bool(valid_liquidity.all())
        trade_share = group[trade_weight_column] / group[trade_weight_column].sum()

        common_values = {
            **group_metadata,
            "traded_security_count": len(group),
            "fully_covered": fully_covered,
            "largest_security_trade_share": float(trade_share.max()),
            "top_5_security_trade_share": float(trade_share.nlargest(5).sum()),
            "effective_traded_security_count": float(1.0 / trade_share.pow(2).sum()),
        }

        for participation_limit in participation_limits:
            capacity = np.nan
            bottleneck_ticker: object = None

            if fully_covered:
                security_capacity = (
                    participation_limit
                    * group[liquidity_column]
                    / group[trade_weight_column]
                )
                bottleneck_index = security_capacity.idxmin()
                capacity = float(security_capacity.loc[bottleneck_index])
                bottleneck_ticker = group.loc[bottleneck_index, ticker_column]

            rows.append(
                {
                    **common_values,
                    "participation_limit": participation_limit,
                    "capacity_usd": capacity,
                    "bottleneck_ticker": bottleneck_ticker,
                }
            )

    return (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values([*group_columns, "participation_limit"])
        .reset_index(drop=True)
    )


def summarise_capacity_by_offset(
    capacity_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise capacity within each portfolio-frequency-offset variant."""
    group_columns = [
        "portfolio",
        "rebalance_frequency",
        "rebalance_offset",
        "participation_limit",
    ]
    required_columns = {
        *group_columns,
        "date",
        "fully_covered",
        "capacity_usd",
        "effective_traded_security_count",
        "largest_security_trade_share",
        "top_5_security_trade_share",
    }
    missing_columns = required_columns - set(capacity_daily.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return (
        capacity_daily.groupby(group_columns, sort=True)
        .agg(
            rebalance_days=("date", "nunique"),
            fully_covered_fraction=("fully_covered", "mean"),
            minimum_capacity_usd=("capacity_usd", "min"),
            fifth_percentile_capacity_usd=(
                "capacity_usd",
                lambda values: values.quantile(0.05),
            ),
            tenth_percentile_capacity_usd=(
                "capacity_usd",
                lambda values: values.quantile(0.10),
            ),
            median_capacity_usd=("capacity_usd", "median"),
            median_effective_traded_names=(
                "effective_traded_security_count",
                "median",
            ),
            maximum_largest_security_trade_share=(
                "largest_security_trade_share",
                "max",
            ),
            maximum_top_5_security_trade_share=(
                "top_5_security_trade_share",
                "max",
            ),
        )
        .reset_index()
    )


def summarise_capacity_across_offsets(
    capacity_by_offset: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate offset-level capacity into the Notebook 06 phase summary."""
    group_columns = [
        "portfolio",
        "rebalance_frequency",
        "participation_limit",
    ]
    required_columns = {
        *group_columns,
        "rebalance_offset",
        "fully_covered_fraction",
        "minimum_capacity_usd",
        "fifth_percentile_capacity_usd",
        "median_capacity_usd",
    }
    missing_columns = required_columns - set(capacity_by_offset.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return (
        capacity_by_offset.groupby(group_columns, sort=True)
        .agg(
            offset_count=("rebalance_offset", "nunique"),
            minimum_adv_coverage=("fully_covered_fraction", "min"),
            median_fifth_percentile_capacity_usd=(
                "fifth_percentile_capacity_usd",
                "median",
            ),
            worst_phase_fifth_percentile_capacity_usd=(
                "fifth_percentile_capacity_usd",
                "min",
            ),
            median_capacity_usd=("median_capacity_usd", "median"),
            worst_historical_capacity_usd=("minimum_capacity_usd", "min"),
        )
        .reset_index()
    )
