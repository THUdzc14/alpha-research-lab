from __future__ import annotations

import pandas as pd

from alpha_research.backtest import (
    BacktestConfig,
    construct_long_short_weights,
    get_rebalance_dates,
)


def build_factor_target_weights(
    panel: pd.DataFrame,
    factor_column: str,
    return_column: str = "forward_ret_1d",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Build dated long-short target weights from one factor.

    A target portfolio is constructed on each scheduled rebalance date.
    High factor values are held long and low factor values are held short,
    following the quantile and exposure settings in ``config``.

    Only dates containing at least one usable forward return are included,
    matching the date convention of the backtest engine.
    """
    if config is None:
        config = BacktestConfig()

    required_columns = {
        "date",
        "ticker",
        factor_column,
        return_column,
    }
    missing_columns = required_columns - set(panel.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = panel[["date", "ticker", factor_column, return_column]].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df[factor_column] = pd.to_numeric(
        df[factor_column],
        errors="coerce",
    )
    df[return_column] = pd.to_numeric(
        df[return_column],
        errors="coerce",
    )

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows found.")

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    valid_return_dates = df.groupby("date")[return_column].apply(
        lambda values: values.notna().any()
    )
    valid_return_dates = pd.DatetimeIndex(valid_return_dates[valid_return_dates].index)

    if valid_return_dates.empty:
        raise ValueError("No dates contain usable asset returns.")

    df = df.loc[df["date"].isin(valid_return_dates)].copy()

    rebalance_dates = get_rebalance_dates(
        valid_return_dates,
        frequency=config.rebalance_frequency,
        offset=config.rebalance_offset,
    )

    targets: list[pd.DataFrame] = []

    for date in rebalance_dates:
        cross_section = df.loc[df["date"] == date]

        weights = construct_long_short_weights(
            cross_section=cross_section,
            factor_column=factor_column,
            quantiles=config.quantiles,
            long_quantile=config.long_quantile,
            short_quantile=config.short_quantile,
            long_gross=config.long_gross,
            short_gross=config.short_gross,
            min_observations=config.min_observations,
        )

        date_targets = weights.rename_axis("ticker").reset_index(name="weight")
        date_targets.insert(0, "date", date)

        targets.append(date_targets)

    if not targets:
        return pd.DataFrame(columns=["date", "ticker", "weight"])

    return (
        pd.concat(targets, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
