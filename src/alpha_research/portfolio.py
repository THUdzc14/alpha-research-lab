from __future__ import annotations

import math
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


def combine_sleeve_target_weights(
    sleeve_targets: dict[str, pd.DataFrame],
    sleeve_allocations: dict[str, float],
) -> pd.DataFrame:
    """Combine independently constructed portfolio sleeves.

    Each sleeve's target weights are multiplied by its capital allocation and
    then summed by date and ticker. Opposing positions naturally offset.

    All sleeves must use the same rebalance dates, and allocations must be
    non-negative and sum to one.
    """
    if not sleeve_targets:
        raise ValueError("sleeve_targets is empty.")

    if set(sleeve_targets) != set(sleeve_allocations):
        raise ValueError(
            "sleeve_targets and sleeve_allocations must have identical keys."
        )

    allocations = {
        name: float(allocation) for name, allocation in sleeve_allocations.items()
    }

    if any(
        not math.isfinite(allocation) or allocation < 0
        for allocation in allocations.values()
    ):
        raise ValueError("Sleeve allocations must be finite and non-negative.")

    if not math.isclose(
        sum(allocations.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Sleeve allocations must sum to one.")

    weighted_frames = []
    reference_dates: pd.DatetimeIndex | None = None

    for sleeve_name, targets in sleeve_targets.items():
        required_columns = {"date", "ticker", "weight"}
        missing_columns = required_columns - set(targets.columns)

        if missing_columns:
            raise ValueError(
                f"{sleeve_name} is missing columns: " f"{sorted(missing_columns)}"
            )

        frame = targets[["date", "ticker", "weight"]].copy()

        frame["date"] = pd.to_datetime(frame["date"])
        frame["ticker"] = frame["ticker"].astype(str)
        frame["weight"] = pd.to_numeric(
            frame["weight"],
            errors="coerce",
        )

        if frame[["date", "ticker"]].duplicated().any():
            raise ValueError(f"{sleeve_name} contains duplicate date/ticker rows.")

        if frame["weight"].isna().any() or not frame["weight"].map(math.isfinite).all():
            raise ValueError(f"{sleeve_name} contains invalid weights.")

        sleeve_dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()

        if reference_dates is None:
            reference_dates = sleeve_dates
        elif not sleeve_dates.equals(reference_dates):
            raise ValueError("All sleeves must use identical rebalance dates.")

        frame["weight"] *= allocations[sleeve_name]
        weighted_frames.append(frame)

    return (
        pd.concat(weighted_frames, ignore_index=True)
        .groupby(["date", "ticker"], as_index=False)["weight"]
        .sum()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def combine_factor_scores(
    panel: pd.DataFrame,
    factor_weights: dict[str, float],
) -> pd.Series:
    """Combine comparable factor scores into one weighted score.

    A row receives a composite score only when all component factor
    scores are available. Factor weights must be non-negative and sum
    to one.
    """
    if not factor_weights:
        raise ValueError("factor_weights is empty.")

    missing_columns = set(factor_weights) - set(panel.columns)

    if missing_columns:
        raise ValueError(f"Missing factor columns: {sorted(missing_columns)}")

    weights = {column: float(weight) for column, weight in factor_weights.items()}

    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Factor weights must be finite and non-negative.")

    if not math.isclose(
        sum(weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Factor weights must sum to one.")

    factor_columns = list(weights)

    scores = panel[factor_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    weighted_scores = scores.mul(
        pd.Series(weights),
        axis="columns",
    )

    return weighted_scores.sum(
        axis=1,
        min_count=len(factor_columns),
    ).rename("composite_factor_score")


def rescale_target_weights_to_gross(
    target_weights: pd.DataFrame,
    target_gross: float | pd.Series,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Rescale each date's target weights to a specified gross exposure.

    target_gross may be either:

    - one constant applied to every date; or
    - a date-indexed Series specifying the desired gross exposure by date.

    Relative stock weights, position signs, and net-exposure proportions are
    preserved. A zero-weight portfolio cannot be scaled to positive exposure.
    """
    required_columns = {"date", "ticker", "weight"}
    missing_columns = required_columns - set(target_weights.columns)

    if missing_columns:
        raise ValueError(
            f"target_weights is missing columns: {sorted(missing_columns)}"
        )

    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative.")

    scaled = target_weights.copy()
    scaled["date"] = pd.to_datetime(scaled["date"])
    scaled["ticker"] = scaled["ticker"].astype(str)
    scaled["weight"] = pd.to_numeric(
        scaled["weight"],
        errors="coerce",
    )

    if scaled[["date", "ticker"]].duplicated().any():
        raise ValueError("target_weights contains duplicate date/ticker rows.")

    if scaled["weight"].isna().any() or not scaled["weight"].map(math.isfinite).all():
        raise ValueError("target_weights contains invalid weights.")

    dates = pd.DatetimeIndex(scaled["date"].drop_duplicates()).sort_values()

    if isinstance(target_gross, pd.Series):
        desired_gross = target_gross.copy()
        desired_gross.index = pd.to_datetime(desired_gross.index)

        if desired_gross.index.duplicated().any():
            raise ValueError("target_gross contains duplicate dates.")

        missing_dates = dates.difference(desired_gross.index)

        if not missing_dates.empty:
            raise ValueError("target_gross is missing target-weight dates.")

        desired_gross = pd.to_numeric(
            desired_gross.reindex(dates),
            errors="coerce",
        )
    else:
        desired_value = float(target_gross)
        desired_gross = pd.Series(
            desired_value,
            index=dates,
            dtype=float,
        )

    if (
        desired_gross.isna().any()
        or not desired_gross.map(math.isfinite).all()
        or (desired_gross < 0).any()
    ):
        raise ValueError("Target gross exposures must be finite and non-negative.")

    current_gross = (
        scaled.groupby("date")["weight"]
        .agg(lambda weights: weights.abs().sum())
        .reindex(dates)
    )

    impossible_dates = (current_gross <= tolerance) & (desired_gross > tolerance)

    if impossible_dates.any():
        raise ValueError(
            "Cannot scale a zero-weight portfolio to positive gross exposure."
        )

    scale_factors = pd.Series(
        0.0,
        index=dates,
        dtype=float,
    )

    scalable_dates = current_gross > tolerance

    scale_factors.loc[scalable_dates] = (
        desired_gross.loc[scalable_dates] / current_gross.loc[scalable_dates]
    )

    scaled["weight"] *= scaled["date"].map(scale_factors)

    return scaled.sort_values(["date", "ticker"]).reset_index(drop=True)
