from __future__ import annotations

import math
import numpy as np
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


def estimate_trailing_sleeve_volatility(
    sleeve_returns: pd.DataFrame,
    lookback: int = 63,
    min_periods: int = 42,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate trailing sleeve volatility without look-ahead bias.

    The rolling estimate is shifted by one observation, so the volatility
    assigned to date t uses returns available only through date t - 1.
    """
    if sleeve_returns.empty:
        raise ValueError("sleeve_returns is empty.")

    if not isinstance(sleeve_returns.index, pd.DatetimeIndex):
        raise ValueError("sleeve_returns must have a DatetimeIndex.")

    if sleeve_returns.index.duplicated().any():
        raise ValueError("sleeve_returns contains duplicate dates.")

    if lookback < 2:
        raise ValueError("lookback must be at least 2.")

    if min_periods < 2 or min_periods > lookback:
        raise ValueError("min_periods must be between 2 and lookback.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    returns = sleeve_returns.sort_index().apply(pd.to_numeric, errors="coerce")

    if np.isinf(returns.to_numpy()).any():
        raise ValueError("sleeve_returns contains infinite values.")

    return (
        returns.rolling(
            window=lookback,
            min_periods=min_periods,
        )
        .std(ddof=1)
        .shift(1)
        .mul(math.sqrt(periods_per_year))
    )


def calculate_rebalance_inverse_volatility_allocations(
    sleeve_returns: pd.DataFrame,
    rebalance_dates: pd.Index,
    lookback: int = 63,
    min_periods: int = 42,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate pure inverse-volatility weights on rebalance dates.

    Each allocation uses the latest jointly observed sleeve returns strictly
    before its rebalance date.  Equal allocations are used until enough joint
    history is available.  This is the exact convention retained from
    Notebook 06.
    """
    if sleeve_returns.empty:
        raise ValueError("sleeve_returns is empty.")

    if not isinstance(sleeve_returns.index, pd.DatetimeIndex):
        raise ValueError("sleeve_returns must have a DatetimeIndex.")

    returns = sleeve_returns.sort_index().apply(
        pd.to_numeric,
        errors="coerce",
    )

    if returns.index.duplicated().any():
        raise ValueError("sleeve_returns contains duplicate dates.")

    if np.isinf(returns.to_numpy()).any():
        raise ValueError("sleeve_returns contains infinite values.")

    if returns.shape[1] < 2:
        raise ValueError("At least two sleeves are required.")

    if lookback < 2:
        raise ValueError("lookback must be at least 2.")

    if min_periods < 2 or min_periods > lookback:
        raise ValueError("min_periods must be between 2 and lookback.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(rebalance_dates).dropna()))

    if dates.empty:
        raise ValueError("rebalance_dates is empty.")

    if dates.duplicated().any():
        raise ValueError("rebalance_dates contains duplicates.")

    dates = dates.sort_values()
    allocations = pd.DataFrame(
        1.0 / returns.shape[1],
        index=dates,
        columns=returns.columns,
        dtype=float,
    )
    allocations.index.name = "date"

    for date in dates:
        history = returns.loc[returns.index < date].dropna(how="any").tail(lookback)

        if len(history) < min_periods:
            continue

        volatility = history.std(ddof=1) * math.sqrt(periods_per_year)

        if (
            volatility.isna().any()
            or not np.isfinite(volatility.to_numpy()).all()
            or volatility.le(0.0).any()
        ):
            continue

        inverse_volatility = 1.0 / volatility
        allocations.loc[date] = inverse_volatility / inverse_volatility.sum()

    if not np.allclose(
        allocations.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Inverse-volatility allocations do not sum to one.")

    return allocations


def calculate_inverse_volatility_allocations(
    sleeve_volatility: pd.DataFrame,
    allocation_floor: float = 0.20,
    volatility_floor: float = 1e-8,
) -> pd.DataFrame:
    """Convert sleeve volatilities into bounded inverse-vol allocations.

    Before valid volatility estimates exist for every sleeve, equal
    allocations are used.

    allocation_floor reserves a minimum allocation for every sleeve.
    The remaining capital is distributed using inverse volatility.
    """
    if sleeve_volatility.empty:
        raise ValueError("sleeve_volatility is empty.")

    volatility = sleeve_volatility.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if np.isinf(volatility.to_numpy()).any():
        raise ValueError("sleeve_volatility contains infinite values.")

    if (volatility.dropna() < 0.0).any().any():
        raise ValueError("Sleeve volatilities cannot be negative.")

    sleeve_count = volatility.shape[1]

    if sleeve_count < 2:
        raise ValueError("At least two sleeves are required.")

    if (
        not math.isfinite(allocation_floor)
        or allocation_floor < 0.0
        or allocation_floor > 1.0 / sleeve_count
    ):
        raise ValueError(
            "allocation_floor must be between zero and " "1 / number of sleeves."
        )

    if not math.isfinite(volatility_floor) or volatility_floor <= 0.0:
        raise ValueError("volatility_floor must be finite and positive.")

    equal_weight = 1.0 / sleeve_count

    raw_allocations = pd.DataFrame(
        equal_weight,
        index=volatility.index,
        columns=volatility.columns,
        dtype=float,
    )

    valid_dates = volatility.notna().all(axis=1) & (volatility > volatility_floor).all(
        axis=1
    )

    inverse_volatility = 1.0 / volatility.loc[valid_dates].clip(lower=volatility_floor)

    raw_allocations.loc[valid_dates] = inverse_volatility.div(
        inverse_volatility.sum(axis=1),
        axis=0,
    )

    remaining_allocation = 1.0 - sleeve_count * allocation_floor

    allocations = allocation_floor + remaining_allocation * raw_allocations

    return allocations


def combine_dynamic_sleeve_target_weights(
    sleeve_targets: dict[str, pd.DataFrame],
    sleeve_allocations: pd.DataFrame,
) -> pd.DataFrame:
    """Combine sleeve targets using date-specific allocations.

    sleeve_allocations must have rebalance dates as its index and one
    column for every sleeve. Allocations must sum to one on each date.
    """
    if not sleeve_targets:
        raise ValueError("sleeve_targets is empty.")

    allocations = sleeve_allocations.copy()

    if not isinstance(allocations.index, pd.DatetimeIndex):
        allocations.index = pd.to_datetime(allocations.index)

    allocations = allocations.sort_index()

    if allocations.index.duplicated().any():
        raise ValueError("sleeve_allocations contains duplicate dates.")

    if set(allocations.columns) != set(sleeve_targets):
        raise ValueError("Allocation columns and sleeve target names " "must match.")

    allocations = allocations.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if (
        allocations.isna().any().any()
        or not np.isfinite(allocations.to_numpy()).all()
        or (allocations < 0.0).any().any()
    ):
        raise ValueError("Sleeve allocations must be finite and non-negative.")

    if not np.allclose(
        allocations.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Sleeve allocations must sum to one on every date.")

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
            raise ValueError(f"{sleeve_name} contains duplicate " "date/ticker rows.")

        if (
            frame["weight"].isna().any()
            or not np.isfinite(frame["weight"].to_numpy()).all()
        ):
            raise ValueError(f"{sleeve_name} contains invalid weights.")

        sleeve_dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()

        if reference_dates is None:
            reference_dates = sleeve_dates
        elif not sleeve_dates.equals(reference_dates):
            raise ValueError("All sleeves must use identical rebalance dates.")

        missing_allocation_dates = sleeve_dates.difference(allocations.index)

        if not missing_allocation_dates.empty:
            raise ValueError("sleeve_allocations is missing rebalance dates.")

        frame["weight"] *= frame["date"].map(allocations[sleeve_name])

        weighted_frames.append(frame)

    assert reference_dates is not None

    extra_allocation_dates = allocations.index.difference(reference_dates)

    if not extra_allocation_dates.empty:
        raise ValueError("sleeve_allocations contains unexpected dates.")

    return (
        pd.concat(weighted_frames, ignore_index=True)
        .groupby(["date", "ticker"], as_index=False)["weight"]
        .sum()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
