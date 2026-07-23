from __future__ import annotations

import numpy as np
import pandas as pd


def winsorise_cross_section(
    panel: pd.DataFrame,
    column: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Winsorise one factor independently on each date.

    Values below the lower quantile are clipped upward.
    Values above the upper quantile are clipped downward.
    Missing values remain missing.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")

    if column not in panel.columns:
        raise ValueError(f"Column not found: {column}")

    def _winsorise(group: pd.Series) -> pd.Series:
        valid = group.dropna()

        if valid.empty:
            return group

        lower = valid.quantile(lower_quantile)
        upper = valid.quantile(upper_quantile)

        return group.clip(lower=lower, upper=upper)

    return panel.groupby("date")[column].transform(_winsorise)


def cross_sectional_zscore(
    panel: pd.DataFrame,
    column: str,
    ddof: int = 0,
) -> pd.Series:
    """Calculate a cross-sectional z-score independently on each date.

    Uses population standard deviation by default, ddof=0.
    Dates with zero cross-sectional dispersion return NaN.
    """
    if column not in panel.columns:
        raise ValueError(f"Column not found: {column}")

    grouped = panel.groupby("date")[column]

    means = grouped.transform("mean")
    stds = grouped.transform(lambda x: x.std(ddof=ddof))

    zscore = (panel[column] - means) / stds
    zscore = zscore.where(stds > 0)

    return zscore


def cross_sectional_rank(
    panel: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Calculate percentile ranks independently on each date.

    Rank values lie in (0, 1], with higher raw values receiving higher ranks.
    Missing values remain missing.
    """
    if column not in panel.columns:
        raise ValueError(f"Column not found: {column}")

    return panel.groupby("date")[column].rank(
        method="average",
        pct=True,
        na_option="keep",
    )


def process_factor(
    panel: pd.DataFrame,
    raw_column: str,
    output_prefix: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Add winsorised, z-score and percentile-rank versions of one factor."""
    df = panel.copy()

    winsorised_column = f"{output_prefix}_winsorised"
    zscore_column = f"{output_prefix}_z"
    rank_column = f"{output_prefix}_rank"

    df[winsorised_column] = winsorise_cross_section(
        df,
        column=raw_column,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    df[zscore_column] = cross_sectional_zscore(
        df,
        column=winsorised_column,
    )

    df[rank_column] = cross_sectional_rank(
        df,
        column=winsorised_column,
    )

    return df


def process_factor_columns(
    panel: pd.DataFrame,
    factor_map: dict[str, str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Process multiple raw factors.

    Args:
        panel:
            Factor panel containing date and raw factor columns.
        factor_map:
            Mapping from raw column to output prefix.

            Example:
                {
                    "mom_12_1m_raw": "mom_12_1m",
                    "mom_3m_raw": "mom_3m",
                }
    """
    df = panel.copy()

    for raw_column, output_prefix in factor_map.items():
        df = process_factor(
            df,
            raw_column=raw_column,
            output_prefix=output_prefix,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    return df


def grouped_zscore(
    panel: pd.DataFrame,
    column: str,
    group_columns: list[str],
    ddof: int = 0,
    min_observations: int = 3,
) -> pd.Series:
    """Calculate z-scores within specified cross-sectional groups.

    Example:
        group_columns=["date", "sector"]

    Groups with fewer than `min_observations`, or zero dispersion,
    receive NaN.
    """
    required = {column, *group_columns}
    missing = required - set(panel.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _zscore(group: pd.Series) -> pd.Series:
        valid = group.dropna()

        result = pd.Series(
            np.nan,
            index=group.index,
            dtype="float64",
        )

        if len(valid) < min_observations:
            return result

        std = valid.std(ddof=ddof)

        if pd.isna(std) or std <= 0:
            return result

        result.loc[valid.index] = (valid - valid.mean()) / std

        return result

    return panel.groupby(group_columns)[column].transform(_zscore)


def add_sector_neutral_factor(
    panel: pd.DataFrame,
    factor_column: str,
    output_column: str,
    sector_column: str = "sector",
    min_sector_observations: int = 3,
) -> pd.DataFrame:
    """Add a factor z-score calculated within each date and sector."""
    df = panel.copy()

    df[output_column] = grouped_zscore(
        df,
        column=factor_column,
        group_columns=["date", sector_column],
        ddof=0,
        min_observations=min_sector_observations,
    )

    return df
