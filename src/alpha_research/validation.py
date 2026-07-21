from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_daily_ic(
    panel: pd.DataFrame,
    factor_column: str,
    forward_return_column: str,
    method: str = "spearman",
    min_observations: int = 20,
) -> pd.DataFrame:
    """Calculate cross-sectional information coefficient by date.

    Args:
        panel:
            DataFrame containing date, factor and forward-return columns.
        factor_column:
            Factor score observed on date t.
        forward_return_column:
            Future return beginning after date t.
        method:
            "pearson" or "spearman".
        min_observations:
            Minimum valid stocks required on a date.

    Returns:
        DataFrame with:
            date, ic, observations
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    required = {"date", factor_column, forward_return_column}
    missing = required - set(panel.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = panel[["date", factor_column, forward_return_column]].copy()

    df["date"] = pd.to_datetime(df["date"])

    results = []

    for date, group in df.groupby("date"):
        valid = group.dropna(subset=[factor_column, forward_return_column])

        observations = len(valid)

        if observations < min_observations:
            ic = np.nan
        else:
            ic = valid[factor_column].corr(
                valid[forward_return_column],
                method=method,
            )

        results.append(
            {
                "date": date,
                "ic": ic,
                "observations": observations,
            }
        )

    return pd.DataFrame(results).sort_values("date").reset_index(drop=True)


def summarise_ic(ic_series: pd.DataFrame) -> pd.DataFrame:
    """Summarise an IC time series.

    The t-statistic is:

        mean(IC) / [std(IC) / sqrt(number of observations)]

    This is a simple IID t-statistic. Later, we may replace it with a
    Newey-West adjusted statistic because IC observations can be serially
    correlated, especially for overlapping forward returns.
    """
    if "ic" not in ic_series.columns:
        raise ValueError("Input must contain an 'ic' column.")

    valid = ic_series["ic"].dropna()
    count = len(valid)

    if count == 0:
        return pd.DataFrame(
            [
                {
                    "count": 0,
                    "mean_ic": np.nan,
                    "std_ic": np.nan,
                    "ic_ir": np.nan,
                    "t_stat": np.nan,
                    "positive_fraction": np.nan,
                }
            ]
        )

    mean_ic = valid.mean()
    std_ic = valid.std(ddof=1)

    if count > 1 and std_ic > 0:
        ic_ir = mean_ic / std_ic
        t_stat = mean_ic / (std_ic / np.sqrt(count))
    else:
        ic_ir = np.nan
        t_stat = np.nan

    return pd.DataFrame(
        [
            {
                "count": count,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "t_stat": t_stat,
                "positive_fraction": (valid > 0).mean(),
            }
        ]
    )


def assign_quantiles(
    panel: pd.DataFrame,
    factor_column: str,
    quantiles: int = 5,
    min_observations: int | None = None,
) -> pd.Series:
    """Assign factor quantiles independently on each date.

    Quantile 1 contains the lowest factor scores.
    Quantile `quantiles` contains the highest factor scores.
    """
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")

    if factor_column not in panel.columns:
        raise ValueError(f"Column not found: {factor_column}")

    if min_observations is None:
        min_observations = quantiles * 2

    def _assign(group: pd.Series) -> pd.Series:
        result = pd.Series(
            np.nan,
            index=group.index,
            dtype="float64",
        )

        valid = group.dropna()

        if len(valid) < min_observations:
            return result

        # Rank first to avoid qcut failures caused by duplicate values.
        ranks = valid.rank(method="first")

        result.loc[valid.index] = (
            pd.qcut(
                ranks,
                q=quantiles,
                labels=False,
            ).astype(float)
            + 1.0
        )

        return result

    return panel.groupby("date")[factor_column].transform(_assign)


def calculate_quantile_returns(
    panel: pd.DataFrame,
    factor_column: str,
    forward_return_column: str,
    quantiles: int = 5,
    min_observations: int | None = None,
) -> pd.DataFrame:
    """Calculate equal-weight forward returns by factor quantile and date."""
    required = {"date", factor_column, forward_return_column}
    missing = required - set(panel.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = panel[["date", "ticker", factor_column, forward_return_column]].copy()

    df["date"] = pd.to_datetime(df["date"])

    df["quantile"] = assign_quantiles(
        df,
        factor_column=factor_column,
        quantiles=quantiles,
        min_observations=min_observations,
    )

    valid = df.dropna(subset=["quantile", forward_return_column]).copy()

    valid["quantile"] = valid["quantile"].astype(int)

    return (
        valid.groupby(["date", "quantile"])
        .agg(
            mean_forward_return=(forward_return_column, "mean"),
            observations=(forward_return_column, "count"),
        )
        .reset_index()
        .sort_values(["date", "quantile"])
        .reset_index(drop=True)
    )


def calculate_long_short_spread(
    quantile_returns: pd.DataFrame,
    top_quantile: int,
    bottom_quantile: int = 1,
) -> pd.DataFrame:
    """Calculate top-minus-bottom quantile return by date."""
    required = {"date", "quantile", "mean_forward_return"}
    missing = required - set(quantile_returns.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pivot = quantile_returns.pivot(
        index="date",
        columns="quantile",
        values="mean_forward_return",
    )

    if bottom_quantile not in pivot.columns:
        raise ValueError(f"Bottom quantile {bottom_quantile} not found.")

    if top_quantile not in pivot.columns:
        raise ValueError(f"Top quantile {top_quantile} not found.")

    spread = (pivot[top_quantile] - pivot[bottom_quantile]).rename("long_short_return")

    return spread.reset_index()


def calculate_ic_by_horizon(
    panel: pd.DataFrame,
    factor_column: str,
    forward_return_columns: list[str],
    method: str = "spearman",
    min_observations: int = 20,
) -> pd.DataFrame:
    """Calculate and summarise IC across several forward-return horizons."""
    summaries = []

    for return_column in forward_return_columns:
        daily_ic = calculate_daily_ic(
            panel=panel,
            factor_column=factor_column,
            forward_return_column=return_column,
            method=method,
            min_observations=min_observations,
        )

        summary = summarise_ic(daily_ic).iloc[0].to_dict()
        summary["forward_return_column"] = return_column

        summaries.append(summary)

    return pd.DataFrame(summaries)


def calculate_subperiod_ic(
    panel: pd.DataFrame,
    factor_column: str,
    forward_return_column: str,
    periods: dict[str, tuple[str, str]],
    method: str = "spearman",
    min_observations: int = 20,
) -> pd.DataFrame:
    """Calculate IC statistics separately for specified date ranges.

    Example:
        periods = {
            "2015-2018": ("2015-01-01", "2018-12-31"),
            "2019-2022": ("2019-01-01", "2022-12-31"),
        }
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    summaries = []

    for period_name, (start_date, end_date) in periods.items():
        subset = df.loc[
            df["date"].between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
            )
        ]

        daily_ic = calculate_daily_ic(
            panel=subset,
            factor_column=factor_column,
            forward_return_column=forward_return_column,
            method=method,
            min_observations=min_observations,
        )

        summary = summarise_ic(daily_ic).iloc[0].to_dict()
        summary["period"] = period_name
        summary["start_date"] = pd.Timestamp(start_date)
        summary["end_date"] = pd.Timestamp(end_date)

        summaries.append(summary)

    return pd.DataFrame(summaries)


def select_non_overlapping_dates(
    panel: pd.DataFrame,
    step: int,
    date_column: str = "date",
    offset: int = 0,
) -> pd.DataFrame:
    """Keep every `step`-th unique date.

    For a 5-day forward return, setting step=5 produces approximately
    non-overlapping prediction periods.

    Different offsets can be tested because the result may depend on which
    weekday or starting date is selected.
    """
    if step <= 0:
        raise ValueError("step must be positive.")

    if not 0 <= offset < step:
        raise ValueError("offset must satisfy 0 <= offset < step.")

    if date_column not in panel.columns:
        raise ValueError(f"Column not found: {date_column}")

    df = panel.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    dates = np.sort(df[date_column].dropna().unique())
    selected_dates = dates[offset::step]

    return df.loc[df[date_column].isin(selected_dates)].copy()


def calculate_non_overlapping_ic(
    panel: pd.DataFrame,
    factor_column: str,
    forward_return_column: str,
    horizon: int,
    method: str = "spearman",
    min_observations: int = 20,
) -> pd.DataFrame:
    """Calculate non-overlapping IC summaries for every possible offset.

    For horizon=5, offsets 0, 1, 2, 3 and 4 are tested separately.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    summaries = []

    for offset in range(horizon):
        subset = select_non_overlapping_dates(
            panel,
            step=horizon,
            offset=offset,
        )

        daily_ic = calculate_daily_ic(
            panel=subset,
            factor_column=factor_column,
            forward_return_column=forward_return_column,
            method=method,
            min_observations=min_observations,
        )

        summary = summarise_ic(daily_ic).iloc[0].to_dict()
        summary["offset"] = offset
        summary["horizon"] = horizon

        summaries.append(summary)

    return pd.DataFrame(summaries)


def calculate_rolling_ic(
    ic_series: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Add a rolling mean IC to a daily IC series."""
    if "date" not in ic_series.columns or "ic" not in ic_series.columns:
        raise ValueError("Input must contain 'date' and 'ic' columns.")

    if window <= 0:
        raise ValueError("window must be positive.")

    if min_periods is None:
        min_periods = max(1, window // 2)

    result = ic_series.copy()
    result = result.sort_values("date").reset_index(drop=True)

    result[f"rolling_ic_{window}"] = (
        result["ic"].rolling(window=window, min_periods=min_periods).mean()
    )

    return result
