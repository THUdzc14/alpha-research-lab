"""Reusable performance, drawdown, rolling-risk, and concentration metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from alpha_research.config.research import (
    DEFAULT_NUMERICAL_TOLERANCE,
    TRADING_DAYS_PER_YEAR,
)

NumericSeries = pd.Series | Iterable[float]


def _as_numeric_series(
    values: NumericSeries,
    *,
    name: str,
) -> pd.Series:
    """Return a numeric Series while preserving an existing index."""
    if isinstance(values, pd.Series):
        result = values.copy()
    else:
        result = pd.Series(values, dtype="float64")

    try:
        result = pd.to_numeric(result, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values.") from exc

    valid_values = result.dropna()

    if not np.isfinite(valid_values.to_numpy()).all():
        raise ValueError(f"{name} contains infinite values.")

    return result


def _prepare_returns(returns: NumericSeries) -> pd.Series:
    result = _as_numeric_series(returns, name="returns")

    if result.dropna().lt(-1.0).any():
        raise ValueError("Returns cannot be less than -1.0.")

    return result


def _validate_periods_per_year(periods_per_year: int) -> None:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")


def compound_return(returns: NumericSeries) -> float:
    """Compound a return series over its observed period."""
    clean_returns = _prepare_returns(returns).dropna()

    if clean_returns.empty:
        return np.nan

    return float((1.0 + clean_returns).prod() - 1.0)


def calculate_wealth_index(
    returns: NumericSeries,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Calculate cumulative wealth after each return observation."""
    if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
        raise ValueError("initial_wealth must be finite and positive.")

    prepared_returns = _prepare_returns(returns)

    return ((1.0 + prepared_returns).cumprod() * initial_wealth).rename("wealth")


def calculate_drawdown_from_wealth(
    wealth: NumericSeries,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Calculate drawdown with the initial capital treated as a peak."""
    if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
        raise ValueError("initial_wealth must be finite and positive.")

    prepared_wealth = _as_numeric_series(wealth, name="wealth")

    if prepared_wealth.dropna().lt(0.0).any():
        raise ValueError("wealth cannot contain negative values.")

    running_peak = prepared_wealth.cummax().clip(lower=initial_wealth)

    return (prepared_wealth / running_peak - 1.0).rename("drawdown")


def calculate_drawdown_from_returns(
    returns: NumericSeries,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Calculate the drawdown path directly from periodic returns."""
    wealth = calculate_wealth_index(
        returns,
        initial_wealth=initial_wealth,
    )

    return calculate_drawdown_from_wealth(
        wealth,
        initial_wealth=initial_wealth,
    )


def calculate_drawdown_duration(drawdown: NumericSeries) -> pd.Series:
    """Count consecutive underwater observations, resetting at each recovery."""
    prepared_drawdown = _as_numeric_series(drawdown, name="drawdown")

    if prepared_drawdown.dropna().gt(DEFAULT_NUMERICAL_TOLERANCE).any():
        raise ValueError("drawdown cannot contain positive values.")

    durations = pd.Series(
        pd.NA,
        index=prepared_drawdown.index,
        dtype="Int64",
        name="drawdown_duration",
    )

    current_duration = 0

    for position, value in enumerate(prepared_drawdown.to_numpy()):
        if np.isnan(value):
            current_duration = 0
            continue

        # if value < -DEFAULT_NUMERICAL_TOLERANCE:
        # Preserve Notebook 08's exact underwater convention. Tiny negative
        # floating-point drawdowns still extend an existing episode.
        if value < 0.0:
            current_duration += 1
        else:
            current_duration = 0

        durations.iloc[position] = current_duration

    if not durations.isna().any():
        return durations.astype("int64")

    return durations


def annualised_geometric_return(
    returns: NumericSeries,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualise compounded growth using the number of valid observations."""
    _validate_periods_per_year(periods_per_year)

    clean_returns = _prepare_returns(returns).dropna()

    if clean_returns.empty:
        return np.nan

    total_growth = float((1.0 + clean_returns).prod())

    if total_growth <= 0.0:
        return np.nan

    return total_growth ** (periods_per_year / len(clean_returns)) - 1.0


def annualised_volatility(
    returns: NumericSeries,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualise the sample standard deviation of periodic returns."""
    _validate_periods_per_year(periods_per_year)

    clean_returns = _prepare_returns(returns).dropna()

    if len(clean_returns) < 2:
        return np.nan

    return float(clean_returns.std(ddof=1) * math.sqrt(periods_per_year))


def annualised_sharpe_ratio(
    returns: NumericSeries,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    zero_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> float:
    """Calculate the zero-risk-free-rate arithmetic Sharpe ratio."""
    _validate_periods_per_year(periods_per_year)

    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")

    clean_returns = _prepare_returns(returns).dropna()

    if len(clean_returns) < 2:
        return np.nan

    return_standard_deviation = float(clean_returns.std(ddof=1))

    if return_standard_deviation <= zero_tolerance:
        return np.nan

    return float(
        clean_returns.mean() / return_standard_deviation * math.sqrt(periods_per_year)
    )


def annualised_downside_deviation(
    returns: NumericSeries,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualise the root mean squared return below a zero threshold."""
    _validate_periods_per_year(periods_per_year)

    clean_returns = _prepare_returns(returns).dropna()

    if clean_returns.empty:
        return np.nan

    squared_downside_returns = clean_returns.clip(upper=0.0).pow(2)

    return float(math.sqrt(squared_downside_returns.mean() * periods_per_year))


def maximum_drawdown(returns: NumericSeries) -> float:
    """Return the minimum drawdown reached by a return series."""
    clean_returns = _prepare_returns(returns).dropna()

    if clean_returns.empty:
        return np.nan

    return float(calculate_drawdown_from_returns(clean_returns).min())


def summarise_returns(
    returns: NumericSeries,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, int | float]:
    """Calculate the standard full-sample return statistics."""
    _validate_periods_per_year(periods_per_year)

    clean_returns = _prepare_returns(returns).dropna()

    return {
        "observations": len(clean_returns),
        "total_return": compound_return(clean_returns),
        "annualised_return": annualised_geometric_return(
            clean_returns,
            periods_per_year=periods_per_year,
        ),
        "annualised_volatility": annualised_volatility(
            clean_returns,
            periods_per_year=periods_per_year,
        ),
        "sharpe_ratio": annualised_sharpe_ratio(
            clean_returns,
            periods_per_year=periods_per_year,
        ),
        "maximum_drawdown": maximum_drawdown(clean_returns),
        "positive_day_fraction": (
            float(clean_returns.gt(0.0).mean()) if not clean_returns.empty else np.nan
        ),
    }


def _maximum_drawdown_from_array(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]

    return float(np.min(wealth / running_peak - 1.0))


def calculate_rolling_return_state(
    data: pd.DataFrame,
    return_column: str = "return",
    date_column: str = "date",
    performance_window: int = 252,
    risk_window: int = 126,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Add the rolling performance and risk state used by monitoring.

    The input represents one return stream. Full rolling windows are
    required, matching the completed monitoring research in Notebook 08.
    """
    required_columns = {date_column, return_column}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if performance_window < 2:
        raise ValueError("performance_window must be at least 2.")

    if risk_window < 2:
        raise ValueError("risk_window must be at least 2.")

    _validate_periods_per_year(periods_per_year)

    result = data.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    if result[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")

    if result[date_column].duplicated().any():
        raise ValueError(f"{date_column} contains duplicate dates.")

    result = result.sort_values(date_column).reset_index(drop=True)
    result[return_column] = _prepare_returns(result[return_column])

    returns = result[return_column]
    wealth = calculate_wealth_index(returns)
    drawdown = calculate_drawdown_from_wealth(wealth)

    result["wealth"] = wealth
    result["drawdown"] = drawdown
    result["drawdown_duration"] = calculate_drawdown_duration(drawdown)

    performance_rolling = returns.rolling(
        performance_window,
        min_periods=performance_window,
    )

    risk_rolling = returns.rolling(
        risk_window,
        min_periods=risk_window,
    )

    result[f"trailing_return_{performance_window}"] = (1.0 + returns).rolling(
        performance_window,
        min_periods=performance_window,
    ).apply(np.prod, raw=True) - 1.0

    rolling_mean = performance_rolling.mean()
    rolling_standard_deviation = performance_rolling.std(ddof=1)
    valid_standard_deviation = rolling_standard_deviation.gt(
        DEFAULT_NUMERICAL_TOLERANCE
    )

    result[f"rolling_sharpe_{performance_window}"] = (
        rolling_mean
        / rolling_standard_deviation.where(valid_standard_deviation)
        * math.sqrt(periods_per_year)
    )

    result[f"annualised_volatility_{risk_window}"] = risk_rolling.std(
        ddof=1
    ) * math.sqrt(periods_per_year)

    squared_downside_returns = returns.clip(upper=0.0).pow(2)

    result[f"annualised_downside_deviation_{risk_window}"] = np.sqrt(
        squared_downside_returns.rolling(
            risk_window,
            min_periods=risk_window,
        ).mean()
        * periods_per_year
    )

    result[f"maximum_drawdown_{performance_window}"] = performance_rolling.apply(
        _maximum_drawdown_from_array,
        raw=True,
    )

    return result


def summarise_concentration(
    values: NumericSeries,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> dict[str, int | float]:
    """Summarise concentration from non-negative position or exposure values."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    prepared_values = _as_numeric_series(values, name="values").dropna()

    if prepared_values.lt(-tolerance).any():
        raise ValueError("Concentration values must be non-negative.")

    active_values = prepared_values.loc[prepared_values.gt(tolerance)]
    total = float(active_values.sum())

    if active_values.empty or total <= tolerance:
        return {
            "count": 0,
            "effective_count": np.nan,
            "largest_share": np.nan,
            "top_three_share": np.nan,
            "top_five_share": np.nan,
        }

    shares = active_values / total

    return {
        "count": len(shares),
        "effective_count": float(1.0 / shares.pow(2).sum()),
        "largest_share": float(shares.max()),
        "top_three_share": float(shares.nlargest(3).sum()),
        "top_five_share": float(shares.nlargest(5).sum()),
    }
