"""Security-level return, cost, and exposure attribution.

The functions in this module extract the attribution construction and
reconciliation originally implemented in Notebook 06.  They consume holdings
produced by the backtest engine; they do not calculate or alter portfolio
weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from alpha_research.config.research import (
    BACKTEST_RETURN_COLUMN,
    BASELINE_TRANSACTION_COST_BPS,
    DEFAULT_NUMERICAL_TOLERANCE,
)
from alpha_research.costs import calculate_linear_transaction_cost

ATTRIBUTION_DAILY_KEY_COLUMNS = (
    "portfolio",
    "rebalance_frequency",
    "rebalance_offset",
    "date",
)

SECURITY_ATTRIBUTION_KEY_COLUMNS = (
    *ATTRIBUTION_DAILY_KEY_COLUMNS,
    "ticker",
)

SECURITY_ATTRIBUTION_EXPORT_COLUMNS = (
    "date",
    "ticker",
    "pre_trade_weight",
    "weight",
    "trade",
    "portfolio",
    "rebalance_frequency",
    "rebalance_offset",
    "role",
    "asset_return",
    "return_record_missing",
    "asset_return_missing",
    "realised_asset_return",
    "absolute_trade_weight",
    "gross_contribution",
    "long_contribution",
    "short_contribution",
    "transaction_cost_contribution",
    "net_contribution",
    "long_exposure_contribution",
    "short_exposure_contribution",
    "missing_return_weight_contribution",
    "holding_side",
)

SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS: Mapping[str, str] = {
    "long_return": "long_contribution",
    "short_return": "short_contribution",
    "gross_return": "gross_contribution",
    "turnover": "absolute_trade_weight",
    "transaction_cost": "transaction_cost_contribution",
    "net_return": "net_contribution",
    "long_exposure": "long_exposure_contribution",
    "short_exposure": "short_exposure_contribution",
    "missing_return_weight": "missing_return_weight_contribution",
}


def _validate_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    *,
    name: str,
) -> None:
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(f"{name} is missing columns: {sorted(missing_columns)}")


def _validate_unique_keys(
    data: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    name: str,
) -> None:
    if data[list(key_columns)].isna().any().any():
        raise ValueError(f"{name} contains missing key values.")

    if data.duplicated(list(key_columns)).any():
        raise ValueError(f"{name} contains duplicate keys.")


def _as_finite_numeric(
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


def prepare_security_attribution(
    security_holdings: pd.DataFrame,
    return_panel: pd.DataFrame,
    return_column: str = BACKTEST_RETURN_COLUMN,
    transaction_cost_bps: float = BASELINE_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Build active security-day attribution from backtest holdings.

    Missing asset returns are realised as zero, matching
    :func:`alpha_research.backtest.run_target_weight_backtest`.  A separate
    flag distinguishes a missing return value from an entirely missing
    date-ticker record.
    """
    holding_columns = set(SECURITY_ATTRIBUTION_EXPORT_COLUMNS[:9])
    _validate_columns(
        security_holdings,
        holding_columns,
        name="security_holdings",
    )
    _validate_columns(
        return_panel,
        {"date", "ticker", return_column},
        name="return_panel",
    )

    cost_bps = float(transaction_cost_bps)

    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")

    holdings = security_holdings[list(SECURITY_ATTRIBUTION_EXPORT_COLUMNS[:9])].copy()
    returns = return_panel[["date", "ticker", return_column]].copy()

    holdings["date"] = pd.to_datetime(holdings["date"], errors="raise")
    returns["date"] = pd.to_datetime(returns["date"], errors="raise")

    _validate_unique_keys(
        holdings,
        SECURITY_ATTRIBUTION_KEY_COLUMNS,
        name="security_holdings",
    )
    _validate_unique_keys(
        returns,
        ("date", "ticker"),
        name="return_panel",
    )

    for column in ("pre_trade_weight", "weight", "trade"):
        holdings[column] = _as_finite_numeric(
            holdings[column],
            name=f"security_holdings.{column}",
            allow_missing=False,
        )

    returns[return_column] = _as_finite_numeric(
        returns[return_column],
        name=f"return_panel.{return_column}",
        allow_missing=True,
    )

    active_holding_mask = (
        holdings[["pre_trade_weight", "weight", "trade"]].ne(0.0).any(axis=1)
    )

    result = (
        holdings.loc[active_holding_mask]
        .merge(
            returns,
            on=["date", "ticker"],
            how="left",
            validate="many_to_one",
            indicator="return_merge_status",
        )
        .rename(columns={return_column: "asset_return"})
        .sort_values(["portfolio", "date", "ticker"])
        .reset_index(drop=True)
    )

    result["return_record_missing"] = result["return_merge_status"].eq("left_only")
    result["asset_return_missing"] = result["asset_return"].isna()
    result["realised_asset_return"] = result["asset_return"].fillna(0.0)
    result["absolute_trade_weight"] = result["trade"].abs()
    result["gross_contribution"] = result["weight"] * result["realised_asset_return"]
    result["long_contribution"] = result["gross_contribution"].where(
        result["weight"].gt(0.0),
        0.0,
    )
    result["short_contribution"] = result["gross_contribution"].where(
        result["weight"].lt(0.0),
        0.0,
    )
    result["transaction_cost_contribution"] = calculate_linear_transaction_cost(
        result["absolute_trade_weight"],
        transaction_cost_bps=cost_bps,
    )
    result["net_contribution"] = (
        result["gross_contribution"] - result["transaction_cost_contribution"]
    )
    result["long_exposure_contribution"] = result["weight"].clip(lower=0.0)
    result["short_exposure_contribution"] = -result["weight"].clip(upper=0.0)
    result["missing_return_weight_contribution"] = (
        result["weight"].abs().where(result["asset_return_missing"], 0.0)
    )
    result["holding_side"] = np.select(
        [result["weight"].gt(0.0), result["weight"].lt(0.0)],
        ["Long", "Short"],
        default="Flat",
    )

    return result.loc[:, SECURITY_ATTRIBUTION_EXPORT_COLUMNS]


def reconstruct_portfolio_daily_attribution(
    security_attribution: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate security attribution to its portfolio-day identities."""
    required_columns = set(SECURITY_ATTRIBUTION_KEY_COLUMNS)
    required_columns.update(SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS.values())
    _validate_columns(
        security_attribution,
        required_columns,
        name="security_attribution",
    )
    _validate_unique_keys(
        security_attribution,
        SECURITY_ATTRIBUTION_KEY_COLUMNS,
        name="security_attribution",
    )

    prepared = security_attribution.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")

    aggregations = {
        portfolio_column: pd.NamedAgg(
            column=security_column,
            aggfunc="sum",
        )
        for portfolio_column, security_column in (
            SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS.items()
        )
    }

    return (
        prepared.groupby(
            list(ATTRIBUTION_DAILY_KEY_COLUMNS),
            sort=True,
            observed=True,
        )
        .agg(**aggregations)
        .reset_index()
    )


def reconcile_security_attribution(
    portfolio_daily: pd.DataFrame,
    security_attribution: pd.DataFrame,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> pd.DataFrame:
    """Audit security sums against portfolio-day backtest outputs."""
    tolerance = float(tolerance)

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    required_portfolio_columns = set(ATTRIBUTION_DAILY_KEY_COLUMNS)
    required_portfolio_columns.update(SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS)
    _validate_columns(
        portfolio_daily,
        required_portfolio_columns,
        name="portfolio_daily",
    )
    _validate_unique_keys(
        portfolio_daily,
        ATTRIBUTION_DAILY_KEY_COLUMNS,
        name="portfolio_daily",
    )

    reference = portfolio_daily[
        [
            *ATTRIBUTION_DAILY_KEY_COLUMNS,
            *SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS,
        ]
    ].copy()
    reference["date"] = pd.to_datetime(reference["date"], errors="raise")

    for column in SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS:
        reference[column] = _as_finite_numeric(
            reference[column],
            name=f"portfolio_daily.{column}",
            allow_missing=False,
        )

    reconstructed = reconstruct_portfolio_daily_attribution(security_attribution)
    key_columns = list(ATTRIBUTION_DAILY_KEY_COLUMNS)
    reference_keys = (
        reference[key_columns].sort_values(key_columns).reset_index(drop=True)
    )
    reconstructed_keys = (
        reconstructed[key_columns].sort_values(key_columns).reset_index(drop=True)
    )

    if not reference_keys.equals(reconstructed_keys):
        raise ValueError(
            "Security attribution portfolio-day coverage does not match "
            "portfolio_daily."
        )

    audit = reference.merge(
        reconstructed,
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_reconstructed"),
    )
    audit_rows = []

    for portfolio_name, portfolio_data in audit.groupby(
        "portfolio",
        sort=False,
    ):
        audit_row: dict[str, str | int | float | bool] = {
            "portfolio": portfolio_name,
            "observations": len(portfolio_data),
        }
        maximum_difference = 0.0

        for column in SECURITY_ATTRIBUTION_RECONCILIATION_COLUMNS:
            difference = float(
                (
                    portfolio_data[f"{column}_reference"]
                    - portfolio_data[f"{column}_reconstructed"]
                )
                .abs()
                .max()
            )
            audit_row[f"max_abs_{column}_difference"] = difference
            maximum_difference = max(maximum_difference, difference)

        audit_row["maximum_absolute_difference"] = maximum_difference
        audit_row["audit_passes"] = maximum_difference < tolerance
        audit_rows.append(audit_row)

    return pd.DataFrame(audit_rows)
