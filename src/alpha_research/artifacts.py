"""Validation and persistence for materialised analytical datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_research.monitoring import DIAGNOSTIC_FLAG_EXPORT_COLUMNS
from alpha_research.workflows import MONITORING_DATASET_NAMES


@dataclass(frozen=True)
class ArtifactContract:
    """Schema and persistence contract for one materialised dataset."""

    filename: str
    key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    date_column: str | None = "date"


MONITORING_ARTIFACT_CONTRACTS = {
    "signal_health": ArtifactContract(
        filename="strategy_monitoring_signal_health_daily.parquet",
        key_columns=("factor", "date"),
        required_columns=(
            "factor",
            "date",
            "signal_coverage",
            "raw_iqr",
            "ic",
            "rolling_mean_ic_252",
            "rank_stability_1d",
            "rank_stability_21d",
        ),
    ),
    "factor_dependence": ArtifactContract(
        filename="strategy_monitoring_factor_dependence_daily.parquet",
        key_columns=("date",),
        required_columns=(
            "date",
            "factor_rank_correlation",
            "observations",
            "rolling_factor_rank_correlation_252",
        ),
    ),
    "performance_risk": ArtifactContract(
        filename="strategy_monitoring_performance_risk_daily.parquet",
        key_columns=("portfolio", "date"),
        required_columns=(
            "portfolio",
            "date",
            "return",
            "wealth",
            "drawdown",
            "drawdown_duration",
            "trailing_return_252",
            "rolling_sharpe_252",
            "annualised_volatility_126",
            "maximum_drawdown_252",
        ),
    ),
    "beta": ArtifactContract(
        filename="strategy_monitoring_beta_daily.parquet",
        key_columns=("portfolio", "date"),
        required_columns=(
            "portfolio",
            "date",
            "beta_coverage",
            "holdings_market_beta",
            "realised_gross_beta_126",
            "beta_measurement_gap",
        ),
    ),
    "concentration": ArtifactContract(
        filename="strategy_monitoring_concentration_daily.parquet",
        key_columns=("portfolio", "date"),
        required_columns=(
            "portfolio",
            "date",
            "effective_position_count",
            "largest_absolute_sector_net_exposure",
            "top_five_absolute_beta_contribution_share",
            "effective_contributor_count_63",
            "top_five_contributor_share_63",
            "effective_contribution_sector_count_63",
        ),
    ),
    "implementation": ArtifactContract(
        filename="strategy_monitoring_implementation_daily.parquet",
        key_columns=("portfolio", "date"),
        required_columns=(
            "portfolio",
            "date",
            "turnover",
            "transaction_cost",
            "trade_count",
            "annualised_turnover_63",
            "largest_trade_weight_63",
            "minimum_trade_capacity_1pct_usd_63",
            "maximum_missing_return_weight_63",
        ),
    ),
    "liquidity_coverage": ArtifactContract(
        filename="strategy_monitoring_liquidity_coverage_daily.parquet",
        key_columns=("portfolio", "date"),
        required_columns=(
            "portfolio",
            "date",
            "turnover",
            "liquidity_covered_turnover",
            "liquidity_coverage",
        ),
    ),
    "diagnostic_flags": ArtifactContract(
        filename="strategy_monitoring_diagnostic_flags.parquet",
        key_columns=(
            "entity_type",
            "entity",
            "category",
            "diagnostic",
        ),
        required_columns=tuple(DIAGNOSTIC_FLAG_EXPORT_COLUMNS),
        date_column="latest_date",
    ),
    "latest_overview": ArtifactContract(
        filename="strategy_monitoring_latest_overview.parquet",
        key_columns=("entity_type", "entity"),
        required_columns=(
            "entity_type",
            "entity",
            "diagnostics",
            "passes",
            "warnings",
            "breaches",
            "unavailable",
            "overall_status",
            "signal_status",
            "market_risk_status",
            "concentration_status",
            "implementation_status",
        ),
        date_column=None,
    ),
}


def _normalise_artifact(
    data: pd.DataFrame,
    contract: ArtifactContract,
) -> pd.DataFrame:
    result = data.copy()

    if contract.date_column is not None:
        result[contract.date_column] = pd.to_datetime(result[contract.date_column])

    return result.sort_values(list(contract.key_columns)).reset_index(drop=True)


def validate_monitoring_artifacts(
    datasets: Mapping[str, pd.DataFrame],
) -> None:
    """Validate schemas, keys, dates, and cross-artifact identities."""
    expected_names = set(MONITORING_DATASET_NAMES)
    actual_names = set(datasets)

    if actual_names != expected_names:
        raise ValueError(
            "Monitoring dataset names do not match the artifact contract. "
            f"Missing: {sorted(expected_names - actual_names)}; "
            f"additional: {sorted(actual_names - expected_names)}"
        )

    prepared: dict[str, pd.DataFrame] = {}

    for name in MONITORING_DATASET_NAMES:
        data = datasets[name]
        contract = MONITORING_ARTIFACT_CONTRACTS[name]

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

        if data.empty:
            raise ValueError(f"{name} is empty.")

        missing_columns = set(contract.required_columns) - set(data.columns)

        if missing_columns:
            raise KeyError(f"{name} is missing columns: {sorted(missing_columns)}")

        if data[list(contract.key_columns)].isna().any().any():
            raise ValueError(f"{name} contains missing key values.")

        if data.duplicated(list(contract.key_columns)).any():
            raise ValueError(f"{name} contains duplicate keys.")

        if contract.date_column is not None:
            converted_dates = pd.to_datetime(
                data[contract.date_column],
                errors="coerce",
            )

            if converted_dates.isna().any():
                raise ValueError(
                    f"{name}.{contract.date_column} contains invalid dates."
                )

        prepared[name] = _normalise_artifact(data, contract)

    portfolio_date_names = (
        "beta",
        "concentration",
        "implementation",
        "liquidity_coverage",
    )
    reference_keys = prepared["beta"][["portfolio", "date"]]

    for name in portfolio_date_names[1:]:
        candidate_keys = prepared[name][["portfolio", "date"]]

        if not reference_keys.equals(candidate_keys):
            raise ValueError(f"{name} portfolio-date coverage does not match beta.")

    flag_entities = (
        prepared["diagnostic_flags"][["entity_type", "entity"]]
        .drop_duplicates()
        .sort_values(["entity_type", "entity"])
        .reset_index(drop=True)
    )
    overview_entities = prepared["latest_overview"][["entity_type", "entity"]]

    if not flag_entities.equals(overview_entities):
        raise ValueError("latest_overview entities do not match diagnostic_flags.")


def write_monitoring_artifacts(
    datasets: Mapping[str, pd.DataFrame],
    output_directory: Path,
) -> pd.DataFrame:
    """Validate and atomically replace all monitoring Parquet artifacts."""
    validate_monitoring_artifacts(datasets)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for name in MONITORING_DATASET_NAMES:
        contract = MONITORING_ARTIFACT_CONTRACTS[name]
        prepared = _normalise_artifact(datasets[name], contract)
        output_path = output_directory / contract.filename
        temporary_path = output_directory / f".{contract.filename}.tmp"

        try:
            prepared.to_parquet(temporary_path, index=False)
            saved = pd.read_parquet(temporary_path)
            pd.testing.assert_frame_equal(
                saved,
                prepared,
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=0.0,
            )
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        if contract.date_column is None:
            start_date = pd.NaT
            end_date = pd.NaT
        else:
            start_date = prepared[contract.date_column].min()
            end_date = prepared[contract.date_column].max()

        manifest_rows.append(
            {
                "dataset": name,
                "file": contract.filename,
                "rows": len(prepared),
                "columns": prepared.shape[1],
                "start_date": start_date,
                "end_date": end_date,
                "read_back_passes": True,
            }
        )

    return pd.DataFrame(manifest_rows)
