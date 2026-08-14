"""Validation and persistence for materialised analytical datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_research.attribution import (
    SECURITY_ATTRIBUTION_EXPORT_COLUMNS,
    reconcile_security_attribution,
)
from alpha_research.config.research import (
    DEFAULT_NUMERICAL_TOLERANCE,
    STRATEGY_SPECIFICATIONS,
)
from alpha_research.monitoring import (
    DIAGNOSTIC_FLAG_EXPORT_COLUMNS,
)
from alpha_research.workflows import (
    ATTRIBUTION_DATASET_NAMES,
    MONITORING_DATASET_NAMES,
    validate_frozen_implementations,
)


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

ATTRIBUTION_ARTIFACT_CONTRACTS = {
    "selected_implementations": ArtifactContract(
        filename=("attribution_selected_implementations.parquet"),
        key_columns=(
            "portfolio",
            "rebalance_frequency",
            "rebalance_offset",
        ),
        required_columns=(
            "portfolio",
            "rebalance_frequency",
            "rebalance_offset",
            "role",
        ),
        date_column=None,
    ),
    "portfolio_daily": ArtifactContract(
        filename="attribution_portfolio_daily.parquet",
        key_columns=("portfolio", "date"),
        required_columns=(
            "date",
            "is_rebalance",
            "long_return",
            "short_return",
            "gross_return",
            "turnover",
            "transaction_cost",
            "net_return",
            "long_exposure",
            "short_exposure",
            "net_exposure",
            "gross_exposure",
            "missing_return_weight",
            "gross_cumulative_return",
            "net_cumulative_return",
            "portfolio",
            "rebalance_frequency",
            "rebalance_offset",
            "transaction_cost_bps",
            "role",
        ),
    ),
    "security_holdings": ArtifactContract(
        filename=("attribution_security_holdings.parquet"),
        key_columns=(
            "portfolio",
            "date",
            "ticker",
        ),
        required_columns=(
            "date",
            "ticker",
            "pre_trade_weight",
            "weight",
            "trade",
            "portfolio",
            "rebalance_frequency",
            "rebalance_offset",
            "role",
        ),
    ),
    "target_weights": ArtifactContract(
        filename="attribution_target_weights.parquet",
        key_columns=(
            "portfolio",
            "date",
            "ticker",
        ),
        required_columns=(
            "date",
            "ticker",
            "weight",
            "portfolio",
            "rebalance_frequency",
            "rebalance_offset",
            "role",
        ),
    ),
    "benchmark_daily": ArtifactContract(
        filename="attribution_benchmark_daily.parquet",
        key_columns=("benchmark", "date"),
        required_columns=(
            "date",
            "benchmark",
            "benchmark_return",
        ),
    ),
    "security_daily": ArtifactContract(
        filename="attribution_security_daily.parquet",
        key_columns=(
            "portfolio",
            "date",
            "ticker",
        ),
        required_columns=(SECURITY_ATTRIBUTION_EXPORT_COLUMNS),
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


def validate_attribution_artifacts(
    datasets: Mapping[str, pd.DataFrame],
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> None:
    """Validate attribution schemas and accounting."""
    expected_names = set(ATTRIBUTION_DATASET_NAMES)
    actual_names = set(datasets)

    if actual_names != expected_names:
        raise ValueError(
            "Attribution dataset names do not "
            "match the artifact contract. "
            f"Missing: "
            f"{sorted(expected_names - actual_names)}; "
            f"additional: "
            f"{sorted(actual_names - expected_names)}"
        )

    prepared: dict[str, pd.DataFrame] = {}

    for name in ATTRIBUTION_DATASET_NAMES:
        data = datasets[name]
        contract = ATTRIBUTION_ARTIFACT_CONTRACTS[name]

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

        if data.empty:
            raise ValueError(f"{name} is empty.")

        missing_columns = set(contract.required_columns) - set(data.columns)

        if missing_columns:
            raise KeyError(f"{name} is missing columns: " f"{sorted(missing_columns)}")

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
                    f"{name}." f"{contract.date_column} " "contains invalid dates."
                )

        prepared[name] = _normalise_artifact(
            data,
            contract,
        )

    portfolios = validate_frozen_implementations(prepared["selected_implementations"])

    specification_columns = [
        "portfolio",
        "rebalance_frequency",
        "rebalance_offset",
        "role",
    ]

    expected_specifications = (
        prepared["selected_implementations"][specification_columns]
        .sort_values("portfolio")
        .reset_index(drop=True)
    )

    for name in (
        "portfolio_daily",
        "security_holdings",
        "target_weights",
        "security_daily",
    ):
        observed_specifications = (
            prepared[name][specification_columns]
            .drop_duplicates()
            .sort_values("portfolio")
            .reset_index(drop=True)
        )

        if not observed_specifications.equals(expected_specifications):
            raise ValueError(
                f"{name} specifications do not " "match selected_implementations."
            )

    benchmark_names = prepared["benchmark_daily"]["benchmark"].drop_duplicates()

    if len(benchmark_names) != 1:
        raise ValueError("benchmark_daily must contain exactly " "one benchmark.")

    if prepared["benchmark_daily"]["benchmark_return"].isna().any():
        raise ValueError("benchmark_daily contains missing returns.")

    benchmark_dates = pd.DatetimeIndex(
        prepared["benchmark_daily"]["date"]
    ).sort_values()

    portfolio_daily = prepared["portfolio_daily"]

    transaction_cost_lookup = {
        specification.portfolio: (specification.transaction_cost_bps)
        for specification in STRATEGY_SPECIFICATIONS
    }

    accounting_columns = {
        "long_short_return": (
            portfolio_daily["long_return"]
            + portfolio_daily["short_return"]
            - portfolio_daily["gross_return"]
        ),
        "net_return": (
            portfolio_daily["gross_return"]
            - portfolio_daily["transaction_cost"]
            - portfolio_daily["net_return"]
        ),
        "transaction_cost": (
            portfolio_daily["turnover"]
            * portfolio_daily["transaction_cost_bps"]
            / 10_000.0
            - portfolio_daily["transaction_cost"]
        ),
        "gross_exposure": (
            portfolio_daily["long_exposure"]
            + portfolio_daily["short_exposure"]
            - portfolio_daily["gross_exposure"]
        ),
        "net_exposure": (
            portfolio_daily["long_exposure"]
            - portfolio_daily["short_exposure"]
            - portfolio_daily["net_exposure"]
        ),
    }

    for (
        identity_name,
        difference,
    ) in accounting_columns.items():
        if difference.abs().max() >= tolerance:
            raise ValueError(
                "portfolio_daily " f"{identity_name} identity does " "not reconcile."
            )

    for portfolio in portfolios:
        portfolio_data = (
            portfolio_daily.loc[portfolio_daily["portfolio"].eq(portfolio)]
            .sort_values("date")
            .reset_index(drop=True)
        )

        portfolio_dates = pd.DatetimeIndex(portfolio_data["date"])

        if not portfolio_dates.equals(benchmark_dates):
            raise ValueError(f"{portfolio} dates do not " "match benchmark_daily.")

        expected_cost_bps = transaction_cost_lookup[portfolio]

        if not portfolio_data["transaction_cost_bps"].eq(expected_cost_bps).all():
            raise ValueError(
                f"{portfolio} transaction costs "
                "do not match the frozen "
                "specification."
            )

        expected_gross_cumulative = (1.0 + portfolio_data["gross_return"]).cumprod()

        expected_net_cumulative = (1.0 + portfolio_data["net_return"]).cumprod()

        gross_difference = (
            (expected_gross_cumulative - portfolio_data["gross_cumulative_return"])
            .abs()
            .max()
        )

        net_difference = (
            (expected_net_cumulative - portfolio_data["net_cumulative_return"])
            .abs()
            .max()
        )

        if gross_difference >= tolerance or net_difference >= tolerance:
            raise ValueError(f"{portfolio} cumulative returns " "do not reconcile.")

    attribution_audit = reconcile_security_attribution(
        portfolio_daily,
        prepared["security_daily"],
        tolerance=tolerance,
    )

    if not attribution_audit["audit_passes"].all():
        raise ValueError("Security-level attribution does " "not reconcile.")


def write_attribution_artifacts(
    datasets: Mapping[str, pd.DataFrame],
    output_directory: Path,
) -> pd.DataFrame:
    """Validate and atomically write attribution data."""
    validate_attribution_artifacts(datasets)

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for name in ATTRIBUTION_DATASET_NAMES:
        contract = ATTRIBUTION_ARTIFACT_CONTRACTS[name]

        prepared = _normalise_artifact(
            datasets[name],
            contract,
        )

        output_path = output_directory / contract.filename

        temporary_path = output_directory / f".{contract.filename}.tmp"

        try:
            prepared.to_parquet(
                temporary_path,
                index=False,
            )

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
                "columns": (prepared.shape[1]),
                "start_date": start_date,
                "end_date": end_date,
                "read_back_passes": True,
            }
        )

    return pd.DataFrame(manifest_rows)
