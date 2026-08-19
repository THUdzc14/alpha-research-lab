"""Dashboard-facing loading and freshness checks for research artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research.artifacts import (
    ATTRIBUTION_ARTIFACT_CONTRACTS,
    MONITORING_ARTIFACT_CONTRACTS,
    ArtifactContract,
    validate_artifact,
    validate_attribution_artifacts,
    validate_monitoring_artifacts,
)
from alpha_research.config.paths import MONITORING_DATA_DIR, PROCESSED_DATA_DIR
from alpha_research.data_loader import load_parquet

DEFAULT_STALE_AFTER_BUSINESS_DAYS = 5

# Undated configuration and snapshot artifacts inherit a freshness reference
# from the dated dataset that determines whether their contents are current.
# These proxies affect freshness metadata only, never structural validation.
DASHBOARD_FRESHNESS_PROXIES = {
    ("attribution", "target_weights"): ("attribution", "portfolio_daily"),
    ("monitoring", "latest_overview"): ("monitoring", "diagnostic_flags"),
}

DASHBOARD_ARTIFACT_METADATA_COLUMNS = (
    "group",
    "dataset",
    "file",
    "path",
    "exists",
    "loaded",
    "artifact_valid",
    "group_valid",
    "rows",
    "columns",
    "start_date",
    "end_date",
    "latest_observation_date",
    "freshness_reference_date",
    "file_modified_at_utc",
    "age_business_days",
    "is_stale",
    "status",
    "error",
)


class DashboardArtifactLoadError(RuntimeError):
    """Raised when strict dashboard-artifact loading does not succeed."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        details = "\n".join(f"- {error}" for error in errors)
        super().__init__(f"Dashboard artifact loading failed:\n{details}")


@dataclass(frozen=True)
class DashboardArtifactBundle:
    """Dashboard datasets with independent readiness and freshness metadata."""

    attribution: dict[str, pd.DataFrame]
    monitoring: dict[str, pd.DataFrame]
    metadata: pd.DataFrame
    errors: tuple[str, ...]
    as_of_date: pd.Timestamp
    stale_after_business_days: int

    @property
    def all_datasets(self) -> dict[str, pd.DataFrame]:
        """Return all successfully loaded datasets in one mapping."""
        return {
            **self.attribution,
            **self.monitoring,
        }

    @property
    def is_ready(self) -> bool:
        """Whether every artifact loaded and passed all validation checks."""
        return not self.errors

    @property
    def has_stale_data(self) -> bool:
        """Whether any dated artifact exceeds the configured freshness limit."""
        return bool(self.metadata["is_stale"].fillna(False).any())


def _normalise_as_of_date(as_of_date: object | None) -> pd.Timestamp:
    value = pd.Timestamp.today() if as_of_date is None else pd.Timestamp(as_of_date)

    if value.tzinfo is not None:
        value = value.tz_localize(None)

    return value.normalize()


def _business_day_age(
    latest_observation_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> int:
    return int(
        np.busday_count(
            latest_observation_date.date(),
            as_of_date.date(),
        )
    )


def _empty_metadata_row(
    group: str,
    name: str,
    contract: ArtifactContract,
    path: Path,
) -> dict[str, object]:
    exists = path.is_file()

    return {
        "group": group,
        "dataset": name,
        "file": contract.filename,
        "path": str(path),
        "exists": exists,
        "loaded": False,
        "artifact_valid": False,
        "group_valid": False,
        "rows": pd.NA,
        "columns": pd.NA,
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "latest_observation_date": pd.NaT,
        "freshness_reference_date": pd.NaT,
        "file_modified_at_utc": (
            pd.Timestamp.fromtimestamp(path.stat().st_mtime, tz="UTC") if exists else pd.NaT
        ),
        "age_business_days": pd.NA,
        "is_stale": pd.NA,
        "status": "MISSING" if not exists else "NOT_LOADED",
        "error": pd.NA,
    }


def _load_artifact_group(
    group: str,
    contracts: Mapping[str, ArtifactContract],
    directory: Path,
    group_validator: Callable[[Mapping[str, pd.DataFrame]], None],
    as_of_date: pd.Timestamp,
    stale_after_business_days: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[str]]:
    datasets: dict[str, pd.DataFrame] = {}
    metadata_rows: list[dict[str, object]] = []
    errors: list[str] = []

    for name, contract in contracts.items():
        path = directory / contract.filename
        row = _empty_metadata_row(group, name, contract, path)

        if not row["exists"]:
            message = f"{group}.{name} is missing: {path}"
            row["error"] = message
            errors.append(message)
            metadata_rows.append(row)
            continue

        try:
            loaded = load_parquet(path)
            row["loaded"] = True
        except Exception as error:
            message = f"{group}.{name} could not be read: {error}"
            row["status"] = "READ_ERROR"
            row["error"] = message
            errors.append(message)
            metadata_rows.append(row)
            continue

        try:
            prepared = validate_artifact(name, loaded, contract)
        except Exception as error:
            message = f"{group}.{name} is invalid: {error}"
            row["status"] = "INVALID"
            row["error"] = message
            errors.append(message)
            metadata_rows.append(row)
            continue

        datasets[name] = prepared
        row["artifact_valid"] = True
        row["rows"] = len(prepared)
        row["columns"] = prepared.shape[1]

        if contract.date_column is None:
            row["status"] = "UNDATED"
        else:
            dates = prepared[contract.date_column]
            start_date = dates.min().normalize()
            end_date = dates.max().normalize()
            age_business_days = _business_day_age(end_date, as_of_date)
            is_stale = age_business_days > stale_after_business_days

            row["start_date"] = start_date
            row["end_date"] = end_date
            row["latest_observation_date"] = end_date
            row["freshness_reference_date"] = end_date
            row["age_business_days"] = age_business_days
            row["is_stale"] = is_stale
            row["status"] = "STALE" if is_stale else "READY"

        metadata_rows.append(row)

    group_is_complete = len(datasets) == len(contracts)

    if group_is_complete:
        try:
            group_validator(datasets)
        except Exception as error:
            message = f"{group} artifact group is invalid: {error}"
            errors.append(message)

            for row in metadata_rows:
                row["group_valid"] = False

                if row["status"] in {"READY", "STALE", "UNDATED"}:
                    row["status"] = "INVALID_GROUP"
                    row["error"] = message
        else:
            for row in metadata_rows:
                row["group_valid"] = True

    return datasets, metadata_rows, errors


def _apply_freshness_proxies(
    metadata: pd.DataFrame,
    as_of_date: pd.Timestamp,
    stale_after_business_days: int,
) -> pd.DataFrame:
    result = metadata.copy()

    for target_key, source_key in DASHBOARD_FRESHNESS_PROXIES.items():
        target_group, target_dataset = target_key
        source_group, source_dataset = source_key
        target_mask = result["group"].eq(target_group) & result["dataset"].eq(target_dataset)
        source_mask = result["group"].eq(source_group) & result["dataset"].eq(source_dataset)

        if not target_mask.any() or not source_mask.any():
            continue

        target_index = result.index[target_mask][0]
        source_index = result.index[source_mask][0]

        if not bool(result.at[target_index, "group_valid"]):
            continue

        source_date = result.at[source_index, "latest_observation_date"]

        if pd.isna(source_date):
            continue

        age_business_days = _business_day_age(source_date, as_of_date)
        is_stale = age_business_days > stale_after_business_days
        result.at[target_index, "freshness_reference_date"] = source_date
        result.at[target_index, "age_business_days"] = age_business_days
        result.at[target_index, "is_stale"] = is_stale
        result.at[target_index, "status"] = "STALE" if is_stale else "READY"

    return result


def load_dashboard_artifacts(
    processed_directory: Path = PROCESSED_DATA_DIR,
    monitoring_directory: Path = MONITORING_DATA_DIR,
    *,
    as_of_date: object | None = None,
    stale_after_business_days: int = DEFAULT_STALE_AFTER_BUSINESS_DAYS,
    strict: bool = True,
) -> DashboardArtifactBundle:
    """Load, validate, and assess all dashboard-facing research artifacts.

    Stale datasets remain loadable and do not make the bundle structurally
    invalid. Use ``has_stale_data`` to surface a dashboard warning. In
    non-strict mode, missing or invalid artifacts are returned as metadata
    errors so that the UI can explain the problem instead of failing opaquely.

    ``as_of_date`` is the reference date for artifact freshness only. Metric-
    specific dates embedded in the datasets, including predictive-statistic
    as-of dates, are loaded unchanged for the analytics layer to interpret.
    """
    if (
        isinstance(stale_after_business_days, bool)
        or not isinstance(stale_after_business_days, int)
        or stale_after_business_days < 0
    ):
        raise ValueError("stale_after_business_days must be a non-negative integer.")

    processed_directory = Path(processed_directory)
    monitoring_directory = Path(monitoring_directory)
    normalised_as_of_date = _normalise_as_of_date(as_of_date)

    attribution, attribution_metadata, attribution_errors = _load_artifact_group(
        group="attribution",
        contracts=ATTRIBUTION_ARTIFACT_CONTRACTS,
        directory=processed_directory,
        group_validator=validate_attribution_artifacts,
        as_of_date=normalised_as_of_date,
        stale_after_business_days=stale_after_business_days,
    )
    monitoring, monitoring_metadata, monitoring_errors = _load_artifact_group(
        group="monitoring",
        contracts=MONITORING_ARTIFACT_CONTRACTS,
        directory=monitoring_directory,
        group_validator=validate_monitoring_artifacts,
        as_of_date=normalised_as_of_date,
        stale_after_business_days=stale_after_business_days,
    )

    metadata = pd.DataFrame(
        [*attribution_metadata, *monitoring_metadata],
        columns=DASHBOARD_ARTIFACT_METADATA_COLUMNS,
    )
    metadata = _apply_freshness_proxies(
        metadata,
        as_of_date=normalised_as_of_date,
        stale_after_business_days=stale_after_business_days,
    )
    errors = tuple([*attribution_errors, *monitoring_errors])

    if strict and errors:
        raise DashboardArtifactLoadError(errors)

    return DashboardArtifactBundle(
        attribution=attribution,
        monitoring=monitoring,
        metadata=metadata,
        errors=errors,
        as_of_date=normalised_as_of_date,
        stale_after_business_days=stale_after_business_days,
    )
