import shutil

import numpy as np
import pandas as pd
import pytest

from alpha_research.artifacts import (
    MONITORING_ARTIFACT_CONTRACTS,
    write_attribution_artifacts,
    write_monitoring_artifacts,
)
from alpha_research.dashboard_data import (
    DashboardArtifactLoadError,
    load_dashboard_artifacts,
)
from alpha_research.refresh import build_complete_research_refresh
from alpha_research.workflows import (
    ATTRIBUTION_DATASET_NAMES,
    MONITORING_DATASET_NAMES,
)


@pytest.fixture(scope="module")
def dashboard_datasets():
    dates = pd.bdate_range("2015-12-15", periods=130)
    tickers = [f"S{number:02d}" for number in range(30)]
    market_returns = np.resize(
        np.array([-0.01, -0.005, 0.005, 0.01]),
        len(dates) - 1,
    )
    benchmark_prices = np.concatenate(
        [
            np.array([100.0]),
            100.0 * np.cumprod(1.0 + market_returns),
        ]
    )
    factor_rows = []

    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            score = float(ticker_number)
            asset_return = (
                market_returns[date_number] + (ticker_number - 14.5) / 10_000.0
                if date_number < len(market_returns)
                else np.nan
            )
            factor_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "mom_12_1m_z": score,
                    "mom_12_1m_raw": score * 10.0,
                    "realised_vol_63_z": score,
                    "realised_vol_63_raw": score / 10.0,
                    "forward_ret_1d": asset_return,
                    "forward_ret_5d": score / 1_000.0,
                    "beta_126": 0.8 + ticker_number / 100.0,
                    "sector": (
                        "Technology" if ticker_number % 2 == 0 else "Financials"
                    ),
                    "dollar_volume": 1_000_000.0 + ticker_number * 10_000.0,
                }
            )

    return build_complete_research_refresh(
        factor_panel=pd.DataFrame(factor_rows),
        benchmark_prices=pd.DataFrame(
            {
                "date": dates,
                "ticker": "SPY",
                "adj_close": benchmark_prices,
            }
        ),
    )


@pytest.fixture(scope="module")
def dashboard_artifact_source(tmp_path_factory, dashboard_datasets):
    processed_directory = tmp_path_factory.mktemp("dashboard_artifacts")
    monitoring_directory = processed_directory / "monitoring"
    write_attribution_artifacts(
        dashboard_datasets["attribution"],
        processed_directory,
    )
    write_monitoring_artifacts(
        dashboard_datasets["monitoring"],
        monitoring_directory,
    )
    return processed_directory


@pytest.fixture()
def dashboard_artifact_directories(tmp_path, dashboard_artifact_source):
    processed_directory = tmp_path / "processed"
    shutil.copytree(dashboard_artifact_source, processed_directory)
    return processed_directory, processed_directory / "monitoring"


def test_dashboard_loader_returns_validated_bundle_and_metadata(
    dashboard_artifact_directories,
    dashboard_datasets,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    latest_input_date = dashboard_datasets["monitoring"]["signal_health"]["date"].max()

    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        as_of_date=latest_input_date + pd.offsets.BDay(2),
    )

    assert bundle.is_ready
    assert not bundle.has_stale_data
    assert tuple(bundle.attribution) == ATTRIBUTION_DATASET_NAMES
    assert tuple(bundle.monitoring) == MONITORING_DATASET_NAMES
    assert len(bundle.all_datasets) == 15
    assert len(bundle.metadata) == 15
    assert bundle.metadata["exists"].all()
    assert bundle.metadata["loaded"].all()
    assert bundle.metadata["artifact_valid"].all()
    assert bundle.metadata["group_valid"].all()
    assert set(bundle.metadata["status"]) == {"READY", "UNDATED"}

    selected = bundle.metadata.loc[
        bundle.metadata["dataset"].eq("selected_implementations")
    ].iloc[0]
    signal_health = bundle.metadata.loc[
        bundle.metadata["dataset"].eq("signal_health")
    ].iloc[0]

    assert pd.isna(selected["latest_observation_date"])
    assert pd.isna(selected["is_stale"])
    assert signal_health["rows"] == len(
        dashboard_datasets["monitoring"]["signal_health"]
    )
    assert signal_health["latest_observation_date"] == latest_input_date


def test_dashboard_loader_uses_proxy_freshness_dates(
    dashboard_artifact_directories,
    dashboard_datasets,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    latest_input_date = dashboard_datasets["monitoring"]["signal_health"]["date"].max()
    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        as_of_date=latest_input_date + pd.offsets.BDay(2),
    )
    metadata = bundle.metadata.set_index("dataset")
    target_weights = metadata.loc["target_weights"]
    portfolio_daily = metadata.loc["portfolio_daily"]
    latest_overview = metadata.loc["latest_overview"]
    diagnostic_flags = metadata.loc["diagnostic_flags"]

    assert target_weights["latest_observation_date"] < portfolio_daily["end_date"]
    assert (
        target_weights["freshness_reference_date"]
        == portfolio_daily["latest_observation_date"]
    )
    assert pd.isna(latest_overview["latest_observation_date"])
    assert (
        latest_overview["freshness_reference_date"]
        == diagnostic_flags["latest_observation_date"]
    )


def test_dashboard_loader_marks_dated_artifacts_as_stale(
    dashboard_artifact_directories,
    dashboard_datasets,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    latest_input_date = dashboard_datasets["monitoring"]["signal_health"]["date"].max()

    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        as_of_date=latest_input_date + pd.offsets.BDay(10),
        stale_after_business_days=5,
    )
    freshness_metadata = bundle.metadata.loc[
        bundle.metadata["freshness_reference_date"].notna()
    ]

    assert bundle.is_ready
    assert bundle.has_stale_data
    assert freshness_metadata["is_stale"].all()
    assert freshness_metadata["status"].eq("STALE").all()


def test_dashboard_loader_reports_missing_artifacts(
    dashboard_artifact_directories,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    missing_path = (
        monitoring_directory / MONITORING_ARTIFACT_CONTRACTS["signal_health"].filename
    )
    missing_path.unlink()

    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        strict=False,
    )
    metadata = bundle.metadata.set_index("dataset").loc["signal_health"]

    assert not bundle.is_ready
    assert metadata["status"] == "MISSING"
    assert not metadata["exists"]
    assert "signal_health" not in bundle.monitoring
    assert "is missing" in bundle.errors[0]

    with pytest.raises(DashboardArtifactLoadError, match="signal_health is missing"):
        load_dashboard_artifacts(
            processed_directory,
            monitoring_directory,
        )


def test_dashboard_loader_reports_unreadable_parquet(
    dashboard_artifact_directories,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    invalid_path = (
        monitoring_directory / MONITORING_ARTIFACT_CONTRACTS["signal_health"].filename
    )
    invalid_path.write_text("not a parquet file", encoding="utf-8")

    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        strict=False,
    )
    metadata = bundle.metadata.set_index("dataset").loc["signal_health"]

    assert not bundle.is_ready
    assert metadata["status"] == "READ_ERROR"
    assert metadata["exists"]
    assert not metadata["loaded"]


def test_dashboard_loader_reports_malformed_artifact(
    dashboard_artifact_directories,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    invalid_path = (
        monitoring_directory / MONITORING_ARTIFACT_CONTRACTS["signal_health"].filename
    )
    malformed = pd.read_parquet(invalid_path).drop(columns="ic")
    malformed.to_parquet(invalid_path, index=False)

    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        strict=False,
    )
    metadata = bundle.metadata.set_index("dataset").loc["signal_health"]

    assert not bundle.is_ready
    assert metadata["status"] == "INVALID"
    assert metadata["loaded"]
    assert not metadata["artifact_valid"]
    assert "missing columns" in metadata["error"]


def test_dashboard_loader_reports_cross_artifact_validation_failure(
    dashboard_artifact_directories,
):
    processed_directory, monitoring_directory = dashboard_artifact_directories
    concentration_path = (
        monitoring_directory / MONITORING_ARTIFACT_CONTRACTS["concentration"].filename
    )
    concentration = pd.read_parquet(concentration_path).iloc[:-1]
    concentration.to_parquet(concentration_path, index=False)

    bundle = load_dashboard_artifacts(
        processed_directory,
        monitoring_directory,
        strict=False,
    )
    monitoring_metadata = bundle.metadata.loc[bundle.metadata["group"].eq("monitoring")]

    assert not bundle.is_ready
    assert not monitoring_metadata["group_valid"].any()
    assert monitoring_metadata["status"].eq("INVALID_GROUP").all()
    assert "coverage does not match beta" in bundle.errors[0]


@pytest.mark.parametrize("invalid_value", [-1, 1.5, True])
def test_dashboard_loader_rejects_invalid_freshness_limit(invalid_value):
    with pytest.raises(ValueError, match="non-negative integer"):
        load_dashboard_artifacts(stale_after_business_days=invalid_value)
