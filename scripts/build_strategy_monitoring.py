"""Rebuild validated monitoring artifacts for the frozen strategies."""

from pathlib import Path

import pandas as pd

from alpha_research.artifacts import write_monitoring_artifacts
from alpha_research.config.paths import (
    MONITORING_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from alpha_research.data_loader import load_parquet
from alpha_research.workflows import build_strategy_monitoring_datasets

INPUT_FILENAMES = {
    "factor_panel": "factor_panel.parquet",
    "selected_implementations": "attribution_selected_implementations.parquet",
    "portfolio_daily": "attribution_portfolio_daily.parquet",
    "security_holdings": "attribution_security_holdings.parquet",
    "security_daily": "attribution_security_daily.parquet",
    "benchmark_daily": "attribution_benchmark_daily.parquet",
}


def rebuild_monitoring_artifacts(
    processed_directory: Path = PROCESSED_DATA_DIR,
    monitoring_directory: Path = MONITORING_DATA_DIR,
) -> pd.DataFrame:
    """Load frozen inputs, rebuild monitoring datasets, and persist them."""
    inputs = {
        name: load_parquet(processed_directory / filename)
        for name, filename in INPUT_FILENAMES.items()
    }
    datasets = build_strategy_monitoring_datasets(**inputs)
    return write_monitoring_artifacts(
        datasets,
        monitoring_directory,
    )


def main() -> None:
    manifest = rebuild_monitoring_artifacts()
    print(manifest.to_string(index=False))
    print(f"\nMonitoring artifacts written to: {MONITORING_DATA_DIR}")


if __name__ == "__main__":
    main()
