"""Rebuild validated monitoring artifacts for the frozen strategies."""

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


def main() -> None:
    inputs = {
        name: load_parquet(PROCESSED_DATA_DIR / filename)
        for name, filename in INPUT_FILENAMES.items()
    }
    datasets = build_strategy_monitoring_datasets(**inputs)
    manifest = write_monitoring_artifacts(
        datasets,
        MONITORING_DATA_DIR,
    )

    print(manifest.to_string(index=False))
    print(f"\nMonitoring artifacts written to: {MONITORING_DATA_DIR}")


if __name__ == "__main__":
    main()
