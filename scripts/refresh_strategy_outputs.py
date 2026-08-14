"""Reconstruct or write all frozen strategy outputs without notebooks."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from alpha_research.artifacts import (
    ATTRIBUTION_ARTIFACT_CONTRACTS,
    MONITORING_ARTIFACT_CONTRACTS,
    ArtifactContract,
    validate_attribution_artifacts,
    validate_monitoring_artifacts,
    write_attribution_artifacts,
    write_monitoring_artifacts,
)
from alpha_research.config.paths import (
    MONITORING_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from alpha_research.data_loader import load_parquet
from alpha_research.refresh import build_complete_research_refresh, key_values_match


def _compare_with_existing_artifacts(
    datasets: Mapping[str, pd.DataFrame],
    contracts: Mapping[str, ArtifactContract],
    directory: Path,
    group: str,
) -> pd.DataFrame:
    audit_rows = []

    for name, reconstructed in datasets.items():
        contract = contracts[name]
        reference = load_parquet(directory / contract.filename)
        key_columns = list(contract.key_columns)
        reconstructed = reconstructed.sort_values(key_columns).reset_index(drop=True)
        reference = reference.sort_values(key_columns).reset_index(drop=True)
        column_sets_match = set(reconstructed.columns) == set(reference.columns)
        keys_match = key_values_match(reference, reconstructed, key_columns)
        values_match = False

        if column_sets_match:
            try:
                pd.testing.assert_frame_equal(
                    reconstructed[reference.columns],
                    reference,
                    check_dtype=False,
                    check_categorical=False,
                    check_exact=False,
                    rtol=0.0,
                    atol=1e-12,
                )
                values_match = True
            except AssertionError as error:
                print(f"\n{group}.{name} mismatch:\n{error}")

        audit_passes = column_sets_match and keys_match and values_match

        audit_rows.append(
            {
                "group": group,
                "dataset": name,
                "reconstructed_rows": len(reconstructed),
                "reference_rows": len(reference),
                "column_sets_match": column_sets_match,
                "keys_match": keys_match,
                "values_match": values_match,
                "audit_passes": audit_passes,
            }
        )

        status = "PASS" if audit_passes else "FAIL"

        print(
            f"      Checked {group}.{name}: {status}",
            flush=True,
        )

    return pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild frozen attribution and monitoring outputs."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write refreshed artifacts; otherwise compare in memory only.",
    )
    args = parser.parse_args()

    mode = "write" if args.write else "dry run"

    print(
        f"[1/6] Loading input datasets ({mode})...",
        flush=True,
    )

    factor_panel = load_parquet(PROCESSED_DATA_DIR / "factor_panel.parquet")
    benchmark_prices = load_parquet(RAW_DATA_DIR / "spy_benchmark.parquet")

    print(
        f"      Loaded {len(factor_panel):,} factor-panel rows and "
        f"{len(benchmark_prices):,} benchmark-price rows.",
        flush=True,
    )

    print(
        "[2/6] Rebuilding attribution and monitoring datasets...",
        flush=True,
    )

    refreshed = build_complete_research_refresh(
        factor_panel,
        benchmark_prices,
    )

    print(
        "      In-memory rebuild complete.",
        flush=True,
    )

    print(
        "[3/6] Validating rebuilt datasets...",
        flush=True,
    )

    validate_attribution_artifacts(refreshed["attribution"])
    validate_monitoring_artifacts(refreshed["monitoring"])

    print(
        "      All artifact contracts and identities passed.",
        flush=True,
    )

    if not args.write:
        print(
            "[4/6] Comparing attribution artifacts...",
            flush=True,
        )

        attribution_audit = _compare_with_existing_artifacts(
            refreshed["attribution"],
            ATTRIBUTION_ARTIFACT_CONTRACTS,
            PROCESSED_DATA_DIR,
            group="attribution",
        )

        print(
            "[5/6] Comparing monitoring artifacts...",
            flush=True,
        )

        monitoring_audit = _compare_with_existing_artifacts(
            refreshed["monitoring"],
            MONITORING_ARTIFACT_CONTRACTS,
            MONITORING_DATA_DIR,
            group="monitoring",
        )

        print(
            "[6/6] Reporting dry-run reconciliation...",
            flush=True,
        )

        audit = pd.concat(
            [
                attribution_audit,
                monitoring_audit,
            ],
            ignore_index=True,
        )

        print(audit.to_string(index=False))
        print("\nAll refresh reconciliations pass: " f"{audit['audit_passes'].all()}")

        if not audit["audit_passes"].all():
            raise ValueError("Research refresh reconciliation failed.")

        print("\nDry run only: no artifacts were written.")
        return

    print(
        "[4/6] Writing attribution artifacts...",
        flush=True,
    )

    attribution_manifest = write_attribution_artifacts(
        refreshed["attribution"],
        PROCESSED_DATA_DIR,
    )

    print(
        "      Attribution artifacts written and read back.",
        flush=True,
    )

    print(
        "[5/6] Writing monitoring artifacts...",
        flush=True,
    )

    monitoring_manifest = write_monitoring_artifacts(
        refreshed["monitoring"],
        MONITORING_DATA_DIR,
    )

    print(
        "      Monitoring artifacts written and read back.",
        flush=True,
    )

    print(
        "[6/6] Reporting refreshed artifacts...",
        flush=True,
    )

    print("Attribution artifacts:")
    print(attribution_manifest.to_string(index=False))

    print("\nMonitoring artifacts:")
    print(monitoring_manifest.to_string(index=False))

    print("\nAll strategy outputs were refreshed successfully.")


if __name__ == "__main__":
    main()
