import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    DIAGNOSTIC_TABLE_COLUMNS,
    MONITORING_OVERVIEW_COLUMNS,
    prepare_diagnostic_table,
    prepare_monitoring_overview,
)


@pytest.fixture()
def diagnostic_inputs():
    flags = pd.DataFrame(
        [
            {
                "entity_type": "Factor",
                "entity": "Momentum",
                "category": "Signal",
                "diagnostic": "Coverage",
                "latest_date": "2026-07-02",
                "latest_value": 0.99,
                "adverse_direction": "Structural failure",
                "calibration": "Complete coverage",
                "threshold_value": 0.30,
                "historical_percentile": float("nan"),
                "status": "PASS",
                "notes": "",
            },
            {
                "entity_type": "Portfolio",
                "entity": "Alpha",
                "category": "Market risk",
                "diagnostic": "Volatility",
                "latest_date": "2026-07-01",
                "latest_value": 0.30,
                "adverse_direction": "Higher",
                "calibration": "Historical upper 10% tail",
                "threshold_value": 0.25,
                "historical_percentile": 0.95,
                "status": "WARNING",
                "notes": "",
            },
            {
                "entity_type": "Portfolio",
                "entity": "Alpha",
                "category": "Implementation",
                "diagnostic": "Liquidity coverage",
                "latest_date": "2026-07-01",
                "latest_value": 1.0,
                "adverse_direction": "Structural failure",
                "calibration": "Complete coverage",
                "threshold_value": 1.0,
                "historical_percentile": float("nan"),
                "status": "PASS",
                "notes": "",
            },
        ]
    )

    overview = pd.DataFrame(
        {
            "entity_type": [
                "Factor",
                "Portfolio",
            ],
            "entity": [
                "Momentum",
                "Alpha",
            ],
            "diagnostics": [1, 2],
            "passes": [1, 1],
            "warnings": [0, 1],
            "breaches": [0, 0],
            "unavailable": [0, 0],
            "overall_status": [
                "PASS",
                "WARNING",
            ],
            "signal_status": [
                "PASS",
                "N/A",
            ],
            "market_risk_status": [
                "N/A",
                "WARNING",
            ],
            "concentration_status": [
                "N/A",
                "N/A",
            ],
            "implementation_status": [
                "N/A",
                "PASS",
            ],
        }
    )

    return flags, overview


def test_prepare_diagnostic_table_filters_and_ranks_statuses(
    diagnostic_inputs,
):
    flags, _ = diagnostic_inputs

    table = prepare_diagnostic_table(
        flags,
        entity_types=["Portfolio"],
        statuses=["WARNING"],
    )

    assert tuple(table.columns) == (DIAGNOSTIC_TABLE_COLUMNS)
    assert len(table) == 1
    assert table.loc[0, "entity"] == "Alpha"
    assert table.loc[0, "diagnostic"] == "Volatility"
    assert table.loc[0, "status_severity"] == 1
    assert table.loc[
        0,
        "latest_date",
    ] == pd.Timestamp("2026-07-01")


def test_prepare_monitoring_overview_validates_and_filters_counts(
    diagnostic_inputs,
):
    _, overview = diagnostic_inputs

    prepared = prepare_monitoring_overview(
        overview,
        entities=["Alpha"],
    )

    assert tuple(prepared.columns) == (MONITORING_OVERVIEW_COLUMNS)
    assert prepared["entity"].tolist() == ["Alpha"]
    assert prepared.loc[0, "diagnostics"] == 2
    assert (
        prepared.loc[
            0,
            "overall_status",
        ]
        == "WARNING"
    )


def test_dashboard_diagnostics_reject_malformed_inputs(
    diagnostic_inputs,
):
    flags, overview = diagnostic_inputs

    duplicated = pd.concat(
        [flags, flags.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate diagnostic flags",
    ):
        prepare_diagnostic_table(duplicated)

    inconsistent = overview.copy()
    inconsistent.loc[0, "diagnostics"] = 2

    with pytest.raises(
        ValueError,
        match="counts do not reconcile",
    ):
        prepare_monitoring_overview(inconsistent)

    with pytest.raises(
        ValueError,
        match="unknown labels",
    ):
        prepare_diagnostic_table(
            flags,
            statuses=["ALERT"],
        )


@pytest.mark.parametrize(
    ("statuses", "error_type", "message"),
    (
        ("WARNING", TypeError, "sequence of labels"),
        ([], ValueError, "must not be empty"),
        (["PASS", "PASS"], ValueError, "unique labels"),
    ),
)
def test_dashboard_diagnostics_reject_invalid_label_filters(
    diagnostic_inputs,
    statuses,
    error_type,
    message,
):
    flags, _ = diagnostic_inputs

    with pytest.raises(error_type, match=message):
        prepare_diagnostic_table(flags, statuses=statuses)
