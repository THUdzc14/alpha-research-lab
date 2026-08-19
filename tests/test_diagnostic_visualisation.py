import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    DIAGNOSTIC_COUNT_STYLES,
    MONITORING_STATUS_LABELS,
    STATUS_CODES,
    STATUS_COLOURS,
    build_diagnostic_count_figure,
    build_monitoring_status_heatmap,
)


@pytest.fixture()
def monitoring_overview():
    return pd.DataFrame(
        {
            "entity_type": [
                "Factor",
                "Portfolio",
                "Portfolio",
            ],
            "entity": [
                "Momentum",
                "Alpha",
                "Beta",
            ],
            "diagnostics": [2, 4, 4],
            "passes": [2, 2, 3],
            "warnings": [0, 1, 0],
            "breaches": [0, 1, 0],
            "unavailable": [0, 0, 1],
            "overall_status": [
                "PASS",
                "BREACH",
                "UNAVAILABLE",
            ],
            "signal_status": [
                "PASS",
                "N/A",
                "N/A",
            ],
            "market_risk_status": [
                "N/A",
                "WARNING",
                "PASS",
            ],
            "concentration_status": [
                "N/A",
                "BREACH",
                "UNAVAILABLE",
            ],
            "implementation_status": [
                "N/A",
                "PASS",
                "PASS",
            ],
        }
    )


def test_monitoring_status_heatmap_encodes_status_matrix(
    monitoring_overview,
):
    figure = build_monitoring_status_heatmap(
        monitoring_overview,
        title="Current Monitoring Status",
        height=500,
    )
    heatmap = figure.data[0]

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Current Monitoring Status"
    assert figure.layout.height == 500
    assert list(heatmap.x) == list(MONITORING_STATUS_LABELS.values())
    assert list(heatmap.y) == [
        "Momentum",
        "Alpha",
        "Beta",
    ]
    assert np.asarray(heatmap.z).shape == (3, 4)
    assert heatmap.z[0][0] == STATUS_CODES["PASS"]
    assert heatmap.z[1][2] == STATUS_CODES["BREACH"]
    assert heatmap.z[2][2] == (STATUS_CODES["UNAVAILABLE"])
    assert heatmap.showscale is False


def test_diagnostic_count_figure_uses_stacked_status_colours(
    monitoring_overview,
):
    figure = build_diagnostic_count_figure(monitoring_overview)

    assert isinstance(figure, go.Figure)
    assert figure.layout.barmode == "stack"
    assert [trace.name for trace in figure.data] == [
        style[0] for style in DIAGNOSTIC_COUNT_STYLES.values()
    ]
    assert [trace.marker.color for trace in figure.data] == [
        STATUS_COLOURS["PASS"],
        STATUS_COLOURS["WARNING"],
        STATUS_COLOURS["BREACH"],
        STATUS_COLOURS["UNAVAILABLE"],
    ]
    assert list(figure.data[1].y) == [0, 1, 0]


def test_diagnostic_figures_reject_malformed_data(
    monitoring_overview,
):
    invalid_status = monitoring_overview.copy()
    invalid_status.loc[
        0,
        "signal_status",
    ] = "ALERT"

    with pytest.raises(
        ValueError,
        match="unknown statuses",
    ):
        build_monitoring_status_heatmap(invalid_status)

    invalid_count = monitoring_overview.copy()
    invalid_count["warnings"] = invalid_count["warnings"].astype("object")
    invalid_count.loc[0, "warnings"] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid counts",
    ):
        build_diagnostic_count_figure(invalid_count)

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        build_monitoring_status_heatmap(
            monitoring_overview,
            height=0,
        )

    missing_entity = monitoring_overview.copy()
    missing_entity.loc[0, "entity"] = pd.NA

    with pytest.raises(ValueError, match="missing entity labels"):
        build_monitoring_status_heatmap(missing_entity)

    with pytest.raises(ValueError, match="entity contains missing values"):
        build_diagnostic_count_figure(missing_entity)
