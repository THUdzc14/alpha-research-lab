import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    FACTOR_COLOURS,
    FACTOR_DEPENDENCE_STYLES,
    SIGNAL_METRIC_SPECIFICATIONS,
    build_factor_dependence_figure,
    build_signal_health_figure,
)


@pytest.fixture()
def signal_history():
    dates = pd.bdate_range("2026-06-29", periods=3)

    return pd.DataFrame(
        [
            {
                "date": date,
                "factor": factor,
                "signal_coverage": 0.98,
                "raw_iqr": 1.0 + date_number * 0.10,
                "ic": 0.01 + date_number * 0.001,
                "rolling_mean_ic_252": 0.02 + factor_number * 0.01,
                "rank_stability_1d": 0.98 - factor_number * 0.01,
                "rank_stability_21d": 0.75 - factor_number * 0.02,
            }
            for factor_number, factor in enumerate(("Realised Volatility", "Momentum"))
            for date_number, date in enumerate(dates)
        ]
    )


@pytest.fixture()
def dependence_history():
    dates = pd.bdate_range("2026-06-29", periods=3)

    return pd.DataFrame(
        {
            "date": dates,
            "factor_rank_correlation": [0.20, 0.30, 0.25],
            "observations": [100, 100, 100],
            "rolling_factor_rank_correlation_252": [0.22, 0.23, 0.24],
        }
    )


@pytest.mark.parametrize(
    ("metric", "tickformat", "expected_zero_lines"),
    [
        ("signal_coverage", ".0%", 0),
        ("raw_iqr", ".3f", 0),
        ("ic", ".3f", 1),
        ("rolling_mean_ic_252", ".3f", 1),
        ("rank_stability_1d", ".2f", 1),
        ("rank_stability_21d", ".2f", 1),
    ],
)
def test_signal_health_figure_uses_metric_styles(
    signal_history,
    metric,
    tickformat,
    expected_zero_lines,
):
    figure = build_signal_health_figure(signal_history, metric)
    specification = SIGNAL_METRIC_SPECIFICATIONS[metric]

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == [
        "Realised Volatility",
        "Momentum",
    ]
    assert figure.data[0].line.color == FACTOR_COLOURS["Realised Volatility"]
    assert figure.data[1].line.color == FACTOR_COLOURS["Momentum"]
    assert all(isinstance(trace, go.Scatter) for trace in figure.data)
    assert all(trace.legendgroup == trace.name for trace in figure.data)
    assert figure.layout.title.text == specification.title
    assert figure.layout.yaxis.tickformat == tickformat
    assert len(figure.layout.shapes) == expected_zero_lines


def test_factor_dependence_figure_uses_daily_and_rolling_styles(
    dependence_history,
):
    figure = build_factor_dependence_figure(dependence_history)
    trace_names = [trace.name for trace in figure.data]

    assert trace_names == [
        "Daily factor rank correlation",
        "Rolling 252-day average",
    ]
    assert len(figure.layout.shapes) == 1

    for trace in figure.data:
        expected = FACTOR_DEPENDENCE_STYLES[trace.name]

        assert trace.line.color == expected["color"]
        assert trace.line.width == expected["width"]
        assert trace.opacity == expected["opacity"]


def test_signal_figures_reject_unsupported_or_malformed_data(
    signal_history,
    dependence_history,
):
    with pytest.raises(ValueError, match="Unsupported signal-health metric"):
        build_signal_health_figure(signal_history, "unsupported")

    with pytest.raises(KeyError, match="rolling_factor_rank_correlation_252"):
        build_factor_dependence_figure(
            dependence_history.drop(columns="rolling_factor_rank_correlation_252")
        )

    duplicated = pd.concat(
        [signal_history, signal_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate factor-date"):
        build_signal_health_figure(duplicated, "ic")

    missing_factor = signal_history.copy()
    missing_factor.loc[0, "factor"] = pd.NA

    with pytest.raises(ValueError, match="data.factor contains missing values"):
        build_signal_health_figure(missing_factor, "ic")
