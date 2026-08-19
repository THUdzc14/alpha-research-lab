import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    PORTFOLIO_COLOURS,
    ROLLING_METRIC_SPECIFICATIONS,
    build_cumulative_performance_figure,
    build_drawdown_figure,
    build_rolling_metric_figure,
)


@pytest.fixture()
def performance_history():
    dates = pd.bdate_range("2026-07-01", periods=3)
    portfolio_values = {
        "Pure Inverse Volatility": {
            "indexed_wealth": [1.0, 1.01, 1.02],
            "drawdown": [0.0, -0.01, -0.005],
        },
        "Composite Score": {
            "indexed_wealth": [1.0, 0.98, 1.03],
            "drawdown": [0.0, -0.02, 0.0],
        },
        "SPY": {
            "indexed_wealth": [1.0, 0.99, 1.01],
            "drawdown": [0.0, -0.01, 0.0],
        },
    }
    rows = []

    for portfolio, values in portfolio_values.items():
        for observation, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "indexed_wealth": values["indexed_wealth"][observation],
                    "drawdown": values["drawdown"][observation],
                    "trailing_return_252": 0.05 + observation * 0.01,
                    "rolling_sharpe_252": 0.50 + observation * 0.10,
                    "annualised_volatility_126": 0.20 + observation * 0.01,
                    "maximum_drawdown_252": -0.15 - observation * 0.01,
                }
            )

    return pd.DataFrame(rows)


def test_cumulative_performance_figure_preserves_order_and_styles(
    performance_history,
):
    figure = build_cumulative_performance_figure(performance_history)
    trace_names = [trace.name for trace in figure.data]

    assert isinstance(figure, go.Figure)
    assert trace_names == ["Pure Inverse Volatility", "Composite Score", "SPY"]
    assert figure.data[0].line.color == PORTFOLIO_COLOURS["Pure Inverse Volatility"]
    assert figure.data[1].line.color == PORTFOLIO_COLOURS["Composite Score"]
    assert figure.data[2].line.color == PORTFOLIO_COLOURS["SPY"]
    assert figure.data[2].line.dash == "10px, 5px"
    assert all(isinstance(trace, go.Scatter) for trace in figure.data)
    assert all(trace.mode == "lines" for trace in figure.data)
    assert all(trace.legendgroup == trace.name for trace in figure.data)
    assert list(figure.data[1].y) == [1.0, 0.98, 1.03]
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.yaxis.title.text == "Indexed wealth (start = 1.0)"
    assert figure.layout.yaxis.tickformat == ".2f"


def test_drawdown_figure_uses_percent_format_and_zero_line(performance_history):
    figure = build_drawdown_figure(
        performance_history,
        title="Historical Drawdowns",
        height=520,
    )

    assert figure.layout.title.text == "Historical Drawdowns"
    assert figure.layout.height == 520
    assert figure.layout.yaxis.tickformat == ".0%"
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].y0 == 0.0
    assert figure.layout.shapes[0].y1 == 0.0
    assert np.allclose(figure.data[0].y, [0.0, -0.01, -0.005])


@pytest.mark.parametrize(
    ("metric", "tickformat", "expected_shapes"),
    [
        ("trailing_return_252", ".0%", 1),
        ("rolling_sharpe_252", ".2f", 1),
        ("annualised_volatility_126", ".0%", 0),
        ("maximum_drawdown_252", ".0%", 1),
    ],
)
def test_rolling_metric_figure_uses_metric_specification(
    performance_history,
    metric,
    tickformat,
    expected_shapes,
):
    figure = build_rolling_metric_figure(performance_history, metric)
    specification = ROLLING_METRIC_SPECIFICATIONS[metric]

    assert figure.layout.title.text == specification.title
    assert figure.layout.yaxis.title.text == specification.yaxis_title
    assert figure.layout.yaxis.tickformat == tickformat
    assert len(figure.layout.shapes) == expected_shapes
    assert len(figure.data) == 3


def test_rolling_metric_figure_rejects_unknown_metric(performance_history):
    with pytest.raises(ValueError, match="Unsupported rolling metric"):
        build_rolling_metric_figure(performance_history, "unknown_metric")


def test_figure_builders_reject_malformed_data(performance_history):
    with pytest.raises(KeyError, match="indexed_wealth"):
        build_cumulative_performance_figure(performance_history.drop(columns="indexed_wealth"))

    duplicated = pd.concat(
        [performance_history, performance_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate portfolio-date"):
        build_drawdown_figure(duplicated)

    invalid_numeric = performance_history.copy()
    invalid_numeric["rolling_sharpe_252"] = invalid_numeric["rolling_sharpe_252"].astype("object")
    invalid_numeric.loc[0, "rolling_sharpe_252"] = "invalid"

    with pytest.raises(ValueError, match="non-numeric"):
        build_rolling_metric_figure(invalid_numeric, "rolling_sharpe_252")

    with pytest.raises(ValueError, match="positive integer"):
        build_drawdown_figure(performance_history, height=0)

    with pytest.raises(ValueError, match="positive integer"):
        build_drawdown_figure(performance_history, height=True)

    with pytest.raises(ValueError, match="must not be empty"):
        build_cumulative_performance_figure(performance_history.iloc[0:0])
