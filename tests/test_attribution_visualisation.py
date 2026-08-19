import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    ATTRIBUTION_COMPONENT_STYLES,
    build_cumulative_attribution_figure,
    build_side_cost_attribution_figure,
)


@pytest.fixture()
def attribution_inputs():
    dates = pd.bdate_range(
        "2026-06-29",
        periods=3,
    )
    history = pd.DataFrame(
        [
            {
                "date": date,
                "portfolio": portfolio,
                "cumulative_long_contribution": (0.01 * (date_number + 1)),
                "cumulative_short_contribution": (-0.003 * (date_number + 1)),
                "cumulative_cost_contribution": (-0.0005 * (date_number + 1)),
                "cumulative_net_contribution": (0.0065 * (date_number + 1)),
            }
            for portfolio in ("Alpha", "Beta")
            for date_number, date in enumerate(dates)
        ]
    )
    summary = pd.DataFrame(
        {
            "portfolio": ["Alpha", "Beta"],
            "annualised_long_contribution": [
                0.12,
                0.10,
            ],
            "annualised_short_contribution": [
                -0.04,
                -0.03,
            ],
            "annualised_cost_drag": [
                0.01,
                0.008,
            ],
        }
    )

    return history, summary


def test_side_cost_figure_uses_signed_bars(
    attribution_inputs,
):
    _, summary = attribution_inputs
    figure = build_side_cost_attribution_figure(summary)

    assert isinstance(figure, go.Figure)
    assert figure.layout.barmode == "group"
    assert [trace.name for trace in figure.data] == [
        "Long side",
        "Short side",
        "Transaction costs",
    ]
    assert list(figure.data[2].y) == [-0.01, -0.008]
    assert len(figure.layout.shapes) == 1


def test_cumulative_figure_selects_portfolio(
    attribution_inputs,
):
    history, _ = attribution_inputs
    figure = build_cumulative_attribution_figure(
        history,
        "Beta",
        title="Beta Attribution",
        height=510,
    )

    assert [trace.name for trace in figure.data] == list(ATTRIBUTION_COMPONENT_STYLES)
    assert figure.layout.title.text == "Beta Attribution"
    assert figure.layout.height == 510
    assert all(isinstance(trace, go.Scatter) for trace in figure.data)
    assert all(len(trace.x) == 3 for trace in figure.data)
    assert figure.data[0].line.color == (ATTRIBUTION_COMPONENT_STYLES["Long side"]["color"])
    assert figure.data[3].line.width == (ATTRIBUTION_COMPONENT_STYLES["Net contribution"]["width"])
    assert len(figure.layout.shapes) == 1


def test_figures_reject_malformed_data(
    attribution_inputs,
):
    history, summary = attribution_inputs

    with pytest.raises(
        KeyError,
        match="annualised_cost_drag",
    ):
        build_side_cost_attribution_figure(summary.drop(columns="annualised_cost_drag"))

    duplicated_summary = pd.concat(
        [summary, summary.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate portfolios",
    ):
        build_side_cost_attribution_figure(duplicated_summary)

    with pytest.raises(
        ValueError,
        match="missing portfolio",
    ):
        build_cumulative_attribution_figure(
            history,
            "Missing",
        )

    missing_portfolio = summary.copy()
    missing_portfolio.loc[0, "portfolio"] = pd.NA

    with pytest.raises(ValueError, match="portfolio contains missing values"):
        build_side_cost_attribution_figure(missing_portfolio)

    with pytest.raises(ValueError, match="height must be a positive integer"):
        build_cumulative_attribution_figure(history, "Alpha", height=False)
