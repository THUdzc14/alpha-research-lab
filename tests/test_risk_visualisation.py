import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    BETA_METRIC_SPECIFICATIONS,
    CONCENTRATION_METRIC_SPECIFICATIONS,
    PORTFOLIO_COLOURS,
    build_beta_figure,
    build_concentration_figure,
)


@pytest.fixture()
def risk_histories():
    dates = pd.bdate_range("2026-06-29", periods=3)
    beta_rows = []
    concentration_rows = []

    portfolios = (
        "Composite Score",
        "Fixed 50/50 Sleeves",
    )

    for portfolio_number, portfolio in enumerate(portfolios):
        for date_number, date in enumerate(dates):
            beta_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "beta_coverage": 1.0,
                    "holdings_market_beta": (0.80 + portfolio_number * 0.10 + date_number * 0.01),
                    "realised_gross_beta_126": (0.70 + date_number * 0.01),
                    "beta_measurement_gap": (0.10 + portfolio_number * 0.10),
                }
            )

            concentration_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "effective_position_count": (40.0 + date_number),
                    "largest_absolute_sector_net_exposure": 0.30,
                    "top_five_absolute_beta_contribution_share": (0.40),
                    "effective_contributor_count_63": 25.0,
                    "top_five_contributor_share_63": 0.35,
                    "effective_contribution_sector_count_63": 5.0,
                }
            )

    return (
        pd.DataFrame(beta_rows),
        pd.DataFrame(concentration_rows),
    )


@pytest.mark.parametrize(
    ("metric", "tickformat", "zero_lines"),
    [
        ("beta_coverage", ".0%", 0),
        ("holdings_market_beta", ".2f", 1),
        ("realised_gross_beta_126", ".2f", 1),
        ("beta_measurement_gap", ".2f", 1),
    ],
)
def test_beta_figure_uses_metric_specification(
    risk_histories,
    metric,
    tickformat,
    zero_lines,
):
    beta, _ = risk_histories
    figure = build_beta_figure(beta, metric)

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == [
        "Composite Score",
        "Fixed 50/50 Sleeves",
    ]
    assert figure.data[0].line.color == (PORTFOLIO_COLOURS["Composite Score"])
    assert figure.layout.title.text == (BETA_METRIC_SPECIFICATIONS[metric].title)
    assert figure.layout.yaxis.tickformat == tickformat
    assert len(figure.layout.shapes) == zero_lines
    assert all(isinstance(trace, go.Scatter) for trace in figure.data)


@pytest.mark.parametrize(
    ("metric", "tickformat"),
    [
        ("effective_position_count", ".1f"),
        (
            "largest_absolute_sector_net_exposure",
            ".0%",
        ),
        (
            "top_five_absolute_beta_contribution_share",
            ".0%",
        ),
        ("effective_contributor_count_63", ".1f"),
        ("top_five_contributor_share_63", ".0%"),
        (
            "effective_contribution_sector_count_63",
            ".1f",
        ),
    ],
)
def test_concentration_figure_uses_metric_specification(
    risk_histories,
    metric,
    tickformat,
):
    _, concentration = risk_histories
    figure = build_concentration_figure(
        concentration,
        metric,
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.layout.title.text == (CONCENTRATION_METRIC_SPECIFICATIONS[metric].title)
    assert figure.layout.yaxis.tickformat == tickformat
    assert len(figure.layout.shapes) == 0


def test_risk_figures_reject_unsupported_or_malformed_data(
    risk_histories,
):
    beta, concentration = risk_histories

    with pytest.raises(
        ValueError,
        match="Unsupported beta metric",
    ):
        build_beta_figure(beta, "unsupported")

    with pytest.raises(
        ValueError,
        match="Unsupported concentration metric",
    ):
        build_concentration_figure(
            concentration,
            "unsupported",
        )

    with pytest.raises(
        KeyError,
        match="effective_position_count",
    ):
        build_concentration_figure(
            concentration.drop(columns="effective_position_count"),
            "effective_position_count",
        )

    with pytest.raises(ValueError, match="height must be a positive integer"):
        build_beta_figure(beta, "beta_coverage", height=False)
