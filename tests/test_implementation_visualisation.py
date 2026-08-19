import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    IMPLEMENTATION_METRIC_SPECIFICATIONS,
    PORTFOLIO_COLOURS,
    build_implementation_figure,
    build_liquidity_coverage_figure,
)


@pytest.fixture()
def implementation_histories():
    dates = pd.bdate_range("2026-06-29", periods=3)
    implementation_rows = []
    liquidity_rows = []

    portfolios = (
        "Composite Score",
        "Fixed 50/50 Sleeves",
    )

    for portfolio_number, portfolio in enumerate(portfolios):
        for date_number, date in enumerate(dates):
            implementation_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "annualised_turnover_63": (8.0 + portfolio_number),
                    "largest_trade_weight_63": (0.04 + date_number * 0.001),
                    "minimum_trade_capacity_1pct_usd_millions_63": (50.0 + portfolio_number * 10.0),
                    "maximum_missing_return_weight_63": 0.0,
                }
            )

            liquidity_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "liquidity_coverage": 1.0,
                }
            )

    return (
        pd.DataFrame(implementation_rows),
        pd.DataFrame(liquidity_rows),
    )


@pytest.mark.parametrize(
    ("metric", "tickformat"),
    [
        ("annualised_turnover_63", ".1f"),
        ("largest_trade_weight_63", ".0%"),
        (
            "minimum_trade_capacity_1pct_usd_millions_63",
            ".1f",
        ),
        ("maximum_missing_return_weight_63", ".1%"),
    ],
)
def test_implementation_figure_uses_metric_specification(
    implementation_histories,
    metric,
    tickformat,
):
    implementation, _ = implementation_histories
    figure = build_implementation_figure(
        implementation,
        metric,
    )

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == [
        "Composite Score",
        "Fixed 50/50 Sleeves",
    ]
    assert figure.data[0].line.color == (PORTFOLIO_COLOURS["Composite Score"])
    assert figure.layout.title.text == (IMPLEMENTATION_METRIC_SPECIFICATIONS[metric].title)
    assert figure.layout.yaxis.tickformat == tickformat
    assert len(figure.layout.shapes) == 0
    assert all(isinstance(trace, go.Scatter) for trace in figure.data)


def test_liquidity_coverage_figure_uses_percent_format(
    implementation_histories,
):
    _, liquidity = implementation_histories
    figure = build_liquidity_coverage_figure(liquidity)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.layout.yaxis.tickformat == ".0%"
    assert len(figure.layout.shapes) == 0


def test_implementation_figures_reject_unsupported_or_malformed_data(
    implementation_histories,
):
    implementation, liquidity = implementation_histories

    with pytest.raises(
        ValueError,
        match="Unsupported implementation metric",
    ):
        build_implementation_figure(
            implementation,
            "unsupported",
        )

    with pytest.raises(
        KeyError,
        match="liquidity_coverage",
    ):
        build_liquidity_coverage_figure(liquidity.drop(columns="liquidity_coverage"))

    with pytest.raises(ValueError, match="height must be a positive integer"):
        build_implementation_figure(
            implementation,
            "annualised_turnover_63",
            height=0,
        )
