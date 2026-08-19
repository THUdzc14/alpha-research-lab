import pandas as pd
import plotly.graph_objects as go
import pytest

from alpha_research.visualisation import (
    build_security_contribution_figure,
    build_security_contribution_share_figure,
)


@pytest.fixture()
def security_summary():
    return pd.DataFrame(
        {
            "portfolio": ["Alpha"] * 5,
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            "cumulative_net_contribution": [
                0.10,
                0.05,
                0.01,
                -0.02,
                -0.08,
            ],
            "absolute_gross_contribution": [
                0.20,
                0.10,
                0.05,
                0.04,
                0.16,
            ],
            "absolute_contribution_share": [
                0.20 / 0.55,
                0.10 / 0.55,
                0.05 / 0.55,
                0.04 / 0.55,
                0.16 / 0.55,
            ],
        }
    )


def test_signed_figure_selects_both_tails(
    security_summary,
):
    figure = build_security_contribution_figure(
        security_summary,
        top_n=2,
        title="Signed Contributors",
        height=540,
    )

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Signed Contributors"
    assert figure.layout.height == 540
    assert len(figure.data) == 1
    assert set(figure.data[0].y) == {
        "AAA",
        "BBB",
        "DDD",
        "EEE",
    }
    assert list(figure.data[0].x) == sorted(figure.data[0].x)
    assert len(figure.layout.shapes) == 1


def test_share_figure_selects_largest(
    security_summary,
):
    figure = build_security_contribution_share_figure(
        security_summary,
        top_n=3,
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert set(figure.data[0].y) == {
        "AAA",
        "BBB",
        "EEE",
    }
    assert list(figure.data[0].x) == sorted(figure.data[0].x)


def test_figures_reject_malformed_data(
    security_summary,
):
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        build_security_contribution_figure(
            security_summary,
            top_n=0,
        )

    with pytest.raises(ValueError, match="positive integer"):
        build_security_contribution_figure(
            security_summary,
            top_n=True,
        )

    with pytest.raises(
        KeyError,
        match="absolute_contribution_share",
    ):
        build_security_contribution_share_figure(
            security_summary.drop(columns="absolute_contribution_share")
        )

    multiple_portfolios = security_summary.copy()
    multiple_portfolios.loc[
        0,
        "portfolio",
    ] = "Beta"

    with pytest.raises(
        ValueError,
        match="exactly one portfolio",
    ):
        build_security_contribution_figure(multiple_portfolios)

    missing_ticker = security_summary.copy()
    missing_ticker.loc[0, "ticker"] = pd.NA

    with pytest.raises(ValueError, match="missing portfolio or ticker labels"):
        build_security_contribution_share_figure(missing_ticker)

    with pytest.raises(ValueError, match="height must be a positive integer"):
        build_security_contribution_share_figure(security_summary, height=0)
