from unittest.mock import Mock

import pandas as pd
import pytest

from dashboard import streamlit_app
from alpha_research.dashboard_ui import DASHBOARD_PAGES


PAGE_CASES = (
    (
        "Strategy Overview",
        "render_strategy_overview",
        ("bundle", ("Portfolio A", "Portfolio B")),
    ),
    (
        "Performance & Drawdowns",
        "render_performance_page",
        (
            "bundle",
            ("Portfolio A", "Portfolio B"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-12-31"),
        ),
    ),
    (
        "Factor & Signal Health",
        "render_signal_health_page",
        (
            "bundle",
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-12-31"),
        ),
    ),
    (
        "Risk & Concentration",
        "render_risk_concentration_page",
        (
            "bundle",
            ("Portfolio A", "Portfolio B"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-12-31"),
        ),
    ),
    (
        "Implementation & Liquidity",
        "render_implementation_liquidity_page",
        (
            "bundle",
            ("Portfolio A", "Portfolio B"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-12-31"),
        ),
    ),
    (
        "Attribution",
        "render_attribution_page",
        (
            "bundle",
            ("Portfolio A", "Portfolio B"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-12-31"),
        ),
    ),
)


@pytest.mark.parametrize(("page", "renderer_name", "expected_arguments"), PAGE_CASES)
def test_render_dashboard_page_dispatches_supported_page(
    monkeypatch,
    page,
    renderer_name,
    expected_arguments,
):
    renderer = Mock()
    monkeypatch.setattr(streamlit_app, renderer_name, renderer)

    streamlit_app.render_dashboard_page(
        page,
        "bundle",
        ("Portfolio A", "Portfolio B"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-12-31"),
    )

    renderer.assert_called_once_with(*expected_arguments)


def test_page_dispatch_cases_cover_sidebar_pages_in_order():
    assert tuple(case[0] for case in PAGE_CASES) == DASHBOARD_PAGES


def test_render_dashboard_page_rejects_unsupported_page(monkeypatch):
    renderers = (
        "render_strategy_overview",
        "render_performance_page",
        "render_signal_health_page",
        "render_risk_concentration_page",
        "render_implementation_liquidity_page",
        "render_attribution_page",
    )
    mocks = []

    for renderer_name in renderers:
        renderer = Mock()
        monkeypatch.setattr(streamlit_app, renderer_name, renderer)
        mocks.append(renderer)

    with pytest.raises(ValueError, match="Unsupported dashboard page: Unknown"):
        streamlit_app.render_dashboard_page(
            "Unknown",
            "bundle",
            ("Portfolio A",),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-12-31"),
        )

    assert all(not renderer.called for renderer in mocks)
