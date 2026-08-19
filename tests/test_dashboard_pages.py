from types import SimpleNamespace
from unittest.mock import Mock, call

import pandas as pd
import pytest

from dashboard import dashboard_pages


def test_render_figure_grid_places_figures_in_alternating_columns(monkeypatch):
    columns = [Mock(), Mock()]
    figures = [Mock(name=f"figure_{position}") for position in range(4)]
    streamlit_columns = Mock(return_value=columns)
    monkeypatch.setattr(dashboard_pages.st, "columns", streamlit_columns)

    dashboard_pages._render_figure_grid(figures)

    streamlit_columns.assert_called_once_with(2)
    columns[0].plotly_chart.assert_has_calls(
        [
            call(
                figures[0],
                width="stretch",
                config=dashboard_pages.PLOTLY_CONFIG,
            ),
            call(
                figures[2],
                width="stretch",
                config=dashboard_pages.PLOTLY_CONFIG,
            ),
        ]
    )
    columns[1].plotly_chart.assert_has_calls(
        [
            call(
                figures[1],
                width="stretch",
                config=dashboard_pages.PLOTLY_CONFIG,
            ),
            call(
                figures[3],
                width="stretch",
                config=dashboard_pages.PLOTLY_CONFIG,
            ),
        ]
    )


def test_build_latest_risk_snapshot_selects_aligned_portfolio_rows():
    beta_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A", "Portfolio A", "Portfolio B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "holdings_market_beta": [0.1, 0.2, 0.3],
        }
    )
    concentration_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A", "Portfolio A", "Portfolio B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "effective_position_count": [10.0, 11.0, 12.0],
        }
    )

    result = dashboard_pages._build_latest_risk_snapshot(
        beta_history,
        concentration_history,
    )

    expected = pd.DataFrame(
        {
            "portfolio": ["Portfolio A", "Portfolio B"],
            "latest_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "holdings_market_beta": [0.2, 0.3],
            "effective_position_count": [11.0, 12.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_latest_risk_snapshot_rejects_misaligned_dates():
    beta_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A"],
            "date": pd.to_datetime(["2024-01-03"]),
        }
    )
    concentration_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A"],
            "date": pd.to_datetime(["2024-01-02"]),
        }
    )

    with pytest.raises(ValueError, match="beta and concentration dates"):
        dashboard_pages._build_latest_risk_snapshot(
            beta_history,
            concentration_history,
        )


def test_build_latest_implementation_snapshot_preserves_window_diagnostics():
    implementation_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A", "Portfolio A", "Portfolio B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "trade_count": [1, 2, 3],
        }
    )
    liquidity_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A", "Portfolio A", "Portfolio B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "liquidity_coverage": [0.75, 1.0, None],
        }
    )

    result = dashboard_pages._build_latest_implementation_snapshot(
        implementation_history,
        liquidity_history,
    )

    expected = pd.DataFrame(
        {
            "portfolio": ["Portfolio A", "Portfolio B"],
            "latest_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "trade_count": [2, 3],
            "latest_liquidity_coverage": [1.0, float("nan")],
            "minimum_liquidity_coverage": [0.75, float("nan")],
            "incomplete_coverage_days": [1, 1],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_latest_implementation_snapshot_rejects_misaligned_dates():
    implementation_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A"],
            "date": pd.to_datetime(["2024-01-03"]),
        }
    )
    liquidity_history = pd.DataFrame(
        {
            "portfolio": ["Portfolio A"],
            "date": pd.to_datetime(["2024-01-02"]),
            "liquidity_coverage": [1.0],
        }
    )

    with pytest.raises(ValueError, match="implementation and liquidity dates"):
        dashboard_pages._build_latest_implementation_snapshot(
            implementation_history,
            liquidity_history,
        )


def test_render_performance_page_preserves_filters_and_chart_metrics(monkeypatch):
    source = Mock(name="performance_source")
    bundle = SimpleNamespace(monitoring={"performance_risk": source})
    start_date = pd.Timestamp("2024-01-02")
    end_date = pd.Timestamp("2024-12-31")
    performance_history = Mock(name="performance_history")
    performance_summary = Mock(name="performance_summary")
    prepare_history = Mock(return_value=performance_history)
    build_summary = Mock(return_value=performance_summary)
    render_summary = Mock()
    render_grid = Mock()
    cumulative_figure = Mock(return_value="cumulative")
    drawdown_figure = Mock(return_value="drawdown")
    rolling_figure = Mock(side_effect=["sharpe", "volatility"])
    monkeypatch.setattr(dashboard_pages, "prepare_performance_history", prepare_history)
    monkeypatch.setattr(dashboard_pages, "build_performance_summary", build_summary)
    monkeypatch.setattr(dashboard_pages, "_render_performance_summary", render_summary)
    monkeypatch.setattr(
        dashboard_pages,
        "build_cumulative_performance_figure",
        cumulative_figure,
    )
    monkeypatch.setattr(dashboard_pages, "build_drawdown_figure", drawdown_figure)
    monkeypatch.setattr(dashboard_pages, "build_rolling_metric_figure", rolling_figure)
    monkeypatch.setattr(dashboard_pages, "_render_figure_grid", render_grid)
    monkeypatch.setattr(dashboard_pages.st, "header", Mock())
    monkeypatch.setattr(dashboard_pages.st, "caption", Mock())
    monkeypatch.setattr(dashboard_pages.st, "subheader", Mock())

    dashboard_pages.render_performance_page(
        bundle,
        ("Portfolio A", "Portfolio B"),
        start_date,
        end_date,
    )

    expected_arguments = {
        "portfolios": ["Portfolio A", "Portfolio B", "SPY"],
        "start_date": start_date,
        "end_date": end_date,
    }
    prepare_history.assert_called_once_with(source, **expected_arguments)
    build_summary.assert_called_once_with(source, **expected_arguments)
    render_summary.assert_called_once_with(performance_summary)
    cumulative_figure.assert_called_once_with(performance_history, height=430)
    drawdown_figure.assert_called_once_with(performance_history, height=430)
    assert rolling_figure.call_args_list == [
        call(performance_history, "rolling_sharpe_252", height=430),
        call(performance_history, "annualised_volatility_126", height=430),
    ]
    assert render_grid.call_args_list == [
        call(["cumulative", "drawdown"]),
        call(["sharpe", "volatility"]),
    ]


def test_render_signal_health_page_preserves_date_filters_and_chart_order(monkeypatch):
    latest_overview = Mock(name="latest_overview")
    signal_source = Mock(name="signal_source")
    dependence_source = Mock(name="dependence_source")
    bundle = SimpleNamespace(
        monitoring={
            "latest_overview": latest_overview,
            "signal_health": signal_source,
            "factor_dependence": dependence_source,
        }
    )
    start_date = pd.Timestamp("2024-01-02")
    end_date = pd.Timestamp("2024-12-31")
    snapshot = pd.DataFrame(
        {
            "factor": ["Momentum", "Value"],
            "latest_date": pd.to_datetime(["2024-12-30", "2024-12-31"]),
            "signal_status": ["PASS", "WARNING"],
        }
    )
    signal_history = Mock(name="signal_history")
    dependence_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-31"]),
            "rolling_factor_rank_correlation_252": [0.125],
        }
    )
    build_snapshot = Mock(return_value=snapshot)
    prepare_signal = Mock(return_value=signal_history)
    prepare_dependence = Mock(return_value=dependence_history)
    render_snapshot = Mock()
    render_grid = Mock()
    signal_figure = Mock(side_effect=["coverage", "mean_ic", "stability"])
    dependence_figure = Mock(return_value="dependence")
    monkeypatch.setattr(dashboard_pages, "build_latest_factor_snapshot", build_snapshot)
    monkeypatch.setattr(dashboard_pages, "prepare_signal_health_history", prepare_signal)
    monkeypatch.setattr(
        dashboard_pages,
        "prepare_factor_dependence_history",
        prepare_dependence,
    )
    monkeypatch.setattr(dashboard_pages, "_render_factor_snapshot", render_snapshot)
    monkeypatch.setattr(dashboard_pages, "build_signal_health_figure", signal_figure)
    monkeypatch.setattr(
        dashboard_pages,
        "build_factor_dependence_figure",
        dependence_figure,
    )
    monkeypatch.setattr(dashboard_pages, "_render_figure_grid", render_grid)
    monkeypatch.setattr(dashboard_pages.st, "header", Mock())
    monkeypatch.setattr(dashboard_pages.st, "caption", Mock())
    monkeypatch.setattr(dashboard_pages.st, "subheader", Mock())
    monkeypatch.setattr(
        dashboard_pages.st,
        "columns",
        Mock(return_value=[Mock() for _ in range(4)]),
    )

    dashboard_pages.render_signal_health_page(bundle, start_date, end_date)

    build_snapshot.assert_called_once_with(latest_overview, signal_source)
    prepare_signal.assert_called_once_with(
        signal_source,
        factors=["Momentum", "Value"],
        start_date=start_date,
        end_date=end_date,
    )
    prepare_dependence.assert_called_once_with(
        dependence_source,
        start_date=start_date,
        end_date=end_date,
    )
    render_snapshot.assert_called_once_with(snapshot)
    assert signal_figure.call_args_list == [
        call(signal_history, "signal_coverage", height=430),
        call(signal_history, "rolling_mean_ic_252", height=430),
        call(signal_history, "rank_stability_21d", height=430),
    ]
    dependence_figure.assert_called_once_with(dependence_history, height=430)
    render_grid.assert_called_once_with(["coverage", "mean_ic", "stability", "dependence"])
