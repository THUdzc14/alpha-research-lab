"""Reusable Plotly figures for the research dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

PORTFOLIO_COLOURS = {
    "Composite Score": "#2563EB",
    "Fixed 50/50 Sleeves": "#F59E0B",
    "Pure Inverse Volatility": "#10B981",
    "SPY": "#64748B",
}

PORTFOLIO_LINE_DASHES = {
    # "SPY": "dash",
    "SPY": "10px, 5px",
}

PORTFOLIO_LINE_WIDTHS = {
    "Composite Score": 2.0,
    "SPY": 1.6,
}

FALLBACK_COLOURS = (
    "#8B5CF6",
    "#EC4899",
    "#06B6D4",
    "#84CC16",
    "#F97316",
)

DEFAULT_FIGURE_HEIGHT = 460


@dataclass(frozen=True)
class RollingMetricSpecification:
    """Display settings for one supported rolling metric."""

    title: str
    yaxis_title: str
    tickformat: str
    hoverformat: str
    show_zero_line: bool


ROLLING_METRIC_SPECIFICATIONS = {
    "trailing_return_252": RollingMetricSpecification(
        title="Trailing 252-Day Return",
        yaxis_title="Trailing return",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=True,
    ),
    "rolling_sharpe_252": RollingMetricSpecification(
        title="Rolling 252-Day Sharpe Ratio",
        yaxis_title="Sharpe ratio",
        tickformat=".2f",
        hoverformat=".2f",
        show_zero_line=True,
    ),
    "annualised_volatility_126": RollingMetricSpecification(
        title="Annualised 126-Day Volatility",
        yaxis_title="Annualised volatility",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=False,
    ),
    "maximum_drawdown_252": RollingMetricSpecification(
        title="Rolling 252-Day Maximum Drawdown",
        yaxis_title="Maximum drawdown",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=True,
    ),
}


def _require_figure_columns(
    data: pd.DataFrame,
    required_columns: set[str],
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(f"data is missing columns: {sorted(missing_columns)}")


def _prepare_figure_data(
    data: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    _require_figure_columns(data, {"date", "portfolio", value_column})
    prepared = data[["date", "portfolio", value_column]].copy()

    if prepared.empty:
        raise ValueError("data must not be empty.")

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError("data.date contains invalid dates.")

    if prepared["portfolio"].isna().any():
        raise ValueError("data.portfolio contains missing values.")

    if prepared.duplicated(["portfolio", "date"]).any():
        raise ValueError("data contains duplicate portfolio-date rows.")

    numeric_values = pd.to_numeric(prepared[value_column], errors="coerce")
    invalid_values = prepared[value_column].notna() & numeric_values.isna()

    if invalid_values.any():
        raise ValueError(f"data.{value_column} contains non-numeric values.")

    prepared[value_column] = numeric_values
    portfolio_order = list(pd.unique(prepared["portfolio"]))
    positions = {
        portfolio: position for position, portfolio in enumerate(portfolio_order)
    }
    prepared["_portfolio_order"] = prepared["portfolio"].map(positions)
    prepared = (
        prepared.sort_values(["_portfolio_order", "date"], kind="stable")
        .drop(columns="_portfolio_order")
        .reset_index(drop=True)
    )

    return prepared, portfolio_order


def _portfolio_colour(portfolio: str, position: int) -> str:
    return PORTFOLIO_COLOURS.get(
        portfolio,
        FALLBACK_COLOURS[position % len(FALLBACK_COLOURS)],
    )


def _line_figure(
    data: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    yaxis_title: str,
    tickformat: str,
    hoverformat: str,
    show_zero_line: bool,
    height: int,
) -> go.Figure:
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer.")

    prepared, portfolio_order = _prepare_figure_data(data, value_column)
    figure = go.Figure()

    for position, portfolio in enumerate(portfolio_order):
        portfolio_data = prepared.loc[prepared["portfolio"].eq(portfolio)]
        figure.add_trace(
            go.Scatter(
                x=portfolio_data["date"],
                y=portfolio_data[value_column],
                mode="lines",
                name=portfolio,
                legendgroup=portfolio,
                connectgaps=False,
                line={
                    "color": _portfolio_colour(portfolio, position),
                    "dash": PORTFOLIO_LINE_DASHES.get(portfolio, "solid"),
                    "width": PORTFOLIO_LINE_WIDTHS.get(portfolio, 2.0),
                },
                hovertemplate=(
                    f"%{{x|%Y-%m-%d}}<br>%{{y:{hoverformat}}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        height=height,
        margin={"l": 64, "r": 24, "t": 72, "b": 48},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
        },
    )
    figure.update_xaxes(
        title_text="",
        showgrid=False,
    )
    figure.update_yaxes(
        title_text=yaxis_title,
        tickformat=tickformat,
        gridcolor="#E2E8F0",
        zeroline=False,
    )

    if show_zero_line:
        figure.add_hline(
            y=0.0,
            line_color="#94A3B8",
            line_dash="dot",
            line_width=1.0,
        )

    return figure


def build_cumulative_performance_figure(
    performance_history: pd.DataFrame,
    *,
    title: str = "Cumulative Performance",
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot display-rebased wealth prepared by dashboard analytics."""
    return _line_figure(
        performance_history,
        value_column="indexed_wealth",
        title=title,
        yaxis_title="Indexed wealth (start = 1.0)",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=False,
        height=height,
    )


def build_drawdown_figure(
    performance_history: pd.DataFrame,
    *,
    title: str = "Drawdown",
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot full-history drawdown values for the selected display window."""
    return _line_figure(
        performance_history,
        value_column="drawdown",
        title=title,
        yaxis_title="Drawdown",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=True,
        height=height,
    )


def build_rolling_metric_figure(
    performance_history: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot one supported rolling metric with consistent formatting."""
    if metric not in ROLLING_METRIC_SPECIFICATIONS:
        raise ValueError(
            "Unsupported rolling metric. Expected one of: "
            f"{sorted(ROLLING_METRIC_SPECIFICATIONS)}"
        )

    specification = ROLLING_METRIC_SPECIFICATIONS[metric]

    return _line_figure(
        performance_history,
        value_column=metric,
        title=specification.title if title is None else title,
        yaxis_title=specification.yaxis_title,
        tickformat=specification.tickformat,
        hoverformat=specification.hoverformat,
        show_zero_line=specification.show_zero_line,
        height=height,
    )


FACTOR_COLOURS = {
    "Momentum": "#7C3AED",
    "Realised Volatility": "#DB2777",
}

FACTOR_DEPENDENCE_STYLES = {
    "Daily factor rank correlation": {
        "color": "#94A3B8",
        "width": 1.25,
        "opacity": 0.35,
    },
    "Rolling 252-day average": {
        "color": "#7C3AED",
        "width": 2.75,
        "opacity": 1.0,
    },
}


@dataclass(frozen=True)
class SignalMetricSpecification:
    """Display settings for one supported retained-signal metric."""

    title: str
    yaxis_title: str
    tickformat: str
    hoverformat: str
    show_zero_line: bool


SIGNAL_METRIC_SPECIFICATIONS = {
    "signal_coverage": SignalMetricSpecification(
        title="Retained-Signal Coverage",
        yaxis_title="Coverage",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=False,
    ),
    "raw_iqr": SignalMetricSpecification(
        title="Retained-Signal Dispersion",
        yaxis_title="Raw-signal IQR",
        tickformat=".3f",
        hoverformat=".4f",
        show_zero_line=False,
    ),
    "ic": SignalMetricSpecification(
        title="Daily Rank Information Coefficient",
        yaxis_title="Rank IC",
        tickformat=".3f",
        hoverformat=".4f",
        show_zero_line=True,
    ),
    "rolling_mean_ic_252": SignalMetricSpecification(
        title="Rolling 252-Day Mean Information Coefficient",
        yaxis_title="Mean rank IC",
        tickformat=".3f",
        hoverformat=".4f",
        show_zero_line=True,
    ),
    "rank_stability_1d": SignalMetricSpecification(
        title="One-Day Signal-Rank Stability",
        yaxis_title="Rank correlation",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=True,
    ),
    "rank_stability_21d": SignalMetricSpecification(
        title="Twenty-One-Day Signal-Rank Stability",
        yaxis_title="Rank correlation",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=True,
    ),
}


def _factor_line_figure(
    signal_history: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    yaxis_title: str,
    tickformat: str,
    hoverformat: str,
    show_zero_line: bool,
    height: int,
) -> go.Figure:
    _require_figure_columns(signal_history, {"date", "factor", value_column})
    figure_data = signal_history[["date", "factor", value_column]].rename(
        columns={"factor": "portfolio"}
    )
    figure = _line_figure(
        figure_data,
        value_column=value_column,
        title=title,
        yaxis_title=yaxis_title,
        tickformat=tickformat,
        hoverformat=hoverformat,
        show_zero_line=show_zero_line,
        height=height,
    )

    for position, trace in enumerate(figure.data):
        trace.line.color = FACTOR_COLOURS.get(
            trace.name,
            FALLBACK_COLOURS[position % len(FALLBACK_COLOURS)],
        )

    return figure


def build_signal_health_figure(
    signal_history: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot one retained-signal health metric for each selected factor."""
    if metric not in SIGNAL_METRIC_SPECIFICATIONS:
        raise ValueError(
            "Unsupported signal-health metric. Expected one of: "
            f"{sorted(SIGNAL_METRIC_SPECIFICATIONS)}"
        )

    specification = SIGNAL_METRIC_SPECIFICATIONS[metric]

    return _factor_line_figure(
        signal_history,
        value_column=metric,
        title=specification.title if title is None else title,
        yaxis_title=specification.yaxis_title,
        tickformat=specification.tickformat,
        hoverformat=specification.hoverformat,
        show_zero_line=specification.show_zero_line,
        height=height,
    )


def build_factor_dependence_figure(
    factor_dependence_history: pd.DataFrame,
    *,
    title: str = "Retained-Factor Dependence",
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot daily and rolling dependence between the two retained factors."""
    required_columns = {
        "date",
        "factor_rank_correlation",
        "rolling_factor_rank_correlation_252",
    }
    _require_figure_columns(factor_dependence_history, required_columns)
    labels = {
        "factor_rank_correlation": "Daily factor rank correlation",
        "rolling_factor_rank_correlation_252": "Rolling 252-day average",
    }
    figure_data = (
        factor_dependence_history[
            [
                "date",
                "factor_rank_correlation",
                "rolling_factor_rank_correlation_252",
            ]
        ]
        .rename(columns=labels)
        .melt(
            id_vars="date",
            var_name="portfolio",
            value_name="correlation",
        )
    )
    figure = _line_figure(
        figure_data,
        value_column="correlation",
        title=title,
        yaxis_title="Rank correlation",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=True,
        height=height,
    )

    for trace in figure.data:
        style = FACTOR_DEPENDENCE_STYLES[trace.name]
        trace.line.color = style["color"]
        trace.line.width = style["width"]
        trace.opacity = style["opacity"]

    return figure
