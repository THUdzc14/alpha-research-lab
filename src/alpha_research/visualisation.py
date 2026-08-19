"""Reusable Plotly figures for the research dashboard."""

from __future__ import annotations

from collections.abc import Mapping
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


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_figure_height(height: int) -> None:
    _validate_positive_integer(height, name="height")


def _validate_top_n(top_n: int) -> None:
    _validate_positive_integer(top_n, name="top_n")


def _prepare_figure_data(
    data: pd.DataFrame,
    value_column: str,
    *,
    series_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and order one dated value series for line plotting."""
    _require_figure_columns(data, {"date", series_column, value_column})
    prepared = data[["date", series_column, value_column]].copy()

    if prepared.empty:
        raise ValueError("data must not be empty.")

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError("data.date contains invalid dates.")

    if prepared[series_column].isna().any():
        raise ValueError(f"data.{series_column} contains missing values.")

    if prepared.duplicated([series_column, "date"]).any():
        raise ValueError(f"data contains duplicate {series_column}-date rows.")

    numeric_values = pd.to_numeric(prepared[value_column], errors="coerce")
    invalid_values = prepared[value_column].notna() & numeric_values.isna()

    if invalid_values.any():
        raise ValueError(f"data.{value_column} contains non-numeric values.")

    prepared[value_column] = numeric_values
    series_order = list(pd.unique(prepared[series_column]))
    positions = {series: position for position, series in enumerate(series_order)}
    order_column = "_figure_series_order"
    prepared[order_column] = prepared[series_column].map(positions)
    prepared = (
        prepared.sort_values([order_column, "date"], kind="stable")
        .drop(columns=order_column)
        .reset_index(drop=True)
    )

    return prepared, series_order


def _series_colour(
    series: str,
    position: int,
    colours: Mapping[str, str],
) -> str:
    return colours.get(
        series,
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
    series_column: str = "portfolio",
    series_colours: Mapping[str, str] = PORTFOLIO_COLOURS,
    line_dashes: Mapping[str, str] = PORTFOLIO_LINE_DASHES,
    line_widths: Mapping[str, float] = PORTFOLIO_LINE_WIDTHS,
) -> go.Figure:
    """Build a consistently styled line figure for ordered dated series."""
    _validate_figure_height(height)

    prepared, series_order = _prepare_figure_data(
        data,
        value_column,
        series_column=series_column,
    )
    figure = go.Figure()

    for position, series in enumerate(series_order):
        series_data = prepared.loc[prepared[series_column].eq(series)]
        figure.add_trace(
            go.Scatter(
                x=series_data["date"],
                y=series_data[value_column],
                mode="lines",
                name=series,
                legendgroup=series,
                connectgaps=False,
                line={
                    "color": _series_colour(series, position, series_colours),
                    "dash": line_dashes.get(series, "solid"),
                    "width": line_widths.get(series, 2.0),
                },
                hovertemplate=(f"%{{x|%Y-%m-%d}}<br>%{{y:{hoverformat}}}<extra></extra>"),
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
            f"Unsupported rolling metric. Expected one of: {sorted(ROLLING_METRIC_SPECIFICATIONS)}"
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
        "width": 1.2,
        "opacity": 0.35,
    },
    "Rolling 252-day average": {
        "color": "#7C3AED",
        "width": 2.0,
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


BETA_METRIC_SPECIFICATIONS = {
    "beta_coverage": RollingMetricSpecification(
        title="Beta Input Coverage",
        yaxis_title="Coverage",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=False,
    ),
    "holdings_market_beta": RollingMetricSpecification(
        title="Holdings-Implied Market Beta",
        yaxis_title="Market beta",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=True,
    ),
    "realised_gross_beta_126": RollingMetricSpecification(
        title="Rolling 126-Day Realised Gross Beta",
        yaxis_title="Market beta",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=True,
    ),
    "beta_measurement_gap": RollingMetricSpecification(
        title="Holdings vs Realised Beta Gap",
        yaxis_title="Beta gap",
        tickformat=".2f",
        hoverformat=".3f",
        show_zero_line=True,
    ),
}

CONCENTRATION_METRIC_SPECIFICATIONS = {
    "effective_position_count": RollingMetricSpecification(
        title="Effective Position Count",
        yaxis_title="Effective count",
        tickformat=".1f",
        hoverformat=".2f",
        show_zero_line=False,
    ),
    "largest_absolute_sector_net_exposure": (
        RollingMetricSpecification(
            title="Largest Absolute Sector Net Exposure",
            yaxis_title="Absolute exposure",
            tickformat=".0%",
            hoverformat=".2%",
            show_zero_line=False,
        )
    ),
    "top_five_absolute_beta_contribution_share": (
        RollingMetricSpecification(
            title="Top-Five Absolute Beta-Contribution Share",
            yaxis_title="Contribution share",
            tickformat=".0%",
            hoverformat=".2%",
            show_zero_line=False,
        )
    ),
    "effective_contributor_count_63": RollingMetricSpecification(
        title="Effective 63-Day Return-Contributor Count",
        yaxis_title="Effective count",
        tickformat=".1f",
        hoverformat=".2f",
        show_zero_line=False,
    ),
    "top_five_contributor_share_63": RollingMetricSpecification(
        title="Top-Five 63-Day Return-Contributor Share",
        yaxis_title="Contribution share",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=False,
    ),
    "effective_contribution_sector_count_63": (
        RollingMetricSpecification(
            title="Effective 63-Day Contribution-Sector Count",
            yaxis_title="Effective count",
            tickformat=".1f",
            hoverformat=".2f",
            show_zero_line=False,
        )
    ),
}

IMPLEMENTATION_METRIC_SPECIFICATIONS = {
    "annualised_turnover_63": RollingMetricSpecification(
        title="Annualised Trailing 63-Day Turnover",
        yaxis_title="Turnover (× NAV)",
        tickformat=".1f",
        hoverformat=".2f",
        show_zero_line=False,
    ),
    "largest_trade_weight_63": RollingMetricSpecification(
        title="Largest Trailing 63-Day Trade Weight",
        yaxis_title="Trade weight",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=False,
    ),
    "minimum_trade_capacity_1pct_usd_millions_63": (
        RollingMetricSpecification(
            title="Minimum Trailing Capacity at 1% Participation",
            yaxis_title="Capacity (USD millions)",
            tickformat=".1f",
            hoverformat=".2f",
            show_zero_line=False,
        )
    ),
    "maximum_missing_return_weight_63": (
        RollingMetricSpecification(
            title="Maximum Trailing 63-Day Missing-Return Weight",
            yaxis_title="Missing-return weight",
            tickformat=".1%",
            hoverformat=".2%",
            show_zero_line=False,
        )
    ),
}

LIQUIDITY_METRIC_SPECIFICATIONS = {
    "liquidity_coverage": RollingMetricSpecification(
        title="Daily Traded-Weight Liquidity Coverage",
        yaxis_title="Liquidity coverage",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=False,
    ),
}


STATUS_COLOURS = {
    "N/A": "#E2E8F0",
    "PASS": "#10B981",
    "WARNING": "#F59E0B",
    "UNAVAILABLE": "#94A3B8",
    "BREACH": "#DC2626",
}

STATUS_CODES = {
    "N/A": -1,
    "PASS": 0,
    "WARNING": 1,
    "UNAVAILABLE": 2,
    "BREACH": 3,
}

MONITORING_STATUS_LABELS = {
    "signal_status": "Signal",
    "market_risk_status": "Market risk",
    "concentration_status": "Concentration",
    "implementation_status": "Implementation",
}

DIAGNOSTIC_COUNT_STYLES = {
    "passes": (
        "Pass",
        STATUS_COLOURS["PASS"],
    ),
    "warnings": (
        "Warning",
        STATUS_COLOURS["WARNING"],
    ),
    "breaches": (
        "Breach",
        STATUS_COLOURS["BREACH"],
    ),
    "unavailable": (
        "Unavailable",
        STATUS_COLOURS["UNAVAILABLE"],
    ),
}


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

    return _line_figure(
        signal_history,
        value_column=metric,
        title=specification.title if title is None else title,
        yaxis_title=specification.yaxis_title,
        tickformat=specification.tickformat,
        hoverformat=specification.hoverformat,
        show_zero_line=specification.show_zero_line,
        height=height,
        series_column="factor",
        series_colours=FACTOR_COLOURS,
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
            var_name="series",
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
        series_column="series",
    )

    for trace in figure.data:
        style = FACTOR_DEPENDENCE_STYLES[trace.name]
        trace.line.color = style["color"]
        trace.line.width = style["width"]
        trace.opacity = style["opacity"]

    return figure


def _build_portfolio_monitoring_figure(
    history: pd.DataFrame,
    metric: str,
    *,
    specifications: Mapping[str, RollingMetricSpecification],
    category: str,
    title: str | None,
    height: int,
) -> go.Figure:
    if metric not in specifications:
        raise ValueError(
            f"Unsupported {category} metric. Expected one of: {sorted(specifications)}"
        )

    specification = specifications[metric]

    return _line_figure(
        history,
        value_column=metric,
        title=(specification.title if title is None else title),
        yaxis_title=specification.yaxis_title,
        tickformat=specification.tickformat,
        hoverformat=specification.hoverformat,
        show_zero_line=specification.show_zero_line,
        height=height,
    )


def build_beta_figure(
    beta_history: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot one supported beta-monitoring metric by portfolio."""
    return _build_portfolio_monitoring_figure(
        beta_history,
        metric,
        specifications=BETA_METRIC_SPECIFICATIONS,
        category="beta",
        title=title,
        height=height,
    )


def build_concentration_figure(
    concentration_history: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot one supported concentration metric by portfolio."""
    return _build_portfolio_monitoring_figure(
        concentration_history,
        metric,
        specifications=CONCENTRATION_METRIC_SPECIFICATIONS,
        category="concentration",
        title=title,
        height=height,
    )


def build_implementation_figure(
    implementation_history: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot one supported implementation-health metric by portfolio."""
    return _build_portfolio_monitoring_figure(
        implementation_history,
        metric,
        specifications=IMPLEMENTATION_METRIC_SPECIFICATIONS,
        category="implementation",
        title=title,
        height=height,
    )


def build_liquidity_coverage_figure(
    liquidity_coverage_history: pd.DataFrame,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot traded-weight liquidity coverage by portfolio."""
    return _build_portfolio_monitoring_figure(
        liquidity_coverage_history,
        "liquidity_coverage",
        specifications=LIQUIDITY_METRIC_SPECIFICATIONS,
        category="liquidity",
        title=title,
        height=height,
    )


def _prepare_monitoring_status_figure_data(
    monitoring_overview: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "entity_type",
        "entity",
        *MONITORING_STATUS_LABELS,
    }
    _require_figure_columns(
        monitoring_overview,
        required_columns,
    )

    prepared = monitoring_overview.loc[
        :,
        [
            "entity_type",
            "entity",
            *MONITORING_STATUS_LABELS,
        ],
    ].copy()

    if prepared.empty:
        raise ValueError("monitoring_overview must not be empty.")

    if prepared[["entity_type", "entity"]].isna().any().any():
        raise ValueError("monitoring_overview contains missing entity labels.")

    if prepared.duplicated(["entity_type", "entity"]).any():
        raise ValueError("monitoring_overview contains duplicate entity rows.")

    for column in MONITORING_STATUS_LABELS:
        unknown = sorted(set(prepared[column].dropna()) - set(STATUS_CODES))

        if unknown or prepared[column].isna().any():
            raise ValueError(f"monitoring_overview.{column} contains unknown statuses: {unknown}")

    return prepared


def build_monitoring_status_heatmap(
    monitoring_overview: pd.DataFrame,
    *,
    title: str = "Monitoring Status by Category",
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot category-level status for each monitored entity."""
    _validate_figure_height(height)

    prepared = _prepare_monitoring_status_figure_data(monitoring_overview)
    status_columns = list(MONITORING_STATUS_LABELS)
    status_text = prepared[status_columns]
    status_codes = status_text.replace(STATUS_CODES).astype(int)

    colourscale = [
        [0.0, STATUS_COLOURS["N/A"]],
        [0.2, STATUS_COLOURS["N/A"]],
        [0.2, STATUS_COLOURS["PASS"]],
        [0.4, STATUS_COLOURS["PASS"]],
        [0.4, STATUS_COLOURS["WARNING"]],
        [0.6, STATUS_COLOURS["WARNING"]],
        [0.6, STATUS_COLOURS["UNAVAILABLE"]],
        [0.8, STATUS_COLOURS["UNAVAILABLE"]],
        [0.8, STATUS_COLOURS["BREACH"]],
        [1.0, STATUS_COLOURS["BREACH"]],
    ]

    figure = go.Figure(
        go.Heatmap(
            z=status_codes.to_numpy(),
            x=[MONITORING_STATUS_LABELS[column] for column in status_columns],
            y=prepared["entity"],
            text=status_text.to_numpy(),
            zmin=-1.5,
            zmax=3.5,
            colorscale=colourscale,
            showscale=False,
            texttemplate="%{text}",
            hovertemplate=("<b>%{y}</b><br>%{x}: %{text}<extra></extra>"),
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
        margin={
            "l": 150,
            "r": 24,
            "t": 72,
            "b": 48,
        },
    )
    figure.update_xaxes(
        side="top",
        showgrid=False,
    )
    figure.update_yaxes(
        autorange="reversed",
        showgrid=False,
        title_text="",
    )

    return figure


def build_diagnostic_count_figure(
    monitoring_overview: pd.DataFrame,
    *,
    title: str = "Diagnostic Status Counts",
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot stacked diagnostic counts for each monitored entity."""
    _validate_figure_height(height)

    required_columns = {
        "entity",
        *DIAGNOSTIC_COUNT_STYLES,
    }
    _require_figure_columns(
        monitoring_overview,
        required_columns,
    )

    if monitoring_overview.empty:
        raise ValueError("monitoring_overview must not be empty.")

    if monitoring_overview["entity"].isna().any():
        raise ValueError("monitoring_overview.entity contains missing values.")

    figure = go.Figure()

    for column, (
        label,
        colour,
    ) in DIAGNOSTIC_COUNT_STYLES.items():
        values = pd.to_numeric(
            monitoring_overview[column],
            errors="coerce",
        )
        invalid = monitoring_overview[column].notna() & values.isna()

        if invalid.any() or values.isna().any() or values.lt(0).any() or values.mod(1).ne(0).any():
            raise ValueError(f"monitoring_overview.{column} contains invalid counts.")

        figure.add_trace(
            go.Bar(
                x=monitoring_overview["entity"],
                y=values,
                name=label,
                marker_color=colour,
                hovertemplate=("%{x}<br>" + label + ": %{y:.0f}<extra></extra>"),
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
        margin={
            "l": 64,
            "r": 24,
            "t": 72,
            "b": 80,
        },
        barmode="stack",
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
        title_text="Diagnostics",
        dtick=1,
        rangemode="tozero",
        gridcolor="#E2E8F0",
    )

    return figure


ATTRIBUTION_COMPONENT_STYLES = {
    "Long side": {
        "color": "#2563EB",
        "dash": "solid",
        "width": 2.0,
    },
    "Short side": {
        "color": "#F59E0B",
        "dash": "solid",
        "width": 2.0,
    },
    "Transaction costs": {
        "color": "#DC2626",
        "dash": "dot",
        "width": 1.5,
    },
    "Net contribution": {
        "color": "#10B981",
        "dash": "solid",
        "width": 2.0,
    },
}


def build_side_cost_attribution_figure(
    attribution_summary: pd.DataFrame,
    *,
    title: str = "Annualised Arithmetic Return Contributions",
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot annualised long, short, and transaction-cost contributions."""
    _validate_figure_height(height)
    required_columns = {
        "portfolio",
        "annualised_long_contribution",
        "annualised_short_contribution",
        "annualised_cost_drag",
    }
    _require_figure_columns(
        attribution_summary,
        required_columns,
    )

    if attribution_summary.empty:
        raise ValueError("attribution_summary must not be empty.")

    if attribution_summary["portfolio"].isna().any():
        raise ValueError("attribution_summary.portfolio contains missing values.")

    if attribution_summary["portfolio"].duplicated().any():
        raise ValueError("attribution_summary contains duplicate portfolios.")

    components = {
        "Long side": (attribution_summary["annualised_long_contribution"]),
        "Short side": (attribution_summary["annualised_short_contribution"]),
        "Transaction costs": (-attribution_summary["annualised_cost_drag"]),
    }
    figure = go.Figure()

    for component, values in components.items():
        numeric_values = pd.to_numeric(
            values,
            errors="coerce",
        )

        if numeric_values.isna().any():
            raise ValueError(f"attribution_summary contains invalid {component} values.")

        figure.add_trace(
            go.Bar(
                x=attribution_summary["portfolio"],
                y=numeric_values,
                name=component,
                marker_color=(ATTRIBUTION_COMPONENT_STYLES[component]["color"]),
                hovertemplate=("%{x}<br>" + component + ": %{y:.2%}<extra></extra>"),
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
        margin={
            "l": 64,
            "r": 24,
            "t": 72,
            "b": 80,
        },
        barmode="group",
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
        title_text="Annualised contribution",
        tickformat=".0%",
        gridcolor="#E2E8F0",
        zeroline=False,
    )
    figure.add_hline(
        y=0.0,
        line_color="#64748B",
        line_width=1.0,
    )

    return figure


def build_cumulative_attribution_figure(
    attribution_history: pd.DataFrame,
    portfolio: str,
    *,
    title: str | None = None,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> go.Figure:
    """Plot cumulative additive attribution for one portfolio."""
    if not isinstance(portfolio, str) or not portfolio:
        raise ValueError("portfolio must be a non-empty string.")

    columns = {
        "cumulative_long_contribution": "Long side",
        "cumulative_short_contribution": "Short side",
        "cumulative_cost_contribution": ("Transaction costs"),
        "cumulative_net_contribution": ("Net contribution"),
    }
    _require_figure_columns(
        attribution_history,
        {"date", "portfolio", *columns},
    )
    selected = attribution_history.loc[
        attribution_history["portfolio"].eq(portfolio),
        ["date", *columns],
    ]

    if selected.empty:
        raise ValueError(f"attribution_history is missing portfolio: {portfolio}")

    figure_data = selected.rename(columns=columns).melt(
        id_vars="date",
        var_name="component",
        value_name="cumulative_contribution",
    )
    component_colours = {
        component: style["color"] for component, style in ATTRIBUTION_COMPONENT_STYLES.items()
    }
    component_dashes = {
        component: style["dash"] for component, style in ATTRIBUTION_COMPONENT_STYLES.items()
    }
    component_widths = {
        component: style["width"] for component, style in ATTRIBUTION_COMPONENT_STYLES.items()
    }
    figure = _line_figure(
        figure_data,
        value_column="cumulative_contribution",
        title=(f"Cumulative Additive Attribution — {portfolio}" if title is None else title),
        yaxis_title="Cumulative contribution",
        tickformat=".0%",
        hoverformat=".2%",
        show_zero_line=True,
        height=height,
        series_column="component",
        series_colours=component_colours,
        line_dashes=component_dashes,
        line_widths=component_widths,
    )

    return figure


def _validate_security_contribution_summary(
    security_summary: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "portfolio",
        "ticker",
        "cumulative_net_contribution",
        "absolute_gross_contribution",
        "absolute_contribution_share",
    }
    _require_figure_columns(
        security_summary,
        required_columns,
    )

    if security_summary.empty:
        raise ValueError("security_summary must not be empty.")

    if security_summary[["portfolio", "ticker"]].isna().any().any():
        raise ValueError("security_summary contains missing portfolio or ticker labels.")

    if security_summary["ticker"].duplicated().any():
        raise ValueError("security_summary contains duplicate tickers.")

    portfolios = security_summary["portfolio"].dropna().unique()

    if len(portfolios) != 1:
        raise ValueError("security_summary must contain exactly one portfolio.")

    prepared = security_summary.copy()

    for column in (
        "cumulative_net_contribution",
        "absolute_gross_contribution",
        "absolute_contribution_share",
    ):
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

        if prepared[column].isna().any():
            raise ValueError(f"security_summary.{column} contains invalid values.")

    if prepared["absolute_gross_contribution"].lt(0.0).any():
        raise ValueError("absolute_gross_contribution must be non-negative.")

    if prepared["absolute_contribution_share"].lt(0.0).any():
        raise ValueError("absolute_contribution_share must be non-negative.")

    return prepared


def build_security_contribution_figure(
    security_summary: pd.DataFrame,
    *,
    top_n: int = 10,
    title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Plot the largest positive and negative net contributors."""
    _validate_figure_height(height)
    _validate_top_n(top_n)
    prepared = _validate_security_contribution_summary(security_summary)
    value_column = "cumulative_net_contribution"

    positive = prepared.loc[prepared[value_column].gt(0.0)].nlargest(
        top_n,
        value_column,
    )
    negative = prepared.loc[prepared[value_column].lt(0.0)].nsmallest(
        top_n,
        value_column,
    )
    selected = pd.concat(
        [negative, positive],
        ignore_index=True,
    )

    if selected.empty:
        selected = prepared.nlargest(
            top_n,
            "absolute_gross_contribution",
        )

    selected = (
        selected.drop_duplicates("ticker")
        .sort_values(
            value_column,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    portfolio = str(prepared["portfolio"].iloc[0])
    colours = (
        selected[value_column]
        .ge(0.0)
        .map(
            {
                True: "#10B981",
                False: "#DC2626",
            }
        )
    )

    figure = go.Figure(
        go.Bar(
            x=selected[value_column],
            y=selected["ticker"],
            orientation="h",
            marker_color=colours,
            name="Net contribution",
            customdata=selected[["absolute_contribution_share"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>"
                "Net contribution: %{x:.2%}"
                "<br>Absolute contribution share: "
                "%{customdata[0]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (f"Largest Security Contributors — {portfolio}" if title is None else title),
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        height=height,
        margin={
            "l": 72,
            "r": 24,
            "t": 72,
            "b": 56,
        },
        showlegend=False,
    )
    figure.update_xaxes(
        title_text="Cumulative net contribution",
        tickformat=".1%",
        gridcolor="#E2E8F0",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="",
        showgrid=False,
    )
    figure.add_vline(
        x=0.0,
        line_color="#64748B",
        line_width=1.0,
    )

    return figure


def build_security_contribution_share_figure(
    security_summary: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Plot the largest absolute contribution shares."""
    _validate_figure_height(height)
    _validate_top_n(top_n)
    prepared = _validate_security_contribution_summary(security_summary)
    selected = (
        prepared.nlargest(
            top_n,
            "absolute_gross_contribution",
        )
        .sort_values(
            "absolute_contribution_share",
            kind="stable",
        )
        .reset_index(drop=True)
    )
    portfolio = str(prepared["portfolio"].iloc[0])

    figure = go.Figure(
        go.Bar(
            x=selected["absolute_contribution_share"],
            y=selected["ticker"],
            orientation="h",
            marker_color="#2563EB",
            name="Absolute contribution share",
            customdata=selected[["cumulative_net_contribution"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>"
                "Absolute contribution share: "
                "%{x:.2%}"
                "<br>Net contribution: "
                "%{customdata[0]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (
                f"Largest Absolute Security Contribution Shares — {portfolio}"
                if title is None
                else title
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        height=height,
        margin={
            "l": 72,
            "r": 24,
            "t": 72,
            "b": 56,
        },
        showlegend=False,
    )
    figure.update_xaxes(
        title_text=("Share of absolute gross contribution"),
        tickformat=".1%",
        gridcolor="#E2E8F0",
        rangemode="tozero",
    )
    figure.update_yaxes(
        title_text="",
        showgrid=False,
    )

    return figure
