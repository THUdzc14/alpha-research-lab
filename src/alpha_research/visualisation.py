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


def _build_portfolio_monitoring_figure(
    history: pd.DataFrame,
    metric: str,
    *,
    specifications: dict[str, RollingMetricSpecification],
    category: str,
    title: str | None,
    height: int,
) -> go.Figure:
    if metric not in specifications:
        raise ValueError(
            f"Unsupported {category} metric. Expected one of: "
            f"{sorted(specifications)}"
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


def _validate_figure_height(height: int) -> None:
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer.")


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

    if prepared.duplicated(["entity_type", "entity"]).any():
        raise ValueError("monitoring_overview contains duplicate entity rows.")

    for column in MONITORING_STATUS_LABELS:
        unknown = sorted(set(prepared[column].dropna()) - set(STATUS_CODES))

        if unknown or prepared[column].isna().any():
            raise ValueError(
                f"monitoring_overview.{column} contains " f"unknown statuses: {unknown}"
            )

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
            hovertemplate=("<b>%{y}</b><br>" "%{x}: %{text}<extra></extra>"),
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

        if (
            invalid.any()
            or values.isna().any()
            or values.lt(0).any()
            or values.mod(1).ne(0).any()
        ):
            raise ValueError(
                f"monitoring_overview.{column} " "contains invalid counts."
            )

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
