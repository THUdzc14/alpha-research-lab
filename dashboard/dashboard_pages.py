"""Streamlit page renderers for the Alpha Research Lab dashboard."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from alpha_research.dashboard_analytics import (
    build_latest_portfolio_snapshot,
    build_performance_summary,
    prepare_diagnostic_table,
    prepare_monitoring_overview,
    prepare_performance_history,
    build_latest_factor_snapshot,
    prepare_factor_dependence_history,
    prepare_signal_health_history,
    prepare_beta_history,
    prepare_concentration_history,
    prepare_implementation_history,
    prepare_liquidity_coverage_history,
    build_security_contribution_summary,
    build_side_cost_attribution_summary,
    prepare_portfolio_attribution_history,
)
from alpha_research.visualisation import (
    build_cumulative_performance_figure,
    build_diagnostic_count_figure,
    build_drawdown_figure,
    build_monitoring_status_heatmap,
    build_rolling_metric_figure,
    build_factor_dependence_figure,
    build_signal_health_figure,
    build_beta_figure,
    build_concentration_figure,
    build_implementation_figure,
    build_liquidity_coverage_figure,
    build_cumulative_attribution_figure,
    build_security_contribution_figure,
    build_security_contribution_share_figure,
    build_side_cost_attribution_figure,
)

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def _build_strategy_overview_tables(
    bundle,
    portfolios: Sequence[str],
):
    snapshot = build_latest_portfolio_snapshot(
        bundle.attribution["selected_implementations"],
        bundle.monitoring["latest_overview"],
        bundle.monitoring["performance_risk"],
        bundle.monitoring["beta"],
        bundle.monitoring["concentration"],
        bundle.monitoring["implementation"],
        bundle.monitoring["liquidity_coverage"],
    )
    snapshot = snapshot.loc[snapshot["portfolio"].isin(portfolios)].copy()

    complete_overview = prepare_monitoring_overview(
        bundle.monitoring["latest_overview"]
    )
    factors = complete_overview.loc[
        complete_overview["entity_type"].eq("Factor"),
        "entity",
    ].tolist()

    displayed_entities = [
        *factors,
        *portfolios,
    ]
    status_summary = prepare_monitoring_overview(
        bundle.monitoring["latest_overview"],
        entities=displayed_entities,
    )
    active_diagnostics = prepare_diagnostic_table(
        bundle.monitoring["diagnostic_flags"],
        entities=displayed_entities,
        statuses=[
            "WARNING",
            "BREACH",
            "UNAVAILABLE",
        ],
    )

    return (
        snapshot,
        status_summary,
        active_diagnostics,
    )


def _render_overview_metrics(
    snapshot,
    status_summary,
) -> None:
    latest_date = snapshot["latest_date"].max()
    columns = st.columns(4)

    columns[0].metric(
        "Latest portfolio date",
        f"{latest_date:%Y-%m-%d}",
    )
    columns[1].metric(
        "Entities monitored",
        len(status_summary),
    )
    columns[2].metric(
        "Active warnings",
        int(status_summary["warnings"].sum()),
    )
    columns[3].metric(
        "Active breaches",
        int(status_summary["breaches"].sum()),
    )


def _render_portfolio_snapshot(
    snapshot,
) -> None:
    display = snapshot.assign(
        drawdown=lambda data: (data["drawdown"] * 100.0),
        annualised_volatility_126=(
            lambda data: (data["annualised_volatility_126"] * 100.0)
        ),
        largest_absolute_sector_net_exposure=(
            lambda data: (data["largest_absolute_sector_net_exposure"] * 100.0)
        ),
        minimum_capacity_usd_millions=(
            snapshot["minimum_trade_capacity_1pct_usd_63"] / 1_000_000.0
        ),
    )[
        [
            "portfolio",
            "implementation_role",
            "rebalance_frequency",
            "latest_date",
            "overall_status",
            "drawdown",
            "rolling_sharpe_252",
            "annualised_volatility_126",
            "holdings_market_beta",
            ("largest_absolute_sector_" "net_exposure"),
            "annualised_turnover_63",
            "minimum_capacity_usd_millions",
        ]
    ]

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "portfolio": (st.column_config.TextColumn("Portfolio")),
            "implementation_role": (st.column_config.TextColumn("Role")),
            "rebalance_frequency": (
                st.column_config.NumberColumn(
                    "Rebalance frequency",
                    format="%d days",
                )
            ),
            "latest_date": (
                st.column_config.DateColumn(
                    "Latest date",
                    format="YYYY-MM-DD",
                )
            ),
            "overall_status": (st.column_config.TextColumn("Status")),
            "drawdown": (
                st.column_config.NumberColumn(
                    "Drawdown",
                    format="%.1f%%",
                )
            ),
            "rolling_sharpe_252": (
                st.column_config.NumberColumn(
                    "Rolling Sharpe",
                    format="%.2f",
                )
            ),
            "annualised_volatility_126": (
                st.column_config.NumberColumn(
                    "126-day volatility",
                    format="%.1f%%",
                )
            ),
            ("holdings_market_beta"): st.column_config.NumberColumn(
                "Holdings beta",
                format="%.2f",
            ),
            ("largest_absolute_sector_" "net_exposure"): st.column_config.NumberColumn(
                "Largest sector net",
                format="%.1f%%",
            ),
            "annualised_turnover_63": (
                st.column_config.NumberColumn(
                    "Annualised turnover",
                    format="%.2fx",
                )
            ),
            ("minimum_capacity_usd_millions"): st.column_config.NumberColumn(
                "Minimum capacity (USDm)",
                format="$%.1f",
            ),
        },
    )


def _render_active_diagnostics(
    active_diagnostics,
) -> None:
    if active_diagnostics.empty:
        st.success("No active warnings, breaches, or unavailable diagnostics.")
        return

    display = active_diagnostics.assign(
        historical_percentile=(active_diagnostics["historical_percentile"] * 100.0)
    )
    display_columns = [
        "entity_type",
        "entity",
        "category",
        "diagnostic",
        "latest_date",
        "latest_value",
        "threshold_value",
        "historical_percentile",
        "status",
    ]

    st.dataframe(
        display.loc[:, display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "entity_type": (st.column_config.TextColumn("Entity type")),
            "entity": (st.column_config.TextColumn("Entity")),
            "category": (st.column_config.TextColumn("Category")),
            "diagnostic": (st.column_config.TextColumn("Diagnostic")),
            "latest_date": (
                st.column_config.DateColumn(
                    "Latest date",
                    format="YYYY-MM-DD",
                )
            ),
            "latest_value": (
                st.column_config.NumberColumn(
                    "Latest value",
                    format="%.4f",
                )
            ),
            "threshold_value": (
                st.column_config.NumberColumn(
                    "Threshold",
                    format="%.4f",
                )
            ),
            "historical_percentile": (
                st.column_config.NumberColumn(
                    "Historical percentile",
                    format="%.1f%%",
                )
            ),
            "status": (st.column_config.TextColumn("Status")),
        },
    )


def render_strategy_overview(
    bundle,
    portfolios: Sequence[str],
) -> None:
    """Render the latest strategy and monitoring overview."""
    st.header("Strategy Overview")
    st.caption(
        "Latest monitoring state for the retained "
        "factors and selected portfolio "
        "implementations. The historical date "
        "filter does not alter this page."
    )

    (
        snapshot,
        status_summary,
        active_diagnostics,
    ) = _build_strategy_overview_tables(
        bundle,
        portfolios,
    )

    _render_overview_metrics(
        snapshot,
        status_summary,
    )

    st.subheader("Monitoring status")
    figure_columns = st.columns(2)

    figure_columns[0].plotly_chart(
        build_monitoring_status_heatmap(
            status_summary,
            height=390,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    figure_columns[1].plotly_chart(
        build_diagnostic_count_figure(
            status_summary,
            height=390,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Latest portfolio snapshot")
    _render_portfolio_snapshot(snapshot)

    st.subheader("Active diagnostics")
    _render_active_diagnostics(active_diagnostics)


def _render_performance_summary(
    performance_summary,
) -> None:
    percentage_columns = (
        "total_return",
        "annualised_return",
        "annualised_volatility",
        "maximum_drawdown",
        "positive_day_fraction",
    )
    display = performance_summary.copy()

    for column in percentage_columns:
        display[column] = display[column] * 100.0

    display = display[
        [
            "portfolio",
            "observations",
            "start_date",
            "end_date",
            "total_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "positive_day_fraction",
        ]
    ]

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "portfolio": (st.column_config.TextColumn("Portfolio")),
            "observations": (
                st.column_config.NumberColumn(
                    "Observations",
                    format="%d",
                )
            ),
            "start_date": (
                st.column_config.DateColumn(
                    "Start date",
                    format="YYYY-MM-DD",
                )
            ),
            "end_date": (
                st.column_config.DateColumn(
                    "End date",
                    format="YYYY-MM-DD",
                )
            ),
            "total_return": (
                st.column_config.NumberColumn(
                    "Total return",
                    format="%.1f%%",
                )
            ),
            "annualised_return": (
                st.column_config.NumberColumn(
                    "Annualised return",
                    format="%.1f%%",
                )
            ),
            "annualised_volatility": (
                st.column_config.NumberColumn(
                    "Annualised volatility",
                    format="%.1f%%",
                )
            ),
            "sharpe_ratio": (
                st.column_config.NumberColumn(
                    "Sharpe ratio",
                    format="%.2f",
                )
            ),
            "maximum_drawdown": (
                st.column_config.NumberColumn(
                    "Maximum drawdown",
                    format="%.1f%%",
                )
            ),
            "positive_day_fraction": (
                st.column_config.NumberColumn(
                    "Positive days",
                    format="%.1f%%",
                )
            ),
        },
    )


def render_performance_page(
    bundle,
    portfolios: Sequence[str],
    start_date,
    end_date,
) -> None:
    """Render filtered performance and risk analytics."""
    st.header("Performance & Drawdowns")
    st.caption(
        "Selected portfolio implementations with "
        "SPY included automatically as the market "
        "benchmark. Cumulative wealth is rebased "
        "to 1.0 at the selected start date."
    )

    displayed_portfolios = [
        *portfolios,
        "SPY",
    ]
    performance_history = prepare_performance_history(
        bundle.monitoring["performance_risk"],
        portfolios=displayed_portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    performance_summary = build_performance_summary(
        bundle.monitoring["performance_risk"],
        portfolios=displayed_portfolios,
        start_date=start_date,
        end_date=end_date,
    )

    st.subheader("Period summary")
    _render_performance_summary(performance_summary)

    st.subheader("Growth and drawdown")
    growth_columns = st.columns(2)

    growth_columns[0].plotly_chart(
        build_cumulative_performance_figure(
            performance_history,
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    growth_columns[1].plotly_chart(
        build_drawdown_figure(
            performance_history,
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Rolling performance and risk")
    rolling_columns = st.columns(2)

    rolling_columns[0].plotly_chart(
        build_rolling_metric_figure(
            performance_history,
            "rolling_sharpe_252",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    rolling_columns[1].plotly_chart(
        build_rolling_metric_figure(
            performance_history,
            "annualised_volatility_126",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )


def _render_factor_snapshot(snapshot) -> None:
    display = snapshot.assign(
        signal_coverage=lambda data: data["signal_coverage"] * 100.0,
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "factor": st.column_config.TextColumn("Factor"),
            "latest_date": st.column_config.DateColumn(
                "Latest signal date",
                format="YYYY-MM-DD",
            ),
            "overall_status": st.column_config.TextColumn("Overall status"),
            "signal_status": st.column_config.TextColumn("Signal status"),
            "signal_coverage": st.column_config.NumberColumn(
                "Coverage",
                format="%.1f%%",
            ),
            "raw_iqr": st.column_config.NumberColumn(
                "Raw-signal IQR",
                format="%.4f",
            ),
            "ic_as_of_date": st.column_config.DateColumn(
                "IC as of",
                format="YYYY-MM-DD",
            ),
            "ic": st.column_config.NumberColumn(
                "Rank IC",
                format="%.4f",
            ),
            "rolling_mean_ic_252_as_of_date": st.column_config.DateColumn(
                "Rolling IC as of",
                format="YYYY-MM-DD",
            ),
            "rolling_mean_ic_252": st.column_config.NumberColumn(
                "Rolling 252-day mean IC",
                format="%.4f",
            ),
            "rank_stability_1d": st.column_config.NumberColumn(
                "One-day rank stability",
                format="%.3f",
            ),
            "rank_stability_21d": st.column_config.NumberColumn(
                "21-day rank stability",
                format="%.3f",
            ),
        },
    )


def render_signal_health_page(bundle, start_date, end_date) -> None:
    """Render latest and historical retained-factor diagnostics."""
    st.header("Factor & Signal Health")
    st.caption(
        "The latest snapshot is independent of the historical date filter. "
        "The charts use the selected start and end dates; the portfolio "
        "selector does not alter factor-level analytics."
    )
    snapshot = build_latest_factor_snapshot(
        bundle.monitoring["latest_overview"],
        bundle.monitoring["signal_health"],
    )
    factors = snapshot["factor"].tolist()
    signal_history = prepare_signal_health_history(
        bundle.monitoring["signal_health"],
        factors=factors,
        start_date=start_date,
        end_date=end_date,
    )
    dependence_history = prepare_factor_dependence_history(
        bundle.monitoring["factor_dependence"],
        start_date=start_date,
        end_date=end_date,
    )
    latest_dependence = dependence_history.iloc[-1]
    latest_rolling_correlation = latest_dependence[
        "rolling_factor_rank_correlation_252"
    ]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Factors monitored", len(snapshot))
    metric_columns[1].metric(
        "Latest signal date",
        f"{snapshot['latest_date'].max():%Y-%m-%d}",
    )
    metric_columns[2].metric(
        "Active signal alerts",
        int(snapshot["signal_status"].ne("PASS").sum()),
    )
    metric_columns[3].metric(
        "Latest rolling dependence",
        (
            "N/A"
            if pd.isna(latest_rolling_correlation)
            else f"{latest_rolling_correlation:.3f}"
        ),
        help=f"As of {latest_dependence['date']:%Y-%m-%d}",
    )

    st.subheader("Latest retained-factor snapshot")
    _render_factor_snapshot(snapshot)

    st.subheader("Historical signal diagnostics")
    signal_columns = st.columns(2)
    signal_columns[0].plotly_chart(
        build_signal_health_figure(
            signal_history,
            "signal_coverage",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    signal_columns[1].plotly_chart(
        build_signal_health_figure(
            signal_history,
            "rolling_mean_ic_252",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    signal_columns[0].plotly_chart(
        build_signal_health_figure(
            signal_history,
            "rank_stability_21d",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    signal_columns[1].plotly_chart(
        build_factor_dependence_figure(
            dependence_history,
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )


def _build_latest_risk_snapshot(beta_history, concentration_history):
    latest_beta = (
        beta_history.sort_values(["portfolio", "date"], kind="stable")
        .groupby("portfolio", sort=False, as_index=False)
        .tail(1)
        .rename(columns={"date": "beta_date"})
    )
    latest_concentration = (
        concentration_history.sort_values(
            ["portfolio", "date"],
            kind="stable",
        )
        .groupby("portfolio", sort=False, as_index=False)
        .tail(1)
        .rename(columns={"date": "concentration_date"})
    )
    snapshot = latest_beta.merge(
        latest_concentration,
        on="portfolio",
        how="inner",
        validate="one_to_one",
    )

    if not snapshot["beta_date"].eq(snapshot["concentration_date"]).all():
        raise ValueError(
            "Latest beta and concentration dates do not align by portfolio."
        )

    return (
        snapshot.rename(columns={"beta_date": "latest_date"})
        .drop(columns="concentration_date")
        .reset_index(drop=True)
    )


def _render_risk_snapshot(snapshot) -> None:
    percentage_columns = (
        "beta_coverage",
        "largest_absolute_sector_net_exposure",
        "top_five_absolute_beta_contribution_share",
        "top_five_contributor_share_63",
    )
    display = snapshot.copy()

    for column in percentage_columns:
        display[column] = display[column] * 100.0

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "portfolio": st.column_config.TextColumn("Portfolio"),
            "latest_date": st.column_config.DateColumn(
                "Latest date",
                format="YYYY-MM-DD",
            ),
            "beta_coverage": st.column_config.NumberColumn(
                "Beta coverage",
                format="%.1f%%",
            ),
            "holdings_market_beta": st.column_config.NumberColumn(
                "Holdings beta",
                format="%.2f",
            ),
            "realised_gross_beta_126": st.column_config.NumberColumn(
                "Realised gross beta",
                format="%.2f",
            ),
            "beta_measurement_gap": st.column_config.NumberColumn(
                "Beta gap",
                format="%.2f",
            ),
            "effective_position_count": st.column_config.NumberColumn(
                "Effective positions",
                format="%.1f",
            ),
            "largest_absolute_sector_net_exposure": (
                st.column_config.NumberColumn(
                    "Largest sector net",
                    format="%.1f%%",
                )
            ),
            "top_five_absolute_beta_contribution_share": (
                st.column_config.NumberColumn(
                    "Top-five beta share",
                    format="%.1f%%",
                )
            ),
            "effective_contributor_count_63": st.column_config.NumberColumn(
                "Effective contributors",
                format="%.1f",
            ),
            "top_five_contributor_share_63": st.column_config.NumberColumn(
                "Top-five contribution share",
                format="%.1f%%",
            ),
            "effective_contribution_sector_count_63": (
                st.column_config.NumberColumn(
                    "Contribution sectors",
                    format="%.1f",
                )
            ),
        },
    )


def render_risk_concentration_page(
    bundle,
    portfolios: Sequence[str],
    start_date,
    end_date,
) -> None:
    """Render filtered beta and concentration monitoring analytics."""
    st.header("Risk & Concentration")
    st.caption(
        "Holdings-implied and realised market exposure together with position, "
        "sector, and realised-contribution concentration for the selected "
        "portfolios and evaluation window."
    )
    beta_history = prepare_beta_history(
        bundle.monitoring["beta"],
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    concentration_history = prepare_concentration_history(
        bundle.monitoring["concentration"],
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    snapshot = _build_latest_risk_snapshot(
        beta_history,
        concentration_history,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected portfolios", len(snapshot))
    metric_columns[1].metric(
        "Latest risk date",
        f"{snapshot['latest_date'].max():%Y-%m-%d}",
    )
    metric_columns[2].metric(
        "Maximum absolute holdings beta",
        f"{snapshot['holdings_market_beta'].abs().max():.2f}",
    )
    metric_columns[3].metric(
        "Maximum absolute sector net",
        f"{snapshot['largest_absolute_sector_net_exposure'].max():.1%}",
    )

    st.subheader("Latest risk and concentration snapshot")
    _render_risk_snapshot(snapshot)

    st.subheader("Market beta")
    beta_columns = st.columns(2)
    beta_columns[0].plotly_chart(
        build_beta_figure(
            beta_history,
            "holdings_market_beta",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    beta_columns[1].plotly_chart(
        build_beta_figure(
            beta_history,
            "realised_gross_beta_126",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_beta_figure(
            beta_history,
            "beta_measurement_gap",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Concentration")
    concentration_columns = st.columns(2)
    concentration_columns[0].plotly_chart(
        build_concentration_figure(
            concentration_history,
            "effective_position_count",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    concentration_columns[1].plotly_chart(
        build_concentration_figure(
            concentration_history,
            "largest_absolute_sector_net_exposure",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    concentration_columns[0].plotly_chart(
        build_concentration_figure(
            concentration_history,
            "top_five_contributor_share_63",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    concentration_columns[1].plotly_chart(
        build_concentration_figure(
            concentration_history,
            "effective_contribution_sector_count_63",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )


def _build_latest_implementation_snapshot(
    implementation_history,
    liquidity_history,
):
    latest_implementation = (
        implementation_history.sort_values(
            ["portfolio", "date"],
            kind="stable",
        )
        .groupby("portfolio", sort=False, as_index=False)
        .tail(1)
        .rename(columns={"date": "implementation_date"})
    )
    latest_liquidity = (
        liquidity_history.sort_values(
            ["portfolio", "date"],
            kind="stable",
        )
        .groupby("portfolio", sort=False, as_index=False)
        .tail(1)
        .loc[:, ["portfolio", "date", "liquidity_coverage"]]
        .rename(
            columns={
                "date": "liquidity_date",
                "liquidity_coverage": "latest_liquidity_coverage",
            }
        )
    )
    liquidity_window = liquidity_history.groupby(
        "portfolio", sort=False, as_index=False
    ).agg(
        minimum_liquidity_coverage=("liquidity_coverage", "min"),
        incomplete_coverage_days=(
            "liquidity_coverage",
            lambda values: int(values.fillna(0.0).lt(1.0 - 1e-12).sum()),
        ),
    )
    snapshot = latest_implementation.merge(
        latest_liquidity,
        on="portfolio",
        how="inner",
        validate="one_to_one",
    ).merge(
        liquidity_window,
        on="portfolio",
        how="left",
        validate="one_to_one",
    )

    if not snapshot["implementation_date"].eq(snapshot["liquidity_date"]).all():
        raise ValueError(
            "Latest implementation and liquidity dates do not align by " "portfolio."
        )

    return (
        snapshot.rename(columns={"implementation_date": "latest_date"})
        .drop(columns="liquidity_date")
        .reset_index(drop=True)
    )


def _render_implementation_snapshot(snapshot) -> None:
    display = snapshot.assign(
        largest_trade_weight_63=lambda data: (data["largest_trade_weight_63"] * 100.0),
        maximum_missing_return_weight_63=lambda data: (
            data["maximum_missing_return_weight_63"] * 100.0
        ),
        latest_liquidity_coverage=lambda data: (
            data["latest_liquidity_coverage"] * 100.0
        ),
        minimum_liquidity_coverage=lambda data: (
            data["minimum_liquidity_coverage"] * 100.0
        ),
    )[
        [
            "portfolio",
            "latest_date",
            "trade_count",
            "annualised_turnover_63",
            "largest_trade_weight_63",
            "minimum_trade_capacity_1pct_usd_millions_63",
            "maximum_missing_return_weight_63",
            "latest_liquidity_coverage",
            "minimum_liquidity_coverage",
            "incomplete_coverage_days",
        ]
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "portfolio": st.column_config.TextColumn("Portfolio"),
            "latest_date": st.column_config.DateColumn(
                "Latest date",
                format="YYYY-MM-DD",
            ),
            "trade_count": st.column_config.NumberColumn(
                "Latest trade count",
                format="%d",
            ),
            "annualised_turnover_63": st.column_config.NumberColumn(
                "Annualised turnover",
                format="%.2fx",
            ),
            "largest_trade_weight_63": st.column_config.NumberColumn(
                "Largest trade weight",
                format="%.2f%%",
            ),
            "minimum_trade_capacity_1pct_usd_millions_63": (
                st.column_config.NumberColumn(
                    "Minimum capacity (USDm)",
                    format="$%.1f",
                )
            ),
            "maximum_missing_return_weight_63": (
                st.column_config.NumberColumn(
                    "Missing-return weight",
                    format="%.2f%%",
                )
            ),
            "latest_liquidity_coverage": st.column_config.NumberColumn(
                "Latest liquidity coverage",
                format="%.1f%%",
            ),
            "minimum_liquidity_coverage": st.column_config.NumberColumn(
                "Minimum window coverage",
                format="%.1f%%",
            ),
            "incomplete_coverage_days": st.column_config.NumberColumn(
                "Incomplete days",
                format="%d",
            ),
        },
    )


def render_implementation_liquidity_page(
    bundle,
    portfolios: Sequence[str],
    start_date,
    end_date,
) -> None:
    """Render filtered trading-implementation and liquidity diagnostics."""
    st.header("Implementation & Liquidity")
    st.caption(
        "Turnover, trade size, capacity, missing-return exposure, and traded-"
        "weight liquidity coverage for the selected portfolios and evaluation "
        "window."
    )
    implementation_history = prepare_implementation_history(
        bundle.monitoring["implementation"],
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    liquidity_history = prepare_liquidity_coverage_history(
        bundle.monitoring["liquidity_coverage"],
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    snapshot = _build_latest_implementation_snapshot(
        implementation_history,
        liquidity_history,
    )
    minimum_capacity = snapshot["minimum_trade_capacity_1pct_usd_millions_63"].min()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected portfolios", len(snapshot))
    metric_columns[1].metric(
        "Latest implementation date",
        f"{snapshot['latest_date'].max():%Y-%m-%d}",
    )
    metric_columns[2].metric(
        "Maximum annualised turnover",
        f"{snapshot['annualised_turnover_63'].max():.2f}x",
    )
    metric_columns[3].metric(
        "Minimum capacity",
        f"${minimum_capacity:.1f}m",
    )

    st.subheader("Latest implementation and liquidity snapshot")
    _render_implementation_snapshot(snapshot)

    st.subheader("Implementation diagnostics")
    implementation_columns = st.columns(2)
    implementation_columns[0].plotly_chart(
        build_implementation_figure(
            implementation_history,
            "annualised_turnover_63",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    implementation_columns[1].plotly_chart(
        build_implementation_figure(
            implementation_history,
            "largest_trade_weight_63",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    implementation_columns[0].plotly_chart(
        build_implementation_figure(
            implementation_history,
            "minimum_trade_capacity_1pct_usd_millions_63",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    implementation_columns[1].plotly_chart(
        build_implementation_figure(
            implementation_history,
            "maximum_missing_return_weight_63",
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Liquidity coverage")
    st.plotly_chart(
        build_liquidity_coverage_figure(
            liquidity_history,
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )


def _render_side_cost_summary(summary) -> None:
    percentage_columns = (
        "annualised_long_contribution",
        "annualised_short_contribution",
        "annualised_gross_contribution",
        "annualised_cost_drag",
        "annualised_net_contribution",
        "average_daily_turnover",
        "average_rebalance_turnover",
    )
    display = summary.copy()

    for column in percentage_columns:
        display[column] = display[column] * 100.0

    display = display[
        [
            "portfolio",
            "observations",
            "rebalance_count",
            "start_date",
            "end_date",
            "annualised_long_contribution",
            "annualised_short_contribution",
            "annualised_gross_contribution",
            "annualised_cost_drag",
            "annualised_net_contribution",
            "average_daily_turnover",
            "average_rebalance_turnover",
        ]
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "portfolio": st.column_config.TextColumn("Portfolio"),
            "observations": st.column_config.NumberColumn(
                "Observations",
                format="%d",
            ),
            "rebalance_count": st.column_config.NumberColumn(
                "Rebalances",
                format="%d",
            ),
            "start_date": st.column_config.DateColumn(
                "Start date",
                format="YYYY-MM-DD",
            ),
            "end_date": st.column_config.DateColumn(
                "End date",
                format="YYYY-MM-DD",
            ),
            "annualised_long_contribution": st.column_config.NumberColumn(
                "Annualised long",
                format="%.2f%%",
            ),
            "annualised_short_contribution": st.column_config.NumberColumn(
                "Annualised short",
                format="%.2f%%",
            ),
            "annualised_gross_contribution": st.column_config.NumberColumn(
                "Annualised gross",
                format="%.2f%%",
            ),
            "annualised_cost_drag": st.column_config.NumberColumn(
                "Annualised cost drag",
                format="%.2f%%",
            ),
            "annualised_net_contribution": st.column_config.NumberColumn(
                "Annualised net",
                format="%.2f%%",
            ),
            "average_daily_turnover": st.column_config.NumberColumn(
                "Average daily turnover",
                format="%.2f%%",
            ),
            "average_rebalance_turnover": st.column_config.NumberColumn(
                "Average rebalance turnover",
                format="%.2f%%",
            ),
        },
    )


def _render_security_contribution_table(security_summary) -> None:
    percentage_columns = (
        "cumulative_gross_contribution",
        "cumulative_transaction_cost",
        "cumulative_net_contribution",
        "absolute_contribution_share",
    )
    display = security_summary.head(15).copy()

    for column in percentage_columns:
        display[column] = display[column] * 100.0

    display = display[
        [
            "ticker",
            "observations",
            "active_days",
            "cumulative_gross_contribution",
            "cumulative_transaction_cost",
            "cumulative_net_contribution",
            "absolute_contribution_share",
        ]
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "observations": st.column_config.NumberColumn(
                "Observations",
                format="%d",
            ),
            "active_days": st.column_config.NumberColumn(
                "Active days",
                format="%d",
            ),
            "cumulative_gross_contribution": st.column_config.NumberColumn(
                "Cumulative gross",
                format="%.2f%%",
            ),
            "cumulative_transaction_cost": st.column_config.NumberColumn(
                "Cumulative cost",
                format="%.2f%%",
            ),
            "cumulative_net_contribution": st.column_config.NumberColumn(
                "Cumulative net",
                format="%.2f%%",
            ),
            "absolute_contribution_share": st.column_config.NumberColumn(
                "Absolute share",
                format="%.2f%%",
            ),
        },
    )


def render_attribution_page(
    bundle,
    portfolios: Sequence[str],
    start_date,
    end_date,
) -> None:
    """Render filtered portfolio-side, cost, and security attribution."""
    st.header("Attribution")
    st.caption(
        "Arithmetic return attribution by long side, short side, transaction "
        "costs, and individual securities for the selected evaluation window."
    )
    attribution_history = prepare_portfolio_attribution_history(
        bundle.attribution["portfolio_daily"],
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )
    side_cost_summary = build_side_cost_attribution_summary(
        bundle.attribution["portfolio_daily"],
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected portfolios", len(side_cost_summary))
    metric_columns[1].metric(
        "Observations per portfolio",
        int(side_cost_summary["observations"].min()),
    )
    metric_columns[2].metric(
        "Total rebalance events",
        int(side_cost_summary["rebalance_count"].sum()),
    )
    metric_columns[3].metric(
        "Maximum annualised cost drag",
        f"{side_cost_summary['annualised_cost_drag'].max():.2%}",
    )

    st.subheader("Side and transaction-cost attribution")
    _render_side_cost_summary(side_cost_summary)
    st.plotly_chart(
        build_side_cost_attribution_figure(
            side_cost_summary,
            height=430,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Cumulative attribution")
    attribution_tabs = st.tabs(list(portfolios))

    for tab, portfolio in zip(attribution_tabs, portfolios, strict=True):
        with tab:
            st.plotly_chart(
                build_cumulative_attribution_figure(
                    attribution_history,
                    portfolio,
                    height=430,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )

    st.subheader("Security contribution analysis")
    security_portfolio = st.selectbox(
        "Portfolio for security-level attribution",
        options=list(portfolios),
        key="security_attribution_portfolio",
    )
    security_summary = build_security_contribution_summary(
        bundle.attribution["security_daily"],
        security_portfolio,
        start_date=start_date,
        end_date=end_date,
    )
    top_five_share = security_summary.nlargest(
        5,
        "absolute_gross_contribution",
    )["absolute_contribution_share"].sum()

    security_metrics = st.columns(4)
    security_metrics[0].metric("Securities", len(security_summary))
    security_metrics[1].metric(
        "Active securities",
        int(security_summary["active_days"].gt(0).sum()),
    )
    security_metrics[2].metric(
        "Cumulative net contribution",
        f"{security_summary['cumulative_net_contribution'].sum():.2%}",
    )
    security_metrics[3].metric(
        "Top-five absolute share",
        f"{top_five_share:.2%}",
    )

    _render_security_contribution_table(security_summary)

    security_columns = st.columns(2)
    security_columns[0].plotly_chart(
        build_security_contribution_figure(
            security_summary,
            top_n=10,
            height=520,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    security_columns[1].plotly_chart(
        build_security_contribution_share_figure(
            security_summary,
            top_n=15,
            height=520,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
    )


def render_page_placeholder(
    page: str,
    portfolios: Sequence[str],
    start_date,
    end_date,
) -> None:
    """Render a filter-aware placeholder."""
    st.header(page)

    columns = st.columns(3)
    columns[0].metric(
        "Selected portfolios",
        len(portfolios),
    )
    columns[1].metric(
        "Start date",
        f"{start_date:%Y-%m-%d}",
    )
    columns[2].metric(
        "End date",
        f"{end_date:%Y-%m-%d}",
    )

    st.info("This page will be assembled in the next dashboard round.")
