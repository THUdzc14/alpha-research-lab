"""Streamlit entry point for the Alpha Research Lab dashboard."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from alpha_research.dashboard_data import (
    DashboardArtifactBundle,
    load_dashboard_artifacts,
)
from alpha_research.dashboard_ui import (
    DASHBOARD_PAGES,
    build_dashboard_filter_options,
    prepare_dashboard_freshness_table,
)

# Streamlit executes this entry point as a file and may omit the repository
# root from Python's import path.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from dashboard.dashboard_pages import (  # noqa: E402
    render_attribution_page,
    render_implementation_liquidity_page,
    render_performance_page,
    render_risk_concentration_page,
    render_signal_health_page,
    render_strategy_overview,
)

st.set_page_config(
    page_title="Alpha Research Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(
    ttl=300,
    show_spinner=("Loading validated research artifacts..."),
)
def load_cached_dashboard_artifacts() -> DashboardArtifactBundle:
    """Load artifacts while allowing the UI to explain failures."""
    return load_dashboard_artifacts(strict=False)


def render_artifact_state(bundle: DashboardArtifactBundle) -> None:
    """Display structural errors or stale-data warnings."""
    freshness_table = prepare_dashboard_freshness_table(bundle.metadata)

    if bundle.errors:
        st.error(
            "The dashboard artifacts are incomplete "
            "or invalid. Run the refresh workflow and "
            "resolve the errors below."
        )

        for error in bundle.errors:
            st.write(f"- {error}")

        with st.expander(
            "Artifact readiness details",
            expanded=True,
        ):
            st.dataframe(
                freshness_table,
                hide_index=True,
                width="stretch",
            )

        st.stop()

    if bundle.has_stale_data:
        stale_table = prepare_dashboard_freshness_table(
            bundle.metadata,
            stale_only=True,
        )
        maximum_age = stale_table["age_business_days"].max()

        st.warning(
            f"{len(stale_table)} dashboard datasets "
            f"are stale as of "
            f"{bundle.as_of_date:%Y-%m-%d}; "
            f"the oldest is {maximum_age} business "
            "days behind its reference date."
        )

        with st.expander("Stale-data details"):
            st.dataframe(
                stale_table,
                hide_index=True,
                width="stretch",
            )
    else:
        st.success(f"All dated dashboard artifacts are current as of {bundle.as_of_date:%Y-%m-%d}.")


def render_sidebar(
    bundle: DashboardArtifactBundle,
) -> tuple[str, tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    """Render navigation and common portfolio/date controls."""
    options = build_dashboard_filter_options(
        bundle.attribution["selected_implementations"],
        bundle.monitoring["performance_risk"],
    )

    st.sidebar.title("Alpha Research Lab")
    page = st.sidebar.radio(
        "Page",
        DASHBOARD_PAGES,
    )
    st.sidebar.divider()
    st.sidebar.subheader("Analysis filters")

    portfolios = st.sidebar.multiselect(
        "Portfolios",
        options=options.portfolios,
        default=options.portfolios,
    )
    start_date_value = st.sidebar.date_input(
        "Start date",
        value=options.minimum_date.date(),
        min_value=options.minimum_date.date(),
        max_value=options.maximum_date.date(),
        key="dashboard_start_date",
    )
    end_date_value = st.sidebar.date_input(
        "End date",
        value=options.maximum_date.date(),
        min_value=options.minimum_date.date(),
        max_value=options.maximum_date.date(),
        key="dashboard_end_date",
    )

    if not portfolios and page != "Factor & Signal Health":
        st.sidebar.error("Select at least one portfolio.")
        st.stop()

    start_date = pd.Timestamp(start_date_value).normalize()
    end_date = pd.Timestamp(end_date_value).normalize()

    if start_date > end_date:
        st.sidebar.error("Start date must not be after end date.")
        st.stop()

    return page, tuple(portfolios), start_date, end_date


def render_dashboard_page(
    page: str,
    bundle: DashboardArtifactBundle,
    portfolios: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Dispatch a supported page while preserving its filter interface."""
    if page == "Strategy Overview":
        render_strategy_overview(
            bundle,
            portfolios,
        )
    elif page == "Performance & Drawdowns":
        render_performance_page(
            bundle,
            portfolios,
            start_date,
            end_date,
        )
    elif page == "Factor & Signal Health":
        render_signal_health_page(
            bundle,
            start_date,
            end_date,
        )
    elif page == "Risk & Concentration":
        render_risk_concentration_page(
            bundle,
            portfolios,
            start_date,
            end_date,
        )
    elif page == "Implementation & Liquidity":
        render_implementation_liquidity_page(
            bundle,
            portfolios,
            start_date,
            end_date,
        )
    elif page == "Attribution":
        render_attribution_page(
            bundle,
            portfolios,
            start_date,
            end_date,
        )
    else:
        raise ValueError(f"Unsupported dashboard page: {page}")


def main() -> None:
    """Run the dashboard shell."""
    bundle = load_cached_dashboard_artifacts()

    st.title("Systematic Alpha Research Framework")
    st.caption(
        "Validated research artifacts, frozen "
        "strategy definitions, and reusable "
        "monitoring analytics."
    )

    render_artifact_state(bundle)

    (
        page,
        portfolios,
        start_date,
        end_date,
    ) = render_sidebar(bundle)

    render_dashboard_page(
        page,
        bundle,
        portfolios,
        start_date,
        end_date,
    )

    with st.expander("All artifact metadata"):
        st.dataframe(
            prepare_dashboard_freshness_table(bundle.metadata),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
