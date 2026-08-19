import pandas as pd
import pytest

from alpha_research.dashboard_ui import (
    DASHBOARD_PAGES,
    FRESHNESS_TABLE_COLUMNS,
    build_dashboard_filter_options,
    prepare_dashboard_freshness_table,
)


def test_filter_options_preserve_order_and_window():
    selected = pd.DataFrame(
        {
            "portfolio": [
                "Composite",
                "Fixed",
                "Inverse",
            ],
        }
    )
    performance = pd.DataFrame(
        [
            {
                "portfolio": portfolio,
                "date": date,
            }
            for portfolio, dates in {
                "Composite": pd.bdate_range(
                    "2020-01-01",
                    periods=5,
                ),
                "Fixed": pd.bdate_range(
                    "2020-01-02",
                    periods=5,
                ),
                "Inverse": pd.bdate_range(
                    "2020-01-01",
                    periods=4,
                ),
                "SPY": pd.bdate_range(
                    "2020-01-01",
                    periods=6,
                ),
            }.items()
            for date in dates
        ]
    )

    result = build_dashboard_filter_options(
        selected,
        performance,
    )

    assert result.portfolios == (
        "Composite",
        "Fixed",
        "Inverse",
    )
    assert result.minimum_date == pd.Timestamp("2020-01-02")
    assert result.maximum_date == pd.Timestamp("2020-01-06")
    assert len(DASHBOARD_PAGES) == 6


def test_filter_options_reject_bad_portfolios():
    selected = pd.DataFrame(
        {
            "portfolio": [
                "Alpha",
                "Alpha",
            ]
        }
    )
    performance = pd.DataFrame(
        {
            "portfolio": ["Alpha"],
            "date": [pd.Timestamp("2020-01-01")],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate portfolios",
    ):
        build_dashboard_filter_options(
            selected,
            performance,
        )

    missing = pd.DataFrame(
        {
            "portfolio": [
                "Alpha",
                "Beta",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing portfolios",
    ):
        build_dashboard_filter_options(
            missing,
            performance,
        )


def test_freshness_table_orders_and_filters():
    metadata = pd.DataFrame(
        {
            "group": [
                "monitoring",
                "attribution",
                "monitoring",
            ],
            "dataset": [
                "ready",
                "missing",
                "stale",
            ],
            "latest_observation_date": [
                pd.Timestamp("2026-08-18"),
                pd.NaT,
                pd.Timestamp("2026-08-01"),
            ],
            "freshness_reference_date": [
                pd.Timestamp("2026-08-18"),
                pd.NaT,
                pd.Timestamp("2026-08-01"),
            ],
            "age_business_days": [
                0,
                pd.NA,
                12,
            ],
            "is_stale": [
                False,
                pd.NA,
                True,
            ],
            "status": [
                "READY",
                "MISSING",
                "STALE",
            ],
            "error": [
                pd.NA,
                "missing file",
                pd.NA,
            ],
        }
    )

    prepared = prepare_dashboard_freshness_table(metadata)
    stale = prepare_dashboard_freshness_table(
        metadata,
        stale_only=True,
    )

    assert tuple(prepared.columns) == (FRESHNESS_TABLE_COLUMNS)
    assert prepared["dataset"].tolist() == [
        "missing",
        "stale",
        "ready",
    ]
    assert stale["dataset"].tolist() == ["stale"]


def test_freshness_table_rejects_invalid_inputs():
    with pytest.raises(
        KeyError,
        match="latest_observation_date",
    ):
        prepare_dashboard_freshness_table(pd.DataFrame({"group": []}))

    metadata = pd.DataFrame(columns=FRESHNESS_TABLE_COLUMNS)

    with pytest.raises(
        TypeError,
        match="stale_only must be Boolean",
    ):
        prepare_dashboard_freshness_table(
            metadata,
            stale_only=1,
        )
