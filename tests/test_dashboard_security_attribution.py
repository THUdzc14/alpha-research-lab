import numpy as np
import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    SECURITY_CONTRIBUTION_SUMMARY_COLUMNS,
    build_security_contribution_summary,
)


@pytest.fixture()
def security_daily():
    dates = pd.bdate_range(
        "2026-06-29",
        periods=3,
    )
    rows = []

    for portfolio in ("Alpha", "Beta"):
        for date_number, date in enumerate(dates):
            for ticker, weight, gross in (
                (
                    "AAA",
                    0.6,
                    0.010 + 0.001 * date_number,
                ),
                (
                    "BBB",
                    -0.4,
                    -0.004 - 0.001 * date_number,
                ),
            ):
                cost = 0.0005 if date_number == 0 else 0.0
                rows.append(
                    {
                        "date": date,
                        "portfolio": portfolio,
                        "ticker": ticker,
                        "weight": weight,
                        "long_contribution": (gross if weight > 0.0 else 0.0),
                        "short_contribution": (gross if weight < 0.0 else 0.0),
                        "gross_contribution": gross,
                        "transaction_cost_contribution": (cost),
                        "net_contribution": (gross - cost),
                    }
                )

    return pd.DataFrame(rows)


def test_summary_reconciles_and_orders(
    security_daily,
):
    summary = build_security_contribution_summary(
        security_daily,
        "Alpha",
    )

    assert tuple(summary.columns) == (SECURITY_CONTRIBUTION_SUMMARY_COLUMNS)
    assert summary["ticker"].tolist() == [
        "AAA",
        "BBB",
    ]
    assert summary["observations"].tolist() == [
        3,
        3,
    ]
    assert summary["active_days"].tolist() == [
        3,
        3,
    ]
    assert summary["absolute_contribution_share"].sum() == pytest.approx(1.0)

    assert summary["cumulative_gross_contribution"].sum() == pytest.approx(
        security_daily.loc[
            security_daily["portfolio"].eq("Alpha"),
            "gross_contribution",
        ].sum()
    )

    assert np.allclose(
        summary["cumulative_net_contribution"],
        summary["cumulative_gross_contribution"]
        - summary["cumulative_transaction_cost"],
    )


def test_summary_filters_dates(
    security_daily,
):
    summary = build_security_contribution_summary(
        security_daily,
        "Beta",
        start_date="2026-06-30",
        end_date="2026-07-01",
    )

    assert summary["observations"].tolist() == [2, 2]
    assert summary["start_date"].min() == pd.Timestamp("2026-06-30")
    assert summary["end_date"].max() == pd.Timestamp("2026-07-01")


def test_summary_rejects_malformed_data(
    security_daily,
):
    duplicated = pd.concat(
        [
            security_daily,
            security_daily.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate portfolio-date-ticker",
    ):
        build_security_contribution_summary(
            duplicated,
            "Alpha",
        )

    invalid_numeric = security_daily.copy()
    invalid_numeric["gross_contribution"] = invalid_numeric[
        "gross_contribution"
    ].astype("object")
    invalid_numeric.loc[
        0,
        "gross_contribution",
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        build_security_contribution_summary(
            invalid_numeric,
            "Alpha",
        )

    invalid_identity = security_daily.copy()
    invalid_identity.loc[
        0,
        "net_contribution",
    ] += 0.01

    with pytest.raises(
        ValueError,
        match="cost contributions do not reconcile",
    ):
        build_security_contribution_summary(
            invalid_identity,
            "Alpha",
        )

    with pytest.raises(
        ValueError,
        match="missing portfolio",
    ):
        build_security_contribution_summary(
            security_daily,
            "Missing",
        )
