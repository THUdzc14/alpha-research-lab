import numpy as np
import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    PORTFOLIO_ATTRIBUTION_HISTORY_COLUMNS,
    SIDE_COST_ATTRIBUTION_SUMMARY_COLUMNS,
    build_side_cost_attribution_summary,
    prepare_portfolio_attribution_history,
)


@pytest.fixture()
def portfolio_daily():
    dates = pd.bdate_range(
        "2026-06-29",
        periods=4,
    )
    rows = []

    for portfolio_number, portfolio in enumerate(("Alpha", "Beta")):
        for date_number, date in enumerate(dates):
            long_return = 0.01 + portfolio_number * 0.001
            short_return = -0.004 + date_number * 0.001
            gross_return = long_return + short_return
            transaction_cost = 0.0005 if date_number % 2 == 0 else 0.0
            rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "is_rebalance": (date_number % 2 == 0),
                    "long_return": long_return,
                    "short_return": short_return,
                    "gross_return": gross_return,
                    "transaction_cost": (transaction_cost),
                    "net_return": (gross_return - transaction_cost),
                    "turnover": (0.50 if date_number % 2 == 0 else 0.0),
                    "long_exposure": 1.0,
                    "short_exposure": 1.0,
                    "net_exposure": 0.0,
                    "gross_exposure": 2.0,
                    "missing_return_weight": 0.0,
                }
            )

    return pd.DataFrame(rows)


def test_prepare_history_filters_and_reanchors(
    portfolio_daily,
):
    history = prepare_portfolio_attribution_history(
        portfolio_daily.sample(
            frac=1.0,
            random_state=7,
        ),
        portfolios=["Beta", "Alpha"],
        start_date="2026-06-30",
    )

    assert tuple(history.columns) == (PORTFOLIO_ATTRIBUTION_HISTORY_COLUMNS)
    assert list(pd.unique(history["portfolio"])) == ["Beta", "Alpha"]
    assert history["date"].min() == pd.Timestamp("2026-06-30")

    first = history.groupby(
        "portfolio",
        sort=False,
    ).head(1)

    assert np.allclose(
        first["cumulative_long_contribution"],
        first["long_return"],
    )
    assert np.allclose(
        first["cumulative_cost_contribution"],
        -first["transaction_cost"],
    )


def test_side_cost_summary_matches_attribution(
    portfolio_daily,
):
    summary = build_side_cost_attribution_summary(
        portfolio_daily,
        portfolios=["Alpha", "Beta"],
        periods_per_year=4,
    ).set_index("portfolio")

    assert tuple(summary.reset_index().columns) == SIDE_COST_ATTRIBUTION_SUMMARY_COLUMNS

    assert (
        summary.loc[
            "Alpha",
            "observations",
        ]
        == 4
    )
    assert (
        summary.loc[
            "Alpha",
            "rebalance_count",
        ]
        == 2
    )
    assert summary.loc[
        "Alpha",
        "annualised_long_contribution",
    ] == pytest.approx(0.04)
    assert summary.loc[
        "Alpha",
        "annualised_cost_drag",
    ] == pytest.approx(0.001)
    assert summary.loc[
        "Alpha",
        "cumulative_gross_contribution",
    ] == pytest.approx(
        summary.loc[
            "Alpha",
            "cumulative_long_contribution",
        ]
        + summary.loc[
            "Alpha",
            "cumulative_short_contribution",
        ]
    )


def test_attribution_rejects_malformed_data(
    portfolio_daily,
):
    duplicated = pd.concat(
        [
            portfolio_daily,
            portfolio_daily.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate portfolio-date",
    ):
        prepare_portfolio_attribution_history(duplicated)

    invalid_numeric = portfolio_daily.copy()
    invalid_numeric["long_return"] = invalid_numeric["long_return"].astype("object")
    invalid_numeric.loc[
        0,
        "long_return",
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        prepare_portfolio_attribution_history(invalid_numeric)

    invalid_identity = portfolio_daily.copy()
    invalid_identity.loc[
        0,
        "gross_return",
    ] += 0.01

    with pytest.raises(
        ValueError,
        match="side attribution does not reconcile",
    ):
        prepare_portfolio_attribution_history(invalid_identity)

    missing_flag = portfolio_daily.copy()
    missing_flag["is_rebalance"] = missing_flag["is_rebalance"].astype("boolean")
    missing_flag.loc[
        0,
        "is_rebalance",
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="non-missing Boolean",
    ):
        prepare_portfolio_attribution_history(missing_flag)
