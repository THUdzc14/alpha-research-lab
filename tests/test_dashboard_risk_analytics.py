import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    BETA_HISTORY_COLUMNS,
    CONCENTRATION_HISTORY_COLUMNS,
    prepare_beta_history,
    prepare_concentration_history,
)


@pytest.fixture()
def risk_histories():
    dates = pd.bdate_range("2026-06-29", periods=4)
    beta_rows = []
    concentration_rows = []

    for portfolio_number, portfolio in enumerate(("Alpha", "Beta")):
        for date_number, date in enumerate(dates):
            beta_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "beta_coverage": 1.0,
                    "holdings_market_beta": (
                        0.80 + portfolio_number * 0.10 + date_number * 0.01
                    ),
                    "realised_gross_beta_126": (
                        0.70 + portfolio_number * 0.10 + date_number * 0.01
                    ),
                    "beta_measurement_gap": 0.10,
                }
            )

            concentration_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "effective_position_count": (40.0 + date_number),
                    "largest_absolute_sector_net_exposure": (
                        0.30 + portfolio_number * 0.05
                    ),
                    "top_five_absolute_beta_contribution_share": (0.40),
                    "effective_contributor_count_63": (25.0 + date_number),
                    "top_five_contributor_share_63": 0.35,
                    "effective_contribution_sector_count_63": 5.0,
                }
            )

    return (
        pd.DataFrame(beta_rows),
        pd.DataFrame(concentration_rows),
    )


def test_prepare_beta_history_filters_and_preserves_portfolio_order(
    risk_histories,
):
    beta, _ = risk_histories

    history = prepare_beta_history(
        beta.sample(frac=1.0, random_state=7),
        portfolios=["Beta", "Alpha"],
        start_date="2026-06-30",
    )

    assert tuple(history.columns) == BETA_HISTORY_COLUMNS
    assert list(pd.unique(history["portfolio"])) == [
        "Beta",
        "Alpha",
    ]
    assert history["date"].min() == pd.Timestamp("2026-06-30")
    assert history.groupby("portfolio")["date"].size().eq(3).all()


def test_prepare_concentration_history_filters_and_preserves_values(
    risk_histories,
):
    _, concentration = risk_histories

    history = prepare_concentration_history(
        concentration,
        portfolios=["Alpha"],
        end_date="2026-07-01",
    )

    assert tuple(history.columns) == CONCENTRATION_HISTORY_COLUMNS
    assert history["portfolio"].eq("Alpha").all()
    assert history["date"].max() == pd.Timestamp("2026-07-01")
    assert history["effective_position_count"].tolist() == [
        40.0,
        41.0,
        42.0,
    ]


def test_risk_history_preparation_rejects_malformed_data(
    risk_histories,
):
    beta, concentration = risk_histories

    duplicated = pd.concat(
        [beta, beta.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate portfolio-date",
    ):
        prepare_beta_history(duplicated)

    invalid_numeric = concentration.copy()
    invalid_numeric["effective_position_count"] = invalid_numeric[
        "effective_position_count"
    ].astype("object")
    invalid_numeric.loc[
        0,
        "effective_position_count",
    ] = "invalid"

    with pytest.raises(ValueError, match="non-numeric"):
        prepare_concentration_history(invalid_numeric)

    with pytest.raises(ValueError, match="missing portfolios"):
        prepare_beta_history(
            beta,
            portfolios=["Missing"],
        )
