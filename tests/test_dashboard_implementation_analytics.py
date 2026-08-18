import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    IMPLEMENTATION_HISTORY_COLUMNS,
    LIQUIDITY_COVERAGE_HISTORY_COLUMNS,
    prepare_implementation_history,
    prepare_liquidity_coverage_history,
)


@pytest.fixture()
def implementation_histories():
    dates = pd.bdate_range("2026-06-29", periods=4)
    implementation_rows = []
    liquidity_rows = []

    for portfolio_number, portfolio in enumerate(("Alpha", "Beta")):
        for date_number, date in enumerate(dates):
            turnover = 0.05 + date_number * 0.01

            implementation_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "turnover": turnover,
                    "transaction_cost": turnover * 0.001,
                    "trade_count": 10 + date_number,
                    "annualised_turnover_63": (8.0 + portfolio_number),
                    "largest_trade_weight_63": (0.04 + date_number * 0.001),
                    "minimum_trade_capacity_1pct_usd_63": (
                        50_000_000.0 + portfolio_number * 10_000_000.0
                    ),
                    "maximum_missing_return_weight_63": 0.0,
                }
            )

            liquidity_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "turnover": turnover,
                    "liquidity_covered_turnover": turnover,
                    "liquidity_coverage": 1.0,
                }
            )

    return (
        pd.DataFrame(implementation_rows),
        pd.DataFrame(liquidity_rows),
    )


def test_prepare_implementation_history_filters_and_scales_capacity(
    implementation_histories,
):
    implementation, _ = implementation_histories

    history = prepare_implementation_history(
        implementation.sample(frac=1.0, random_state=7),
        portfolios=["Beta", "Alpha"],
        start_date="2026-06-30",
    )

    assert tuple(history.columns) == (IMPLEMENTATION_HISTORY_COLUMNS)
    assert list(pd.unique(history["portfolio"])) == [
        "Beta",
        "Alpha",
    ]
    assert history["date"].min() == pd.Timestamp("2026-06-30")
    assert history.groupby("portfolio")["date"].size().eq(3).all()

    assert (
        history.loc[
            history["portfolio"].eq("Beta"),
            "minimum_trade_capacity_1pct_usd_millions_63",
        ]
        .eq(60.0)
        .all()
    )


def test_prepare_liquidity_history_filters_and_preserves_coverage(
    implementation_histories,
):
    _, liquidity = implementation_histories

    history = prepare_liquidity_coverage_history(
        liquidity,
        portfolios=["Alpha"],
        end_date="2026-07-01",
    )

    assert tuple(history.columns) == (LIQUIDITY_COVERAGE_HISTORY_COLUMNS)
    assert history["portfolio"].eq("Alpha").all()
    assert history["date"].max() == pd.Timestamp("2026-07-01")
    assert history["liquidity_coverage"].eq(1.0).all()


def test_implementation_history_rejects_malformed_data(
    implementation_histories,
):
    implementation, liquidity = implementation_histories

    duplicated = pd.concat(
        [implementation, implementation.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate portfolio-date",
    ):
        prepare_implementation_history(duplicated)

    invalid_numeric = liquidity.copy()
    invalid_numeric["liquidity_coverage"] = invalid_numeric[
        "liquidity_coverage"
    ].astype("object")
    invalid_numeric.loc[
        0,
        "liquidity_coverage",
    ] = "invalid"

    with pytest.raises(ValueError, match="non-numeric"):
        prepare_liquidity_coverage_history(invalid_numeric)

    with pytest.raises(ValueError, match="missing portfolios"):
        prepare_implementation_history(
            implementation,
            portfolios=["Missing"],
        )
