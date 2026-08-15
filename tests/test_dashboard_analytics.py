import numpy as np
import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    LATEST_PORTFOLIO_SNAPSHOT_COLUMNS,
    build_latest_portfolio_snapshot,
    build_performance_summary,
    prepare_performance_history,
)
from alpha_research.metrics import summarise_returns


@pytest.fixture()
def performance_risk():
    dates = pd.bdate_range("2026-07-01", periods=4)
    returns_by_portfolio = {
        "Alpha": [0.01, -0.02, 0.03, 0.01],
        "Beta": [0.005, 0.01, -0.005, 0.02],
        "SPY": [0.008, -0.01, 0.015, 0.005],
    }
    rows = []

    for portfolio, returns in returns_by_portfolio.items():
        wealth = np.cumprod(1.0 + np.asarray(returns))
        running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
        drawdown = wealth / running_peak - 1.0

        for date, daily_return, wealth_value, drawdown_value in zip(
            dates,
            returns,
            wealth,
            drawdown,
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "return": daily_return,
                    "wealth": wealth_value,
                    "drawdown": drawdown_value,
                    "trailing_return_252": 0.10,
                    "rolling_sharpe_252": 0.80,
                    "annualised_volatility_126": 0.20,
                    "maximum_drawdown_252": -0.15,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture()
def latest_snapshot_inputs(performance_risk):
    portfolios = ["Alpha", "Beta"]
    dates = pd.bdate_range("2026-06-30", periods=2)
    selected_implementations = pd.DataFrame(
        {
            "portfolio": portfolios,
            "rebalance_frequency": [21, 10],
            "rebalance_offset": [0, 0],
            "role": ["Primary", "Benchmark"],
        }
    )
    latest_overview = pd.DataFrame(
        {
            "entity_type": ["Factor", "Portfolio", "Portfolio"],
            "entity": ["Momentum", "Alpha", "Beta"],
            "overall_status": ["PASS", "WARNING", "PASS"],
            "market_risk_status": ["N/A", "WARNING", "PASS"],
            "concentration_status": ["N/A", "WARNING", "PASS"],
            "implementation_status": ["N/A", "PASS", "PASS"],
        }
    )

    def build_state(values):
        return pd.DataFrame(
            [
                {
                    "portfolio": portfolio,
                    "date": date,
                    **values,
                }
                for portfolio in portfolios
                for date in dates
            ]
        )

    performance_state = performance_risk.loc[
        performance_risk["portfolio"].isin(portfolios)
        & performance_risk["date"].isin(dates)
    ].copy()
    beta = build_state(
        {
            "holdings_market_beta": 1.10,
            "realised_gross_beta_126": 0.90,
            "beta_measurement_gap": 0.20,
        }
    )
    concentration = build_state(
        {
            "effective_position_count": 40.0,
            "largest_absolute_sector_net_exposure": 0.40,
            "top_five_absolute_beta_contribution_share": 0.30,
            "effective_contribution_sector_count_63": 5.0,
            "top_five_contributor_share_63": 0.25,
        }
    )
    implementation = build_state(
        {
            "annualised_turnover_63": 10.0,
            "largest_trade_weight_63": 0.05,
            "minimum_trade_capacity_1pct_usd_63": 50_000_000.0,
            "maximum_missing_return_weight_63": 0.0,
        }
    )
    liquidity_coverage = build_state({"liquidity_coverage": 1.0})

    return {
        "selected_implementations": selected_implementations,
        "latest_overview": latest_overview,
        "performance_risk": performance_state,
        "beta": beta,
        "concentration": concentration,
        "implementation": implementation,
        "liquidity_coverage": liquidity_coverage,
    }


def test_prepare_performance_history_filters_orders_and_rebases(performance_risk):
    start_date = performance_risk["date"].drop_duplicates().sort_values().iloc[1]
    history = prepare_performance_history(
        performance_risk,
        portfolios=["Beta", "Alpha"],
        start_date=start_date,
    )

    assert list(pd.unique(history["portfolio"])) == ["Beta", "Alpha"]
    assert history["date"].min() == start_date
    assert history.groupby("portfolio")["indexed_wealth"].first().eq(1.0).all()

    expected_drawdown = performance_risk.loc[
        performance_risk["portfolio"].isin(["Beta", "Alpha"])
        & performance_risk["date"].ge(start_date),
        ["portfolio", "date", "drawdown"],
    ]
    drawdown_audit = history[["portfolio", "date", "drawdown"]].merge(
        expected_drawdown,
        on=["portfolio", "date"],
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )

    assert np.allclose(
        drawdown_audit["drawdown_observed"],
        drawdown_audit["drawdown_expected"],
    )


def test_build_performance_summary_uses_shared_metrics(performance_risk):
    summary = build_performance_summary(
        performance_risk,
        portfolios=["Alpha", "SPY"],
    ).set_index("portfolio")
    expected = summarise_returns(
        performance_risk.loc[
            performance_risk["portfolio"].eq("Alpha"),
            "return",
        ]
    )

    assert list(summary.index) == ["Alpha", "SPY"]
    assert summary.loc["Alpha", "observations"] == 4
    assert summary.loc["Alpha", "annualised_return"] == pytest.approx(
        expected["annualised_return"]
    )
    assert summary.loc["Alpha", "sharpe_ratio"] == pytest.approx(
        expected["sharpe_ratio"]
    )
    assert summary.loc["Alpha", "maximum_drawdown"] == pytest.approx(
        expected["maximum_drawdown"]
    )


def test_prepare_performance_history_rejects_invalid_filters(performance_risk):
    with pytest.raises(ValueError, match="start_date must not be after"):
        prepare_performance_history(
            performance_risk,
            start_date="2026-07-10",
            end_date="2026-07-01",
        )

    with pytest.raises(ValueError, match="missing portfolios"):
        prepare_performance_history(
            performance_risk,
            portfolios=["Missing"],
        )


def test_latest_portfolio_snapshot_combines_aligned_states(latest_snapshot_inputs):
    snapshot = build_latest_portfolio_snapshot(**latest_snapshot_inputs)

    assert tuple(snapshot.columns) == LATEST_PORTFOLIO_SNAPSHOT_COLUMNS
    assert snapshot["portfolio"].tolist() == ["Alpha", "Beta"]
    assert snapshot["implementation_role"].tolist() == ["Primary", "Benchmark"]
    assert snapshot["latest_date"].nunique() == 1
    assert snapshot["latest_date"].iloc[0] == pd.Timestamp("2026-07-01")
    assert snapshot.loc[0, "overall_status"] == "WARNING"
    assert snapshot.loc[0, "implementation_status"] == "PASS"


def test_latest_portfolio_snapshot_rejects_misaligned_dates(latest_snapshot_inputs):
    invalid_inputs = dict(latest_snapshot_inputs)
    invalid_beta = invalid_inputs["beta"].copy()
    latest_alpha = invalid_beta["portfolio"].eq("Alpha") & invalid_beta["date"].eq(
        invalid_beta["date"].max()
    )
    invalid_beta.loc[latest_alpha, "date"] += pd.offsets.BDay(1)
    invalid_inputs["beta"] = invalid_beta

    with pytest.raises(ValueError, match="dates do not align"):
        build_latest_portfolio_snapshot(**invalid_inputs)
