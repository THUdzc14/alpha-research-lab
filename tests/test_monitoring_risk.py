import numpy as np
import pandas as pd
import pytest

from alpha_research.risk import (
    BETA_STATE_EXPORT_COLUMNS,
    calculate_beta_concentration_state,
    calculate_beta_state,
    calculate_concentration_state,
    calculate_contribution_concentration_state,
    calculate_holdings_beta_state,
    calculate_position_concentration_state,
    calculate_realised_beta_state,
    calculate_sector_concentration_state,
    prepare_concentration_security_detail,
    prepare_holdings_beta_detail,
    calculate_rolling_contribution_detail,
)


def make_beta_inputs(observations=130):
    dates = pd.bdate_range("2025-01-01", periods=observations)
    market_returns = np.resize(
        np.array([-0.02, -0.01, 0.01, 0.02]),
        observations,
    )
    portfolio_daily = pd.DataFrame(
        {
            "date": dates,
            "portfolio": "Strategy",
            "long_return": 1.4 * market_returns,
            "short_return": -0.6 * market_returns,
            "gross_return": 0.8 * market_returns,
            "net_return": 0.8 * market_returns,
            "long_exposure": 1.0,
            "short_exposure": 1.0,
            "net_exposure": 0.0,
            "gross_exposure": 2.0,
        }
    )
    benchmark_daily = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": market_returns,
        }
    )
    holdings_rows = []
    beta_rows = []
    security_rows = []
    sector_rows = []

    for date in dates:
        for ticker, weight, beta, contribution, sector in [
            ("AAA", 1.0, 1.4, 0.01, "Technology"),
            ("BBB", -1.0, 0.6, -0.005, "Financials"),
        ]:
            holdings_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "portfolio": "Strategy",
                    "weight": weight,
                }
            )
            beta_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "beta_126": beta,
                }
            )
            security_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "portfolio": "Strategy",
                    "weight": weight,
                    "gross_contribution": contribution,
                }
            )
            sector_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                }
            )

    return (
        portfolio_daily,
        benchmark_daily,
        pd.DataFrame(holdings_rows),
        pd.DataFrame(beta_rows),
        pd.DataFrame(security_rows),
        pd.DataFrame(sector_rows),
    )


def test_realised_beta_state_recovers_each_return_stream():
    portfolio_daily, benchmark_daily, *_ = make_beta_inputs(observations=8)

    result = calculate_realised_beta_state(
        portfolio_daily,
        benchmark_daily,
        window=4,
        min_periods=4,
    )
    latest = result.groupby("return_stream").tail(1).set_index("return_stream")

    assert latest.loc["long", "realised_beta_4"] == pytest.approx(1.4)
    assert latest.loc["short", "realised_beta_4"] == pytest.approx(-0.6)
    assert latest.loc["gross", "realised_beta_4"] == pytest.approx(0.8)
    assert latest.loc["net", "realised_beta_4"] == pytest.approx(0.8)
    assert np.allclose(latest["market_correlation_4"].abs(), 1.0)


def test_holdings_beta_state_tracks_contributions_and_missing_coverage():
    _, _, holdings, betas, *_ = make_beta_inputs(observations=1)
    betas.loc[betas["ticker"].eq("BBB"), "beta_126"] = np.nan

    detail = prepare_holdings_beta_detail(holdings, betas)
    result = calculate_holdings_beta_state(detail).iloc[0]

    assert result["beta_coverage"] == pytest.approx(0.5)
    assert result["holdings_long_beta_contribution"] == pytest.approx(1.4)
    assert result["holdings_short_beta_contribution"] == pytest.approx(0.0)
    assert result["holdings_market_beta"] == pytest.approx(1.4)
    assert result["long_basket_beta"] == pytest.approx(1.4)
    assert result["short_basket_beta"] == pytest.approx(0.0)


def test_combined_beta_state_reconciles_holdings_and_realised_beta():
    portfolio_daily, benchmark_daily, holdings, betas, *_ = make_beta_inputs()

    result = calculate_beta_state(
        portfolio_daily,
        benchmark_daily,
        holdings,
        betas,
    )
    latest = result.iloc[-1]

    assert tuple(result.columns) == BETA_STATE_EXPORT_COLUMNS
    assert len(result) == 130
    assert result["realised_gross_beta_126"].iloc[:125].isna().all()
    assert latest["beta_coverage"] == pytest.approx(1.0)
    assert latest["holdings_market_beta"] == pytest.approx(0.8)
    assert latest["realised_gross_beta_126"] == pytest.approx(0.8)
    assert latest["beta_measurement_gap"] == pytest.approx(0.0)
    assert latest["long_basket_beta"] == pytest.approx(1.4)
    assert latest["short_basket_beta"] == pytest.approx(0.6)


def test_beta_inputs_reject_duplicate_security_keys():
    _, _, holdings, betas, *_ = make_beta_inputs(observations=2)
    duplicated = pd.concat([betas, betas.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_holdings_beta_detail(holdings, duplicated)


def test_concentration_preparation_requires_sector_coverage_for_held_weight():
    *_, security_daily, sectors = make_beta_inputs(observations=2)
    sectors.loc[
        sectors["ticker"].eq("AAA"),
        "sector",
    ] = np.nan

    with pytest.raises(ValueError, match="lacks sector metadata"):
        prepare_concentration_security_detail(
            security_daily,
            sectors,
        )


def test_position_sector_and_beta_concentration_statistics():
    _, _, holdings, betas, security_daily, sectors = make_beta_inputs(observations=1)
    concentration_detail = prepare_concentration_security_detail(
        security_daily,
        sectors,
    )
    beta_detail = prepare_holdings_beta_detail(holdings, betas)

    position = calculate_position_concentration_state(concentration_detail).iloc[0]
    sector = calculate_sector_concentration_state(concentration_detail).iloc[0]
    beta = calculate_beta_concentration_state(beta_detail).iloc[0]

    assert position["gross_exposure"] == pytest.approx(2.0)
    assert position["held_position_count"] == 2
    assert position["effective_position_count"] == pytest.approx(2.0)
    assert position["largest_position_gross_share"] == pytest.approx(0.5)
    assert sector["effective_sector_count"] == pytest.approx(2.0)
    assert sector["largest_sector_gross_share"] == pytest.approx(0.5)
    assert sector["largest_absolute_sector_net_exposure"] == pytest.approx(1.0)
    assert beta["effective_beta_contributor_count"] == pytest.approx(
        1.0 / (0.7**2 + 0.3**2)
    )
    assert beta["largest_absolute_beta_contribution_share"] == pytest.approx(0.7)


def test_rolling_contribution_concentration_uses_complete_trailing_window():
    *_, security_daily, sectors = make_beta_inputs(observations=3)
    detail = prepare_concentration_security_detail(
        security_daily,
        sectors,
    )

    security_state, sector_state = calculate_contribution_concentration_state(
        detail,
        window=2,
    )

    assert pd.isna(security_state["effective_contributor_count_2"].iloc[0])
    assert security_state["effective_contributor_count_2"].iloc[1] == pytest.approx(1.8)
    assert security_state["largest_contributor_2"].iloc[1] == "AAA"
    assert sector_state["effective_contribution_sector_count_2"].iloc[1] == (
        pytest.approx(1.8)
    )
    assert sector_state["largest_contribution_sector_2"].iloc[1] == "Technology"


def test_rolling_contribution_concentration_zero_fills_sparse_security_rows():
    dates = pd.bdate_range(
        "2025-01-01",
        periods=3,
    )

    detail = pd.DataFrame(
        [
            {
                "portfolio": "Strategy",
                "date": date,
                "ticker": "AAA",
                "sector": "Technology",
                "gross_contribution": 0.01,
                "absolute_gross_contribution": 0.01,
            }
            for date in dates
        ]
        + [
            {
                "portfolio": "Strategy",
                "date": dates[0],
                "ticker": "BBB",
                "sector": "Financials",
                "gross_contribution": 0.02,
                "absolute_gross_contribution": 0.02,
            },
            {
                "portfolio": "Strategy",
                "date": dates[2],
                "ticker": "BBB",
                "sector": "Financials",
                "gross_contribution": 0.02,
                "absolute_gross_contribution": 0.02,
            },
        ]
    )

    (
        security_detail,
        sector_detail,
    ) = calculate_rolling_contribution_detail(
        detail,
        window=2,
    )

    (
        security_state,
        sector_state,
    ) = calculate_contribution_concentration_state(
        detail,
        window=2,
    )

    # BBB is absent on the middle date. Its contribution on that date
    # must be treated as zero rather than removing the date from its window.
    bbb_detail = security_detail.loc[security_detail["ticker"].eq("BBB")].sort_values(
        "date"
    )

    assert bbb_detail["rolling_absolute_gross_contribution_2"].iloc[1] == pytest.approx(
        0.02
    )

    assert bbb_detail["rolling_absolute_gross_contribution_2"].iloc[2] == pytest.approx(
        0.02
    )

    financials_detail = sector_detail.loc[
        sector_detail["sector"].eq("Financials")
    ].sort_values("date")

    assert financials_detail["rolling_absolute_gross_contribution_2"].iloc[
        1
    ] == pytest.approx(0.02)

    assert security_state["effective_contributor_count_2"].iloc[1] == pytest.approx(2.0)

    assert security_state["effective_contributor_count_2"].iloc[2] == pytest.approx(2.0)

    assert sector_state["effective_contribution_sector_count_2"].iloc[
        1
    ] == pytest.approx(2.0)

    assert sector_state["effective_contribution_sector_count_2"].iloc[
        2
    ] == pytest.approx(2.0)


def test_complete_concentration_state_has_notebook_schema():
    _, _, holdings, betas, security_daily, sectors = make_beta_inputs(observations=65)
    beta_detail = prepare_holdings_beta_detail(holdings, betas)

    result = calculate_concentration_state(
        security_daily,
        sectors,
        beta_detail,
    )
    latest = result.iloc[-1]

    assert len(result) == 65
    assert len(result.columns) == 27
    assert latest["effective_contributor_count_63"] == pytest.approx(1.8)
    assert latest["top_five_contributor_share_63"] == pytest.approx(1.0)
    assert latest["largest_contributor_63"] == "AAA"
    assert latest["effective_contribution_sector_count_63"] == pytest.approx(1.8)
    assert latest["largest_contribution_sector_63"] == "Technology"
