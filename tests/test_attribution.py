import numpy as np
import pandas as pd
import pytest

from alpha_research.attribution import (
    SECURITY_ATTRIBUTION_EXPORT_COLUMNS,
    prepare_security_attribution,
    reconcile_security_attribution,
    reconstruct_portfolio_daily_attribution,
)


def make_attribution_inputs():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    holdings = pd.DataFrame(
        [
            {
                "date": dates[0],
                "ticker": "AAA",
                "pre_trade_weight": 0.0,
                "weight": 1.0,
                "trade": 1.0,
            },
            {
                "date": dates[0],
                "ticker": "BBB",
                "pre_trade_weight": 0.0,
                "weight": -1.0,
                "trade": -1.0,
            },
            {
                "date": dates[0],
                "ticker": "ZZZ",
                "pre_trade_weight": 0.0,
                "weight": 0.0,
                "trade": 0.0,
            },
            {
                "date": dates[1],
                "ticker": "AAA",
                "pre_trade_weight": 1.0,
                "weight": 1.0,
                "trade": 0.0,
            },
            {
                "date": dates[1],
                "ticker": "BBB",
                "pre_trade_weight": -1.0,
                "weight": 0.0,
                "trade": 1.0,
            },
        ]
    )
    holdings = holdings.assign(
        portfolio="Strategy",
        rebalance_frequency=2,
        rebalance_offset=0,
        role="Test implementation",
    )
    returns = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1]],
            "ticker": ["AAA", "BBB", "AAA"],
            "forward_ret_1d": [0.02, -0.01, np.nan],
        }
    )

    return holdings, returns


def test_security_attribution_reproduces_notebook_formulas():
    holdings, returns = make_attribution_inputs()

    result = prepare_security_attribution(
        holdings,
        returns,
        transaction_cost_bps=10.0,
    )
    keyed = result.set_index(["date", "ticker"])
    first_date = holdings["date"].min()
    second_date = holdings["date"].max()

    assert tuple(result.columns) == SECURITY_ATTRIBUTION_EXPORT_COLUMNS
    assert len(result) == 4
    assert (first_date, "ZZZ") not in keyed.index
    assert keyed.loc[(first_date, "AAA"), "gross_contribution"] == pytest.approx(0.02)
    assert keyed.loc[(first_date, "BBB"), "short_contribution"] == pytest.approx(0.01)
    assert keyed.loc[
        (first_date, "AAA"), "transaction_cost_contribution"
    ] == pytest.approx(0.001)
    assert keyed.loc[(first_date, "AAA"), "net_contribution"] == pytest.approx(0.019)
    assert keyed.loc[(second_date, "BBB"), "holding_side"] == "Flat"


def test_security_attribution_distinguishes_missing_record_and_value():
    holdings, returns = make_attribution_inputs()

    result = prepare_security_attribution(holdings, returns)
    keyed = result.set_index(["date", "ticker"])
    second_date = holdings["date"].max()

    assert not bool(keyed.loc[(second_date, "AAA"), "return_record_missing"])
    assert bool(keyed.loc[(second_date, "AAA"), "asset_return_missing"])
    assert keyed.loc[(second_date, "AAA"), "realised_asset_return"] == pytest.approx(
        0.0
    )
    assert bool(keyed.loc[(second_date, "BBB"), "return_record_missing"])
    assert bool(keyed.loc[(second_date, "BBB"), "asset_return_missing"])
    assert keyed.loc[
        (second_date, "AAA"), "missing_return_weight_contribution"
    ] == pytest.approx(1.0)
    assert keyed.loc[
        (second_date, "BBB"), "missing_return_weight_contribution"
    ] == pytest.approx(0.0)


def test_security_attribution_reconstructs_portfolio_daily():
    holdings, returns = make_attribution_inputs()
    security = prepare_security_attribution(holdings, returns)
    portfolio_daily = reconstruct_portfolio_daily_attribution(security)

    audit = reconcile_security_attribution(portfolio_daily, security)

    assert len(portfolio_daily) == 2
    assert audit["audit_passes"].all()
    assert audit["maximum_absolute_difference"].max() == pytest.approx(0.0)


def test_security_attribution_audit_detects_formula_difference():
    holdings, returns = make_attribution_inputs()
    security = prepare_security_attribution(holdings, returns)
    portfolio_daily = reconstruct_portfolio_daily_attribution(security)
    portfolio_daily.loc[0, "gross_return"] += 0.01

    audit = reconcile_security_attribution(portfolio_daily, security)

    assert not audit["audit_passes"].all()
    assert audit["max_abs_gross_return_difference"].iloc[0] == pytest.approx(0.01)


def test_security_attribution_rejects_duplicate_return_keys():
    holdings, returns = make_attribution_inputs()
    duplicated_returns = pd.concat(
        [returns, returns.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="return_panel contains duplicate keys"):
        prepare_security_attribution(holdings, duplicated_returns)


def test_security_attribution_audit_rejects_incomplete_date_coverage():
    holdings, returns = make_attribution_inputs()
    security = prepare_security_attribution(holdings, returns)
    portfolio_daily = reconstruct_portfolio_daily_attribution(security)

    with pytest.raises(ValueError, match="portfolio-day coverage"):
        reconcile_security_attribution(
            portfolio_daily.iloc[:-1],
            security,
        )
