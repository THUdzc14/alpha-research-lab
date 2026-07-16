import numpy as np
import pandas as pd
import pytest

from alpha_research.factors import (
    add_raw_factors,
    momentum_12_1m,
    momentum_3m,
    realised_volatility,
    reversal_1m,
)


def make_constant_growth_panel(
    ticker: str = "AAA",
    observations: int = 300,
    daily_growth: float = 0.01,
) -> pd.DataFrame:
    """Create a deterministic adjusted-price path.

    Price at observation t:

        100 * (1 + daily_growth) ** t
    """
    dates = pd.bdate_range("2020-01-01", periods=observations)
    index = np.arange(observations)

    adj_close = 100.0 * (1.0 + daily_growth) ** index

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "adj_close": adj_close,
        }
    )


def test_momentum_3m_matches_manual_formula():
    panel = make_constant_growth_panel(
        observations=100,
        daily_growth=0.01,
    )

    factor = momentum_3m(panel, lookback=63)

    expected = (1.01**63) - 1.0

    assert factor.iloc[:63].isna().all()
    assert factor.iloc[63] == pytest.approx(expected)
    assert factor.iloc[-1] == pytest.approx(expected)


def test_reversal_is_negative_one_month_return():
    panel = make_constant_growth_panel(
        observations=50,
        daily_growth=0.01,
    )

    factor = reversal_1m(panel, lookback=21)

    expected_return = (1.01**21) - 1.0
    expected_reversal = -expected_return

    assert factor.iloc[:21].isna().all()
    assert factor.iloc[21] == pytest.approx(expected_reversal)
    assert factor.iloc[-1] == pytest.approx(expected_reversal)


def test_momentum_12_1m_skips_most_recent_period():
    panel = make_constant_growth_panel(
        observations=300,
        daily_growth=0.01,
    )

    factor = momentum_12_1m(
        panel,
        long_lag=252,
        skip_lag=21,
    )

    # The measured return covers 252 - 21 = 231 daily growth periods.
    expected = (1.01 ** (252 - 21)) - 1.0

    assert factor.iloc[:252].isna().all()
    assert factor.iloc[252] == pytest.approx(expected)
    assert factor.iloc[-1] == pytest.approx(expected)


def test_realised_volatility_matches_manual_calculation():
    returns = np.array(
        [
            0.01,
            -0.02,
            0.03,
            -0.01,
            0.02,
        ]
    )

    prices = [100.0]

    for ret in returns:
        prices.append(prices[-1] * (1.0 + ret))

    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=len(prices)),
            "ticker": "AAA",
            "adj_close": prices,
        }
    )

    factor = realised_volatility(
        panel,
        window=5,
        annualisation_factor=252,
    )

    expected = returns.std(ddof=1) * np.sqrt(252)

    # The first daily return is NaN, so five valid returns become available
    # at the sixth price observation.
    assert factor.iloc[:5].isna().all()
    assert factor.iloc[5] == pytest.approx(expected)


def test_factors_are_calculated_separately_by_ticker():
    panel_a = make_constant_growth_panel(
        ticker="AAA",
        observations=80,
        daily_growth=0.01,
    )

    panel_b = make_constant_growth_panel(
        ticker="BBB",
        observations=80,
        daily_growth=-0.005,
    )

    panel = pd.concat([panel_b, panel_a], ignore_index=True)

    result = add_raw_factors(
        panel,
        momentum_12m_lag=40,
        momentum_skip_lag=5,
        momentum_3m_lookback=20,
        reversal_lookback=10,
        volatility_window=20,
    )

    a_last = result.query("ticker == 'AAA'").iloc[-1]
    b_last = result.query("ticker == 'BBB'").iloc[-1]

    assert a_last["mom_3m_raw"] > 0
    assert b_last["mom_3m_raw"] < 0

    assert a_last["reversal_1m_raw"] < 0
    assert b_last["reversal_1m_raw"] > 0


def test_add_raw_factors_preserves_existing_columns():
    panel = make_constant_growth_panel(observations=300)
    panel["sector"] = "Technology"
    panel["forward_ret_5d"] = 0.02

    result = add_raw_factors(panel)

    expected_columns = {
        "sector",
        "forward_ret_5d",
        "mom_12_1m_raw",
        "mom_3m_raw",
        "reversal_1m_raw",
        "realised_vol_63_raw",
    }

    assert expected_columns.issubset(result.columns)


def test_duplicate_date_ticker_rows_raise_error():
    panel = make_constant_growth_panel(observations=20)
    panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        momentum_3m(panel, lookback=5)


def test_invalid_12_1m_lags_raise_error():
    panel = make_constant_growth_panel(observations=50)

    with pytest.raises(
        ValueError,
        match="long_lag must be greater than skip_lag",
    ):
        momentum_12_1m(
            panel,
            long_lag=21,
            skip_lag=21,
        )
