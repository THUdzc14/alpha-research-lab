import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest import BacktestConfig
from alpha_research.portfolio import build_factor_target_weights


def make_portfolio_panel() -> pd.DataFrame:
    rows = []

    for date in pd.bdate_range("2024-01-01", periods=6):
        for stock_number in range(10):
            rows.append(
                {
                    "date": date,
                    "ticker": f"S{stock_number:02d}",
                    "factor": float(stock_number),
                    "forward_ret_1d": 0.0,
                }
            )

    return pd.DataFrame(rows)


def test_factor_targets_use_only_rebalance_dates():
    panel = make_portfolio_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    expected_dates = [
        panel["date"].drop_duplicates().iloc[0],
        panel["date"].drop_duplicates().iloc[5],
    ]

    assert targets["date"].drop_duplicates().tolist() == expected_dates


def test_factor_targets_have_expected_exposures_and_direction():
    panel = make_portfolio_panel()

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    for _, cross_section in targets.groupby("date"):
        long_gross = cross_section["weight"].clip(lower=0.0).sum()
        short_gross = -cross_section["weight"].clip(upper=0.0).sum()

        assert long_gross == pytest.approx(1.0)
        assert short_gross == pytest.approx(1.0)
        assert cross_section["weight"].sum() == pytest.approx(0.0)

        weights = cross_section.set_index("ticker")["weight"]

        assert weights["S09"] > 0
        assert weights["S08"] > 0
        assert weights["S00"] < 0
        assert weights["S01"] < 0


def test_factor_targets_exclude_dates_without_usable_returns():
    panel = make_portfolio_panel()

    last_date = panel["date"].max()
    panel.loc[
        panel["date"] == last_date,
        "forward_ret_1d",
    ] = np.nan

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    assert last_date not in targets["date"].unique()


def test_factor_targets_are_zero_when_coverage_is_insufficient():
    panel = make_portfolio_panel()

    first_date = panel["date"].min()
    first_date_rows = panel["date"] == first_date

    panel.loc[
        panel.index[first_date_rows][:2],
        "factor",
    ] = np.nan

    config = BacktestConfig(
        rebalance_frequency=5,
        min_observations=10,
    )

    targets = build_factor_target_weights(
        panel=panel,
        factor_column="factor",
        config=config,
    )

    first_targets = targets.loc[
        targets["date"] == first_date,
        "weight",
    ]

    assert np.all(first_targets == 0.0)
