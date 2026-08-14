import numpy as np
import pandas as pd
import pytest

from alpha_research.portfolio import (
    calculate_rebalance_inverse_volatility_allocations,
)
from alpha_research.workflows import (
    build_frozen_strategy_target_weights,
)


def test_rebalance_inverse_volatility_uses_only_prior_joint_history():
    dates = pd.bdate_range("2025-01-02", periods=4)
    sleeve_returns = pd.DataFrame(
        {
            "Momentum": [0.01, -0.01, 0.01, -0.01],
            "Realised Volatility": [0.02, -0.02, 0.02, -0.02],
        },
        index=dates,
    )

    result = calculate_rebalance_inverse_volatility_allocations(
        sleeve_returns,
        rebalance_dates=pd.DatetimeIndex([dates[1], dates[3]]),
        lookback=3,
        min_periods=2,
        periods_per_year=252,
    )

    assert result.loc[dates[1], "Momentum"] == pytest.approx(0.5)
    assert result.loc[dates[1], "Realised Volatility"] == pytest.approx(0.5)
    assert result.loc[dates[3], "Momentum"] == pytest.approx(2.0 / 3.0)
    assert result.loc[dates[3], "Realised Volatility"] == pytest.approx(1.0 / 3.0)


def test_frozen_strategy_workflow_builds_all_selected_targets():
    dates = pd.bdate_range("2025-01-02", periods=60)
    tickers = [f"S{number:02d}" for number in range(30)]
    rows = []

    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            score = float(ticker_number)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "mom_12_1m_z": score,
                    "realised_vol_63_z": score,
                    "forward_ret_1d": (
                        (ticker_number % 5 - 2) / 1_000.0
                        + (date_number % 3 - 1) / 10_000.0
                    ),
                }
            )

    result = build_frozen_strategy_target_weights(pd.DataFrame(rows))

    assert tuple(result) == (
        "Composite Score",
        "Fixed 50/50 Sleeves",
        "Pure Inverse Volatility",
    )
    assert result["Composite Score"]["date"].nunique() == 3
    assert result["Fixed 50/50 Sleeves"]["date"].nunique() == 6
    assert result["Pure Inverse Volatility"]["date"].nunique() == 6

    for targets in result.values():
        gross_exposure = targets.groupby("date")["weight"].agg(
            lambda weights: weights.abs().sum()
        )

        assert np.allclose(gross_exposure, 2.0)


def test_frozen_strategy_workflow_rejects_duplicate_panel_keys():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ticker": ["AAA", "AAA"],
            "mom_12_1m_z": [1.0, 1.0],
            "realised_vol_63_z": [1.0, 1.0],
            "forward_ret_1d": [0.01, 0.01],
        }
    )

    with pytest.raises(ValueError, match="duplicate date-ticker"):
        build_frozen_strategy_target_weights(panel)
