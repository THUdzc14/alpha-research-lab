import numpy as np
import pandas as pd
import pytest

from alpha_research.artifacts import (
    MONITORING_ARTIFACT_CONTRACTS,
    validate_monitoring_artifacts,
    write_monitoring_artifacts,
)
from alpha_research.config.research import selected_implementations_frame
from alpha_research.workflows import (
    MONITORING_DATASET_NAMES,
    build_strategy_monitoring_datasets,
    validate_frozen_implementations,
)


@pytest.fixture(scope="module")
def monitoring_inputs():
    dates = pd.bdate_range("2025-01-01", periods=130)
    tickers = ["AAA", "BBB", *[f"S{i:02d}" for i in range(28)]]
    portfolios = selected_implementations_frame()["portfolio"].tolist()
    market_returns = np.resize(np.array([-0.02, -0.01, 0.01, 0.02]), len(dates))
    factor_rows = []
    holding_rows = []
    security_rows = []
    portfolio_rows = []

    for date_number, (date, market_return) in enumerate(
        zip(dates, market_returns, strict=True)
    ):
        for ticker_number, ticker in enumerate(tickers):
            momentum = float(ticker_number)
            realised_volatility = float(len(tickers) - ticker_number)
            factor_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "mom_12_1m_z": momentum,
                    "mom_12_1m_raw": momentum * 10.0,
                    "realised_vol_63_z": realised_volatility,
                    "realised_vol_63_raw": realised_volatility / 10.0,
                    "forward_ret_5d": momentum / 1_000.0,
                    "beta_126": (
                        1.4 if ticker == "AAA" else 0.6 if ticker == "BBB" else 1.0
                    ),
                    "sector": (
                        "Technology" if ticker_number % 2 == 0 else "Financials"
                    ),
                    "dollar_volume": 1_000_000.0 + ticker_number * 10_000.0,
                }
            )

        for portfolio in portfolios:
            portfolio_rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "long_return": 1.4 * market_return,
                    "short_return": -0.6 * market_return,
                    "gross_return": 0.8 * market_return,
                    "net_return": 0.8 * market_return,
                    "long_exposure": 1.0,
                    "short_exposure": 1.0,
                    "net_exposure": 0.0,
                    "gross_exposure": 2.0,
                }
            )

            for ticker, weight, contribution in (
                ("AAA", 1.0, 1.4 * market_return),
                ("BBB", -1.0, -0.6 * market_return),
            ):
                trade = weight if date_number == 21 else 0.0
                holding_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "portfolio": portfolio,
                        "weight": weight,
                    }
                )
                security_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "portfolio": portfolio,
                        "weight": weight,
                        "gross_contribution": contribution,
                        "trade": trade,
                        "absolute_trade_weight": abs(trade),
                        "transaction_cost_contribution": abs(trade) * 0.001,
                        "return_record_missing": False,
                        "missing_return_weight_contribution": 0.0,
                    }
                )

    return {
        "factor_panel": pd.DataFrame(factor_rows),
        "selected_implementations": selected_implementations_frame(),
        "portfolio_daily": pd.DataFrame(portfolio_rows),
        "security_holdings": pd.DataFrame(holding_rows),
        "security_daily": pd.DataFrame(security_rows),
        "benchmark_daily": pd.DataFrame(
            {
                "date": dates,
                "benchmark_return": market_returns,
            }
        ),
    }


@pytest.fixture(scope="module")
def monitoring_datasets(monitoring_inputs):
    return build_strategy_monitoring_datasets(**monitoring_inputs)


def test_frozen_implementation_validation_rejects_methodology_change():
    changed = selected_implementations_frame()
    changed.loc[0, "rebalance_frequency"] = 10

    with pytest.raises(ValueError, match="frozen research specification"):
        validate_frozen_implementations(changed)


def test_monitoring_workflow_builds_complete_dataset_set(monitoring_datasets):
    assert tuple(monitoring_datasets) == MONITORING_DATASET_NAMES
    assert len(monitoring_datasets["signal_health"]) == 260
    assert len(monitoring_datasets["factor_dependence"]) == 130
    assert len(monitoring_datasets["performance_risk"]) == 520
    assert len(monitoring_datasets["beta"]) == 390
    assert len(monitoring_datasets["concentration"]) == 390
    assert len(monitoring_datasets["implementation"]) == 390
    assert len(monitoring_datasets["liquidity_coverage"]) == 390
    assert len(monitoring_datasets["diagnostic_flags"]) == 43
    assert len(monitoring_datasets["latest_overview"]) == 5


def test_artifact_validation_rejects_duplicate_keys(monitoring_datasets):
    invalid = dict(monitoring_datasets)
    invalid["beta"] = pd.concat(
        [invalid["beta"], invalid["beta"].iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="beta contains duplicate keys"):
        validate_monitoring_artifacts(invalid)


def test_monitoring_artifacts_round_trip(monitoring_datasets, tmp_path):
    manifest = write_monitoring_artifacts(
        monitoring_datasets,
        tmp_path,
    )

    assert len(manifest) == len(MONITORING_DATASET_NAMES)
    assert manifest["read_back_passes"].all()

    for contract in MONITORING_ARTIFACT_CONTRACTS.values():
        assert (tmp_path / contract.filename).exists()
