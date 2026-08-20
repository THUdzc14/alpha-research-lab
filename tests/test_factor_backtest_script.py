from inspect import signature

import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest import BacktestConfig
from alpha_research.config.research import BACKTEST_RETURN_COLUMN
from scripts import run_factor_backtests as backtest_script


def make_factor_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=6)
    rows = []

    for date_number, current_date in enumerate(dates):
        for ticker_number in range(30):
            score = float(ticker_number)
            rows.append(
                {
                    "date": current_date,
                    "ticker": f"T{ticker_number:02d}",
                    "mom_12_1m_z": score,
                    "realised_vol_63_z": -score,
                    "forward_ret_1d": ((ticker_number - 14.5) / 10_000.0 + date_number / 100_000.0),
                }
            )

    return pd.DataFrame(rows)


def test_factor_backtest_defaults_and_filenames_match_the_frozen_contract():
    parameters = signature(backtest_script.build_factor_backtests).parameters

    assert backtest_script.FACTOR_BACKTEST_CONFIG == BacktestConfig()
    assert parameters["return_column"].default == BACKTEST_RETURN_COLUMN
    assert backtest_script.factor_filename_stem("12-1 Momentum") == "12_1_momentum"
    assert backtest_script.factor_filename_stem("Realised Volatility") == "realised_volatility"


def test_build_factor_backtests_isolates_factors_without_mutating_the_panel():
    panel = make_factor_panel()
    original = panel.copy(deep=True)

    daily_results, holdings_results, summary = backtest_script.build_factor_backtests(panel)

    assert list(daily_results) == list(backtest_script.FACTOR_COLUMNS)
    assert list(holdings_results) == list(backtest_script.FACTOR_COLUMNS)
    assert summary.index.tolist() == list(backtest_script.FACTOR_COLUMNS)
    assert summary.index.name == "factor"

    for daily in daily_results.values():
        assert tuple(daily.columns) == backtest_script.FACTOR_BACKTEST_DAILY_COLUMNS

    for holdings in holdings_results.values():
        assert tuple(holdings.columns) == backtest_script.FACTOR_BACKTEST_HOLDINGS_COLUMNS

    first_date = panel["date"].min()
    momentum_weights = (
        holdings_results["12-1 Momentum"]
        .loc[lambda data: data["date"].eq(first_date)]
        .set_index("ticker")["weight"]
        .sort_index()
    )
    volatility_weights = (
        holdings_results["Realised Volatility"]
        .loc[lambda data: data["date"].eq(first_date)]
        .set_index("ticker")["weight"]
        .sort_index()
    )

    assert np.allclose(momentum_weights, -volatility_weights)
    pd.testing.assert_frame_equal(panel, original)


def test_main_writes_the_established_backtest_artifacts(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "factor_panel.parquet"
    make_factor_panel().to_parquet(input_path, index=False)
    persisted_panel = pd.read_parquet(input_path)
    _, _, expected_summary = backtest_script.build_factor_backtests(persisted_panel)

    monkeypatch.setattr(backtest_script, "PROCESSED_DATA_DIR", tmp_path)

    backtest_script.main()

    expected_filenames = {
        "backtest_12_1_momentum_daily.parquet",
        "backtest_12_1_momentum_holdings.parquet",
        "backtest_realised_volatility_daily.parquet",
        "backtest_realised_volatility_holdings.parquet",
        "factor_backtest_summary.parquet",
    }
    assert {path.name for path in tmp_path.glob("*.parquet")} == {
        "factor_panel.parquet",
        *expected_filenames,
    }

    actual_summary = pd.read_parquet(tmp_path / "factor_backtest_summary.parquet")
    pd.testing.assert_frame_equal(actual_summary, expected_summary.reset_index())

    output = capsys.readouterr().out
    assert "12-1 Momentum" in output
    assert "Realised Volatility" in output
    assert "Combined summary" in output


@pytest.mark.parametrize(
    ("argv", "expected_exit_code"),
    [(["--help"], 0), (["--not-an-option"], 2)],
    ids=["help", "invalid-argument"],
)
def test_cli_inspection_exits_before_loading_data(
    argv,
    expected_exit_code,
    monkeypatch,
):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("CLI inspection attempted to load pipeline data.")

    monkeypatch.setattr(backtest_script, "load_parquet", unexpected_load)

    with pytest.raises(SystemExit) as error:
        backtest_script.main(argv)

    assert error.value.code == expected_exit_code
