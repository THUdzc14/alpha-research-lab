import numpy as np
import pandas as pd
import pytest

from alpha_research.validation import (
    calculate_daily_signal_state,
    calculate_factor_dependence,
    calculate_factor_signal_health,
    calculate_rank_stability,
    calculate_signal_health,
)


def make_signal_monitoring_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=6)
    score_patterns = [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    rows = []

    for date, scores in zip(dates, score_patterns, strict=True):
        for ticker_number, score_a in enumerate(scores):
            rows.append(
                {
                    "date": date,
                    "ticker": f"S{ticker_number}",
                    "score_a": score_a,
                    "raw_a": score_a * 10.0,
                    "score_b": 5.0 - score_a,
                    "raw_b": (5.0 - score_a) * 10.0,
                    "forward_return": score_a / 100.0,
                }
            )

    return pd.DataFrame(rows)


def test_daily_signal_state_calculates_coverage_and_dispersion():
    panel = make_signal_monitoring_panel()
    first_date = panel["date"].min()
    panel.loc[
        panel["date"].eq(first_date) & panel["ticker"].eq("S0"),
        "score_a",
    ] = np.nan

    result = calculate_daily_signal_state(
        panel,
        factor_name="Factor A",
        score_column="score_a",
        raw_column="raw_a",
        forward_return_column="forward_return",
    )
    first_row = result.iloc[0]

    assert first_row["universe_observations"] == 4
    assert first_row["signal_observations"] == 3
    assert first_row["evaluation_observations"] == 3
    assert first_row["signal_coverage"] == pytest.approx(0.75)
    assert first_row["evaluation_coverage"] == pytest.approx(0.75)
    assert first_row["raw_median"] == pytest.approx(25.0)
    assert first_row["raw_iqr"] == pytest.approx(15.0)
    assert first_row["factor"] == "Factor A"


def test_rank_stability_uses_prior_trading_observations():
    result = calculate_rank_stability(
        make_signal_monitoring_panel(),
        factor_name="Factor A",
        score_column="score_a",
        lags=(1, 2),
        min_observations=3,
    )

    assert np.isnan(result.loc[0, "rank_stability_1d"])
    assert result.loc[1, "rank_stability_1d"] == pytest.approx(1.0)
    assert result.loc[2, "rank_stability_1d"] == pytest.approx(-1.0)
    assert result.loc[3, "rank_stability_1d"] == pytest.approx(1.0)
    assert result.loc[2, "rank_stability_2d"] == pytest.approx(-1.0)
    assert result.loc[2, "rank_stability_2d_observations"] == 4


def test_rank_stability_requires_sufficient_overlap():
    panel = make_signal_monitoring_panel()
    second_date = np.sort(panel["date"].unique())[1]
    panel.loc[
        panel["date"].eq(second_date) & panel["ticker"].isin(["S0", "S1"]),
        "score_a",
    ] = np.nan

    result = calculate_rank_stability(
        panel,
        factor_name="Factor A",
        score_column="score_a",
        lags=(1,),
        min_observations=3,
    )

    assert result.loc[1, "rank_stability_1d_observations"] == 2
    assert np.isnan(result.loc[1, "rank_stability_1d"])


def test_factor_dependence_calculates_daily_and_rolling_rank_correlation():
    result = calculate_factor_dependence(
        make_signal_monitoring_panel(),
        factor_columns=("score_a", "score_b"),
        min_observations=3,
        window=2,
        rolling_min_periods=2,
    )

    assert np.allclose(result["factor_rank_correlation"], -1.0)
    assert np.isnan(result.loc[0, "rolling_factor_rank_correlation_2"])
    assert result.loc[1, "rolling_factor_rank_correlation_2"] == pytest.approx(-1.0)


def test_factor_dependence_reports_insufficient_cross_section():
    panel = make_signal_monitoring_panel()
    first_date = panel["date"].min()
    panel.loc[
        panel["date"].eq(first_date) & panel["ticker"].isin(["S0", "S1"]),
        "score_b",
    ] = np.nan

    result = calculate_factor_dependence(
        panel,
        factor_columns=("score_a", "score_b"),
        min_observations=3,
        window=2,
        rolling_min_periods=1,
    )

    assert result.loc[0, "observations"] == 2
    assert np.isnan(result.loc[0, "factor_rank_correlation"])


def test_factor_signal_health_builds_complete_daily_history():
    result = calculate_factor_signal_health(
        make_signal_monitoring_panel(),
        factor_name="Factor A",
        score_column="score_a",
        raw_column="raw_a",
        forward_return_column="forward_return",
        min_observations=3,
        window=3,
        rolling_min_periods=2,
        stability_lags=(1, 2),
    )

    expected_columns = {
        "date",
        "factor",
        "signal_coverage",
        "evaluation_coverage",
        "raw_iqr",
        "ic",
        "ic_observations",
        "rolling_mean_ic_3",
        "rolling_positive_ic_fraction_3",
        "rolling_ic_observations_3",
        "rank_stability_1d",
        "rank_stability_2d",
        "ic_observation_difference",
        "trailing_raw_iqr_median_3",
        "raw_iqr_relative_to_trailing_median",
    }

    assert len(result) == 6
    assert expected_columns.issubset(result.columns)
    assert result["ic_observation_difference"].eq(0).all()
    assert np.allclose(result["ic"], 1.0)
    assert np.isnan(result.loc[0, "rolling_mean_ic_3"])
    assert result.loc[1, "rolling_mean_ic_3"] == pytest.approx(1.0)
    assert result.loc[1, "rolling_positive_ic_fraction_3"] == pytest.approx(1.0)
    assert result.loc[1, "rolling_ic_observations_3"] == 2


def test_signal_health_rolling_ic_uses_valid_ic_observations():
    panel = make_signal_monitoring_panel()
    third_date = np.sort(panel["date"].unique())[2]
    panel.loc[
        panel["date"].eq(third_date) & panel["ticker"].isin(["S0", "S1"]),
        "forward_return",
    ] = np.nan

    result = calculate_factor_signal_health(
        panel,
        factor_name="Factor A",
        score_column="score_a",
        raw_column="raw_a",
        forward_return_column="forward_return",
        min_observations=3,
        window=2,
        rolling_min_periods=2,
        stability_lags=(1,),
    )

    assert np.isnan(result.loc[2, "ic"])
    assert np.isnan(result.loc[2, "rolling_mean_ic_2"])
    assert result.loc[3, "rolling_mean_ic_2"] == pytest.approx(1.0)
    assert result.loc[3, "rolling_ic_observations_2"] == 2


def test_trailing_dispersion_baseline_excludes_current_date():
    panel = make_signal_monitoring_panel()

    baseline = calculate_factor_signal_health(
        panel,
        factor_name="Factor A",
        score_column="score_a",
        raw_column="raw_a",
        forward_return_column="forward_return",
        min_observations=3,
        window=3,
        rolling_min_periods=2,
        stability_lags=(1,),
    )

    changed_panel = panel.copy()
    last_date = changed_panel["date"].max()
    changed_panel.loc[changed_panel["date"].eq(last_date), "raw_a"] *= 100.0

    changed = calculate_factor_signal_health(
        changed_panel,
        factor_name="Factor A",
        score_column="score_a",
        raw_column="raw_a",
        forward_return_column="forward_return",
        min_observations=3,
        window=3,
        rolling_min_periods=2,
        stability_lags=(1,),
    )

    assert changed.loc[5, "trailing_raw_iqr_median_3"] == pytest.approx(
        baseline.loc[5, "trailing_raw_iqr_median_3"]
    )
    assert changed.loc[5, "raw_iqr"] != pytest.approx(baseline.loc[5, "raw_iqr"])


def test_calculate_signal_health_handles_named_factor_set():
    result = calculate_signal_health(
        make_signal_monitoring_panel(),
        factor_columns={
            "Factor A": "score_a",
            "Factor B": "score_b",
        },
        raw_factor_columns={
            "Factor A": "raw_a",
            "Factor B": "raw_b",
        },
        forward_return_column="forward_return",
        min_observations=3,
        window=3,
        rolling_min_periods=2,
        stability_lags=(1, 2),
    )

    assert len(result) == 12
    assert result[["factor", "date"]].duplicated().sum() == 0
    assert set(result["factor"]) == {"Factor A", "Factor B"}


def test_signal_monitoring_rejects_duplicate_keys():
    panel = make_signal_monitoring_panel()
    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate date/ticker"):
        calculate_daily_signal_state(
            duplicated,
            factor_name="Factor A",
            score_column="score_a",
            raw_column="raw_a",
            forward_return_column="forward_return",
        )
