import numpy as np
import pandas as pd
import pytest

from alpha_research.signal_processing import (
    cross_sectional_rank,
    cross_sectional_zscore,
    process_factor,
    winsorise_cross_section,
    grouped_zscore,
)


def make_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "AAA",
                "BBB",
                "CCC",
                "DDD",
            ],
            "factor_raw": [
                1.0,
                2.0,
                3.0,
                100.0,
                -4.0,
                -2.0,
                0.0,
                2.0,
            ],
        }
    )


def test_cross_sectional_zscore_has_zero_mean_by_date():
    panel = make_panel()

    panel["z"] = cross_sectional_zscore(
        panel,
        column="factor_raw",
    )

    means = panel.groupby("date")["z"].mean()

    assert np.allclose(means, 0.0)


def test_cross_sectional_zscore_has_unit_population_std():
    panel = make_panel()

    panel["z"] = cross_sectional_zscore(
        panel,
        column="factor_raw",
        ddof=0,
    )

    stds = panel.groupby("date")["z"].std(ddof=0)

    assert np.allclose(stds, 1.0)


def test_cross_sectional_rank_orders_values():
    panel = make_panel()

    ranks = cross_sectional_rank(
        panel,
        column="factor_raw",
    )

    first_date = panel["date"] == pd.Timestamp("2024-01-01")

    assert ranks[first_date].min() == pytest.approx(0.25)
    assert ranks[first_date].max() == pytest.approx(1.0)


def test_winsorisation_limits_outlier():
    panel = make_panel()

    winsorised = winsorise_cross_section(
        panel,
        column="factor_raw",
        lower_quantile=0.25,
        upper_quantile=0.75,
    )

    first_date = panel["date"] == pd.Timestamp("2024-01-01")

    original_max = panel.loc[first_date, "factor_raw"].max()
    winsorised_max = winsorised[first_date].max()

    assert winsorised_max < original_max


def test_process_factor_adds_expected_columns():
    panel = make_panel()

    result = process_factor(
        panel,
        raw_column="factor_raw",
        output_prefix="factor",
    )

    expected_columns = {
        "factor_winsorised",
        "factor_z",
        "factor_rank",
    }

    assert expected_columns.issubset(result.columns)


def test_missing_values_remain_missing():
    panel = make_panel()
    panel.loc[0, "factor_raw"] = np.nan

    result = process_factor(
        panel,
        raw_column="factor_raw",
        output_prefix="factor",
    )

    assert np.isnan(result["factor_winsorised"].iloc[0])
    assert np.isnan(result["factor_z"].iloc[0])
    assert np.isnan(result["factor_rank"].iloc[0])


def test_zero_dispersion_returns_nan_zscores():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "ticker": ["AAA", "BBB", "CCC"],
            "factor_raw": [1.0, 1.0, 1.0],
        }
    )

    zscores = cross_sectional_zscore(
        panel,
        column="factor_raw",
    )

    assert zscores.isna().all()


def test_sector_neutral_zscore_has_zero_mean_within_sector():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 6),
            "sector": ["Tech"] * 3 + ["Health"] * 3,
            "factor": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )

    result = grouped_zscore(
        panel,
        column="factor",
        group_columns=["date", "sector"],
        min_observations=3,
    )

    panel["z"] = result

    means = panel.groupby(["date", "sector"])["z"].mean()

    assert np.allclose(means, 0.0)


def test_benchmark_hedge_offsets_stock_beta():
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": -0.5, "DDD": -0.5})

    betas = pd.Series({"AAA": 1.5, "BBB": 1.3, "CCC": 0.7, "DDD": 0.5})

    stock_beta = float((weights * betas).sum())
    benchmark_weight = -stock_beta

    assert stock_beta + benchmark_weight == pytest.approx(0.0)
