import pandas as pd
import pytest

from alpha_research.dashboard_analytics import (
    FACTOR_DEPENDENCE_HISTORY_COLUMNS,
    LATEST_FACTOR_SNAPSHOT_COLUMNS,
    SIGNAL_HEALTH_HISTORY_COLUMNS,
    build_latest_factor_snapshot,
    prepare_factor_dependence_history,
    prepare_signal_health_history,
)


@pytest.fixture()
def signal_health():
    dates = pd.bdate_range("2026-06-29", periods=4)
    rows = []

    for factor_number, factor in enumerate(("Momentum", "Realised Volatility")):
        for date_number, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "signal_coverage": 0.95 + factor_number * 0.01,
                    "raw_iqr": 1.0 + factor_number + date_number * 0.10,
                    "ic": 0.01 + factor_number * 0.01 + date_number * 0.001,
                    "rolling_mean_ic_252": 0.02 + factor_number * 0.01,
                    "rank_stability_1d": 0.98 - factor_number * 0.01,
                    "rank_stability_21d": 0.75 - factor_number * 0.02,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture()
def factor_dependence():
    dates = pd.bdate_range("2026-06-29", periods=4)

    return pd.DataFrame(
        {
            "date": dates[::-1],
            "factor_rank_correlation": [0.40, 0.30, 0.20, 0.10],
            "observations": [100, 100, 100, 100],
            "rolling_factor_rank_correlation_252": [0.25, 0.24, 0.23, 0.22],
        }
    )


@pytest.fixture()
def latest_overview():
    return pd.DataFrame(
        {
            "entity_type": ["Factor", "Factor", "Portfolio"],
            "entity": ["Momentum", "Realised Volatility", "Composite Score"],
            "overall_status": ["PASS", "WARNING", "WARNING"],
            "signal_status": ["PASS", "WARNING", "N/A"],
        }
    )


def test_prepare_signal_health_history_filters_and_preserves_factor_order(
    signal_health,
):
    start_date = pd.Timestamp("2026-06-30")
    end_date = pd.Timestamp("2026-07-01")
    history = prepare_signal_health_history(
        signal_health,
        factors=["Realised Volatility", "Momentum"],
        start_date=start_date,
        end_date=end_date,
    )

    assert tuple(history.columns) == SIGNAL_HEALTH_HISTORY_COLUMNS
    assert list(pd.unique(history["factor"])) == ["Realised Volatility", "Momentum"]
    assert history["date"].min() == start_date
    assert history["date"].max() == end_date
    assert history.groupby("factor")["date"].size().eq(2).all()


def test_prepare_factor_dependence_history_filters_and_sorts(factor_dependence):
    history = prepare_factor_dependence_history(
        factor_dependence,
        end_date="2026-07-01",
    )

    assert tuple(history.columns) == FACTOR_DEPENDENCE_HISTORY_COLUMNS
    assert history["date"].is_monotonic_increasing
    assert history["date"].max() == pd.Timestamp("2026-07-01")
    assert len(history) == 3


def test_latest_factor_snapshot_combines_status_and_latest_metrics(
    signal_health,
    latest_overview,
):
    snapshot = build_latest_factor_snapshot(
        latest_overview,
        signal_health,
        factors=["Realised Volatility", "Momentum"],
    )

    assert tuple(snapshot.columns) == LATEST_FACTOR_SNAPSHOT_COLUMNS
    assert snapshot["factor"].tolist() == ["Realised Volatility", "Momentum"]
    assert snapshot["latest_date"].eq(pd.Timestamp("2026-07-02")).all()
    assert snapshot["overall_status"].tolist() == ["WARNING", "PASS"]
    assert snapshot["signal_status"].tolist() == ["WARNING", "PASS"]
    assert snapshot.loc[0, "ic"] == pytest.approx(0.023)
    assert snapshot["ic_as_of_date"].eq(pd.Timestamp("2026-07-02")).all()
    assert snapshot["rolling_mean_ic_252_as_of_date"].eq(pd.Timestamp("2026-07-02")).all()


def test_latest_factor_snapshot_dates_trailing_predictive_metrics(
    signal_health,
    latest_overview,
):
    latest_date = signal_health["date"].max()
    trailing = signal_health.copy()

    trailing.loc[trailing["date"].eq(latest_date), "ic"] = float("nan")
    trailing.loc[
        trailing["date"].ge(pd.Timestamp("2026-07-01")),
        "rolling_mean_ic_252",
    ] = float("nan")

    snapshot = build_latest_factor_snapshot(
        latest_overview,
        trailing,
    )
    momentum = snapshot.set_index("factor").loc["Momentum"]

    assert momentum["latest_date"] == latest_date

    assert momentum["ic_as_of_date"] == pd.Timestamp("2026-07-01")
    assert momentum["ic"] == pytest.approx(0.012)

    assert momentum["rolling_mean_ic_252_as_of_date"] == pd.Timestamp("2026-06-30")
    assert momentum["rolling_mean_ic_252"] == pytest.approx(0.02)


def test_latest_factor_snapshot_marks_unavailable_predictive_metric(
    signal_health,
    latest_overview,
):
    unavailable = signal_health.copy()
    unavailable["ic"] = float("nan")

    snapshot = build_latest_factor_snapshot(latest_overview, unavailable)

    assert snapshot["latest_date"].eq(signal_health["date"].max()).all()
    assert snapshot["ic_as_of_date"].isna().all()
    assert snapshot["ic"].isna().all()
    assert snapshot["rolling_mean_ic_252_as_of_date"].notna().all()


@pytest.mark.parametrize(
    ("factors", "error_type", "message"),
    (
        ("Momentum", TypeError, "sequence of factor names"),
        ([], ValueError, "must not be empty"),
        (["Momentum", "Momentum"], ValueError, "unique names"),
        (["Missing"], ValueError, "missing factors"),
    ),
)
def test_prepare_signal_health_history_rejects_invalid_factor_selections(
    signal_health,
    factors,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        prepare_signal_health_history(signal_health, factors=factors)


def test_signal_analytics_reject_malformed_inputs(
    signal_health,
    factor_dependence,
    latest_overview,
):
    duplicated_signals = pd.concat(
        [signal_health, signal_health.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate factor-date"):
        prepare_signal_health_history(duplicated_signals)

    duplicated_dependence = pd.concat(
        [factor_dependence, factor_dependence.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        prepare_factor_dependence_history(duplicated_dependence)

    with pytest.raises(ValueError, match="No factor-dependence observations"):
        prepare_factor_dependence_history(
            factor_dependence,
            start_date="2027-01-01",
        )

    incomplete_overview = latest_overview.loc[latest_overview["entity"].ne("Momentum")]

    with pytest.raises(ValueError, match="missing factors"):
        build_latest_factor_snapshot(incomplete_overview, signal_health)
