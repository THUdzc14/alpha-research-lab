import pandas as pd
import pytest

from alpha_research.artifacts import (
    validate_attribution_artifacts,
    write_attribution_artifacts,
)
from alpha_research.attribution import reconcile_security_attribution
from alpha_research.config.research import selected_implementations_frame
from alpha_research.workflows import (
    ATTRIBUTION_DATASET_NAMES,
    build_selected_attribution_datasets,
)


@pytest.fixture()
def attribution_workflow_inputs():
    dates = pd.bdate_range("2025-01-02", periods=4)
    return_panel = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "forward_ret_1d": asset_return,
            }
            for date, returns in zip(
                dates,
                (
                    (0.01, -0.01),
                    (0.02, 0.01),
                    (-0.01, 0.02),
                    (0.00, -0.02),
                ),
                strict=True,
            )
            for ticker, asset_return in zip(
                ("AAA", "BBB"),
                returns,
                strict=True,
            )
        ]
    )
    target_template = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[2], dates[2]],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [1.0, -1.0, 1.0, -1.0],
        }
    )
    implementations = selected_implementations_frame()
    targets = {
        portfolio: target_template.copy() for portfolio in implementations["portfolio"]
    }
    benchmark_daily = pd.DataFrame(
        {
            "date": dates,
            "benchmark": "SPY",
            "benchmark_return": [0.005, 0.01, -0.005, 0.0],
        }
    )

    return {
        "return_panel": return_panel,
        "target_weights_by_portfolio": targets,
        "benchmark_daily": benchmark_daily,
        "analysis_dates": dates,
        "selected_implementations": implementations,
    }


@pytest.fixture()
def attribution_datasets(attribution_workflow_inputs):
    return build_selected_attribution_datasets(**attribution_workflow_inputs)


def test_attribution_workflow_builds_the_complete_handoff(
    attribution_datasets,
):
    assert tuple(attribution_datasets) == ATTRIBUTION_DATASET_NAMES
    assert len(attribution_datasets["selected_implementations"]) == 3
    assert len(attribution_datasets["portfolio_daily"]) == 12
    assert len(attribution_datasets["benchmark_daily"]) == 4

    audit = reconcile_security_attribution(
        attribution_datasets["portfolio_daily"],
        attribution_datasets["security_daily"],
    )

    assert audit["audit_passes"].all()


def test_attribution_workflow_recomputes_cumulative_returns(
    attribution_datasets,
):
    portfolio_daily = attribution_datasets["portfolio_daily"]

    for _, portfolio_data in portfolio_daily.groupby("portfolio"):
        portfolio_data = portfolio_data.sort_values("date")
        expected_net = (1.0 + portfolio_data["net_return"]).cumprod()

        assert (
            portfolio_data["net_cumulative_return"]
            .reset_index(drop=True)
            .equals(expected_net.reset_index(drop=True))
        )


def test_attribution_workflow_reanchors_cumulative_returns(
    attribution_workflow_inputs,
):
    inputs = dict(attribution_workflow_inputs)

    full_analysis_dates = pd.DatetimeIndex(inputs["analysis_dates"])

    # Exclude the first backtest date so that the test
    # detects cumulative wealth carried in from history
    # before the exported analysis window.
    inputs["analysis_dates"] = full_analysis_dates[1:]

    datasets = build_selected_attribution_datasets(**inputs)

    portfolio_daily = datasets["portfolio_daily"]

    for _, portfolio_data in portfolio_daily.groupby("portfolio"):
        portfolio_data = portfolio_data.sort_values("date").reset_index(drop=True)

        first_day = portfolio_data.iloc[0]

        assert first_day["gross_cumulative_return"] == pytest.approx(
            1.0 + first_day["gross_return"]
        )

        assert first_day["net_cumulative_return"] == pytest.approx(
            1.0 + first_day["net_return"]
        )


def test_attribution_workflow_requires_every_frozen_target_set(
    attribution_workflow_inputs,
):
    incomplete_inputs = dict(attribution_workflow_inputs)
    incomplete_targets = dict(incomplete_inputs["target_weights_by_portfolio"])
    incomplete_targets.pop("Composite Score")
    incomplete_inputs["target_weights_by_portfolio"] = incomplete_targets

    with pytest.raises(ValueError, match="frozen portfolio set"):
        build_selected_attribution_datasets(**incomplete_inputs)


def test_attribution_artifacts_reject_stale_cumulative_return(
    attribution_datasets,
):
    invalid = dict(attribution_datasets)
    invalid["portfolio_daily"] = invalid["portfolio_daily"].copy()
    invalid["portfolio_daily"].loc[
        invalid["portfolio_daily"]["portfolio"].eq("Composite Score"),
        "net_cumulative_return",
    ] = invalid["portfolio_daily"].loc[
        invalid["portfolio_daily"]["portfolio"].eq("Composite Score"),
        "gross_cumulative_return",
    ]

    with pytest.raises(ValueError, match="cumulative returns"):
        validate_attribution_artifacts(invalid)


def test_attribution_artifacts_round_trip(
    attribution_datasets,
    tmp_path,
):
    manifest = write_attribution_artifacts(
        attribution_datasets,
        tmp_path,
    )

    assert tuple(manifest["dataset"]) == ATTRIBUTION_DATASET_NAMES
    assert manifest["read_back_passes"].all()

    for filename in manifest["file"]:
        assert (tmp_path / filename).exists()
