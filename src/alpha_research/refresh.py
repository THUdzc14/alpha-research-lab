"""End-to-end, in-memory refresh for frozen strategy outputs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from alpha_research.backtest import run_target_weight_backtest
from alpha_research.config.research import (
    BACKTEST_RETURN_COLUMN,
    DEFAULT_NUMERICAL_TOLERANCE,
    STRATEGY_EVALUATION_START_DATE,
    selected_implementations_frame,
)
from alpha_research.risk import prepare_benchmark_returns
from alpha_research.workflows import (
    build_frozen_strategy_target_weights,
    build_selected_attribution_datasets,
    build_strategy_monitoring_datasets,
)

REFRESH_DATASET_GROUPS = (
    "attribution",
    "monitoring",
)


def key_values_match(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    key_columns: Sequence[str],
) -> bool:
    """Compare key values while allowing equivalent storage dtypes."""
    columns = list(key_columns)

    try:
        pd.testing.assert_frame_equal(
            reference[columns],
            reconstructed[columns],
            check_dtype=False,
            check_categorical=False,
            check_exact=True,
        )
    except AssertionError:
        return False

    return True


def _prepare_refresh_benchmark(
    benchmark_prices: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:
    """Prepare one forward-return benchmark series for refresh alignment."""
    required_columns = {"date", "adj_close"}
    missing_columns = required_columns - set(benchmark_prices.columns)

    if missing_columns:
        raise KeyError(f"benchmark_prices is missing columns: {sorted(missing_columns)}")

    prepared = benchmark_prices.copy()

    if "ticker" in prepared.columns:
        available_tickers = set(prepared["ticker"].dropna().astype(str))

        if benchmark_name in available_tickers:
            prepared = prepared.loc[prepared["ticker"].astype(str).eq(benchmark_name)].copy()
        elif len(available_tickers) != 1:
            raise ValueError(
                f"benchmark_prices contains multiple tickers and does not contain {benchmark_name}."
            )

    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")

    if prepared["date"].isna().any():
        raise ValueError("benchmark_prices contains missing dates.")

    if prepared["date"].duplicated().any():
        raise ValueError("benchmark_prices contains duplicate dates.")

    benchmark_returns = prepare_benchmark_returns(prepared)

    return (
        benchmark_returns.dropna(subset=["benchmark_return"])
        .assign(benchmark=benchmark_name)[["date", "benchmark", "benchmark_return"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def _derive_analysis_dates(
    factor_panel: pd.DataFrame,
    target_weights_by_portfolio: dict[str, pd.DataFrame],
    benchmark_daily: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Return common active dates without moving the frozen evaluation start."""
    return_panel = factor_panel[["date", "ticker", BACKTEST_RETURN_COLUMN]].copy()
    common_dates = pd.DatetimeIndex(benchmark_daily["date"].unique()).sort_values()
    active_start_dates = []

    for portfolio, targets in target_weights_by_portfolio.items():
        daily, _ = run_target_weight_backtest(
            return_panel,
            targets,
            return_column=BACKTEST_RETURN_COLUMN,
            transaction_cost_bps=0.0,
        )
        daily["date"] = pd.to_datetime(daily["date"], errors="raise")
        active_daily = daily.loc[daily["gross_exposure"].gt(DEFAULT_NUMERICAL_TOLERANCE)]

        if active_daily.empty:
            raise ValueError(f"{portfolio} never becomes economically active.")

        active_start_dates.append(active_daily["date"].min())
        common_dates = common_dates.intersection(pd.DatetimeIndex(daily["date"]))

    # Newly refreshed data may extend the end date but must not move the
    # evaluation start frozen by Notebook 04.
    common_start = max(
        max(active_start_dates),
        STRATEGY_EVALUATION_START_DATE,
    )
    analysis_dates = common_dates[common_dates >= common_start].sort_values()

    if analysis_dates.empty:
        raise ValueError(
            "No common dates remain after aligning active strategies and the benchmark."
        )

    return analysis_dates


def build_complete_research_refresh(
    factor_panel: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    benchmark_name: str = "SPY",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Rebuild frozen attribution and monitoring datasets in memory."""
    selected_implementations = selected_implementations_frame()
    target_weights_by_portfolio = build_frozen_strategy_target_weights(factor_panel)
    complete_benchmark = _prepare_refresh_benchmark(
        benchmark_prices,
        benchmark_name=benchmark_name,
    )
    analysis_dates = _derive_analysis_dates(
        factor_panel,
        target_weights_by_portfolio,
        complete_benchmark,
    )
    benchmark_daily = complete_benchmark.loc[complete_benchmark["date"].isin(analysis_dates)].copy()
    return_panel = factor_panel[["date", "ticker", BACKTEST_RETURN_COLUMN]].copy()
    attribution = build_selected_attribution_datasets(
        return_panel=return_panel,
        target_weights_by_portfolio=target_weights_by_portfolio,
        benchmark_daily=benchmark_daily,
        analysis_dates=analysis_dates,
        selected_implementations=selected_implementations,
    )
    monitoring = build_strategy_monitoring_datasets(
        factor_panel=factor_panel,
        selected_implementations=attribution["selected_implementations"],
        portfolio_daily=attribution["portfolio_daily"],
        security_holdings=attribution["security_holdings"],
        security_daily=attribution["security_daily"],
        benchmark_daily=attribution["benchmark_daily"],
    )
    result = {
        "attribution": attribution,
        "monitoring": monitoring,
    }

    if tuple(result) != REFRESH_DATASET_GROUPS:
        raise RuntimeError("Research refresh returned invalid dataset groups.")

    return result
