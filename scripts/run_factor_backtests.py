"""Run and persist the initial single-factor long-short backtests."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence

import pandas as pd

from alpha_research.backtest import (
    BacktestConfig,
    run_long_short_backtest,
    summarise_backtest,
)
from alpha_research.config.paths import PROCESSED_DATA_DIR
from alpha_research.config.research import BACKTEST_RETURN_COLUMN
from alpha_research.data_loader import load_parquet, save_parquet

FACTOR_COLUMNS = {
    "12-1 Momentum": "mom_12_1m_z",
    "Realised Volatility": "realised_vol_63_z",
}

FACTOR_BACKTEST_DAILY_COLUMNS = (
    "date",
    "is_rebalance",
    "long_return",
    "short_return",
    "gross_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "long_exposure",
    "short_exposure",
    "net_exposure",
    "gross_exposure",
    "missing_return_weight",
    "gross_cumulative_return",
    "net_cumulative_return",
)
FACTOR_BACKTEST_HOLDINGS_COLUMNS = ("date", "ticker", "weight")

FACTOR_BACKTEST_CONFIG = BacktestConfig()


def factor_filename_stem(factor_name: str) -> str:
    """Return the established filesystem-safe name for one factor."""
    return factor_name.lower().replace(" ", "_").replace("-", "_")


def build_factor_backtests(
    panel: pd.DataFrame,
    factor_columns: Mapping[str, str] = FACTOR_COLUMNS,
    config: BacktestConfig = FACTOR_BACKTEST_CONFIG,
    return_column: str = BACKTEST_RETURN_COLUMN,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Build all single-factor results and their combined summary in memory."""
    daily_results = {}
    holdings_results = {}
    summary_rows = []

    for factor_name, factor_column in factor_columns.items():
        daily, holdings = run_long_short_backtest(
            panel=panel,
            factor_column=factor_column,
            return_column=return_column,
            config=config,
        )
        daily = daily.loc[:, FACTOR_BACKTEST_DAILY_COLUMNS].copy()
        holdings = holdings.loc[:, FACTOR_BACKTEST_HOLDINGS_COLUMNS].copy()
        daily_results[factor_name] = daily
        holdings_results[factor_name] = holdings

        summary = summarise_backtest(daily).iloc[0].to_dict()
        summary["factor"] = factor_name
        summary_rows.append(summary)

    summary_table = pd.DataFrame(summary_rows).set_index("factor")

    return daily_results, holdings_results, summary_table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(() if argv is None else argv)

    input_path = PROCESSED_DATA_DIR / "factor_panel.parquet"
    panel = load_parquet(input_path)
    daily_results, holdings_results, summary_table = build_factor_backtests(panel)

    for factor_name in FACTOR_COLUMNS:
        daily = daily_results[factor_name]
        holdings = holdings_results[factor_name]
        safe_name = factor_filename_stem(factor_name)

        daily_path = PROCESSED_DATA_DIR / f"backtest_{safe_name}_daily.parquet"

        holdings_path = PROCESSED_DATA_DIR / f"backtest_{safe_name}_holdings.parquet"

        save_parquet(daily, daily_path)
        save_parquet(holdings, holdings_path)

        print(f"\n{factor_name}")
        print(summarise_backtest(daily).T)

    summary_path = PROCESSED_DATA_DIR / "factor_backtest_summary.parquet"

    save_parquet(
        summary_table.reset_index(),
        summary_path,
    )

    print("\nCombined summary")
    print(summary_table)


if __name__ == "__main__":
    main(sys.argv[1:])
