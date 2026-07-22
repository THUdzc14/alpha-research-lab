from __future__ import annotations

import pandas as pd

from alpha_research.backtest import (
    BacktestConfig,
    run_long_short_backtest,
    summarise_backtest,
)
from alpha_research.config.paths import PROCESSED_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet

FACTOR_COLUMNS = {
    "12-1 Momentum": "mom_12_1m_z",
    "Realised Volatility": "realised_vol_63_z",
}


def main() -> None:
    input_path = PROCESSED_DATA_DIR / "factor_panel.parquet"

    panel = load_parquet(input_path)

    config = BacktestConfig(
        rebalance_frequency=5,
        quantiles=5,
        long_quantile=5,
        short_quantile=1,
        long_gross=1.0,
        short_gross=1.0,
        transaction_cost_bps=10.0,
        min_observations=30,
        rebalance_offset=0,
    )

    summary_rows = []

    for factor_name, factor_column in FACTOR_COLUMNS.items():
        daily, holdings = run_long_short_backtest(
            panel=panel,
            factor_column=factor_column,
            return_column="forward_ret_1d",
            config=config,
        )

        safe_name = factor_name.lower().replace(" ", "_").replace("-", "_")

        daily_path = PROCESSED_DATA_DIR / f"backtest_{safe_name}_daily.parquet"

        holdings_path = PROCESSED_DATA_DIR / f"backtest_{safe_name}_holdings.parquet"

        save_parquet(daily, daily_path)
        save_parquet(holdings, holdings_path)

        summary = summarise_backtest(daily).iloc[0].to_dict()
        summary["factor"] = factor_name

        summary_rows.append(summary)

        print(f"\n{factor_name}")
        print(summarise_backtest(daily).T)

    summary_table = pd.DataFrame(summary_rows).set_index("factor")

    summary_path = PROCESSED_DATA_DIR / "factor_backtest_summary.parquet"

    save_parquet(
        summary_table.reset_index(),
        summary_path,
    )

    print("\nCombined summary")
    print(summary_table)


if __name__ == "__main__":
    main()
