from __future__ import annotations

from alpha_research.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.factors import add_raw_factors
from alpha_research.signal_processing import (
    add_sector_neutral_factor,
    process_factor_columns,
)
from alpha_research.risk import calculate_rolling_stock_beta

FACTOR_MAP = {
    "mom_12_1m_raw": "mom_12_1m",
    "mom_3m_raw": "mom_3m",
    "reversal_1m_raw": "reversal_1m",
    "realised_vol_63_raw": "realised_vol_63",
}


def main() -> None:
    input_path = PROCESSED_DATA_DIR / "equity_panel.parquet"
    output_path = PROCESSED_DATA_DIR / "factor_panel.parquet"

    panel = load_parquet(input_path)

    factor_panel = add_raw_factors(panel)

    factor_panel = process_factor_columns(
        factor_panel,
        factor_map=FACTOR_MAP,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    # Add sector-neutral factors
    for factor_prefix in [
        "mom_12_1m",
        "mom_3m",
        "reversal_1m",
        "realised_vol_63",
    ]:
        factor_panel = add_sector_neutral_factor(
            factor_panel,
            factor_column=f"{factor_prefix}_winsorised",
            output_column=f"{factor_prefix}_sector_neutral_z",
            sector_column="sector",
            min_sector_observations=3,
        )

    # Add rolling beta to SPY
    spy = load_parquet(RAW_DATA_DIR / "spy_benchmark.parquet")

    if "ticker" in spy.columns:
        spy = spy.loc[spy["ticker"] == "SPY"].copy()

    stock_beta = calculate_rolling_stock_beta(
        factor_panel,
        benchmark=spy,
        stock_return_column="ret_1d",
        benchmark_price_column="adj_close",
        window=126,
        min_periods=63,
        output_column="beta_126",
    )

    factor_panel = factor_panel.merge(
        stock_beta,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    save_parquet(factor_panel, output_path)

    print(f"Saved factor panel: {output_path}")
    print(f"Rows: {len(factor_panel):,}")
    print(f"Tickers: {factor_panel['ticker'].nunique()}")
    print(f"Dates: {factor_panel['date'].nunique()}")

    print("\nFactor coverage:")
    for raw_column in FACTOR_MAP:
        coverage = factor_panel[raw_column].notna().mean()
        print(f"  {raw_column}: {coverage:.2%}")


if __name__ == "__main__":
    main()
