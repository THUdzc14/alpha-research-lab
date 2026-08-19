"""Build raw, processed, neutralised, and risk-model factor features."""

from __future__ import annotations

import pandas as pd

from alpha_research.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from alpha_research.config.research import (
    FACTOR_WINSOR_LOWER_QUANTILE,
    FACTOR_WINSOR_UPPER_QUANTILE,
    MONITORING_SPECIFICATION,
    REALISED_VOLATILITY_WINDOW,
    TRADING_DAYS_PER_YEAR,
)
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.factors import add_raw_factors
from alpha_research.risk import (
    calculate_rolling_market_model,
    calculate_rolling_stock_beta,
)
from alpha_research.signal_processing import (
    add_sector_neutral_factor,
    process_factor_columns,
)

FACTOR_MAP = {
    "mom_12_1m_raw": "mom_12_1m",
    "mom_3m_raw": "mom_3m",
    "reversal_1m_raw": "reversal_1m",
    "realised_vol_63_raw": "realised_vol_63",
    "idio_vol_63_raw": "idio_vol_63",
}

MARKET_MODEL_WINDOW = REALISED_VOLATILITY_WINDOW
MARKET_MODEL_MIN_PERIODS = REALISED_VOLATILITY_WINDOW
MARKET_MODEL_PREFIX = f"market_model_{MARKET_MODEL_WINDOW}"
SECTOR_NEUTRAL_MINIMUM_OBSERVATIONS = 3
STOCK_BETA_WINDOW = MONITORING_SPECIFICATION.risk_window
STOCK_BETA_MIN_PERIODS = REALISED_VOLATILITY_WINDOW
STOCK_BETA_COLUMN = f"beta_{STOCK_BETA_WINDOW}"


def build_factor_panel(panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Construct the complete frozen factor panel without writing artifacts."""
    spy = benchmark

    if "ticker" in benchmark.columns:
        spy = benchmark.loc[benchmark["ticker"] == "SPY"].copy()

    # Build factors that depend only on the equity panel.
    factor_panel = add_raw_factors(panel)

    # Estimate the trailing stock-SPY model through each observation date.
    factor_panel = calculate_rolling_market_model(
        factor_panel,
        benchmark=spy,
        window=MARKET_MODEL_WINDOW,
        min_periods=MARKET_MODEL_MIN_PERIODS,
        annualisation_factor=TRADING_DAYS_PER_YEAR,
        output_prefix=MARKET_MODEL_PREFIX,
    )

    # Publish only idiosyncratic volatility as a raw factor.
    factor_panel = factor_panel.rename(
        columns={
            f"{MARKET_MODEL_PREFIX}_idio_vol": "idio_vol_63_raw",
        }
    ).drop(
        columns=[
            f"{MARKET_MODEL_PREFIX}_alpha",
            f"{MARKET_MODEL_PREFIX}_beta",
            f"{MARKET_MODEL_PREFIX}_residual",
        ]
    )

    # Add winsorised, z-score, and rank variants.
    factor_panel = process_factor_columns(
        factor_panel,
        factor_map=FACTOR_MAP,
        lower_quantile=FACTOR_WINSOR_LOWER_QUANTILE,
        upper_quantile=FACTOR_WINSOR_UPPER_QUANTILE,
    )

    # Add sector-neutral variants.
    for factor_prefix in FACTOR_MAP.values():
        factor_panel = add_sector_neutral_factor(
            factor_panel,
            factor_column=f"{factor_prefix}_winsorised",
            output_column=f"{factor_prefix}_sector_neutral_z",
            sector_column="sector",
            min_sector_observations=SECTOR_NEUTRAL_MINIMUM_OBSERVATIONS,
        )

    # Retain the separate 126-day beta used for portfolio risk analysis.
    stock_beta = calculate_rolling_stock_beta(
        factor_panel,
        benchmark=spy,
        stock_return_column="ret_1d",
        benchmark_price_column="adj_close",
        window=STOCK_BETA_WINDOW,
        min_periods=STOCK_BETA_MIN_PERIODS,
        output_column=STOCK_BETA_COLUMN,
    )

    factor_panel = factor_panel.merge(
        stock_beta,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    return factor_panel


def main() -> None:
    input_path = PROCESSED_DATA_DIR / "equity_panel.parquet"
    output_path = PROCESSED_DATA_DIR / "factor_panel.parquet"
    panel = load_parquet(input_path)
    benchmark = load_parquet(RAW_DATA_DIR / "spy_benchmark.parquet")
    factor_panel = build_factor_panel(panel, benchmark)

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
