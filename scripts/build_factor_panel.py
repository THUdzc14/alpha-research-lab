from __future__ import annotations

from alpha_research.config.paths import PROCESSED_DATA_DIR
from alpha_research.data_loader import load_parquet, save_parquet
from alpha_research.factors import add_raw_factors
from alpha_research.signal_processing import process_factor_columns

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
