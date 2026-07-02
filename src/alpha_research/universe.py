from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

SP100_WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"


def _yfinance_symbol(symbol: str) -> str:
    """Convert symbols to yfinance format.

    Example:
        BRK.B -> BRK-B
    """
    return symbol.replace(".", "-")


def _read_html_with_user_agent(url: str) -> list[pd.DataFrame]:
    """Read HTML tables using a browser-like user agent.

    This avoids occasional HTTP 403 errors from sites that block Python's
    default urllib user agent.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return pd.read_html(StringIO(response.text))


def load_sp100_universe() -> pd.DataFrame:
    """Load current S&P 100 constituents from Wikipedia.

    Note:
        This creates a current-constituent universe and therefore introduces
        survivorship bias when used for historical backtests.
    """
    tables = _read_html_with_user_agent(SP100_WIKI_URL)
    components = None

    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if {"symbol", "name", "sector"}.issubset(cols):
            components = table.copy()
            break

    if components is None:
        raise ValueError("Could not find S&P 100 components table.")

    components.columns = [str(c).strip().lower() for c in components.columns]
    components = components.rename(columns={"symbol": "ticker"})

    components["ticker"] = components["ticker"].astype(str).str.strip()
    components["yf_ticker"] = components["ticker"].map(_yfinance_symbol)

    components = components[["ticker", "yf_ticker", "name", "sector"]]
    components = components.sort_values("ticker").reset_index(drop=True)

    return components


def get_universe(name: str) -> pd.DataFrame:
    name = name.lower()

    if name == "sp100":
        return load_sp100_universe()

    raise ValueError(f"Unsupported universe: {name!r}. Available: 'sp100'.")
