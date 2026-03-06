"""
fetch_data.py – Fetch live and historical DSE data using the bdshare library.
"""

import pandas as pd

try:
    import bdshare
except ImportError:
    bdshare = None  # allow import without bdshare installed


def fetch_live_data() -> pd.DataFrame:
    """Fetch current trade data for all DSE stocks.

    Returns
    -------
    pd.DataFrame
        DataFrame with live market data for all listed stocks.
    """
    if bdshare is None:
        raise ImportError("bdshare package is required. Install it with: pip install bdshare")

    df = bdshare.get_current_trade_all()
    # Coerce numeric columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def fetch_historical_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical OHLCV data for a specific DSE stock ticker.

    Parameters
    ----------
    symbol : str
        DSE stock ticker, e.g. ``"GP"``.
    start : str
        Start date in ``"YYYY-MM-DD"`` format.
    end : str
        End date in ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with OHLCV columns converted to numeric,
        sorted in ascending date order.
    """
    if bdshare is None:
        raise ImportError("bdshare package is required. Install it with: pip install bdshare")

    df = bdshare.get_hist(symbol, start, end)

    # Ensure numeric data types
    for col in df.columns:
        if col.lower() not in ("date", "symbol", "ticker"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse and sort index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)

    return df


if __name__ == "__main__":
    # Example: fetch Grameenphone (GP) historical data
    print("Fetching historical data for GP (Grameenphone)...")
    gp_data = fetch_historical_data("GP", "2023-01-01", "2024-01-01")
    print(gp_data.head())
    print(f"\nShape: {gp_data.shape}")
    print(f"\nColumns: {list(gp_data.columns)}")
