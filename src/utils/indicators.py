"""
indicators.py – Technical indicators for DSE stock data.

All indicators are added as new columns to the supplied DataFrame
by :func:`add_technical_indicators`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append a comprehensive set of technical indicators to *df*.

    Expects the DataFrame to contain at least the columns
    ``close``, ``high``, ``low``, and ``volume`` (case-insensitive).

    Indicators added
    ----------------
    * SMA_7, SMA_21 – Simple Moving Averages
    * EMA_12, EMA_26 – Exponential Moving Averages
    * MACD, MACD_Signal – Moving Average Convergence/Divergence
    * RSI – Relative Strength Index (14-period)
    * BB_Upper, BB_Middle, BB_Lower – Bollinger Bands (20-period, 2σ)
    * Volume_MA – Volume Moving Average (10-period)
    * Daily_Return – Day-over-day percentage change of close
    * Volatility – Rolling 10-period std of Daily_Return
    * Lag_1, Lag_2, Lag_3, Lag_5, Lag_7 – Close price lag features (1, 2, 3, 5, 7 days)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new indicator columns appended.
    """
    df = df.copy()

    # Normalise column names to lower-case for lookup, keep originals
    col_map = {c.lower(): c for c in df.columns}
    close = df[col_map["close"]]
    high = df[col_map["high"]]
    low = df[col_map["low"]]
    volume = df[col_map["volume"]]

    # --- Moving averages ---
    df["SMA_7"] = close.rolling(window=7).mean()
    df["SMA_21"] = close.rolling(window=21).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- RSI ---
    df["RSI"] = _rsi(close, period=14)

    # --- Bollinger Bands (20-period, 2σ) ---
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std

    # --- Volume Moving Average ---
    df["Volume_MA"] = volume.rolling(window=10).mean()

    # --- Daily Return & Volatility ---
    df["Daily_Return"] = close.pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=10).std()

    # --- Lag features ---
    for lag in (1, 2, 3, 5, 7):
        df[f"Lag_{lag}"] = close.shift(lag)

    return df
