"""
preprocess.py – Data cleaning, normalization, train/test splitting,
and feature engineering for DSE stock data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.indicators import add_technical_indicators


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and forward-fill missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def normalize_data(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Scale selected columns to the [0, 1] range using MinMaxScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str] | None
        Columns to scale. Defaults to all numeric columns.

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler]
        Scaled DataFrame and fitted scaler.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame maintaining chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame sorted by date index.
    test_size : float
        Fraction of data to use for testing (default 0.2).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` split at the corresponding index.
    """
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and add all technical indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-enriched DataFrame, with NaN rows dropped.
    """
    df = clean_data(df)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    return df
