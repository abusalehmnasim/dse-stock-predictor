"""
evaluate.py – Model evaluation metrics and visualisation for DSE stock predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoids division by zero)."""
    y_true = np.array(y_true, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute a standard set of regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``rmse``, ``mae``, ``mape``, and ``r2``.
    """
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
) -> None:
    """Plot actual and predicted stock prices with matplotlib.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth price series.
    y_pred : np.ndarray
        Predicted price series.
    title : str
        Chart title.
    """
    import matplotlib.pyplot as plt

    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="Actual", linewidth=1.5)
    ax.plot(y_pred, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
