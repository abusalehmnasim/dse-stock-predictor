"""
xgboost_model.py – XGBoost-based stock price predictor for DSE data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


class DSEXGBoostPredictor:
    """XGBoost regressor for DSE stock price prediction.

    The predictor uses a tabular feature representation: the input window
    is flattened to a 1-D feature vector per sample, matching the expected
    input format of tree-based models.

    Parameters
    ----------
    window_size : int
        Number of past time steps used as input features. Default is 60.
    """

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self.model = None
        self.feature_scaler: MinMaxScaler | None = None
        self.target_scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scale data and create flat feature vectors.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (NaN-free).
        feature_cols : list[str]
            Column names to use as input features.
        target_col : str
            Column to predict.
        test_size : float
            Fraction held out for validation (chronological).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(X_train, y_train, X_val, y_val)``
        """
        features = df[feature_cols].values
        target = df[[target_col]].values

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)

        X, y = [], []
        for i in range(self.window_size, len(features_scaled)):
            # Flatten the window into a single feature vector
            X.append(features_scaled[i - self.window_size : i].flatten())
            y.append(target_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        split = int(len(X) * (1 - test_size))
        return X[:split], y[:split], X[split:], y[split:]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        early_stopping_rounds: int = 20,
    ) -> None:
        """Fit the XGBoost model.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data.
        X_val, y_val : np.ndarray
            Validation data used for early stopping.
        n_estimators : int
            Maximum number of boosting rounds.
        learning_rate : float
            Step size shrinkage.
        max_depth : int
            Maximum depth of each tree.
        early_stopping_rounds : int
            Stop after this many rounds without validation improvement.
        """
        from xgboost import XGBRegressor

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and inverse-transform to original price scale.

        Parameters
        ----------
        X : np.ndarray
            Flat feature array shaped ``(n_samples, window_size * n_features)``.

        Returns
        -------
        np.ndarray
            Predictions in original price scale, shape ``(n_samples,)``.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        preds_scaled = self.model.predict(X).reshape(-1, 1)
        return self.target_scaler.inverse_transform(preds_scaled).flatten()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str) -> None:
        """Persist the model and scalers to disk.

        Parameters
        ----------
        model_path : str
            File path (without extension) for saving.
            Saves model to ``<model_path>.pkl`` and metadata to
            ``<model_path>_meta.pkl``.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        joblib.dump(self.model, f"{model_path}.pkl")
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "window_size": self.window_size,
            },
            f"{model_path}_meta.pkl",
        )

    def load(self, model_path: str) -> None:
        """Load a previously saved model and its metadata.

        Parameters
        ----------
        model_path : str
            Path prefix used when :meth:`save` was called.
        """
        self.model = joblib.load(f"{model_path}.pkl")
        meta = joblib.load(f"{model_path}_meta.pkl")
        self.feature_scaler = meta["feature_scaler"]
        self.target_scaler = meta["target_scaler"]
        self.window_size = meta["window_size"]
