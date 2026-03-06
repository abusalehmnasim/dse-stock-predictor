"""
lstm_model.py – LSTM-based stock price predictor for DSE data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


class DSEStockPredictor:
    """Three-layer stacked LSTM model for DSE stock price prediction.

    Parameters
    ----------
    window_size : int
        Number of past time steps used as input (look-back window).
        Default is 60.
    """

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self.model = None
        self.feature_scaler: MinMaxScaler | None = None
        self.target_scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, input_shape: tuple[int, int]):
        """Build and compile the LSTM network.

        Parameters
        ----------
        input_shape : tuple[int, int]
            ``(window_size, n_features)``
        """
        # Lazy import so the package stays importable without TensorFlow
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )
        self.model = model
        return model

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
        """Scale data and create overlapping sequences.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (NaN-free).
        feature_cols : list[str]
            Column names to use as input features.
        target_col : str
            Column to predict.
        test_size : float
            Fraction of data held out for validation (chronological).

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
            X.append(features_scaled[i - self.window_size : i])
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
        epochs: int = 100,
        batch_size: int = 32,
    ):
        """Fit the LSTM model with early stopping.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training sequences and labels.
        X_val, y_val : np.ndarray
            Validation sequences and labels.
        epochs : int
            Maximum number of training epochs.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        keras.callbacks.History
            Keras training history object.
        """
        from tensorflow.keras.callbacks import EarlyStopping

        if self.model is None:
            self._build_model((X_train.shape[1], X_train.shape[2]))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )
        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference and inverse-transform predictions to original scale.

        Parameters
        ----------
        X : np.ndarray
            Input sequences shaped ``(n_samples, window_size, n_features)``.

        Returns
        -------
        np.ndarray
            Predictions in original price scale, shape ``(n_samples,)``.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded yet.")
        preds_scaled = self.model.predict(X)
        return self.target_scaler.inverse_transform(preds_scaled).flatten()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str) -> None:
        """Save model weights and scalers.

        Parameters
        ----------
        model_path : str
            Path prefix. Keras model is saved to ``<model_path>.h5``
            and scalers to ``<model_path>_scalers.pkl``.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(f"{model_path}.h5")
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "window_size": self.window_size,
            },
            f"{model_path}_scalers.pkl",
        )

    def load(self, model_path: str) -> None:
        """Load a previously saved model and its scalers.

        Parameters
        ----------
        model_path : str
            Path prefix used when :meth:`save` was called.
        """
        from tensorflow import keras

        self.model = keras.models.load_model(f"{model_path}.h5")
        meta = joblib.load(f"{model_path}_scalers.pkl")
        self.feature_scaler = meta["feature_scaler"]
        self.target_scaler = meta["target_scaler"]
        self.window_size = meta["window_size"]
