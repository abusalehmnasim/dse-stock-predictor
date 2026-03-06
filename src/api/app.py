"""
app.py – FastAPI backend for the DSE Stock Predictor application.

Endpoints
---------
GET  /                              Health check
GET  /api/live                      Live DSE market data
GET  /api/historical/{symbol}       Historical OHLCV + technical indicators
POST /api/predict                   Predict stock price N days ahead
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DSE Stock Predictor API",
    description="Machine-learning powered stock prediction for the Dhaka Stock Exchange.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    symbol: str
    days_ahead: int = 7
    model_type: str = "xgboost"  # "lstm" or "xgboost"


class PredictResponse(BaseModel):
    symbol: str
    days_ahead: int
    predicted_price: float
    current_price: float
    price_change_pct: float
    signal: str  # BUY / SELL / HOLD
    model_type: str


class HistoricalResponse(BaseModel):
    symbol: str
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Startup: load pre-trained models (best-effort)
# ---------------------------------------------------------------------------

_lstm_predictor = None
_xgb_predictor = None


@app.on_event("startup")
async def load_models() -> None:
    global _lstm_predictor, _xgb_predictor

    lstm_path = os.environ.get("LSTM_MODEL_PATH", "models/lstm_model")
    xgb_path = os.environ.get("XGB_MODEL_PATH", "models/xgb_model")

    try:
        from src.models.lstm_model import DSEStockPredictor

        predictor = DSEStockPredictor()
        predictor.load(lstm_path)
        _lstm_predictor = predictor
    except Exception:
        pass  # Model not yet trained – predictions will be skipped

    try:
        from src.models.xgboost_model import DSEXGBoostPredictor

        predictor = DSEXGBoostPredictor()
        predictor.load(xgb_path)
        _xgb_predictor = predictor
    except Exception:
        pass  # Model not yet trained – predictions will be skipped


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _signal(change_pct: float) -> str:
    """Convert a predicted percentage change to a trading signal."""
    if change_pct > 1.5:
        return "BUY"
    if change_pct < -1.5:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "DSE Stock Predictor API"}


@app.get("/api/live", tags=["Market Data"])
async def get_live_data() -> dict[str, Any]:
    """Return current live trading data for all DSE stocks."""
    try:
        from src.data.fetch_data import fetch_live_data

        df = fetch_live_data()
        return {"data": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/historical/{symbol}", tags=["Market Data"], response_model=HistoricalResponse)
async def get_historical_data(
    symbol: str,
    start: str = Query(default=str(date.today() - timedelta(days=365)), description="Start date YYYY-MM-DD"),
    end: str = Query(default=str(date.today()), description="End date YYYY-MM-DD"),
) -> HistoricalResponse:
    """Return historical OHLCV data enriched with technical indicators."""
    try:
        from src.data.fetch_data import fetch_historical_data
        from src.data.preprocess import feature_engineering_pipeline

        df = fetch_historical_data(symbol, start, end)
        df = feature_engineering_pipeline(df)
        df.index = df.index.astype(str)
        records = df.reset_index().to_dict(orient="records")
        return HistoricalResponse(symbol=symbol, data=records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict", tags=["Prediction"], response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict the stock price *days_ahead* days in the future.

    Buy/Sell/Hold signal logic:
    * **BUY**  – predicted price change > +1.5 %
    * **SELL** – predicted price change < -1.5 %
    * **HOLD** – otherwise
    """
    symbol = request.symbol.upper()
    model_type = request.model_type.lower()

    # Fetch & preprocess recent data
    try:
        from src.data.fetch_data import fetch_historical_data
        from src.data.preprocess import feature_engineering_pipeline

        end = str(date.today())
        start = str(date.today() - timedelta(days=365))
        df = fetch_historical_data(symbol, start, end)
        df = feature_engineering_pipeline(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {exc}") from exc

    # Select model
    predictor = _xgb_predictor if model_type == "xgboost" else _lstm_predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' is not loaded. Train and save the model first.",
        )

    try:
        col_map = {c.lower(): c for c in df.columns}
        close_col = col_map["close"]
        feature_cols = [c for c in df.columns if c != close_col]

        window = predictor.window_size
        if len(df) < window:
            raise HTTPException(status_code=400, detail="Not enough data to form a prediction window.")

        features = df[feature_cols].values
        features_scaled = predictor.feature_scaler.transform(features)

        # Use the last available window
        if model_type == "xgboost":
            X = features_scaled[-window:].flatten().reshape(1, -1)
        else:
            X = features_scaled[-window:].reshape(1, window, -1)

        predicted_price = float(predictor.predict(X)[0])
        current_price = float(df[close_col].iloc[-1])
        change_pct = ((predicted_price - current_price) / current_price) * 100

        return PredictResponse(
            symbol=symbol,
            days_ahead=request.days_ahead,
            predicted_price=round(predicted_price, 4),
            current_price=round(current_price, 4),
            price_change_pct=round(change_pct, 4),
            signal=_signal(change_pct),
            model_type=model_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
