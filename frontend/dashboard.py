"""
dashboard.py – Streamlit frontend for the DSE Stock Predictor application.

Run with:
    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

DSE_SYMBOLS = [
    "GP", "BRAC", "SQURPHARMA", "RENATA", "BEXIMCO",
    "ISLAMIBANK", "DUTCHBANGL", "BRACBANK", "CITYBANK", "PRIMEBANK",
]

st.set_page_config(
    page_title="DSE Stock Predictor",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post(path: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

symbol = st.sidebar.selectbox("Stock Symbol", DSE_SYMBOLS, index=0)
model_type = st.sidebar.radio("Model", ["xgboost", "lstm"], index=0)
days_ahead = st.sidebar.slider("Days Ahead", min_value=1, max_value=30, value=7)

today = date.today()
start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=today)

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.title("📈 DSE Stock Predictor Dashboard")
st.markdown(
    "Dhaka Stock Exchange • ML-powered price prediction using LSTM & XGBoost"
)

# ---------------------------------------------------------------------------
# Live market data
# ---------------------------------------------------------------------------

st.header("🔴 Live Market Data")
with st.spinner("Fetching live data…"):
    live_response = _api_get("/api/live")

if live_response and "data" in live_response:
    live_df = pd.DataFrame(live_response["data"])
    st.dataframe(live_df, use_container_width=True, height=250)
else:
    st.info("Live data unavailable. Make sure the API server is running.")

# ---------------------------------------------------------------------------
# Historical data & charts
# ---------------------------------------------------------------------------

st.header(f"📊 Historical Data – {symbol}")
with st.spinner("Fetching historical data…"):
    hist_response = _api_get(
        f"/api/historical/{symbol}",
        params={"start": str(start_date), "end": str(end_date)},
    )

if hist_response and "data" in hist_response:
    hist_df = pd.DataFrame(hist_response["data"])

    # Identify date column
    date_col = next(
        (c for c in hist_df.columns if c.lower() in ("date", "index", "datetime")),
        hist_df.columns[0],
    )
    hist_df[date_col] = pd.to_datetime(hist_df[date_col])
    hist_df.sort_values(date_col, inplace=True)

    col_map = {c.lower(): c for c in hist_df.columns}
    open_col = col_map.get("open", "open")
    high_col = col_map.get("high", "high")
    low_col = col_map.get("low", "low")
    close_col = col_map.get("close", "close")
    volume_col = col_map.get("volume", "volume")

    # ---- Candlestick + SMA + Bollinger Bands ----
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=(
            f"{symbol} – Price & Indicators",
            "Volume",
            "RSI",
        ),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=hist_df[date_col],
            open=hist_df[open_col],
            high=hist_df[high_col],
            low=hist_df[low_col],
            close=hist_df[close_col],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # SMA overlay
    for sma_col, color in (("SMA_7", "orange"), ("SMA_21", "blue")):
        if sma_col in hist_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist_df[date_col],
                    y=hist_df[sma_col],
                    name=sma_col,
                    line=dict(color=color, width=1),
                ),
                row=1,
                col=1,
            )

    # Bollinger Bands overlay
    if "BB_Upper" in hist_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hist_df[date_col],
                y=hist_df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(128,0,128,0.5)", dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df[date_col],
                y=hist_df["BB_Lower"],
                name="BB Lower",
                fill="tonexty",
                fillcolor="rgba(128,0,128,0.05)",
                line=dict(color="rgba(128,0,128,0.5)", dash="dot"),
            ),
            row=1,
            col=1,
        )

    # Volume bar chart
    if volume_col in hist_df.columns:
        fig.add_trace(
            go.Bar(
                x=hist_df[date_col],
                y=hist_df[volume_col],
                name="Volume",
                marker_color="rgba(0,128,0,0.4)",
            ),
            row=2,
            col=1,
        )

    # RSI
    if "RSI" in hist_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hist_df[date_col],
                y=hist_df["RSI"],
                name="RSI",
                line=dict(color="purple", width=1),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- MACD chart ----
    if "MACD" in hist_df.columns:
        st.subheader("MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(
            go.Scatter(
                x=hist_df[date_col],
                y=hist_df["MACD"],
                name="MACD",
                line=dict(color="blue", width=1.5),
            )
        )
        if "MACD_Signal" in hist_df.columns:
            macd_fig.add_trace(
                go.Scatter(
                    x=hist_df[date_col],
                    y=hist_df["MACD_Signal"],
                    name="Signal",
                    line=dict(color="red", width=1.5),
                )
            )
        macd_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        macd_fig.update_layout(height=250)
        st.plotly_chart(macd_fig, use_container_width=True)

else:
    st.info("Historical data unavailable. Make sure the API server is running.")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

st.header("🔮 Price Prediction")

if st.button("Predict", type="primary"):
    with st.spinner("Running model inference…"):
        pred_response = _api_post(
            "/api/predict",
            {
                "symbol": symbol,
                "days_ahead": days_ahead,
                "model_type": model_type,
            },
        )

    if pred_response:
        col1, col2, col3, col4 = st.columns(4)
        signal = pred_response.get("signal", "HOLD")
        signal_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(signal, "gray")

        col1.metric("Current Price", f"৳ {pred_response['current_price']:,.2f}")
        col2.metric(
            f"Predicted ({days_ahead}d)",
            f"৳ {pred_response['predicted_price']:,.2f}",
            delta=f"{pred_response['price_change_pct']:+.2f}%",
        )
        col3.metric("Signal", signal)
        col4.metric("Model", pred_response.get("model_type", model_type).upper())

        if signal == "BUY":
            st.success(f"✅ **BUY** – predicted upside of {pred_response['price_change_pct']:+.2f}%")
        elif signal == "SELL":
            st.error(f"🔴 **SELL** – predicted downside of {pred_response['price_change_pct']:+.2f}%")
        else:
            st.warning("⚠️ **HOLD** – predicted change within ±1.5%")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational purposes only and does **not** constitute "
    "financial advice. Past predictions do not guarantee future results."
)
