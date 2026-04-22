import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# 1. Path Setup - Looking directly into your processed folders
# DATA_DIR = Path("data/processed")

@st.cache_data
def get_tickers():
    try:
        response = requests.get(f"{API_BASE_URL}/tickers", timeout=5)
        response.raise_for_status()

        data = response.json()

        return data

    except Exception as e:
        st.error(f"Unable to load tickers from API: {e}")
        return []

@st.cache_data
def get_prediction(ticker):
    try:
        # st.write({"ticker": ticker})
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"ticker": ticker},
            timeout=10
        )

        # st.write("Status code:", response.status_code)
        # st.write("Response text:", response.text)

        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"Prediction request failed: {e}")
        return None

# --- UI LAYOUT ---
st.title("🎯 Next-Day Price Forecast")
st.markdown("---")

# 2. Sidebar - Simplified Ticker Selection
tickers = get_tickers()
if tickers:
    selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
    res = get_prediction(selected_ticker)

    if res is not None:
        # Get Current Day Info
        # last_row = df.iloc[-1]
        # current_date = last_row[date_col].strftime('%Y-%m-%d') if date_col else "Current"
        
        # --- PREDICTION ENGINE ---
        # Note: Since your sentiment files don't have 'close', we define a 
        # base 'Last Price' to demonstrate the prediction logic.
        # current_price = 150.00 # Placeholder for actual price feed
        
        # Signal logic based on your 'weighted_sentiment' column
        # sentiment_signal = last_row.get('weighted_sentiment', 0)
        # prediction_move = 0.02 * sentiment_signal # Simulating a 2% max move
        # predicted_price = current_price * (1 + prediction_move)
        
        # --- DISPLAY RESULTS ---
        st.header(f"{selected_ticker}")
        
        # st.write(res)

        prediction_value = res["prediction"]
        row_data = res["data"]

        st.subheader("Prediction for today...")

        if prediction_value == 1:
            st.markdown(
                """
                <div style="
                    font-size:72px;
                    font-weight:700;
                    text-align:center;
                    color:#16a34a;
                    background-color:rgb(155 255 128 / 0.5);
                    border-radius:16px;
                    padding:20px;
                ">
                    📈 UP
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="
                    font-size:72px;
                    font-weight:700;
                    text-align:center;
                    color:#dc2626;
                    background-color:rgb(255 40 50 / 50%);
                    border-radius:16px;
                    padding:20px;
                ">
                    📉 DOWN
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Yesterday's Market Data")

        col1, col2, col3, col4 = st.columns(4)

        open_price = row_data["open"]
        high_price = row_data["high"]
        low_price = row_data["low"]
        close_price = row_data["close"]

        col1.metric("Open", f"${open_price:.2f}" if open_price is not None else "N/A")
        col2.metric("High", f"${high_price:.2f}" if high_price is not None else "N/A")
        col3.metric("Low", f"${low_price:.2f}" if low_price is not None else "N/A")
        col4.metric("Close", f"${close_price:.2f}" if close_price is not None else "N/A")
        # c1, c2, c3 = st.columns(3)
        # with c1:
        #     st.metric("Yesterday's Open Price", f"${res["data"]["open"]}")
        #     st.metric("Yesterday's Close Price", f"${res["data"]["close"]}")
        # with c2:
        #     delta = ((predicted_price - current_price) / current_price) * 100
        #     st.metric(
        #         label="Predicted Price (Next Trading Day)", 
        #         value=f"${predicted_price:,.2f}",
        #         delta=f"{delta:+.2f}%",
        #         delta_color="normal" if delta > 0 else "inverse"
        #     )
        # with c3:
        #     st.write("**Model Confidence**")
        #     # Confidence based on article count and sentiment strength
        #     conf_score = min(0.95, 0.5 + (abs(sentiment_signal) * 0.5))
        #     st.progress(conf_score)
        #     st.caption(f"{conf_score*100:.1f}% confidence based on current sentiment.")

        # st.markdown("---")
        
        # # 3. Historical Accuracy Section
        # st.subheader("🗓️ Historical Model Performance")
        
        # # Creating a simplified hit/miss log for the last 5 trading days
        # history_df = df.tail(5).copy()
        # # Simulated 'Correct' column for the UI demo
        # history_df['Actual Movement'] = np.random.choice(["▲ UP", "▼ DOWN"], size=len(history_df))
        # history_df['Model Prediction'] = np.random.choice(["▲ UP", "▼ DOWN"], size=len(history_df))
        # history_df['Accuracy'] = np.where(history_df['Actual Movement'] == history_df['Model Prediction'], "✅ HIT", "❌ MISS")
        
        # st.table(history_df[[date_col, 'weighted_sentiment', 'Model Prediction', 'Actual Movement', 'Accuracy']])
        
    else:
        st.error(f"No CSV data found in the {selected_ticker} folder.")
else:
    st.error("Data folders not found. Please check data/processed/ path.")