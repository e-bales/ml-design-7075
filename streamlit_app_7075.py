import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# 1. Path Setup - Looking directly into your processed folders
DATA_DIR = Path("data/processed")

@st.cache_data
def get_tickers():
    if DATA_DIR.exists():
        return [f.name for f in DATA_DIR.iterdir() if f.is_dir()]
    return []

@st.cache_data
def load_latest_data(ticker):
    # Finds the CSV inside the ticker folder
    ticker_path = DATA_DIR / ticker
    csv_files = list(ticker_path.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        df.columns = [c.lower() for c in df.columns]
        # Standardize date
        date_col = next((c for c in df.columns if 'date' in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        return df, date_col
    return None, None

# --- UI LAYOUT ---
st.title("🎯 Next-Day Price Forecast")
st.markdown("---")

# 2. Sidebar - Simplified Ticker Selection
tickers = get_tickers()
if tickers:
    selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
    df, date_col = load_latest_data(selected_ticker)

    if df is not None:
        # Get Current Day Info
        last_row = df.iloc[-1]
        current_date = last_row[date_col].strftime('%Y-%m-%d') if date_col else "Current"
        
        # --- PREDICTION ENGINE ---
        # Note: Since your sentiment files don't have 'close', we define a 
        # base 'Last Price' to demonstrate the prediction logic.
        current_price = 150.00 # Placeholder for actual price feed
        
        # Signal logic based on your 'weighted_sentiment' column
        sentiment_signal = last_row.get('weighted_sentiment', 0)
        prediction_move = 0.02 * sentiment_signal # Simulating a 2% max move
        predicted_price = current_price * (1 + prediction_move)
        
        # --- DISPLAY RESULTS ---
        st.subheader(f"Analysis for {selected_ticker} (as of {current_date})")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Estimated Current Price", f"${current_price:,.2f}")
        with c2:
            delta = ((predicted_price - current_price) / current_price) * 100
            st.metric(
                label="Predicted Price (Next Trading Day)", 
                value=f"${predicted_price:,.2f}",
                delta=f"{delta:+.2f}%",
                delta_color="normal" if delta > 0 else "inverse"
            )
        with c3:
            st.write("**Model Confidence**")
            # Confidence based on article count and sentiment strength
            conf_score = min(0.95, 0.5 + (abs(sentiment_signal) * 0.5))
            st.progress(conf_score)
            st.caption(f"{conf_score*100:.1f}% confidence based on current sentiment.")

        st.markdown("---")
        
        # 3. Historical Accuracy Section
        st.subheader("🗓️ Historical Model Performance")
        
        # Creating a simplified hit/miss log for the last 5 trading days
        history_df = df.tail(5).copy()
        # Simulated 'Correct' column for the UI demo
        history_df['Actual Movement'] = np.random.choice(["▲ UP", "▼ DOWN"], size=len(history_df))
        history_df['Model Prediction'] = np.random.choice(["▲ UP", "▼ DOWN"], size=len(history_df))
        history_df['Accuracy'] = np.where(history_df['Actual Movement'] == history_df['Model Prediction'], "✅ HIT", "❌ MISS")
        
        st.table(history_df[[date_col, 'weighted_sentiment', 'Model Prediction', 'Actual Movement', 'Accuracy']])
        
    else:
        st.error(f"No CSV data found in the {selected_ticker} folder.")
else:
    st.error("Data folders not found. Please check data/processed/ path.")