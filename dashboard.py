"""
dashboard.py — Streamlit analyst dashboard for next-day stock predictions.

Run (with API already running on port 8000):
    streamlit run dashboard.py
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Stock Direction Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_tickers() -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE}/tickers", timeout=10)
        resp.raise_for_status()
        return resp.json()["tickers"]
    except Exception as e:
        st.error(f"Cannot reach API at {API_BASE}. Start it with: uvicorn api:app --reload --port 8000\n\nError: {e}")
        return []


def fetch_prediction(ticker: str) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/predict/{ticker}", timeout=60)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Prediction failed ({resp.status_code}): {resp.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out — Alpha Vantage may be slow. Try again.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def fmt_pct(v: float) -> str:
    return f"{v:+.1%}" if v is not None else "N/A"


def fmt_sharpe(v: float) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Stock Direction Prediction Dashboard")
st.caption("Next-day UP/DOWN signal — ML model trained on price, sentiment, and macro features")

ticker_data = fetch_tickers()
if not ticker_data:
    st.stop()

ticker_map = {t["ticker"]: t for t in ticker_data}
ticker_list = sorted(ticker_map.keys())

# Sidebar
with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Select Ticker", ticker_list)
    run_btn = st.button("Get Prediction", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Historical Performance")
    perf = ticker_map[selected]["historical_performance"]
    best_model = ticker_map[selected]["best_model"]

    st.markdown(f"**Best Model:** `{best_model}`")
    ann_ret = perf["ann_ret"]
    alpha = perf["alpha"]
    sharpe = perf["sharpe"]

    col1, col2 = st.columns(2)
    col1.metric("Ann. Return", fmt_pct(ann_ret))
    col2.metric("Jensen's α", fmt_pct(alpha))
    st.metric("Sharpe Ratio", fmt_sharpe(sharpe))

    st.divider()
    st.caption("All metrics from historical backtest (test set, long-or-flat strategy vs. SPY benchmark).")

# Main content
if run_btn:
    with st.spinner(f"Fetching live data and predicting for {selected}..."):
        result = fetch_prediction(selected)

    if result:
        pred = result["prediction"]
        conf = result["confidence"]
        as_of = result["as_of_date"]
        model_used = result["model_used"]

        # Prediction banner
        st.markdown("---")
        col_pred, col_conf, col_meta = st.columns([2, 2, 3])

        with col_pred:
            if pred == "UP":
                st.success(f"## ▲ UP", icon="✅")
            else:
                st.error(f"## ▼ DOWN", icon="🔴")
            st.caption(f"As of {as_of}")

        with col_conf:
            st.metric("Confidence (P[UP])", f"{conf:.1%}")
            conf_pct = int(conf * 100)
            bar_color = "#28a745" if pred == "UP" else "#dc3545"
            st.markdown(
                f"""
                <div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%">
                    <div style="background:{bar_color};border-radius:8px;height:18px;width:{conf_pct}%"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_meta:
            st.markdown(f"**Ticker:** {result['ticker']}")
            st.markdown(f"**Model:** `{model_used}`")
            st.markdown(f"**Strategy:** Long when UP predicted, flat otherwise")

        st.markdown("---")

        # Price chart and key features side by side
        col_chart, col_feat = st.columns([3, 2])

        with col_chart:
            st.subheader("30-Day Price History")
            prices = result.get("recent_prices", [])
            if prices:
                price_df = pd.DataFrame(prices)
                price_df["date"] = pd.to_datetime(price_df["date"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_df["date"],
                    y=price_df["close"],
                    mode="lines+markers",
                    line=dict(color="#2196F3", width=2),
                    marker=dict(size=4),
                    name="Close Price",
                    hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No local price data available for chart.")

        with col_feat:
            st.subheader("Key Features")
            feats = result.get("key_features", {})
            feature_labels = {
                "return_1d": ("1-Day Return", ".2%"),
                "rsi_14": ("RSI (14)", ".1f"),
                "macd_hist": ("MACD Histogram", ".4f"),
                "avg_sentiment": ("Avg Sentiment", ".4f"),
                "bullish_share": ("Bullish News %", ".1%"),
                "volatility_5d": ("5-Day Volatility", ".4f"),
                "has_news": ("Has News Today", "d"),
                "vix": ("VIX", ".1f"),
                "yield_curve": ("Yield Curve", ".4f"),
            }
            rows = []
            for key, (label, fmt) in feature_labels.items():
                if key in feats:
                    val = feats[key]
                    if fmt.endswith("%"):
                        display = f"{val:{fmt}}"
                    elif fmt == "d":
                        display = "Yes" if val else "No"
                    else:
                        display = f"{val:{fmt}}"
                    rows.append({"Feature": label, "Value": display})
            if rows:
                feat_df = pd.DataFrame(rows).set_index("Feature")
                st.dataframe(feat_df, use_container_width=True)

else:
    # Landing state
    st.markdown("---")
    st.info(f"Select a ticker and click **Get Prediction** to fetch live data and generate a next-day signal.")

    # Show all tickers summary table
    st.subheader("Model Performance Summary (Historical Backtest)")
    rows = []
    for t in ticker_list:
        perf = ticker_map[t]["historical_performance"]
        rows.append({
            "Ticker": t,
            "Best Model": ticker_map[t]["best_model"],
            "Ann. Return": f"{perf['ann_ret']:+.1%}",
            "Jensen's α": f"{perf['alpha']:+.1%}",
            "Sharpe": f"{perf['sharpe']:.3f}",
        })
    summary_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(summary_df, use_container_width=True)
    st.caption(
        "Backtest period: ~172 trading days test set (30% holdout). Long-or-flat strategy, SPY as market benchmark. "
        "Past performance does not guarantee future results."
    )
