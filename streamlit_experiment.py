import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "mlruns" / "1" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

PRICE_FEATURE_COLS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volume_change_1d",
    "hl_spread",
    "price_to_sma5",
    "price_to_sma10",
    "price_to_sma20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
]

SENTIMENT_FEATURE_COLS = [
    "has_news",
    "article_count",
    "avg_sentiment",
    "median_sentiment",
    "sentiment_std",
    "avg_relevance",
    "bullish_share",
    "bearish_share",
    "neutral_share",
    "weighted_sentiment",
]

MACRO_FEATURE_COLS = [
    "vix",
    "vix_ma5",
    "sp500_return_1d",
    "yield_curve",
    "high_yield_spread",
]

USE_MACRO = False


def find_latest_model_artifact(models_dir: Path) -> Path:
    model_dirs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not model_dirs:
        raise FileNotFoundError(f"No model artifacts found in {models_dir}")
    latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
    artifact_path = latest / "artifacts"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact folder not found: {artifact_path}")
    return artifact_path


def load_latest_model(models_dir: Path):
    artifact_dir = find_latest_model_artifact(models_dir)
    model = mlflow.pyfunc.load_model(str(artifact_dir))
    return model, artifact_dir


def load_processed_data(processed_dir: Path) -> pd.DataFrame:
    dfs = []
    for ticker_dir in sorted(processed_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        modeling_table = ticker_dir / "modeling_table.csv"
        if not modeling_table.exists():
            continue
        data = pd.read_csv(modeling_table, parse_dates=["date"])
        data["ticker"] = ticker_dir.name
        dfs.append(data)

    if not dfs:
        raise FileNotFoundError(
            f"No processed modeling tables found under {processed_dir}"
        )
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    return combined


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_to_sma5"] = df["close"] / df["sma_5"]
    df["price_to_sma10"] = df["close"] / df["sma_10"]
    df["price_to_sma20"] = df["close"] / df["sma_20"]

    ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    df = pd.concat([df, ticker_dummies], axis=1)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
    macro_cols = [c for c in MACRO_FEATURE_COLS if c in df.columns] if USE_MACRO else []
    return PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS + macro_cols + ticker_cols


FEATURE_DESCRIPTIONS = {
    "return_1d": "Daily return for the sample day.",
    "return_3d": "3-day cumulative return.",
    "return_5d": "5-day cumulative return.",
    "volatility_5d": "5-day return volatility.",
    "volume_change_1d": "Volume change from previous day.",
    "hl_spread": "High-low price spread as a ratio.",
    "price_to_sma5": "Price divided by 5-day moving average.",
    "price_to_sma10": "Price divided by 10-day moving average.",
    "price_to_sma20": "Price divided by 20-day moving average.",
    "rsi_14": "Relative Strength Index over 14 days.",
    "macd": "MACD value (trend momentum).",
    "macd_signal": "MACD signal line value.",
    "macd_hist": "MACD histogram value.",
    "has_news": "Whether news events were available.",
    "article_count": "Number of news articles in the window.",
    "avg_sentiment": "Mean sentiment score across articles.",
    "median_sentiment": "Median sentiment score.",
    "sentiment_std": "Standard deviation of sentiment.",
    "avg_relevance": "Average relevance of news to the ticker.",
    "bullish_share": "Share of articles tagged bullish.",
    "bearish_share": "Share of articles tagged bearish.",
    "neutral_share": "Share of neutral articles.",
    "weighted_sentiment": "Sentiment score weighted by relevance.",
    "vix": "VIX index level.",
    "vix_ma5": "5-day moving average of VIX.",
    "sp500_return_1d": "S&P 500 one-day return.",
    "yield_curve": "Spread between long and short rates.",
    "high_yield_spread": "Spread between high-yield and risk-free debt.",
}


def _init_what_if_state(defaults: dict[str, float], sample_key: str) -> None:
    if st.session_state.get("whatif_sample_id") != sample_key:
        for col, value in defaults.items():
            st.session_state[f"whatif_{col}"] = float(value)
        st.session_state["whatif_sample_id"] = sample_key
        return

    for col, value in defaults.items():
        key = f"whatif_{col}"
        if key not in st.session_state:
            st.session_state[key] = float(value)


def _reset_what_if(defaults: dict[str, float], sample_key: str) -> None:
    for col, value in defaults.items():
        st.session_state[f"whatif_{col}"] = float(value)
    st.session_state["whatif_sample_id"] = sample_key


def _build_feature_frame(feature_input: dict[str, float], feature_cols: list[str], ticker_cols: list[str], ticker_col_name: str) -> pd.DataFrame:
    all_inputs = feature_input.copy()
    for tc in ticker_cols:
        all_inputs[tc] = 1.0 if tc == ticker_col_name else 0.0
    return pd.DataFrame([all_inputs], columns=feature_cols)


def _sensitivity_analysis(model, feature_cols: list[str], current_values: dict[str, float], ticker_cols: list[str], ticker_col_name: str, editable_cols: list[str]) -> pd.DataFrame:
    base_frame = _build_feature_frame(current_values, feature_cols, ticker_cols, ticker_col_name)
    base_pred = int(model.predict(base_frame)[0])
    base_proba = None
    if hasattr(model, "predict_proba"):
        try:
            base_proba = model.predict_proba(base_frame)[0]
        except Exception:
            base_proba = None

    rows = []
    for col in editable_cols:
        if col in PRICE_FEATURE_COLS:
            delta = max(0.01, abs(current_values[col]) * 0.05)
        elif col in SENTIMENT_FEATURE_COLS or col in MACRO_FEATURE_COLS:
            delta = max(0.01, abs(current_values[col]) * 0.1)
        else:
            delta = 0.01

        for direction, label in [(delta, "up"), (-delta, "down")]:
            test_values = current_values.copy()
            test_values[col] = float(test_values[col] + direction)
            test_frame = _build_feature_frame(test_values, feature_cols, ticker_cols, ticker_col_name)
            try:
                pred = int(model.predict(test_frame)[0])
            except Exception:
                continue
            prob = None
            prob_change = None
            if base_proba is not None:
                try:
                    new_proba = model.predict_proba(test_frame)[0]
                    prob = float(new_proba[1]) if len(new_proba) > 1 else None
                    prob_change = prob - float(base_proba[1]) if prob is not None else None
                except Exception:
                    prob = None
            rows.append({
                "feature": col,
                "direction": label,
                "delta": direction,
                "pred": pred,
                "flip": pred != base_pred,
                "prob": prob,
                "prob_change": prob_change,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "prob_change" in df.columns:
        df = df.sort_values(["flip", "prob_change"], ascending=[False, False])
    else:
        df = df.sort_values(["flip"], ascending=[False])
    return df.head(10)


def render_prediction(model, feature_cols, row: pd.Series) -> None:
    editable_cols = [c for c in feature_cols if not c.startswith("ticker_")]
    ticker_cols = [c for c in feature_cols if c.startswith("ticker_")]
    default_values = row[editable_cols].astype(float).to_dict()
    ticker_col_name = f"ticker_{row['ticker']}"
    sample_key = f"{row['ticker']}_{row['date'].strftime('%Y-%m-%d')}"

    _init_what_if_state(default_values, sample_key)

    st.markdown("### What-if scenario")
    st.write("Change a few inputs below, then press Run prediction to see the model response.")

    st.button(
        "Reset inputs to sample values",
        on_click=_reset_what_if,
        args=(default_values, sample_key),
    )

    price_cols = [c for c in editable_cols if c in PRICE_FEATURE_COLS]
    sentiment_cols = [c for c in editable_cols if c in SENTIMENT_FEATURE_COLS]
    macro_cols = [c for c in editable_cols if c in MACRO_FEATURE_COLS and c in default_values]

    left, right = st.columns(2)

    with left:
        if price_cols:
            st.markdown("**Price & technical features**")
            price_left, price_right = st.columns(2)
            for i, col in enumerate(price_cols):
                target_col = price_left if i % 2 == 0 else price_right
                step = 0.01 if "return" in col or "volatility" in col or "ratio" in col else 1.0
                fmt = "%.4f" if step < 1 else "%.2f"
                target_col.number_input(
                    col.replace("_", " ").title(),
                    value=st.session_state[f"whatif_{col}"],
                    step=step,
                    format=fmt,
                    help=FEATURE_DESCRIPTIONS.get(col, ""),
                    key=f"whatif_{col}",
                )

        if macro_cols:
            with st.expander("Macro feature inputs"):
                macro_left, macro_right = st.columns(2)
                for i, col in enumerate(macro_cols):
                    target_col = macro_left if i % 2 == 0 else macro_right
                    target_col.number_input(
                        col.replace("_", " ").title(),
                        value=st.session_state[f"whatif_{col}"],
                        step=0.01,
                        format="%.4f",
                        help=FEATURE_DESCRIPTIONS.get(col, ""),
                        key=f"whatif_{col}",
                    )

    with right:
        if sentiment_cols:
            st.markdown("**Sentiment features**")
            sent_left, sent_right = st.columns(2)
            for i, col in enumerate(sentiment_cols):
                target_col = sent_left if i % 2 == 0 else sent_right
                target_col.number_input(
                    col.replace("_", " ").title(),
                    value=st.session_state[f"whatif_{col}"],
                    step=0.01,
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS.get(col, ""),
                    key=f"whatif_{col}",
                )

    run_pressed = st.button("Run prediction")
    analyze_pressed = st.button("Analyze feature impact")

    current_values = {col: float(st.session_state[f"whatif_{col}"]) for col in editable_cols}
    sample_feature_df = pd.DataFrame.from_dict(current_values, orient="index", columns=["value"])
    sample_feature_df.index.name = "feature"
    st.markdown("**Current feature values**")
    st.dataframe(sample_feature_df, use_container_width=True)

    show_prediction = run_pressed or analyze_pressed
    base_prediction = None
    base_prob = None
    if show_prediction:
        feature_input = current_values.copy()
        for col in ticker_cols:
            feature_input[col] = 1.0 if col == ticker_col_name else 0.0

        feature_df = pd.DataFrame([feature_input], columns=feature_cols)
        try:
            prediction = model.predict(feature_df)
            base_prediction = int(prediction[0])
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(feature_df)[0]
                    base_prob = float(proba[1]) if len(proba) > 1 else None
                except Exception:
                    base_prob = None

            direction = "📈 Up" if base_prediction == 1 else "📉 Down"
            st.markdown("### Prediction result")
            st.metric("Predicted direction", direction)
            if base_prob is not None:
                st.metric("Confidence", f"{base_prob:.2%}")

            with st.expander("Prediction details"):
                st.write({
                    "prediction": base_prediction,
                    "confidence": base_prob,
                })
                st.write(feature_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if analyze_pressed:
        analysis_df = _sensitivity_analysis(
            model,
            feature_cols,
            current_values,
            ticker_cols,
            ticker_col_name,
            editable_cols,
        )
        if analysis_df.empty:
            st.warning("No sensitivity results could be computed for this selection.")
        else:
            st.markdown("### Impact analysis")
            if base_prob is not None:
                st.markdown("This table shows the most impactful one-step changes and the direction flip probability.")
                display_df = analysis_df["feature direction delta pred flip prob prob_change".split()]
                st.dataframe(display_df, use_container_width=True)
            else:
                st.markdown("This table shows candidate changes that may flip the predicted class.")
                display_df = analysis_df[["feature", "direction", "delta", "pred", "flip"]]
                st.dataframe(display_df, use_container_width=True)

    if not show_prediction:
        st.info("Adjust the inputs above and press Run prediction to evaluate a what-if case.")

    with st.expander("Advanced raw JSON output"):
        raw_json = json.dumps({**current_values, **{col: 1.0 if col == ticker_col_name else 0.0 for col in ticker_cols}}, indent=2)
        st.code(raw_json, language="json")


st.set_page_config(
    page_title="ML Design 7075 Prediction App",
    page_icon="📊",
    layout="wide",
)
st.title("ML Design 7075 Stock Direction Prediction")
st.markdown(
    "This app uses the `ml-design-7075` package, the latest saved MLflow model artifact, and processed ticker data to predict next-day stock direction."
)

try:
    model, model_path = load_latest_model(MODELS_DIR)
    st.success(f"Loaded model from: {model_path}")
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

try:
    raw_data = load_processed_data(PROCESSED_DIR)
    processed_df = prepare_features(raw_data)
    feature_columns = get_feature_cols(processed_df)
except Exception as exc:
    st.error(f"Unable to load processed data: {exc}")
    st.stop()

st.sidebar.header("Model details")
st.sidebar.write(f"**Latest artifact:** {model_path.name}")
st.sidebar.write(f"**Path:** {model_path}")
st.sidebar.write(f"**Features used:** {len(feature_columns)}")
st.sidebar.markdown(
    "---\n" "Select a sample row to inspect the processed features and run a prediction. "
    "Use the JSON editor to override feature values for what-if analysis."
)

st.sidebar.header("Usage")
st.sidebar.markdown(
    "1. Choose a ticker and date.\n"
    "2. Review the sample data.\n"
    "3. Edit feature values if needed.\n"
    "4. Click **Predict next-day direction**."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Run command:**\n```bash\nstreamlit run ml-design-7075/streamlit_app_7075.py\n```"
)

st.sidebar.header("Sample selection")
selected_ticker = st.sidebar.selectbox(
    "Ticker", sorted(processed_df["ticker"].unique())
)
available_dates = (
    processed_df[processed_df["ticker"] == selected_ticker]["date"]
    .dt.strftime("%Y-%m-%d")
    .sort_values(ascending=False)
    .unique()
)
selected_date = st.sidebar.selectbox("Date", available_dates)

selected_row = processed_df[
    (processed_df["ticker"] == selected_ticker)
    & (processed_df["date"].dt.strftime("%Y-%m-%d") == selected_date)
]
if selected_row.empty:
    st.warning("No data row found for the selected ticker/date.")
    st.stop()
row = selected_row.iloc[0]

left, right = st.columns([1, 1])

with left:
    st.subheader("Selected sample")
    st.markdown(
        f"**Ticker:** {row['ticker']}  \n"
        f"**Date:** {row['date'].strftime('%Y-%m-%d')}"
    )
    if "target_up" in row.index and pd.notna(row["target_up"]):
        st.metric(
            "Actual direction",
            "📈 Up" if int(row["target_up"]) == 1 else "📉 Down",
        )

    key_columns = [
        "close",
        "next_day_return",
        "article_count",
        "avg_sentiment",
        "bullish_share",
        "bearish_share",
        "neutral_share",
    ]
    present_cols = [c for c in key_columns if c in row.index]
    if present_cols:
        st.markdown("**Key sample values**")
        summary = row[present_cols].to_frame().T
        summary = summary.rename(columns={
            c: c.replace("_", " ").title() for c in present_cols
        })
        st.dataframe(summary)

    with st.expander("Show full raw selected row"):
        display_cols = ["date", "ticker"] + [c for c in row.index if c not in {"date", "ticker"}]
        st.dataframe(row.loc[display_cols].to_frame().T)

with right:
    st.subheader("Prediction")
    render_prediction(model, feature_columns, row)

st.markdown("---")
st.info("Use the JSON editor to tweak feature values and test model sensitivity.")
st.caption("This app loads the most recent MLflow model artifact from the ml-design-7075 package.")