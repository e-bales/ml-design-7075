import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.models.predict import load_latest_model, predict

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

st.set_page_config(page_title="Stock Movement Prediction", page_icon="📈")
st.title("Stock Movement Prediction Demo")

st.markdown(
    "This app loads the latest saved model and predicts whether a stock's next-day closing price will rise or fall based on input features."
)

try:
    model, model_path = load_latest_model(MODEL_DIR)
    st.success(f"Loaded latest model: {model_path.name}")
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

st.markdown("### Input features")
feature_text = st.text_area(
    "Enter feature values as JSON",
    value=json.dumps(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1200000,
        },
        indent=2,
    ),
    height=200,
)

if st.button("Predict"):
    try:
        features = json.loads(feature_text)
        if not isinstance(features, dict):
            raise ValueError("Input must be a JSON object mapping feature names to numeric values.")

        result = predict(model, features)
        st.write("### Prediction Result")
        st.write(result)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("---")
st.markdown(
    "To use the Streamlit app, run `streamlit run src/app/streamlit_app.py` from the project root."
)
