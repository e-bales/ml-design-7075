import glob
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data.alphavantage import save_daily_adjusted
from src.data.feature_engineering import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

st.title('Stock Direction Prediction Demo')
st.write('This demo loads the latest registered model and makes a direction prediction for a sample stock.')

REGISTRY_FILE = MODELS_DIR / 'registry.json'
if not REGISTRY_FILE.exists():
    st.error('Model registry not found. Run the training pipeline first to create `models/registry.json`.')
    st.stop()

with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
    registry = json.load(f)

if not registry:
    st.error('Model registry is empty. Train a model and register it first.')
    st.stop()

latest_entry = sorted(registry, key=lambda entry: entry.get('registered_at', ''), reverse=True)[0]
model_filename = latest_entry.get('model_filename')
if model_filename is None:
    st.error('Invalid registry entry: missing model_filename.')
    st.stop()

model_path = MODELS_DIR / model_filename
if not model_path.exists():
    st.error(f'Model file not found: {model_path}')
    st.stop()

model = joblib.load(model_path)

st.sidebar.header('Alpha Vantage')
api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
if not api_key:
    st.sidebar.warning('Set ALPHAVANTAGE_API_KEY to fetch data directly from Alpha Vantage.')

symbol = st.sidebar.text_input('Alpha Vantage symbol', value='AAPL')
if st.sidebar.button('Fetch daily data'):
    if not api_key:
        st.sidebar.error('Unable to fetch: missing ALPHAVANTAGE_API_KEY.')
    else:
        try:
            raw_path = save_daily_adjusted(symbol, ROOT / 'data' / 'raw')
            st.sidebar.success(f'Raw data saved to {raw_path.name}')
            st.write(f'Successfully fetched daily adjusted data for {symbol.upper()}.')
            data_preview = pd.read_csv(raw_path, parse_dates=['date']).head()
            st.dataframe(data_preview)
        except Exception as exc:
            st.sidebar.error(str(exc))

st.sidebar.header('Demo settings')

csv_files = sorted(
    glob.glob(str(PROCESSED_DIR / 'versions' / '*.csv')),
    key=lambda path: Path(path).stat().st_mtime,
    reverse=True,
)

if not csv_files:
    st.warning('Processed dataset not found. Run the notebook or pipeline to generate processed feature CSVs.')
    st.stop()

processed_file = Path(csv_files[0])
data = pd.read_csv(processed_file, parse_dates=['date'])

st.sidebar.markdown('### Sample selection')
if 'ticker' not in data.columns or 'date' not in data.columns:
    st.error('Processed file is missing required columns `ticker` and/or `date`.')
    st.stop()

tickers = sorted(data['ticker'].unique())
ticker = st.sidebar.selectbox('Ticker', tickers)
dates = data[data['ticker'] == ticker]['date'].dt.strftime('%Y-%m-%d').tolist()
date = st.sidebar.selectbox('Date', dates)
selected = data[(data['ticker'] == ticker) & (data['date'].dt.strftime('%Y-%m-%d') == date)].head(1)

if selected.empty:
    st.warning('No sample found for selected ticker/date.')
else:
    row = selected.iloc[0]
    st.subheader('Selected sample')
    display_columns = [col for col in ['date', 'ticker', 'close', 'sentiment_compound'] if col in row.index]
    st.write(row[display_columns])
    if not all(feature in row.index for feature in FEATURE_COLUMNS):
        st.error('Processed row is missing expected feature columns for prediction.')
    else:
        features = row[FEATURE_COLUMNS].values.reshape(1, -1)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        st.markdown('**Predicted next-day direction:**')
        st.write('📈 Up' if prediction == 1 else '📉 Down')
        if prob is not None:
            st.write('Confidence:', float(prob.max()))

st.sidebar.header('Latest registered model')
st.sidebar.write(f"Model file: {model_filename}")
st.sidebar.write(latest_entry.get('metrics', {}))
