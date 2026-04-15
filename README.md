# ML Design Final Project

This repository implements the MVP prototype for a stock movement prediction system.
The system is designed for a private investment company and targets Portfolio Managers, Data Analysts, Engineers, and Quants.

## Project Overview

- Goal: predict whether a selected stock's closing price will increase or decrease on the next trading day.
- Data sources: historical end-of-day market data, engineered features, and sentiment analysis inputs.
- Primary users: Portfolio Managers who use predictions to support buy/sell decisions.
- Supporting stakeholders: Data Analysts, Engineers, and Quants responsible for maintenance and feature updates.

## Prototype Requirements Covered

1. Data Pipeline
   - Batch ingestion from `data/raw/`.
   - Validation rules implemented in `src/data/pipeline.py`.
   - Data versioning for processed datasets in `data/processed/versions/`.
   - Processed data stored for reproducible model training.

2. ML Model Development
   - Experiment tracking using MLflow under `mlruns/`.
   - Model training and logging in `src/models/train.py`.
   - Model version registry stored in `models/registry.json`.
   - Saved model artifacts in `models/`.

3. Basic System Integration
   - FastAPI demo service available at `src/app/main.py`.
   - Predict endpoint supports JSON feature input.

## Folder Structure

- `data/raw/` - raw CSV datasets to ingest
- `data/processed/` - processed and versioned data exports
- `src/data/` - ingestion, validation, and data pipeline code
- `src/models/` - training, registry, and prediction code
- `src/app/` - demonstration API service
- `models/` - versioned model artifacts and registry metadata
- `notebooks/` - exploratory notebooks and analysis
- `tests/` - automated unit tests

## Setup

Recommended environment setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For conda users:

```powershell
conda env create -f environment.yml
conda activate ml-design
```

## Running the Prototype

1. Add raw stock data to `data/raw/`.
2. Fetch raw daily stock data from Alpha Vantage (optional):

```powershell
streamlit run streamlit_app.py
```

Use the sidebar to download raw data for a symbol. The file is saved to `data/raw/<SYMBOL>_daily.csv`.

3. Convert raw Alpha Vantage data into processed features:

```powershell
python .\run_feature_engineering.py data/raw/AAPL_daily.csv --ticker AAPL
```

This creates a processed CSV file in `data/processed/versions/` with training features and a `target` column.

4. Or run the full fetch/process/train pipeline for one symbol:

```powershell
python .\run_full_pipeline.py AAPL
```

Or using the helper script:

```powershell
.\run_full_pipeline.ps1 AAPL
```

5. Train a model:

```powershell
python .\run_train.py
```

If you want to train manually, use a processed CSV with a `target` column:

```powershell
python -c "from pathlib import Path; import pandas as pd; from src.data.load_data import split_features_target; from src.models.train import train_model; df=pd.read_csv('data/processed/versions/<latest>.csv'); X,y=split_features_target(df,'target'); train_model(X,y,Path('models'))"
```

> Note: the training and Streamlit demo both use the same engineered feature set from `src/data/feature_engineering.py`. This keeps the model input contract consistent and avoids mismatches between training and inference.

4. Run the API demo:

```powershell
uvicorn src.app.main:app --reload
```

5. Fetch data from Alpha Vantage:

- The project loads `ALPHAVANTAGE_API_KEY` from the local `.env` file in the repository root.
- A `.env` file has been created with the configured key.

- Start the Streamlit demo and use the Alpha Vantage panel to download raw daily data by symbol:

```powershell
# Activate the virtual environment first
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

- Or, if you prefer to run directly from the local Python interpreter:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### Running the new ml-design-7075 Streamlit app

If you want to launch the separate app for the `ml-design-7075` package, use:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run ml-design-7075/streamlit_app_7075.py
```

This is the preferred command when `streamlit` is not found directly on PATH.

- The raw output CSV will be written to `data/raw/<SYMBOL>_daily.csv`.

## Notes

- Experiment tracking is stored in `mlruns/`.
- Versioned processed data files are saved in `data/processed/versions/`.
- Model registry metadata is saved in `models/registry.json`.
- Use the FastAPI endpoint for a minimal inference demonstration.
