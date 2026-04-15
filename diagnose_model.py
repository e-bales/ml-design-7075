import pathlib
import pandas as pd
import mlflow

root = pathlib.Path('ml-design-7075')
models_dir = root / 'mlruns' / '1' / 'models'
model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if not model_dirs:
    raise SystemExit('No model dirs found')
artifact_dir = model_dirs[0] / 'artifacts'
print('artifact', artifact_dir)
model = mlflow.pyfunc.load_model(str(artifact_dir))
print('loaded model type', type(model))

processed_rows = []
for ticker_dir in sorted((root / 'data' / 'processed').iterdir()):
    if ticker_dir.is_dir():
        mt = ticker_dir / 'modeling_table.csv'
        if mt.exists():
            df = pd.read_csv(mt, parse_dates=['date'])
            df['ticker'] = ticker_dir.name
            processed_rows.append(df)

combined = pd.concat(processed_rows, ignore_index=True).sort_values(['date', 'ticker']).reset_index(drop=True)
print('combined shape', combined.shape)
for name in ['target_up', 'target', 'next_day_return']:
    if name in combined.columns:
        print(name, combined[name].dropna().astype(int).value_counts().to_dict())
    else:
        print(name, 'missing')

row = combined.iloc[10]
print('sample', row['ticker'], row['date'], 'target_up', row.get('target_up', None))

feature_cols = [
    'return_1d','return_3d','return_5d','volatility_5d','volume_change_1d','hl_spread',
    'price_to_sma5','price_to_sma10','price_to_sma20','rsi_14','macd','macd_signal','macd_hist',
    'has_news','article_count','avg_sentiment','median_sentiment','sentiment_std','avg_relevance',
    'bullish_share','bearish_share','neutral_share','weighted_sentiment'
]

ticker_cols = [f'ticker_{t}' for t in sorted(combined['ticker'].unique())]
feature_cols += ticker_cols
input_dict = {}
for col in feature_cols:
    if col in row.index:
        input_dict[col] = float(row[col])
    elif col == 'price_to_sma5':
        input_dict[col] = float(row['close'] / row['sma_5']) if row['sma_5'] != 0 else 0.0
    elif col == 'price_to_sma10':
        input_dict[col] = float(row['close'] / row['sma_10']) if row['sma_10'] != 0 else 0.0
    elif col == 'price_to_sma20':
        input_dict[col] = float(row['close'] / row['sma_20']) if row['sma_20'] != 0 else 0.0
    else:
        input_dict[col] = 0.0
for tc in ticker_cols:
    input_dict[tc] = 1.0 if tc == f'ticker_{row['ticker']}' else 0.0

X = pd.DataFrame([input_dict], columns=feature_cols)
print('X sample columns', X.columns[:20].tolist())
print('X sample values', X.iloc[0].head(20).to_dict())
print('prediction', model.predict(X))
if hasattr(model, 'predict_proba'):
    print('proba', model.predict_proba(X)[0])
