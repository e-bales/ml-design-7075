# Pipeline README

## Overview

`pipeline.py` is the first ingestion stage for the stock prediction project. It pulls raw stock price data and raw news sentiment data from Alpha Vantage, then saves both a current copy and an archived copy of each pull as CSV files.

The script is designed to support a machine learning workflow where:

- stock price history is used for price-based features
- news sentiment is used for sentiment-based features
- both datasets are later merged into a modeling table


## What The Pipeline Does

When you run `pipeline.py`, it does the following:

1. Loads your Alpha Vantage API key from `.env`
2. Pulls daily stock price history for a ticker using `TIME_SERIES_DAILY`
3. Pulls news sentiment history for the same ticker using `NEWS_SENTIMENT`
4. Breaks news requests into time windows so it can collect more than 1000 articles
5. Automatically splits busy windows into smaller windows if a request hits the 1000-article cap
6. Deduplicates the collected news articles
7. Saves the outputs as CSV files


## Environment Variable

The script expects your Alpha Vantage key in `.env`.

Accepted variable names:

- `ALPHA_VANTAGE_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `ALPHA_VANTAGE_KEY`
- `ALPHAVANTAGE_KEY`
- `API_KEY`

Example `.env`:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
```


## How To Run

Basic run:

```powershell
python pipeline.py --ticker AAPL
```

Example with custom settings:

```powershell
python pipeline.py --ticker AAPL --news-months 6 --news-chunk-days 30 --pause-seconds 2.5
```


## Main Arguments

- `--ticker`
  - required
  - stock ticker to pull, such as `AAPL`

- `--price-outputsize`
  - default: `compact`
  - choices: `compact`, `full`
  - on the free tier, `compact` is usually the practical choice

- `--news-months`
  - default: `6`
  - number of months of news history to attempt to collect

- `--news-chunk-days`
  - default: `30`
  - initial news request window size in days

- `--news-limit`
  - default: `1000`
  - max articles Alpha Vantage will return per news request

- `--pause-seconds`
  - default: `1.2`
  - pause between requests to reduce rate-limit issues

- `--output-dir`
  - default: `data/raw`
  - location of the latest raw outputs

- `--archive-dir`
  - default: `past_data_pulled`
  - location of timestamped archived pulls


## Output Files

For ticker `AAPL`, the script writes:

Latest raw files:

- `data/raw/AAPL/prices_daily.csv`
- `data/raw/AAPL/news_sentiment.csv`
- `data/raw/AAPL/news_chunk_summary.csv`

Archived files:

- `past_data_pulled/AAPL_YYYYMMDD_HHMMSS_prices_daily.csv`
- `past_data_pulled/AAPL_YYYYMMDD_HHMMSS_news_sentiment.csv`
- `past_data_pulled/AAPL_YYYYMMDD_HHMMSS_news_chunk_summary.csv`


## What Each Output Contains

`prices_daily.csv`

- daily open, high, low, close, and volume
- one row per trading day

`news_sentiment.csv`

- one row per article
- article timestamp
- title
- source
- url
- summary
- overall sentiment score and label
- ticker-specific relevance score
- ticker-specific sentiment score and label

`news_chunk_summary.csv`

- one row per attempted news window
- start and end timestamps for each request
- number of rows returned
- whether the window hit the article limit
- whether the window was automatically split into smaller windows
- split depth
- whether the daily Alpha Vantage limit was reached


## Adaptive News Pulling

Alpha Vantage only returns up to `1000` news articles per request. Because of that, the script does not rely on a single request for news history.

Instead, it:

- starts with larger windows such as 30 days
- checks whether a window returns the max of 1000 rows
- if the window is full, it assumes that period may be truncated
- splits that window into two smaller windows
- repeats until coverage is better or the window becomes small enough

This helps the pipeline capture more complete news history for high-volume tickers like `AAPL`.


## Rate Limits And Free-Tier Constraints

Alpha Vantage free-tier usage is the main practical limitation of this script.

Important limits:

- about `1 request per second`
- about `25 requests per day`

The script already handles some of this by:

- retrying when it hits the per-second rate limit
- backing off and waiting before retrying
- saving partial results if the daily limit is reached

For very busy tickers, a full 6-month backfill may take multiple days on the free plan.


## Typical Workflow

1. Add your API key to `.env`
2. Run the ingestion script for a ticker
3. Review the raw CSVs in `data/raw/<TICKER>/`
4. Check `news_chunk_summary.csv` to see which windows were split or truncated
5. Use the raw files for downstream feature engineering and model training


## Notes

- Do not commit `.env`
- `prices_daily.csv` may be limited on the free Alpha Vantage plan
- `news_sentiment.csv` is deduplicated before saving
- partial data is still saved if the daily request cap is reached


## Next Suggested Step

The natural next stage after this script is a feature-engineering pipeline that:

- aggregates news sentiment by day
- computes price-based technical features
- merges price and sentiment data into one modeling table
- saves the result into a `data/processed/` folder
