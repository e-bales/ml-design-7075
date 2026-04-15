# ml-design-7075

Our preliminary README for the stock market prediction model operations workflow.

# Pipeline README

## Overview

`pipeline.py` ingests raw stock price data and news sentiment data from Alpha Vantage for a selected ticker. It saves both the latest pull and a timestamped archive as CSV files for later feature engineering and modeling.

## What It Does

When you run the script, it:

1. Loads your Alpha Vantage API key from `.env`
2. Pulls daily stock price data with `TIME_SERIES_DAILY`
3. Pulls news sentiment data with `NEWS_SENTIMENT`
4. Splits large news windows into smaller ones when a request hits the 1000-article cap
5. Deduplicates articles
6. Saves the results as CSV files

## API Key

Add your Alpha Vantage key to `.env`. Accepted names include:

- `ALPHA_VANTAGE_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `ALPHA_VANTAGE_KEY`
- `ALPHAVANTAGE_KEY`
- `API_KEY`

Example:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
