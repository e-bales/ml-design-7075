# Market Confidence Modeling: Predicting Next-Day Stock Direction

Our preliminary README for the stock market prediction model operations workflow. Read on for more information on the project details, including the problem statement, purpose of the ML system, stakeholders, workflow, and data pipeline setup instructions.

## Project Team

- Devin Walker
- Vivian Comer
- Eli Bales
- Andrew Mccurrach
- Colton Jones

## Project Overview

This project builds a machine learning system to predict whether a selected stock’s closing price will increase or decrease on the next trading day. The system combines historical stock price data, engineered market features, and news sentiment signals to produce an interpretable, data-driven forecast for short-term investment decision support.

## Problem Statement

Humans and rule-based code alone cannot reliably process all relevant short-term market signals quickly enough to make consistent next-day directional forecasts. Stock price movement is influenced by many interacting factors, and the relationships between them are noisy, complex, and difficult to capture. Our goal is not to build a fully automated trading system, but to develop a robust ML pipeline that produces a timely, repeatable, and useful signal for real decisions.

## Purpose of the ML System

The purpose of this system is to provide an interpretable next-day directional forecast that improves consistency, efficiency, and decision support in short-term investing. The model is designed to support Portfolio Managers and Analysts by providing a reproducible signal based on both historical market behavior and article/news sentiment.

## Stakeholders

- Portfolio Managers who use the model output to inform buy/sell decisions
- Data Analysts, Engineers, and Quants who maintain and improve the pipeline
- Clients whose capital is being managed by the investment company

## System Workflow

The project follows a workflow that includes:
1. Data ingestion from Alpha Vantage API
2. Data validation
3. Processing and feature engineering
4. Model training
5. Model validation and testing
6. Deployment
7. Monitoring and logging
8. Data and model versioning

## Current Pipeline

`pipeline.py` ingests raw stock price data and news sentiment data from Alpha Vantage for a selected ticker. It saves both the latest pull and a timestamped archive as CSV files for later feature engineering and modeling.

## What It Does

When you run the script, it:

1. Loads your Alpha Vantage API key from `.env`
2. Pulls daily stock price data with `TIME_SERIES_DAILY`
3. Pulls news sentiment data with `NEWS_SENTIMENT`
4. Splits large news windows into smaller ones when a request hits the 1000-article cap
5. Deduplicates articles
6. Saves the results as CSV files

## Model Development and Experiment Tracking

The project uses MLflow as the main experiment tracking library. MLflow logs model parameters, evaluation metrics, and run metadata to support reproducibility and comparison across experiments. Each model run is assigned a unique UUID, and artifacts such as evaluation graphs and confusion matrices are stored for later review. Reproducibility is further supported through generated environment files such as `conda.yaml`.

## API Key

Log onto [Alpha Vantage](https://www.alphavantage.co/) and click on "GET FREE API KEY" to generate your own API key. Add your Alpha Vantage key to `.env`. Accepted names include:

- `ALPHA_VANTAGE_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `ALPHA_VANTAGE_KEY`
- `ALPHAVANTAGE_KEY`
- `API_KEY`

Example:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
```
