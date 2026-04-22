"""
analyze.py — Annualized Sharpe ratio and Jensen's Alpha for each model strategy.

Risk-free rate: actual daily federal funds rate averaged over the test period.
Market proxy:   equal-weighted daily return across all 9 tickers (buy & hold).
Alpha/Beta:     OLS regression of daily excess strategy returns on daily excess
                market returns (CAPM).

Run:
    python analyze.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from model import (
    DEFAULT_PROCESSED_DIR,
    TARGET_COL,
    build_models,
    get_feature_cols,
    get_feature_cols_single_ticker,
    load_all_tickers,
    prepare_features,
    time_split,
)

TRADING_DAYS = 252
TEST_FRAC = 0.3
RF_RATE_PATH = Path("data/raw/macro/federal_funds_rate.csv")
SPY_PRICE_PATH = Path("data/raw/SPY/prices_daily.csv")
RF_FALLBACK = 0.045


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def load_spy_returns(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    if not SPY_PRICE_PATH.exists():
        raise FileNotFoundError(
            f"SPY price data not found at {SPY_PRICE_PATH}. "
            "Run: python pipeline.py --ticker SPY --skip-news --price-outputsize full --skip-macro"
        )
    spy = pd.read_csv(SPY_PRICE_PATH, parse_dates=["date"])
    spy = spy.sort_values("date").set_index("date")
    spy["return"] = spy["close"].pct_change()
    mask = (spy.index >= start_date) & (spy.index <= end_date)
    return spy.loc[mask, "return"].dropna()


def load_rf_rate(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if not RF_RATE_PATH.exists():
        return RF_FALLBACK
    rf = pd.read_csv(RF_RATE_PATH, parse_dates=["date"])
    mask = (rf["date"] >= start_date) & (rf["date"] <= end_date)
    mean_annual = rf.loc[mask, "value"].mean() / 100
    return mean_annual if not np.isnan(mean_annual) else RF_FALLBACK


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sharpe(daily_returns: pd.Series, rf_daily: float) -> float:
    excess = daily_returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def alpha_beta(
    strategy_daily: pd.Series, market_daily: pd.Series, rf_daily: float
) -> tuple[float, float]:
    excess_strat = (strategy_daily - rf_daily).values
    excess_mkt = (market_daily - rf_daily).values
    reg = LinearRegression().fit(excess_mkt.reshape(-1, 1), excess_strat)
    beta = float(reg.coef_[0])
    alpha_annual = float((1 + reg.intercept_) ** TRADING_DAYS - 1)
    return alpha_annual, beta


def annualized_return(daily_returns: pd.Series) -> float:
    return float((1 + daily_returns).prod() ** (TRADING_DAYS / len(daily_returns)) - 1)


# ---------------------------------------------------------------------------
# Strategy return builder
# ---------------------------------------------------------------------------

def strategy_daily_returns(
    X: pd.DataFrame,
    y: pd.Series,
    next_day_returns: pd.Series,
    dates: pd.Series,
    tickers: pd.Series,
    model,
    scale: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X)
    else:
        X_tr = X_train.values
        X_te = X.values

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    return pd.DataFrame({
        "date": dates.values,
        "ticker": tickers.values,
        "pred": preds,
        "next_day_return": next_day_returns.values,
        "strategy_return": next_day_returns.values * preds,
        "bah_return": next_day_returns.values,
    })


# ---------------------------------------------------------------------------
# Callable API for external use (e.g. api.py startup)
# ---------------------------------------------------------------------------

def compute_ticker_metrics(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    test_frac: float = TEST_FRAC,
) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Train per-ticker models and compute Sharpe, Alpha, Ann.Return.

    Returns:
        best_model:     ticker -> model name with highest Sharpe
        historical_perf: ticker -> {"sharpe", "alpha", "ann_ret"}
    """
    df = load_all_tickers(processed_dir)
    df = prepare_features(df)
    train_df, test_df, _ = time_split(df, test_frac=test_frac)

    test_start = test_df["date"].min()
    test_end = test_df["date"].max()
    rf_annual = load_rf_rate(test_start, test_end)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    mkt_daily = load_spy_returns(test_start, test_end)

    ticker_feature_cols = get_feature_cols_single_ticker(df)
    ticker_results = []

    for ticker in sorted(df["ticker"].unique()):
        t_train = train_df[train_df["ticker"] == ticker]
        t_test = test_df[test_df["ticker"] == ticker]

        X_tr = t_train[ticker_feature_cols]
        y_tr = t_train[TARGET_COL]
        X_te = t_test[ticker_feature_cols]
        ticker_mkt = mkt_daily.reindex(t_test["date"].values).dropna()

        for run_name, model, scale in build_models():
            daily = strategy_daily_returns(
                X=X_te,
                y=t_test[TARGET_COL],
                next_day_returns=t_test["next_day_return"],
                dates=t_test["date"],
                tickers=t_test["ticker"],
                model=model,
                scale=scale,
                X_train=X_tr,
                y_train=y_tr,
            )
            strat = daily.set_index("date")["strategy_return"].sort_index()
            strat, mkt_t = strat.align(ticker_mkt, join="inner")

            ann_ret = annualized_return(strat)
            sr = sharpe(strat, rf_daily)
            alp, bet = alpha_beta(strat, mkt_t, rf_daily)

            ticker_results.append({
                "ticker": ticker,
                "model": run_name,
                "ann_ret": ann_ret,
                "sharpe": sr,
                "alpha": alp,
                "beta": bet,
            })

    results_df = pd.DataFrame(ticker_results)
    best_rows = (
        results_df.sort_values("sharpe", ascending=False)
        .groupby("ticker", sort=False)
        .first()
        .reset_index()
    )

    best_model = {}
    historical_perf = {}
    for _, row in best_rows.iterrows():
        t = row["ticker"]
        best_model[t] = row["model"]
        historical_perf[t] = {
            "sharpe": round(float(row["sharpe"]), 3),
            "alpha": round(float(row["alpha"]), 4),
            "ann_ret": round(float(row["ann_ret"]), 4),
        }

    return best_model, historical_perf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    df = load_all_tickers(DEFAULT_PROCESSED_DIR)
    df = prepare_features(df)

    train_df, test_df, cutoff_date = time_split(df, test_frac=TEST_FRAC)

    test_start = test_df["date"].min()
    test_end = test_df["date"].max()
    rf_annual = load_rf_rate(test_start, test_end)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

    print(f"Test period:  {test_start.date()} → {test_end.date()}")
    print(f"Risk-free rate (annualized, avg fed funds): {rf_annual:.2%}")
    print(f"Market proxy: SPY (S&P 500 ETF)\n")

    # SPY daily returns as market proxy
    mkt_daily = load_spy_returns(test_start, test_end)

    # -----------------------------------------------------------------------
    # Pooled models
    # -----------------------------------------------------------------------
    pooled_feature_cols = get_feature_cols(df)
    X_train_pool = train_df[pooled_feature_cols]
    y_train_pool = train_df[TARGET_COL]
    X_test_pool = test_df[pooled_feature_cols]

    pooled_results = []
    print("=" * 68)
    print("  POOLED MODELS")
    print("=" * 68)
    print(f"  {'Model':<24} {'Ann.Ret':>8} {'Sharpe':>8} {'Alpha':>8} {'Beta':>6}")
    print(f"  {'-'*56}")

    for run_name, model, scale in build_models():
        daily = strategy_daily_returns(
            X=X_test_pool,
            y=test_df[TARGET_COL],
            next_day_returns=test_df["next_day_return"],
            dates=test_df["date"],
            tickers=test_df["ticker"],
            model=model,
            scale=scale,
            X_train=X_train_pool,
            y_train=y_train_pool,
        )

        port_daily = daily.groupby("date")["strategy_return"].mean().sort_index()
        port_daily, mkt_aligned = port_daily.align(mkt_daily, join="inner")

        ann_ret = annualized_return(port_daily)
        sr = sharpe(port_daily, rf_daily)
        alp, bet = alpha_beta(port_daily, mkt_aligned, rf_daily)

        pooled_results.append({
            "model": run_name, "ann_ret": ann_ret,
            "sharpe": sr, "alpha": alp, "beta": bet,
        })
        print(f"  {run_name:<24} {ann_ret:>8.2%} {sr:>8.3f} {alp:>8.2%} {bet:>6.3f}")

    # Buy & hold benchmark
    bah_daily = mkt_daily
    bah_ann = annualized_return(bah_daily)
    bah_sr = sharpe(bah_daily, rf_daily)
    print(f"  {'buy_and_hold':<24} {bah_ann:>8.2%} {bah_sr:>8.3f} {'0.00%':>8} {'1.000':>6}")

    # -----------------------------------------------------------------------
    # Per-ticker models — all 3 models, show only the best per ticker by Sharpe
    # -----------------------------------------------------------------------
    ticker_feature_cols = get_feature_cols_single_ticker(df)

    # Collect every ticker × model result first, then filter to the winner
    ticker_results = []

    for ticker in sorted(df["ticker"].unique()):
        t_train = train_df[train_df["ticker"] == ticker]
        t_test = test_df[test_df["ticker"] == ticker]

        X_tr = t_train[ticker_feature_cols]
        y_tr = t_train[TARGET_COL]
        X_te = t_test[ticker_feature_cols]

        ticker_mkt = mkt_daily.reindex(t_test["date"].values).dropna()
        bah_cum_t = float((1 + t_test["next_day_return"]).prod() - 1)

        for run_name, model, scale in build_models():
            daily = strategy_daily_returns(
                X=X_te,
                y=t_test[TARGET_COL],
                next_day_returns=t_test["next_day_return"],
                dates=t_test["date"],
                tickers=t_test["ticker"],
                model=model,
                scale=scale,
                X_train=X_tr,
                y_train=y_tr,
            )

            strat = daily.set_index("date")["strategy_return"].sort_index()
            strat, mkt_t = strat.align(ticker_mkt, join="inner")

            ann_ret = annualized_return(strat)
            strat_cum = float((1 + strat).prod() - 1)
            sr = sharpe(strat, rf_daily)
            alp, bet = alpha_beta(strat, mkt_t, rf_daily)

            ticker_results.append({
                "ticker": ticker,
                "model": run_name,
                "ann_ret": ann_ret,
                "strat_cum": strat_cum,
                "sharpe": sr,
                "alpha": alp,
                "beta": bet,
                "bah_cum": bah_cum_t,
                "edge": strat_cum - bah_cum_t,
            })

    # Keep only the best model per ticker (highest Sharpe)
    results_df = pd.DataFrame(ticker_results)
    best_per_ticker = (
        results_df.sort_values("sharpe", ascending=False)
        .groupby("ticker", sort=False)
        .first()
        .reset_index()
        .sort_values("ticker")
    )

    print(f"\n{'=' * 84}")
    print("  BEST MODEL PER TICKER (by Sharpe ratio)  |  vs B&H = cumulative return over test period")
    print(f"{'=' * 84}")
    print(f"  {'Ticker':<6} {'Best Model':<24} {'Ann.Ret':>8} {'Cum.Ret':>8} {'Sharpe':>8} {'Alpha':>8} {'Beta':>6} {'vs B&H':>8}")
    print(f"  {'-'*78}")
    for _, row in best_per_ticker.iterrows():
        print(
            f"  {row['ticker']:<6} {row['model']:<24} {row['ann_ret']:>8.2%} "
            f"{row['strat_cum']:>8.2%} {row['sharpe']:>8.3f} {row['alpha']:>8.2%} "
            f"{row['beta']:>6.3f} {row['edge']:>+8.2%}"
        )


if __name__ == "__main__":
    main()
