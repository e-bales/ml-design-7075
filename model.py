import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DEFAULT_PROCESSED_DIR = Path("data") / "processed"
EXPERIMENT_NAME = "stock-direction-prediction"

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

# Macro features require sufficient training data to be useful.
# Set to False until full price history is available.
USE_MACRO = False

TARGET_COL = "target_up"


def load_all_tickers(processed_dir: Path) -> pd.DataFrame:
    dfs = []
    for ticker_dir in sorted(processed_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        table_path = ticker_dir / "modeling_table.csv"
        if not table_path.exists():
            print(f"  Skipping {ticker_dir.name} — no modeling_table.csv found.")
            continue
        df = pd.read_csv(table_path)
        df["ticker"] = ticker_dir.name
        dfs.append(df)
        print(f"  Loaded {ticker_dir.name}: {len(df)} rows")

    if not dfs:
        raise FileNotFoundError(f"No modeling tables found in {processed_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined.sort_values("date").reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Price-to-MA ratios — scale-invariant across tickers
    df["price_to_sma5"] = df["close"] / df["sma_5"]
    df["price_to_sma10"] = df["close"] / df["sma_10"]
    df["price_to_sma20"] = df["close"] / df["sma_20"]

    # One-hot encode ticker so the model knows which stock each row belongs to
    ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    df = pd.concat([df, ticker_dummies], axis=1)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
    macro_cols = [c for c in MACRO_FEATURE_COLS if c in df.columns] if USE_MACRO else []
    return PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS + macro_cols + ticker_cols


def time_split(
    df: pd.DataFrame, test_frac: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = df["date"].sort_values().unique()
    cutoff_idx = int(len(dates) * (1 - test_frac))
    cutoff_date = pd.Timestamp(dates[cutoff_idx])
    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()
    return train, test, cutoff_date


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    run_name: str,
    artifact_dir: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"])
    ax.set_yticklabels(["Down", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    path = artifact_dir / f"confusion_matrix_{run_name}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_feature_importance(
    importance: np.ndarray,
    feature_cols: list[str],
    run_name: str,
    artifact_dir: Path,
) -> Path:
    top_n = min(15, len(feature_cols))
    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importance[indices], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features — {run_name}")
    plt.tight_layout()
    path = artifact_dir / f"feature_importance_{run_name}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def compute_backtest_return(
    y_pred: np.ndarray, actual_returns: pd.Series, tickers: pd.Series
) -> tuple[float, float]:
    """
    Compute average per-ticker strategy return and buy-and-hold return.
    Long when model predicts up, flat otherwise.
    Returns (avg_strategy_return, avg_buy_hold_return).
    """
    results = pd.DataFrame({
        "ticker": tickers.values,
        "actual_return": actual_returns.values,
        "pred": y_pred,
    })
    results["signal_return"] = results["actual_return"] * results["pred"]

    def compound(s):
        return float(np.prod(1 + s) - 1)

    per_ticker = results.groupby("ticker").agg(
        strategy=("signal_return", compound),
        buy_hold=("actual_return", compound),
    )
    return float(per_ticker["strategy"].mean()), float(per_ticker["buy_hold"].mean())


def compute_per_ticker_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    actual_returns: pd.Series,
    tickers: pd.Series,
) -> pd.DataFrame:
    rows = []
    for ticker in sorted(tickers.unique()):
        mask = tickers.values == ticker
        yt = y_test.values[mask]
        yp = y_pred[mask]
        ret = actual_returns.values[mask]
        signal_ret = ret * yp

        accuracy = accuracy_score(yt, yp)
        try:
            roc_auc = roc_auc_score(yt, y_proba[mask]) if y_proba is not None else None
        except ValueError:
            roc_auc = None

        report = classification_report(yt, yp, output_dict=True, zero_division=0)
        strategy_ret = float(np.prod(1 + signal_ret) - 1)
        buy_hold_ret = float(np.prod(1 + ret) - 1)

        rows.append({
            "ticker": ticker,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision_up": report.get("1", {}).get("precision", 0.0),
            "recall_up": report.get("1", {}).get("recall", 0.0),
            "f1_up": report.get("1", {}).get("f1-score", 0.0),
            "n_test": int(mask.sum()),
            "strategy_return": strategy_ret,
            "buy_hold_return": buy_hold_ret,
            "vs_buy_hold": strategy_ret - buy_hold_ret,
        })

    return pd.DataFrame(rows).set_index("ticker")


def save_per_ticker_chart(
    ticker_df: pd.DataFrame, run_name: str, artifact_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Per-Ticker Performance — {run_name}", fontsize=13)

    tickers = ticker_df.index.tolist()
    x = range(len(tickers))

    # Accuracy per ticker
    axes[0].bar(x, ticker_df["accuracy"], color="steelblue")
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers, rotation=45)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Ticker")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Strategy vs buy & hold
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], ticker_df["strategy_return"], width, label="Strategy", color="steelblue")
    axes[1].bar([i + width / 2 for i in x], ticker_df["buy_hold_return"], width, label="Buy & Hold", color="coral")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers, rotation=45)
    axes[1].set_ylabel("Return")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_title("Strategy vs Buy & Hold by Ticker")
    axes[1].legend()

    plt.tight_layout()
    path = artifact_dir / f"per_ticker_{run_name}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def run_model(
    model,
    run_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list[str],
    test_returns: pd.Series,
    test_tickers: pd.Series,
    cutoff_date: pd.Timestamp,
    artifact_dir: Path,
    scale: bool = False,
) -> None:
    if scale:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit = X_train.values
        X_test_fit = X_test.values

    with mlflow.start_run(run_name=run_name):
        # Parameters
        mlflow.log_param("model_type", run_name)
        mlflow.log_param("train_cutoff_date", str(cutoff_date.date()))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("scaled", scale)
        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                mlflow.log_param(k, v)

        # Train
        model.fit(X_train_fit, y_train)

        # Predict
        y_pred = model.predict(X_test_fit)
        y_proba = (
            model.predict_proba(X_test_fit)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        report = classification_report(y_test, y_pred, output_dict=True)
        backtest_ret, buy_hold_ret = compute_backtest_return(y_pred, test_returns, test_tickers)

        mlflow.log_metric("accuracy", accuracy)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision_up", report["1"]["precision"])
        mlflow.log_metric("recall_up", report["1"]["recall"])
        mlflow.log_metric("f1_up", report["1"]["f1-score"])
        mlflow.log_metric("backtest_return", backtest_ret)
        mlflow.log_metric("buy_hold_return", buy_hold_ret)

        # Console summary
        print(f"\n{'='*52}")
        print(f"  {run_name}")
        print(f"{'='*52}")
        print(f"  Accuracy:          {accuracy:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC:           {roc_auc:.4f}")
        print(f"  Precision (Up):    {report['1']['precision']:.4f}")
        print(f"  Recall (Up):       {report['1']['recall']:.4f}")
        print(f"  F1 (Up):           {report['1']['f1-score']:.4f}")
        print(f"  Backtest Return:   {backtest_ret:.2%}")
        print(f"  Buy & Hold:        {buy_hold_ret:.2%}")

        # Per-ticker breakdown
        ticker_df = compute_per_ticker_metrics(
            y_test, y_pred, y_proba, test_returns, test_tickers
        )
        print(f"\n  {'Ticker':<8} {'Acc':>6} {'AUC':>6} {'Strat':>8} {'B&H':>8} {'Edge':>8}")
        print(f"  {'-'*50}")
        for ticker, row in ticker_df.iterrows():
            auc_str = f"{row['roc_auc']:.3f}" if row["roc_auc"] is not None else "  N/A"
            print(
                f"  {ticker:<8} {row['accuracy']:>6.3f} {auc_str:>6} "
                f"{row['strategy_return']:>8.2%} {row['buy_hold_return']:>8.2%} "
                f"{row['vs_buy_hold']:>+8.2%}"
            )

        # Log per-ticker metrics to MLflow
        for ticker, row in ticker_df.iterrows():
            mlflow.log_metric(f"{ticker}_accuracy", row["accuracy"])
            mlflow.log_metric(f"{ticker}_strategy_return", row["strategy_return"])
            mlflow.log_metric(f"{ticker}_vs_buy_hold", row["vs_buy_hold"])

        pt_path = save_per_ticker_chart(ticker_df, run_name, artifact_dir)
        mlflow.log_artifact(str(pt_path))

        # Artifacts
        cm_path = save_confusion_matrix(y_test, y_pred, run_name, artifact_dir)
        mlflow.log_artifact(str(cm_path))

        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])

        if importance is not None:
            fi_path = save_feature_importance(
                importance, feature_cols, run_name, artifact_dir
            )
            mlflow.log_artifact(str(fi_path))

        # Log model
        if "xgboost" in run_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate stock direction models with MLflow tracking."
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing processed ticker folders.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        help="Fraction of dates to hold out as test set. Default 0.3.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="mlflow_artifacts",
        help="Local directory for plot artifacts before logging to MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print("Loading all ticker data...")
    df = load_all_tickers(processed_dir)
    print(f"\nCombined dataset: {len(df)} rows across {df['ticker'].nunique()} tickers.")
    print(f"Date range:       {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Class balance:    {df[TARGET_COL].mean():.1%} up days")

    df = prepare_features(df)
    feature_cols = get_feature_cols(df)

    train_df, test_df, cutoff_date = time_split(df, test_frac=args.test_frac)
    print(f"\nTrain: {len(train_df)} rows  (up to {cutoff_date.date()})")
    print(f"Test:  {len(test_df)} rows  (from {cutoff_date.date()})")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    test_returns = test_df["next_day_return"]
    test_tickers = test_df["ticker"]

    mlflow.set_experiment(EXPERIMENT_NAME)

    models = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=42),
            True,
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            False,
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42,
            ),
            False,
        ),
    ]

    for run_name, model, scale in models:
        run_model(
            model=model,
            run_name=run_name,
            X_train=X_train.copy(),
            X_test=X_test.copy(),
            y_train=y_train,
            y_test=y_test,
            feature_cols=feature_cols,
            test_returns=test_returns,
            test_tickers=test_tickers,
            cutoff_date=cutoff_date,
            artifact_dir=artifact_dir,
            scale=scale,
        )

    print("\n" + "=" * 52)
    print("All runs complete. Launch MLflow UI with:")
    print("  mlflow ui")
    print("Then open http://localhost:5000")


if __name__ == "__main__":
    main()
