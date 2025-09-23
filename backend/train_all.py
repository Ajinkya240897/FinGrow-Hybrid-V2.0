# backend/train_all.py
"""
Auto-fetch training script.

This script automatically fetches historical OHLCV for a list of tickers using yfinance,
computes features, trains models for multiple horizons, and saves joblib models to backend/models/.

Usage:
 - set environment variable TICKERS (comma-separated), e.g. "RELIANCE.NS,TCS.NS,INFY.NS"
 - optionally set MAX_ROWS (int) to limit historical rows fetched (defaults to 2500)
 - run: python backend/train_all.py

Notes:
 - The script trains per-symbol models and saves them as:
     backend/models/model_<safe_ticker>_<interval>.joblib
 - Intervals: '3-15d' (10 days ahead), '1-3m' (60 days), '3-6m' (120 days), '1-3y' (365 days)
 - If used on Render, ensure yfinance and other dependencies are in backend/requirements.txt
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# attempt to import yfinance
try:
    import yfinance as yf
except Exception:
    print("ERROR: yfinance is required. Add 'yfinance' to backend/requirements.txt and reinstall.", file=sys.stderr)
    raise

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# mapping intervals to horizon (days ahead)
HORIZONS = {
    "3-15d": 10,
    "1-3m": 60,
    "3-6m": 120,
    "1-3y": 365,
}

FEATURE_COLS = ["ret1", "ret3", "ma5", "ma10", "ma20", "vol10", "rsi14"]

# Configuration via env (tune as needed)
TICKERS_ENV = os.getenv("TICKERS", "").strip()
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS", "LT.NS"
]
TICKERS = [t.strip() for t in (TICKERS_ENV.split(",") if TICKERS_ENV else DEFAULT_TICKERS) if t.strip()]

MAX_ROWS = int(os.getenv("MAX_ROWS", "2500"))  # cap rows fetched to limit runtime
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "150"))  # RF complexity
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "10"))

MIN_ROWS_FOR_TRAIN = int(os.getenv("MIN_ROWS_FOR_TRAIN", "60"))  # minimal rows required


def safe_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def fetch_history_yf(ticker: str, max_rows: int = MAX_ROWS) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV via yfinance, return a DataFrame with lowercase columns.
    """
    try:
        t = yf.Ticker(ticker)
        # Use 'max' to get as much history as possible, then trim
        hist = t.history(period="max", auto_adjust=False)
        if hist is None or hist.empty:
            print(f"[{ticker}] yfinance returned no history.")
            return None
        # Keep standard columns Open, High, Low, Close, Volume if present
        # Reset index to integer index for consistent processing
        hist = hist.reset_index()
        # Ensure 'Close' present
        if "Close" not in hist.columns and "close" not in hist.columns:
            print(f"[{ticker}] history missing 'Close' column.")
            return None
        # limit rows
        if hist.shape[0] > max_rows:
            hist = hist.tail(max_rows).reset_index(drop=True)
        # normalize column names to lowercase
        hist.columns = [c.lower() for c in hist.columns]
        if "close" not in hist.columns:
            print(f"[{ticker}] normalized history missing 'close'.")
            return None
        return hist
    except Exception as e:
        print(f"[{ticker}] fetch_history_yf error: {e}")
        return None


def compute_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a 'close' column, compute feature columns identical to modeling.
    """
    dfc = df.copy()
    s = dfc["close"].astype(float)

    dfc["ret1"] = s.pct_change(1)
    dfc["ret3"] = s.pct_change(3)
    dfc["ma5"] = s.rolling(window=5, min_periods=1).mean()
    dfc["ma10"] = s.rolling(window=10, min_periods=1).mean()
    dfc["ma20"] = s.rolling(window=20, min_periods=1).mean()
    dfc["vol10"] = s.pct_change(1).rolling(window=10, min_periods=1).std().fillna(0)

    # RSI14
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    dfc["rsi14"] = rsi.fillna(50)

    # fill NaNs
    dfc = dfc.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return dfc


def prepare_X_y(df_feat: pd.DataFrame, horizon: int):
    """
    Prepare X (features) and y (target shifted -horizon) arrays.
    """
    df = df_feat.copy()
    df["target"] = df["close"].shift(-horizon)
    df = df.dropna(subset=["target"])
    if df.shape[0] < MIN_ROWS_FOR_TRAIN:
        return None, None
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y


def train_and_save_model_for(ticker_safe: str, interval: str, X, y) -> bool:
    """
    Train RandomForest on X,y and save to models dir with consistent filename.
    """
    try:
        # Time-series split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, n_jobs=1, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        fname = f"model_{ticker_safe}_{interval}.joblib"
        path = os.path.join(MODELS_DIR, fname)
        joblib.dump(model, path)
        print(f"[{ticker_safe} {interval}] trained. MAE={mae:.4f}. saved -> {path}")
        return True
    except Exception as e:
        print(f"[{ticker_safe} {interval}] training error: {e}")
        return False


def train_for_ticker(ticker: str):
    """
    Fetch history, compute features, and train models for each horizon for this ticker.
    """
    print(f"=== Starting training for {ticker} ===")
    hist = fetch_history_yf(ticker)
    if hist is None or hist.shape[0] < MIN_ROWS_FOR_TRAIN:
        print(f"[{ticker}] insufficient history rows ({0 if hist is None else hist.shape[0]}). Skipping.")
        return

    df_feat = compute_features_from_df(hist)
    ticker_safe = safe_filename(ticker)

    for interval, horizon in HORIZONS.items():
        X, y = prepare_X_y(df_feat, horizon)
        if X is None or y is None:
            print(f"[{ticker_safe} {interval}] not enough rows for horizon {horizon}. Skipping.")
            continue
        # limit rows for speed
        if X.shape[0] > MAX_ROWS:
            X = X[-MAX_ROWS:]
            y = y[-MAX_ROWS:]
        print(f"[{ticker_safe} {interval}] training using {X.shape[0]} samples...")
        ok = train_and_save_model_for(ticker_safe, interval, X, y)
        if not ok:
            print(f"[{ticker_safe} {interval}] failed training.")
        # avoid aggressive fetches
        time.sleep(0.5)


def main():
    print("train_all.py (auto-fetch) starting...")
    print("Tickers to train:", TICKERS)
    for t in TICKERS:
        try:
            train_for_ticker(t)
        except Exception as e:
            print(f"[{t}] unexpected error during training loop: {e}")
    print("train_all.py finished.")


if __name__ == "__main__":
    main()
