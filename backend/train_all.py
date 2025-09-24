# backend/train_all.py
"""
Optional batch trainer - auto-fetch history using yfinance and train per-ticker models.
Set TICKERS env var as comma-separated list to override default.
"""
import os, time, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = {"3-15d":10, "1-3m":60, "3-6m":120, "1-3y":365}
FEATURE_COLS = ["ret1","ret3","ma5","ma10","ma20","vol10","rsi14"]

TICKERS_ENV = os.getenv("TICKERS", "").strip()
DEFAULT_TICKERS = ["RELIANCE.NS","TCS.NS","INFY.NS"]
TICKERS = [t.strip() for t in (TICKERS_ENV.split(",") if TICKERS_ENV else DEFAULT_TICKERS) if t.strip()]
MAX_ROWS = int(os.getenv("MAX_ROWS", "2500"))

try:
    import yfinance as yf
except Exception:
    raise

def fetch_history(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="max", auto_adjust=False)
    if hist is None or hist.empty:
        return None
    hist = hist.reset_index()
    hist.columns = [c.lower() for c in hist.columns]
    if "close" not in hist.columns: return None
    return hist

def compute_features(df):
    s = df["close"].astype(float)
    df["ret1"] = s.pct_change(1)
    df["ret3"] = s.pct_change(3)
    df["ma5"] = s.rolling(5, min_periods=1).mean()
    df["ma10"] = s.rolling(10, min_periods=1).mean()
    df["ma20"] = s.rolling(20, min_periods=1).mean()
    df["vol10"] = s.pct_change(1).rolling(10, min_periods=1).std().fillna(0)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df["rsi14"] = rsi.fillna(50)
    return df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

def train_for_ticker(ticker):
    hist = fetch_history(ticker)
    if hist is None or hist.shape[0] < 60:
        print(f"Skipping {ticker} - insufficient history")
        return
    df = compute_features(hist)
    for interval, days in HORIZONS.items():
        df_local = df.copy()
        df_local["target"] = df_local["close"].shift(-days)
        df_local = df_local.dropna(subset=["target"])
        if df_local.shape[0] < 60: 
            print(f"[{ticker} {interval}] not enough rows")
            continue
        X = df_local[FEATURE_COLS].values
        y = df_local["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        fname = f"model_{ticker.replace('/','_')}_{interval}.joblib"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
        print(f"[{ticker} {interval}] saved model, MAE {mae:.4f}")

if __name__ == "__main__":
    for t in TICKERS:
        try:
            train_for_ticker(t)
        except Exception as e:
            print("Train error", e)
