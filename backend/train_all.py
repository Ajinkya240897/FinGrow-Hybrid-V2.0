"""
train_all.py (enhanced)

Trains RandomForest models per horizon using richer features:
- returns (1,3)
- moving averages (5,10,20)
- volatility (rolling std)
- RSI(14)

Saves models as: backend/models/model_<interval>.joblib
Example intervals: '3-15d', '1-3m', '3-6m', '1-3y'
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "historical.csv")
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = {
    "3-15d": 10,
    "1-3m": 60,
    "3-6m": 120,
    "1-3y": 365
}

def compute_features(df):
    """
    Expect df with at least ['open','high','low','close','volume'].
    Returns df with feature columns and aligned index.
    """
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df.get('high', df['close']), errors='coerce')
    df['low'] = pd.to_numeric(df.get('low', df['close']), errors='coerce')
    df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)

    # Returns
    df['ret1'] = df['close'].pct_change(1)
    df['ret3'] = df['close'].pct_change(3)

    # Moving averages
    df['ma5']  = df['close'].rolling(window=5, min_periods=1).mean()
    df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()

    # Volatility (rolling std of returns)
    df['vol10'] = df['ret1'].rolling(window=10, min_periods=1).std().fillna(0)

    # RSI(14) simple implementation
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df['rsi14'] = rsi.fillna(50)  # neutral value for insufficient data

    # Fill any remaining NA conservatively
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df

def prepare_dataset(df, days_ahead):
    """
    Creates X, y aligned for a horizon of days_ahead.
    X: features at time t, y: close at time t + days_ahead
    """
    df_feat = compute_features(df)
    df_feat['target'] = df_feat['close'].shift(-days_ahead)
    df_feat = df_feat.dropna(subset=['target'])
    feature_cols = ['ret1','ret3','ma5','ma10','ma20','vol10','rsi14']
    X = df_feat[feature_cols].iloc[:-0].copy()
    y = df_feat['target'].copy()
    # Align X to y: last rows where target exists are fine because we dropped NaN target
    # But ensure shapes match
    if len(X) > len(y):
        X = X.iloc[:len(y), :]
    elif len(y) > len(X):
        y = y.iloc[:len(X)]
    return X, y

def train_and_save(df, interval_name, days, random_search=False):
    X, y = prepare_dataset(df, days)
    if len(X) < 20:
        raise ValueError(f"Not enough samples to train for {interval_name}. Need >=20, got {len(X)}")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if random_search:
        param_dist = {
            'n_estimators': [100,200,300],
            'max_depth': [5,10,15,None],
            'min_samples_split': [2,5,10]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        rs = RandomizedSearchCV(model, param_dist, n_iter=6, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1, random_state=42)
        rs.fit(X, y)
        model = rs.best_estimator_
    else:
        model.fit(X, y)
    out_path = os.path.join(MODELS_DIR, f"model_{interval_name}.joblib")
    joblib.dump(model, out_path)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    return out_path, len(X), {'mae': float(mae), 'rmse': float(rmse)}

def main(random_search=False):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Historical CSV not found at {DATA_PATH}. Place your OHLCV CSV there.")
    print("Loading historical data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=True)
    summary = {}
    for interval_name, days in HORIZONS.items():
        try:
            print(f"\nTraining for {interval_name} (horizon {days} days)...")
            out_path, n_samples, metrics = train_and_save(df, interval_name, days, random_search=random_search)
            summary[interval_name] = {'model_path': out_path, 'samples': n_samples, 'metrics': metrics}
            print(f"Saved model -> {out_path} | samples={n_samples} | MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            summary[interval_name] = {'error': str(e)}
            print(f"Failed to train {interval_name}: {e}")
    print("\nTraining summary:")
    for k, v in summary.items():
        print(k, "=>", v)

if __name__ == "__main__":
    # set random_search=True to enable hyperparameter tuning (slower)
    main(random_search=False)
