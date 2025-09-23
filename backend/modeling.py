# backend/modeling.py
"""
On-demand modeling helper.

- load_or_train_model(resolved_symbol, interval, history_dict=None) -> model or None
- predict_with_model(model, data) -> (pred_price, confidence_pct)
"""

import os, joblib, numpy as np, pandas as pd
from typing import Any, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MIN_ROWS_FOR_TRAIN = 60
MAX_ROWS_TO_USE = 2000
MODEL_TYPE = os.getenv("MODEL_TYPE", "ridge")  # 'ridge' or 'rf'

def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

def _compute_features_from_history(history: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        if not history or "Close" not in history:
            return None
        closes = list(history["Close"].values())
        if len(closes) < 21:
            return None
        s = pd.Series(closes).astype(float)
        ret1 = s.pct_change(1).iloc[-1] if len(s) >= 2 else 0.0
        ret3 = s.pct_change(3).iloc[-1] if len(s) >= 4 else 0.0
        ma5  = s.rolling(window=5, min_periods=1).mean().iloc[-1]
        ma10 = s.rolling(window=10, min_periods=1).mean().iloc[-1]
        ma20 = s.rolling(window=20, min_periods=1).mean().iloc[-1]
        vol10 = s.pct_change(1).rolling(window=10, min_periods=1).std().iloc[-1] if len(s) >= 2 else 0.0
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14, min_periods=1).mean()
        roll_down = down.rolling(14, min_periods=1).mean()
        rs = roll_up / (roll_down.replace(0, pd.NA))
        rsi = 100 - (100 / (1 + rs))
        rsi14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        feats = np.array([ret1, ret3, ma5, ma10, ma20, vol10, rsi14], dtype=float).reshape(1, -1)
        return feats
    except Exception:
        return None

def load_model(resolved_symbol: str, interval: str) -> Optional[Any]:
    fname = f"model_{_safe_filename(resolved_symbol)}_{interval}.joblib"
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

def save_model_for_symbol(model: Any, resolved_symbol: str, interval: str) -> Optional[str]:
    try:
        fname = f"model_{_safe_filename(resolved_symbol)}_{interval}.joblib"
        path = os.path.join(MODELS_DIR, fname)
        joblib.dump(model, path)
        print(f"[MODEL SAVED] {path}")
        return path
    except Exception:
        return None

def _compute_feature_df(history: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        if not history or "Close" not in history:
            return None
        closes = list(history["Close"].values())
        if len(closes) < 2:
            return None
        s = pd.Series(closes).astype(float)
        df = pd.DataFrame({"close": s})
        df["ret1"] = s.pct_change(1)
        df["ret3"] = s.pct_change(3)
        df["ma5"] = s.rolling(window=5, min_periods=1).mean()
        df["ma10"] = s.rolling(window=10, min_periods=1).mean()
        df["ma20"] = s.rolling(window=20, min_periods=1).mean()
        df["vol10"] = s.pct_change(1).rolling(window=10, min_periods=1).std().fillna(0)
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14, min_periods=1).mean()
        roll_down = down.rolling(14, min_periods=1).mean()
        rs = roll_up / (roll_down.replace(0, pd.NA))
        rsi = 100 - (100 / (1 + rs))
        df["rsi14"] = rsi.fillna(50)
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        return df
    except Exception:
        return None

def _prepare_X_y_from_df(df_feat: pd.DataFrame, days_ahead: int):
    try:
        df = df_feat.copy()
        df["target"] = df["close"].shift(-days_ahead)
        df = df.dropna(subset=["target"])
        if df.shape[0] < MIN_ROWS_FOR_TRAIN:
            return None, None
        feature_cols = ["ret1", "ret3", "ma5", "ma10", "ma20", "vol10", "rsi14"]
        X = df[feature_cols].values
        y = df["target"].values
        if X.shape[0] > MAX_ROWS_TO_USE:
            X = X[-MAX_ROWS_TO_USE:]
            y = y[-MAX_ROWS_TO_USE:]
        return X, y
    except Exception:
        return None, None

def train_on_demand_and_get_model(resolved_symbol: str, interval: str, history_dict = None):
    """
    Train a small model (Ridge by default) using history_dict (pandas-like dict).
    """
    try:
        mapping = {"3-15d": 10, "1-3m": 60, "3-6m": 120, "1-3y": 365}
        days = mapping.get(interval)
        if days is None:
            return None

        if not history_dict:
            # if no history passed, we can't fetch (providers should have provided history)
            return None

        df_feat = _compute_feature_df(history_dict)
        if df_feat is None:
            return None
        X, y = _prepare_X_y_from_df(df_feat, days)
        if X is None or y is None:
            return None

        if MODEL_TYPE == "rf":
            model = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42)
        else:
            model = Ridge(alpha=1.0)
        model.fit(X, y)
        save_model_for_symbol(model, resolved_symbol, interval)
        return model
    except Exception:
        return None

def load_or_train_model(resolved_symbol: str, interval: str, history_dict = None):
    try:
        m = load_model(resolved_symbol, interval)
        if m is not None:
            return m
        m2 = train_on_demand_and_get_model(resolved_symbol, interval, history_dict=history_dict)
        return m2
    except Exception:
        return None

def predict_with_model(model: Any, data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    try:
        if model is None:
            return None, None
        cur = data.get("current_price", None)
        history = data.get("history", None)
        X = None
        if history:
            df_feat = _compute_feature_df(history)
            if df_feat is not None:
                last_row = df_feat.tail(1)
                cols = ["ret1", "ret3", "ma5", "ma10", "ma20", "vol10", "rsi14"]
                try:
                    X = last_row[cols].values.astype(float)
                except Exception:
                    X = None
        if X is None and cur is not None:
            X = np.array([[float(cur)]])
        if X is None:
            return None, None
        preds = model.predict(X)
        pred_value = float(preds[0])
        conf = None
        try:
            if hasattr(model, "estimators_"):
                tree_preds = np.array([est.predict(X)[0] for est in model.estimators_])
                spread = float(tree_preds.max() - tree_preds.min())
                conf = max(0.0, 100.0 - spread)
            else:
                conf = 60.0
        except Exception:
            conf = None
        return pred_value, conf
    except Exception:
        return None, None
