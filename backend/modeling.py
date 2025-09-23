# backend/modeling.py
"""
Robust modeling helper.

- load_model(interval) -> loads joblib model from backend/models/model_<interval>.joblib
- predict_with_model(model, data) -> tries to build feature vector from data['history'] or fallbacks.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_model(interval: str) -> Optional[Any]:
    filename = f"model_{interval}.joblib"
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _compute_features_from_history(history: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Build feature vector matching the enhanced train_all features:
    ['ret1','ret3','ma5','ma10','ma20','vol10','rsi14']
    history is expected in the pandas.DataFrame.to_dict() format (as used in providers.py).
    """
    try:
        if not history or "Close" not in history:
            return None
        # Reconstruct a pandas Series of Close values from the dict
        # history['Close'] is expected to be {ts: val, ...}
        closes = list(history["Close"].values())
        if len(closes) < 21:  # need at least 21 for MA20 + RSI window
            return None
        s = pd.Series(closes).astype(float)

        # features
        ret1 = s.pct_change(1).iloc[-1] if len(s) >= 2 else 0.0
        ret3 = s.pct_change(3).iloc[-1] if len(s) >= 4 else 0.0
        ma5  = s.rolling(window=5, min_periods=1).mean().iloc[-1]
        ma10 = s.rolling(window=10, min_periods=1).mean().iloc[-1]
        ma20 = s.rolling(window=20, min_periods=1).mean().iloc[-1]
        vol10 = s.pct_change(1).rolling(window=10, min_periods=1).std().iloc[-1] if len(s) >= 2 else 0.0

        # RSI14
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


def predict_with_model(model: Any, data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Attempt to predict using best-effort feature construction.

    Input 'data' should contain:
      - 'current_price' (float)
      - 'history' (optional) in pandas.DataFrame.to_dict() style (from yfinance)

    Returns (predicted_price, confidence_pct) or (None, None) on failure.
    """
    try:
        cur = data.get("current_price", None)
        history = data.get("history", None)

        # 1) Try to construct full feature vector from history (preferred)
        X = None
        if history:
            X = _compute_features_from_history(history)

        # 2) If that failed, try single-feature fallback [[current_price]]
        if X is None and cur is not None:
            X_try = np.array([[float(cur)]], dtype=float)
            # Test if model accepts shape (1,1)
            try:
                preds = model.predict(X_try)
                # If predict succeeds, compute confidence if possible
                pred_value = float(preds[0])
                conf = None
                if hasattr(model, "estimators_"):
                    # estimate spread among trees on X_try
                    tree_preds = np.array([est.predict(X_try)[0] for est in model.estimators_])
                    spread = float(tree_preds.max() - tree_preds.min())
                    conf = max(0.0, 100.0 - spread)
                return pred_value, conf
            except Exception:
                X = None  # fallthrough, will try other options

        # 3) If we have a full X from history, attempt predict
        if X is not None:
            try:
                preds = model.predict(X)
                pred_value = float(preds[0])
                conf = None
                if hasattr(model, "estimators_"):
                    tree_preds = np.array([est.predict(X)[0] for est in model.estimators_])
                    spread = float(tree_preds.max() - tree_preds.min())
                    conf = max(0.0, 100.0 - spread)
                return pred_value, conf
            except Exception:
                # prediction failed with full features; try fallback single-feature again (if cur exists)
                if cur is not None:
                    try:
                        X_try = np.array([[float(cur)]], dtype=float)
                        preds = model.predict(X_try)
                        pred_value = float(preds[0])
                        conf = None
                        if hasattr(model, "estimators_"):
                            tree_preds = np.array([est.predict(X_try)[0] for est in model.estimators_])
                            spread = float(tree_preds.max() - tree_preds.min())
                            conf = max(0.0, 100.0 - spread)
                        return pred_value, conf
                    except Exception:
                        return None, None
                return None, None

        # 4) No features available
        return None, None

    except Exception:
        return None, None
