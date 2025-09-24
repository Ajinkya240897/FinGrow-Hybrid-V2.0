# backend/modeling.py
import os, joblib, numpy as np, pandas as pd, traceback
from typing import Any, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# More permissive defaults to reduce NA; keep env override
MIN_ROWS_FOR_TRAIN = int(os.getenv("MIN_ROWS_FOR_TRAIN", "30"))
MIN_ROWS_FOR_FAST_TRAIN = int(os.getenv("MIN_ROWS_FOR_FAST_TRAIN", "20"))
MAX_ROWS_TO_USE = int(os.getenv("MAX_ROWS_TO_USE", "2000"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "ridge")  # 'ridge' or 'rf'

def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

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
        traceback.print_exc()
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
        traceback.print_exc()
        return None, None

def _prepare_X_y_from_df_min(df_feat: pd.DataFrame, days_ahead: int):
    try:
        df = df_feat.copy()
        df["target"] = df["close"].shift(-days_ahead)
        df = df.dropna(subset=["target"])
        if df.shape[0] < MIN_ROWS_FOR_FAST_TRAIN:
            return None, None
        feature_cols = ["ret1", "ret3", "ma5", "ma10", "ma20", "vol10", "rsi14"]
        X = df[feature_cols].values
        y = df["target"].values
        if X.shape[0] > MAX_ROWS_TO_USE:
            X = X[-MAX_ROWS_TO_USE:]
            y = y[-MAX_ROWS_TO_USE:]
        return X, y
    except Exception:
        traceback.print_exc()
        return None, None

def _model_path(resolved_symbol: str, interval: str) -> str:
    fname = f"model_{_safe_filename(resolved_symbol)}_{interval}.joblib"
    return os.path.join(MODELS_DIR, fname)

def load_model(resolved_symbol: str, interval: str) -> Optional[Any]:
    path = _model_path(resolved_symbol, interval)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        traceback.print_exc()
        return None

def save_model_for_symbol(model: Any, resolved_symbol: str, interval: str) -> Optional[str]:
    try:
        path = _model_path(resolved_symbol, interval)
        joblib.dump(model, path)
        print(f"[MODEL SAVED] {path}")
        return path
    except Exception:
        traceback.print_exc()
        return None

def train_on_demand_and_get_model(resolved_symbol: str, interval: str, history_dict = None) -> Tuple[Optional[Any], str]:
    try:
        mapping = {"3-15d": 10, "1-3m": 60, "3-6m": 120, "1-3y": 365}
        days = mapping.get(interval)
        if days is None:
            return None, f"Unknown interval {interval}"
        if not history_dict:
            return None, "No history provided to train_on_demand"
        df_feat = _compute_feature_df(history_dict)
        if df_feat is None:
            return None, "Could not compute feature dataframe from history"
        X, y = _prepare_X_y_from_df(df_feat, days)
        if X is None or y is None:
            X2, y2 = _prepare_X_y_from_df_min(df_feat, days)
            if X2 is None or y2 is None:
                return None, f"Insufficient rows: have {len(df_feat)} rows; need >= {MIN_ROWS_FOR_TRAIN} for robust train"
            else:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(X2, y2)
                    save_model_for_symbol(model, resolved_symbol, interval)
                    return model, "Trained fallback Ridge model on limited data (low-confidence)"
                except Exception as e:
                    traceback.print_exc()
                    return None, f"Fallback Ridge training failed: {e}"
        try:
            if MODEL_TYPE == "rf":
                model = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42)
            else:
                model = Ridge(alpha=1.0)
            model.fit(X, y)
            save_model_for_symbol(model, resolved_symbol, interval)
            return model, "Trained model successfully"
        except Exception as e:
            traceback.print_exc()
            return None, f"Training failed: {e}"
    except Exception as e:
        traceback.print_exc()
        return None, f"Unexpected exception in training: {e}"

def load_or_train_model(resolved_symbol: str, interval: str, history_dict = None) -> Tuple[Optional[Any], str]:
    try:
        m = load_model(resolved_symbol, interval)
        if m is not None:
            return m, "Loaded cached model"
        m2, reason = train_on_demand_and_get_model(resolved_symbol, interval, history_dict=history_dict)
        return m2, reason
    except Exception as e:
        traceback.print_exc()
        return None, f"Error in load_or_train_model: {e}"

def _simple_momentum_fallback(history: Dict[str, Any], current_price: float) -> Tuple[Optional[float], Optional[float], str]:
    try:
        if not history or "Close" not in history:
            return None, None, "No history for momentum fallback"
        closes_map = history["Close"]
        if not closes_map:
            return None, None, "No close points for momentum fallback"
        items = sorted(closes_map.items(), key=lambda kv: kv[0])
        if len(items) < 2:
            return None, None, "Not enough close points for momentum fallback"
        # use very recent returns
        recent = [v for (_,v) in items[-5:] if v is not None]
        if len(recent) < 2:
            return None, None, "Not enough recent points for momentum fallback"
        rets = []
        for i in range(1, len(recent)):
            if recent[i-1] != 0:
                rets.append((recent[i]-recent[i-1])/recent[i-1])
        if not rets:
            return None, None, "Zero returns -> cannot estimate momentum"
        mean_ret = float(np.mean(rets))
        capped = max(min(mean_ret, 0.05), -0.05)
        pred = float(current_price) * (1.0 + capped)
        conf = min(50.0, 10.0 + len(recent) * 4.0)
        return round(pred, 4), round(conf, 2), f"Momentum fallback applied (mean_ret={mean_ret:.6f}, capped={capped:.4f})"
    except Exception:
        traceback.print_exc()
        return None, None, "Exception during momentum fallback"

def _fundamentals_based_fallback(current_price: float, fundamentals_score: Optional[int], interval: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Conservative fallback when there is no history/model but we have current_price and fundamentals_score.
    We map interval -> base expected return and scale by fundamentals_score (50 means neutral).
    """
    try:
        if current_price is None:
            return None, None, "No current_price for fundamentals fallback"
        mapping = {"3-15d": 0.005, "1-3m": 0.02, "3-6m": 0.05, "1-3y": 0.18}
        base = mapping.get(interval, 0.02)
        # default fundamentals neutral at 50 -> scale factor 1.0
        try:
            fs = float(fundamentals_score) if fundamentals_score is not None else 50.0
        except Exception:
            fs = 50.0
        # scale: each 10 points above 50 adds 10% to expected return
        scale = 1.0 + ((fs - 50.0) / 100.0)
        scale = max(0.5, min(scale, 1.8))  # limit scaling
        pred = current_price * (1.0 + base * scale)
        # low confidence for this heuristic; longer horizons slightly higher conf
        conf = 10.0 + min(40.0, base * 100.0 * (1.0 + (fs-50.0)/50.0))
        conf = round(max(5.0, min(conf, 60.0)), 2)
        return round(pred, 4), conf, f"Fundamentals fallback (base={base}, fs={fs}, scale={scale:.3f})"
    except Exception:
        traceback.print_exc()
        return None, None, "Exception in fundamentals fallback"

def predict_with_model_or_fallback(model: Any, data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], bool, str]:
    """
    data must include:
      - current_price (float)
      - history (dict or None)
      - fundamentals_score (optional)
      - interval (optional) -> use heuristic if no model
    Returns: (predicted_price, confidence_pct, used_fallback_bool, reason_str)
    """
    try:
        cur = data.get("current_price", None)
        history = data.get("history", None)
        interval = data.get("interval", None)
        fundamentals_score = data.get("fundamentals_score", None)

        # 1) If model exists, attempt to predict using latest features
        if model is not None:
            X = None
            if history:
                df_feat = _compute_feature_df(history)
                if df_feat is not None and df_feat.shape[0] >= 1:
                    last_row = df_feat.tail(1)
                    cols = ["ret1", "ret3", "ma5", "ma10", "ma20", "vol10", "rsi14"]
                    try:
                        X = last_row[cols].values.astype(float)
                    except Exception:
                        X = None
            if X is None and cur is not None:
                # try to craft a simple feature vector using current price proxies
                try:
                    X = np.zeros((1,7))
                    X[0,0] = 0.0
                    X[0,1] = 0.0
                    X[0,2] = cur
                    X[0,3] = cur
                    X[0,4] = cur
                    X[0,5] = 0.0
                    X[0,6] = 50.0
                except Exception:
                    X = None
            if X is not None:
                try:
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
                        conf = 50.0
                    return pred_value, conf, False, "Model-based prediction"
                except Exception:
                    traceback.print_exc()

        # 2) If we have history, try momentum fallback (short-term)
        if history and cur is not None:
            pred, conf, reason = _simple_momentum_fallback(history, cur)
            if pred is not None:
                return pred, conf, True, reason

        # 3) If no history but we have fundamentals and current price, use fundamentals fallback
        if cur is not None:
            pred_fund, conf_fund, reason_fund = _fundamentals_based_fallback(cur, fundamentals_score, interval or "1-3m")
            if pred_fund is not None:
                return pred_fund, conf_fund, True, reason_fund

        # 4) final generic fallback: tiny drift (very low confidence)
        if cur is not None:
            pred = cur * 1.01
            return round(pred,4), 5.0, True, "Generic conservative fallback (1% drift)"

        return None, None, False, "No model, no history, no cur price"
    except Exception:
        traceback.print_exc()
        return None, None, False, "Unexpected exception in predict_with_model_or_fallback"
