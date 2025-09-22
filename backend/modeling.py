"""
modeling.py

Provides functions to load models and predict using them.
Models are saved in backend/models/ as joblib files per interval.
"""

import os
import joblib
import numpy as np
from typing import Any, Dict, Tuple, Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def load_model(interval: str) -> Optional[Any]:
    """
    Load a trained model for a given interval.
    Interval values are expected like: '3-15d', '1-3m', '3-6m', '1-3y'
    """
    filename = f"model_{interval}.joblib"
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception:
        return None


def predict_with_model(model: Any, data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Run prediction with a loaded model.
    - model: a trained scikit-learn model
    - data: dict returned by providers.fetch_data

    Returns:
      (predicted_price, confidence_pct)
    """
    try:
        current_price = data.get("current_price")
        if current_price is None:
            return None, None

        X = np.array([[current_price]])  # very simple feature placeholder
        y_pred = model.predict(X)

        # Estimate confidence as inverse of variance of trees (if RandomForest)
        conf = None
        if hasattr(model, "estimators_"):
            preds = np.array([est.predict(X)[0] for est in model.estimators_])
            if preds.size > 0:
                spread = preds.max() - preds.min()
                conf = max(0, 100 - spread)  # heuristic confidence %

        return float(y_pred[0]), float(conf) if conf is not None else None
    except Exception:
        return None, None
