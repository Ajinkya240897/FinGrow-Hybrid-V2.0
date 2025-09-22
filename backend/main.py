from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from providers import fetch_data  # your AlphaVantage + yfinance provider module
from modeling import load_model, predict_with_model  # ML helpers

# ----------------------------------------------------
# App initialization
# ----------------------------------------------------
app = FastAPI(title="Fingrow-Hybrid API v2.0")

# --- CORS middleware ---
origins = [
    "https://fingrow-hybrid-v2-0.vercel.app"  # ✅ replace with your actual Vercel frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Request/response schemas
# ----------------------------------------------------
class PredictRequest(BaseModel):
    symbol: str
    interval: str
    alpha_key: Optional[str] = None

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/model/predict")
def predict(req: PredictRequest):
    """
    Predicts stock price for given ticker + interval.
    Returns current price, predicted price, implied return %, confidence,
    momentum, fundamentals score, and beginner-friendly recommendation.
    """
    try:
        # Fetch current + historical data
        data = fetch_data(req.symbol, req.alpha_key)
        if not data or "current_price" not in data:
            return {
                "symbol": req.symbol,
                "current_price": "NA",
                "predicted_price": "NA",
                "implied_return_pct": "NA",
                "confidence_pct": "NA",
                "momentum_pct": "NA",
                "fundamentals_score": "NA",
                "recommendation": {
                    "action": "NA",
                    "target_price": "NA",
                    "explanation": "NA"
                },
                "provider": "NA"
            }

        current_price = data["current_price"]

        # Load correct model
        model = load_model(req.interval)
        if model is None:
            return {
                "symbol": req.symbol,
                "current_price": current_price,
                "predicted_price": "NA",
                "implied_return_pct": "NA",
                "confidence_pct": "NA",
                "momentum_pct": data.get("momentum_pct", "NA"),
                "fundamentals_score": data.get("fundamentals_score", "NA"),
                "recommendation": {
                    "action": "NA",
                    "target_price": "NA",
                    "explanation": "Model not trained"
                },
                "provider": data.get("provider", "NA")
            }

        # Run prediction
        prediction, conf = predict_with_model(model, data)

        implied_return = None
        if prediction and current_price and prediction != "NA":
            implied_return = round(((prediction - current_price) / current_price) * 100, 2)

        # Beginner-friendly recommendation logic
        if prediction == "NA":
            rec = {"action": "NA", "target_price": "NA", "explanation": "Prediction unavailable"}
        else:
            if implied_return is not None and implied_return > 5:
                rec = {"action": "Buy", "target_price": prediction,
                       "explanation": "Predicted growth and positive fundamentals suggest buying."}
            elif implied_return is not None and implied_return < -5:
                rec = {"action": "Sell", "target_price": prediction,
                       "explanation": "Predicted decline indicates you may want to sell."}
            else:
                rec = {"action": "Hold", "target_price": prediction,
                       "explanation": "Little expected movement — best to hold for now."}

        return {
            "symbol": req.symbol,
            "current_price": current_price,
            "predicted_price": prediction if prediction is not None else "NA",
            "implied_return_pct": implied_return if implied_return is not None else "NA",
            "confidence_pct": conf if conf is not None else "NA",
            "momentum_pct": data.get("momentum_pct", "NA"),
            "fundamentals_score": data.get("fundamentals_score", "NA"),
            "recommendation": rec,
            "provider": data.get("provider", "NA")
        }

    except Exception as e:
        # Fail-safe return (everything = NA)
        return {
            "symbol": req.symbol,
            "current_price": "NA",
            "predicted_price": "NA",
            "implied_return_pct": "NA",
            "confidence_pct": "NA",
            "momentum_pct": "NA",
            "fundamentals_score": "NA",
            "recommendation": {
                "action": "NA",
                "target_price": "NA",
                "explanation": f"Error: {str(e)}"
            },
            "provider": "NA"
        }

# ----------------------------------------------------
# Run locally (optional)
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
