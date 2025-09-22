from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, logging

# import providers and modeling
from providers import fetch_data
import modeling

app = FastAPI(title="Fingrow-Hybrid API v2.0")

# Allow all origins for now; replace "*" with your Vercel URL for production
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("fingrow")

# ---------------- Root & health ----------------
@app.get("/")
def root():
    return {
        "service": "Fingrow-Hybrid backend",
        "version": "v2.0",
        "status": "ok"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Prediction request ----------------
class PredictRequest(BaseModel):
    symbol: str
    interval: str  # '3-15d','1-3m','3-6m','1-3y'
    indianapi_key: Optional[str] = None  # new key name

@app.post("/model/predict")
def predict(req: PredictRequest):
    symbol = req.symbol.strip().upper()
    interval = req.interval.strip()
    ind_key = req.indianapi_key or None

    # default NA response
    NA = lambda: "NA"
    out = {
        "symbol": symbol,
        "current_price": NA(),
        "predicted_price": NA(),
        "implied_return_pct": NA(),
        "confidence_pct": NA(),
        "momentum_pct": NA(),
        "fundamentals_score": NA(),
        "recommendation": {
            "action": NA(),
            "target_price": NA(),
            "explanation": NA()
        },
        "provider": "NA"
    }

    # Step 1: Fetch data
    try:
        q = fetch_data(symbol, indianapi_key=ind_key)
    except Exception as e:
        logger.exception("fetch_data error")
        q = None

    if not q or q.get("current_price") is None:
        return out

    out["current_price"] = float(q.get("current_price"))
    out["provider"] = q.get("provider") or "NA"
    out["momentum_pct"] = q.get("momentum_pct") if q.get("momentum_pct") is not None else "NA"
    out["fundamentals_score"] = q.get("fundamentals_score") if q.get("fundamentals_score") is not None else "NA"

    # Step 2: Prediction
    try:
        model = modeling.load_model(interval)
        if model is None:
            return out

        pred_price, conf = modeling.predict_with_model(
            model,
            {"current_price": out["current_price"], "history": q.get("history")}
        )

        out["predicted_price"] = pred_price if pred_price is not None else "NA"
        if pred_price is not None:
            out["implied_return_pct"] = round(
                ((pred_price - out["current_price"]) / out["current_price"]) * 100.0,
                3
            )
        out["confidence_pct"] = conf if conf is not None else "NA"
    except FileNotFoundError:
        return out
    except Exception:
        logger.exception("prediction error")
        return out

    # Step 3: Recommendation
    try:
        impl = out["implied_return_pct"]
        rec = {"action": "NA", "target_price": "NA", "explanation": "NA"}
        if impl != "NA":
            implf = float(impl)
            if implf >= 10.0:
                rec["action"] = "Buy"
                rec["target_price"] = out["predicted_price"]
                rec["explanation"] = f"Predicted upside {implf:.2f}%. Suggest buying if it fits your risk profile."
            elif implf >= 5.0:
                rec["action"] = "Consider Buy"
                rec["target_price"] = out["predicted_price"]
                rec["explanation"] = f"Moderate upside {implf:.2f}%. You may accumulate gradually and monitor fundamentals."
            elif implf >= -5.0:
                rec["action"] = "Hold"
                rec["target_price"] = out["predicted_price"]
                rec["explanation"] = f"Predicted change {implf:.2f}%. Best to hold and re-evaluate; consider stop-loss if downside risk matters."
            else:
                rec["action"] = "Sell"
                rec["target_price"] = out["predicted_price"]
                rec["explanation"] = f"Predicted downside {implf:.2f}%. Consider selling to limit losses or set a tight stop-loss."
        out["recommendation"] = rec
    except Exception:
        pass

    return out
