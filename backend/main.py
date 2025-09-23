# backend/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, logging, json

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
# Keep Pydantic model for docs, but we'll parse the raw body ourselves to be tolerant.
class PredictRequest(BaseModel):
    symbol: str
    interval: str  # '3-15d','1-3m','3-6m','1-3y'
    indianapi_key: Optional[str] = None  # new key name


async def _parse_json_tolerant(request: Request):
    """
    Read request body and attempt to parse JSON.
    - First, try standard json.loads
    - If that fails with JSONDecodeError ("Extra data"), attempt to raw_decode the first JSON object
      and return that object (ignoring trailing garbage).
    Returns parsed dict on success or raises HTTPException(400) on failure.
    """
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(status_code=400, detail="Empty request body")
    text = None
    try:
        text = body_bytes.decode('utf-8')
    except Exception:
        # fallback: try latin-1 decode
        try:
            text = body_bytes.decode('latin-1')
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode request body as text")

    # Try normal parse
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            # if parsed is not an object, reject
            raise HTTPException(status_code=400, detail="Expected JSON object in request body")
        return parsed
    except json.JSONDecodeError as e:
        # Attempt tolerant parse: extract first JSON object using JSONDecoder.raw_decode
        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(text)
            if isinstance(obj, dict):
                return obj
            else:
                raise HTTPException(status_code=400, detail="First JSON value is not an object")
        except Exception as e2:
            # Could not recover
            logger.exception("JSON parse failed", exc_info=True)
            raise HTTPException(status_code=400, detail=f"JSON decode error: {str(e)}")


@app.post("/model/predict")
async def predict(request: Request):
    """
    Tolerant predict endpoint: parses JSON tolerantly and then proceeds with normal flow.
    Returns safe 'NA' fields on any failure.
    """
    # default NA response
    NA = lambda: "NA"
    out = {
        "symbol": NA(),
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

    # Parse JSON tolerantly
    try:
        payload = await _parse_json_tolerant(request)
    except HTTPException as he:
        # return FastAPI JSON error for client; frontend should show a friendly message
        raise he
    except Exception as e:
        logger.exception("Unexpected parse error")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Validate expected fields
    symbol = payload.get("symbol", "")
    interval = payload.get("interval", "")
    ind_key = payload.get("indianapi_key", None)

    if not symbol or not interval:
        # Return 400: missing fields
        raise HTTPException(status_code=400, detail="Missing required fields 'symbol' and/or 'interval'")

    symbol = str(symbol).strip().upper()
    interval = str(interval).strip()

    out["symbol"] = symbol

    # Step 1: Fetch data
    try:
        q = fetch_data(symbol, indianapi_key=ind_key)
    except Exception:
        logger.exception("fetch_data error")
        q = None

    if not q or q.get("current_price") is None:
        # safe NA response
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
            try:
                out["implied_return_pct"] = round(
                    ((pred_price - out["current_price"]) / out["current_price"]) * 100.0,
                    3
                )
            except Exception:
                out["implied_return_pct"] = "NA"
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
