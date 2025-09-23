# backend/main.py
import traceback, json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import providers
import modeling

app = FastAPI(title="Fingrow-Hybrid V2.0")

# In production restrict origins to your frontend site for security
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"service": "Fingrow-Hybrid backend", "version": "v2.0", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/model/predict")
async def predict(request: Request):
    """
    Accepts JSON:
    { "symbol": "RELIANCE", "interval": "3-15d", "indianapi_key": "<optional-key>" }

    Returns:
      - current_price
      - predicted_price
      - implied_return_pct
      - confidence_pct
      - momentum_pct
      - fundamentals_score
      - recommendation (action, target_price, explanation)
    """
    # safe NA helper
    NA = lambda: "NA"
    out = {
        "symbol": NA(),
        "interval": NA(),
        "current_price": NA(),
        "predicted_price": NA(),
        "implied_return_pct": NA(),
        "confidence_pct": NA(),
        "momentum_pct": NA(),
        "fundamentals_score": NA(),
        "recommendation": {"action": NA(), "target_price": NA(), "explanation": NA()},
        "provider": "NA"
    }

    # tolerant parse — get JSON body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    symbol = str(payload.get("symbol", "")).strip().upper()
    interval = str(payload.get("interval", "")).strip()
    indianapi_key = payload.get("indianapi_key", None)

    out["symbol"] = symbol
    out["interval"] = interval

    if not symbol or not interval:
        raise HTTPException(status_code=400, detail="Missing 'symbol' or 'interval'")

    # Step 1: fetch data (IndianAPI primary, yfinance fallback)
    try:
        q = providers.fetch_data(symbol, indianapi_key=indianapi_key)
    except Exception:
        q = None

    if not q or q.get("current_price") is None:
        # safe NA output when no live price
        return out

    out["current_price"] = float(q.get("current_price"))
    out["provider"] = q.get("provider") or "NA"
    out["momentum_pct"] = q.get("momentum_pct") if q.get("momentum_pct") is not None else "NA"
    out["fundamentals_score"] = q.get("fundamentals_score") if q.get("fundamentals_score") is not None else "NA"

    # Step 2: load or train model on-demand & predict
    try:
        resolved = q.get("resolved_symbol") or symbol
        model = modeling.load_or_train_model(resolved, interval, history_dict=q.get("history"))
        if model is None:
            return out

        pred_price, conf = modeling.predict_with_model(model, {
            "current_price": out["current_price"],
            "history": q.get("history")
        })

        if pred_price is None:
            return out

        out["predicted_price"] = round(pred_price, 2)
        try:
            out["implied_return_pct"] = round(100.0 * (out["predicted_price"] - out["current_price"]) / out["current_price"], 3)
        except Exception:
            out["implied_return_pct"] = "NA"
        out["confidence_pct"] = round(conf, 2) if conf is not None else "NA"

    except Exception:
        traceback.print_exc()
        return out

    # Step 3: beginner-friendly recommendation
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
