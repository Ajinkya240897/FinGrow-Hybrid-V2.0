# backend/main.py (DEBUG version)
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import providers
import modeling
import importlib

app = FastAPI(title="Fingrow-Hybrid V2.0 (debug)")

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/debug/provider")
async def debug_provider(request: Request):
    """
    Diagnostic endpoint — POST JSON { "symbol":"TCS", "indianapi_key": "..." }
    Returns: providers.fetch_data raw result + extra diagnostics:
      - yfinance_available (bool)
      - history_present (bool)
      - history_counts (per column)
      - small sample of history (first/last up to 3 rows)
      - raw_provider_payload
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    symbol = str(payload.get("symbol","")).strip().upper()
    indianapi_key = payload.get("indianapi_key", None)
    if not symbol:
        raise HTTPException(status_code=400, detail="Missing 'symbol' in body")

    # check yfinance import availability
    try:
        import yfinance as yf  # noqa
        yfinance_available = True
    except Exception as e:
        yfinance_available = False

    resp = {"symbol": symbol, "yfinance_available": yfinance_available, "provider_result": None, "error": None}
    try:
        q = providers.fetch_data(symbol, indianapi_key=indianapi_key)
        resp["provider_result"] = q
        # analyze history
        history = None
        if q:
            history = q.get("history")
        if history and isinstance(history, dict):
            counts = {}
            for k, mp in history.items():
                try:
                    counts[k] = len(mp) if isinstance(mp, dict) else 0
                except Exception:
                    counts[k] = 0
            resp["history_present"] = any(v>0 for v in counts.values())
            resp["history_counts"] = counts
            # sample first/last 3 dates of close
            sample = {"first": None, "last": None}
            try:
                if "Close" in history and isinstance(history["Close"], dict) and len(history["Close"])>0:
                    items = sorted(history["Close"].items(), key=lambda kv: kv[0])
                    first = items[:3]
                    last = items[-3:]
                    sample["first"] = first
                    sample["last"] = last
            except Exception:
                sample["first"] = None; sample["last"] = None
            resp["history_sample"] = sample
        else:
            resp["history_present"] = False
            resp["history_counts"] = {}
            resp["history_sample"] = {"first":None,"last":None}
    except Exception as e:
        resp["error"] = str(e)
        resp["traceback"] = traceback.format_exc()
    return resp

# Original predict endpoint (keeps diagnostics)
@app.post("/model/predict")
async def predict(request: Request):
    NA = lambda: "NA"
    out = {
        "symbol": NA(), "interval": NA(), "current_price": NA(),
        "predicted_price": NA(), "implied_return_pct": NA(), "confidence_pct": NA(),
        "momentum_pct": NA(), "fundamentals_score": NA(),
        "recommendation": {"action":NA(),"target_price":NA(),"explanation":NA()},
        "provider": NA(), "model_status": NA(), "prediction_source": NA()
    }

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    symbol = str(payload.get("symbol","")).strip().upper()
    interval = str(payload.get("interval","")).strip()
    indianapi_key = payload.get("indianapi_key", None)
    out["symbol"] = symbol; out["interval"] = interval

    if not symbol or not interval:
        raise HTTPException(status_code=400, detail="Missing 'symbol' or 'interval'")

    q = None
    try:
        q = providers.fetch_data(symbol, indianapi_key=indianapi_key)
    except Exception:
        q = None

    if not q or q.get("current_price") is None:
        out["model_status"] = "No current_price from providers"
        return out

    out["current_price"] = q.get("current_price")
    out["provider"] = q.get("provider") or "NA"
    out["momentum_pct"] = q.get("momentum_pct") or "NA"
    out["fundamentals_score"] = q.get("fundamentals_score") or "NA"

    try:
        resolved = q.get("resolved_symbol") or symbol
        model, model_reason = modeling.load_or_train_model(resolved, interval, history_dict=q.get("history"))
        out["model_status"] = model_reason or "No model_reason"
    except Exception as e:
        model = None
        out["model_status"] = f"Exception in load_or_train_model: {e}"

    try:
        pred_price, conf, used_fallback, pred_reason = modeling.predict_with_model_or_fallback(
            model, {"current_price": out["current_price"], "history": q.get("history")}
        )
        out["prediction_source"] = "fallback" if used_fallback else ("model" if pred_price is not None else "none")
        out["model_status"] = (out.get("model_status","") or "") + f"; predict_step: {pred_reason}"
    except Exception as e:
        traceback.print_exc()
        pred_price = None
        conf = None
        out["prediction_source"] = "error"
        out["model_status"] = (out.get("model_status","") or "") + f"; predict exception: {e}"

    if pred_price is None:
        return out

    out["predicted_price"] = round(pred_price,4)
    try:
        out["implied_return_pct"] = round(100.0*(out["predicted_price"] - out["current_price"])/out["current_price"],3)
    except Exception:
        out["implied_return_pct"] = "NA"
    out["confidence_pct"] = round(conf,2) if conf is not None else "NA"

    try:
        impl = out["implied_return_pct"]
        rec = {"action":"NA","target_price":out["predicted_price"],"explanation":"NA"}
        if impl != "NA":
            implf = float(impl)
            if implf >= 10.0:
                rec["action"]="Buy"; rec["explanation"]=f"Predicted upside {implf:.2f}%"
            elif implf >= 5.0:
                rec["action"]="Consider Buy"; rec["explanation"]=f"Moderate upside {implf:.2f}%"
            elif implf >= -5.0:
                rec["action"]="Hold"; rec["explanation"]=f"Predicted change {implf:.2f}%"
            else:
                rec["action"]="Sell"; rec["explanation"]=f"Predicted downside {implf:.2f}%"
        out["recommendation"] = rec
    except Exception:
        pass

    return out
