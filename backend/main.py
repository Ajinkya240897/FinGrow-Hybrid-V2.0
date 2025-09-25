# backend/main.py
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import providers
import modeling

app = FastAPI(title="Fingrow-Hybrid V2.0 (final recommendations)")

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/model/predict")
async def predict(request: Request):
    NA = lambda: "NA"
    out = {
        "symbol": NA(), "interval": NA(), "current_price": NA(),
        "predicted_price": NA(), "implied_return_pct": NA(), "confidence_pct": NA(),
        "momentum_pct": NA(), "fundamentals_score": NA(),
        "recommendation": {"action":NA(),"target_price":NA(),"explanation":NA(),"buy_at":NA(),"sell_at":NA(),"stop_loss":NA()},
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
    out["momentum_pct"] = q.get("momentum_pct") if q.get("momentum_pct") is not None else "NA"
    out["fundamentals_score"] = q.get("fundamentals_score") if q.get("fundamentals_score") is not None else "NA"

    try:
        resolved = q.get("resolved_symbol") or symbol
        model, model_reason = modeling.load_or_train_model(resolved, interval, history_dict=q.get("history"))
        out["model_status"] = model_reason or "No model_reason"
    except Exception as e:
        model = None
        out["model_status"] = f"Exception in load_or_train_model: {e}"

    try:
        pred_price, conf, used_fallback, pred_reason = modeling.predict_with_model_or_fallback(
            model,
            {
                "current_price": out["current_price"],
                "history": q.get("history"),
                "fundamentals_score": q.get("fundamentals_score"),
                "interval": interval
            }
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

    # Build detailed recommendation (beginner-friendly)
    try:
        impl = out["implied_return_pct"]
        rec = {"action":"NA","target_price":out["predicted_price"],"explanation":"NA","buy_at":"NA","sell_at":"NA","stop_loss":"NA"}
        if impl != "NA":
            implf = float(impl)
            # thresholds for action
            if implf >= 10.0:
                rec["action"] = "Buy"
            elif implf >= 5.0:
                rec["action"] = "Consider Buy"
            elif implf >= -5.0:
                rec["action"] = "Hold"
            else:
                rec["action"] = "Sell"

            # compute buy/sell/stop guidance:
            # - buy_at = current * (1 - 0.02) for 'Buy' / (1 - 0.01) for 'Consider Buy' else NA
            # - sell_at = predicted_price
            # - stop_loss = current * (1 - 0.06) (conservative) or smaller for 'Consider Buy'
            cp = float(out["current_price"])
            tp = float(out["predicted_price"])
            if rec["action"] == "Buy":
                rec["buy_at"] = round(cp * 0.99, 2)
                rec["stop_loss"] = round(cp * 0.92, 2)
            elif rec["action"] == "Consider Buy":
                rec["buy_at"] = round(cp * 0.995, 2)
                rec["stop_loss"] = round(cp * 0.95, 2)
            elif rec["action"] == "Hold":
                rec["buy_at"] = "Only on dip"
                rec["stop_loss"] = round(cp * 0.94, 2)
            else:
                rec["buy_at"] = "No"
                rec["stop_loss"] = round(cp * 0.97, 2)

            rec["sell_at"] = round(tp, 2)

            # Build a beginner-friendly explanation string
            expl_parts = []
            expl_parts.append(f"Model predicts ~{implf:.2f}% change over the chosen horizon.")
            if out["prediction_source"] == "model":
                expl_parts.append("This prediction is model-backed (trained on historical OHLCV).")
            elif out["prediction_source"] == "fallback":
                expl_parts.append("This is a fallback prediction (from fundamentals or momentum) — treat it as lower confidence guidance.")
            if out["confidence_pct"] != "NA":
                expl_parts.append(f"Confidence: {out['confidence_pct']}%.")
            if out["momentum_pct"] != "NA":
                expl_parts.append(f"Momentum (recent trend): {out['momentum_pct']}%.")
            if out["fundamentals_score"] != "NA":
                expl_parts.append(f"Fundamentals score (0-100): {out['fundamentals_score']}. Higher is generally better.")
            # final practical guidance
            if rec["action"] in ["Buy", "Consider Buy"]:
                expl_parts.append(f"Suggested buy around {rec['buy_at']}, target {rec['sell_at']}. Use stop-loss ~{rec['stop_loss']} to limit downside.")
            elif rec["action"] == "Hold":
                expl_parts.append(f"Consider holding; set stop-loss around {rec['stop_loss']} and re-evaluate if trend weakens.")
            else:
                expl_parts.append(f"Consider selling if you are risk-averse; target exit ~{rec['sell_at']} or earlier if momentum turns negative.")
            rec["explanation"] = " ".join(expl_parts)

        out["recommendation"] = rec
    except Exception:
        pass

    return out
