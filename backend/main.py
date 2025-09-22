from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging, os, traceback, math
load_dotenv()
from providers import fetch_quote
import modeling
try:
    import yfinance as yf
except Exception:
    yf = None
app = FastAPI(title='Fingrow-Hybrid API v2.0')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fingrow')
class PredictRequest(BaseModel):
    symbol: str
    interval: str  # one of keys: 3-15d,1-3m,3-6m,1-3y
    alpha_key: str = None
@app.post('/model/predict')
async def predict(req: PredictRequest):
    symbol = req.symbol.strip().upper()
    interval = req.interval.strip()
    alpha_key = req.alpha_key or None
    # fetch current quote (alpha key can be provided by user)
    q = fetch_quote(symbol, alpha_key=alpha_key)
    current_price = q.get('price')
    # default NA structure
    NA = lambda: 'NA'
    out = {
        'symbol': symbol,
        'current_price': NA(),
        'predicted_price': NA(),
        'implied_return_pct': NA(),
        'confidence_pct': NA(),
        'momentum_pct': NA(),
        'fundamentals_score': NA(),
        'recommendation': {'action': NA(), 'target_price': NA(), 'explanation': NA()},
        'provider': q.get('provider') or 'NA'
    }
    if current_price is None:
        return out
    out['current_price'] = float(current_price)
    # compute momentum (14-day return) via yfinance history if available
    momentum = None
    try:
        if q.get('history') and isinstance(q.get('history'), dict):
            hist = q.get('history')
            # hist is pandas to_dict; try to get close series
            closes = None
            if 'Close' in hist:
                closes = list(hist['Close'].values())
            else:
                # fallback: use yfinance to fetch
                if yf is not None:
                    t = yf.Ticker(symbol)
                    dfhist = t.history(period='3mo')
                    if dfhist is not None and not dfhist.empty:
                        closes = dfhist['Close'].tolist()
            if closes and len(closes)>=2:
                momentum = (closes[-1]-closes[0])/closes[0]*100.0
    except Exception:
        momentum = None
    out['momentum_pct'] = ('NA' if momentum is None else round(momentum,3))
    # fundamentals score via yfinance info (PE, PB, dividend yield)
    fundamentals_score = None
    try:
        if yf is not None:
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t,'info') else {}
            pe = info.get('trailingPE') or info.get('trailing_pe') or None
            pb = info.get('priceToBook') or info.get('priceToBook') or None
            div = info.get('dividendYield') or info.get('dividend_yield') or None
            scores = []
            if pe is not None and pe>0:
                scores.append(max(0, min(1, 1.0/(pe/15.0))))  # ideal PE ~15
            if pb is not None and pb>0:
                scores.append(max(0, min(1, 1.0 - ((pb-2.0)/10.0))))  # ideal PB <=2
            if div is not None and div>0:
                scores.append(max(0, min(1, div*5)))  # scaled
            if scores:
                fundamentals_score = int(sum(scores)/len(scores)*100)
    except Exception:
        fundamentals_score = None
    out['fundamentals_score'] = ('NA' if fundamentals_score is None else fundamentals_score)
    # prediction using trained models per horizon
    try:
        horizon_map = modeling.HORIZON_MAP
        if interval not in horizon_map:
            return out
        days = horizon_map[interval]
        # prepare features: ret1, ret3, ma5, ma10 from latest history if available
        features = [0.0,0.0,out['current_price'],out['current_price']]
        # attempt to compute real features via yfinance
        try:
            if yf is not None:
                t = yf.Ticker(symbol)
                df = t.history(period='1y')
                if df is not None and not df.empty and 'Close' in df.columns:
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    df = df.dropna(subset=['Close'])
                    if len(df)>=10:
                        closes = df['Close']
                        ret1 = (closes.iloc[-1]-closes.iloc[-2])/closes.iloc[-2]
                        ret3 = (closes.iloc[-1]-closes.iloc[-4])/closes.iloc[-4]
                        ma5 = closes.rolling(window=5).mean().iloc[-1]
                        ma10 = closes.rolling(window=10).mean().iloc[-1]
                        features = [ret1, ret3, ma5, ma10]
        except Exception:
            pass
        pred, std = modeling.predict_for_horizon(features, days)
        out['predicted_price'] = round(float(pred),4)
        implied = (pred - out['current_price'])/out['current_price']*100.0
        out['implied_return_pct'] = round(float(implied),3)
        # confidence: map std/price to 0-100 (higher std -> lower confidence)
        try:
            rel_std = (std / pred) if pred and pred!=0 else 1.0
            confidence = max(0.0, min(100.0, 100.0 - rel_std*200.0))  # heuristic
            out['confidence_pct'] = round(confidence,2)
        except Exception:
            out['confidence_pct'] = 'NA'
    except FileNotFoundError:
        # model missing - return NA for prediction fields
        return out
    except Exception as e:
        logger.exception('prediction error')
        return out
    # recommendation logic
    try:
        impl = out['implied_return_pct']
        rec = {'action':'NA','target_price':'NA','explanation':'NA'}
        if impl == 'NA':
            out['recommendation'] = rec
            return out
        implf = float(impl)
        # thresholds (beginner-friendly): Buy if >=10%, Consider Buy 5-10, Hold  -5 to 5, Sell if <-5
        if implf >= 10.0:
            rec['action'] = 'Buy'
            rec['target_price'] = out['predicted_price']
            rec['explanation'] = f"Predicted upside {implf:.2f}% over current price. Suggest buying if it fits your risk profile."
        elif implf >= 5.0:
            rec['action'] = 'Consider Buy'
            rec['target_price'] = out['predicted_price']
            rec['explanation'] = f"Moderate upside {implf:.2f}%. You may accumulate gradually and monitor fundamentals."
        elif implf >= -5.0:
            rec['action'] = 'Hold'
            rec['target_price'] = out['predicted_price']
            rec['explanation'] = f"Predicted change {implf:.2f}%. Best to hold and re-evaluate; consider stop-loss if downside risk matters."
        else:
            rec['action'] = 'Sell'
            rec['target_price'] = out['predicted_price']
            rec['explanation'] = f"Predicted downside {implf:.2f}%. Consider selling to limit losses or set a tight stop-loss."
        out['recommendation'] = rec
    except Exception:
        pass
    return out
