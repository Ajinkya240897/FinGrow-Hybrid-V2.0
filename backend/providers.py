import os, time, requests
from typing import Optional
try:
    import yfinance as yf
except Exception:
    yf = None
ALPHA_BASE = os.getenv('ALPHAVANTAGE_BASE', 'https://www.alphavantage.co/query')
ALPHA_KEY = os.getenv('ALPHAVANTAGE_KEY', '') or None
ALPHA_CALLS_PER_MIN = int(os.getenv('ALPHAVANTAGE_CALLS_PER_MIN', '5'))
_LAST_CALL_FILE = '.alpha_last_call'
def _float(v):
    try:
        return float(v)
    except:
        return None
def _respect_rate_limit():
    if ALPHA_KEY is None: return
    interval = 60.0 / max(1, ALPHA_CALLS_PER_MIN)
    try:
        ts = 0.0
        if os.path.exists(_LAST_CALL_FILE):
            with open(_LAST_CALL_FILE,'r') as f:
                ts = float(f.read().strip() or '0')
        elapsed = time.time() - ts
        if elapsed < interval:
            time.sleep(interval - elapsed)
    finally:
        try:
            with open(_LAST_CALL_FILE,'w') as f:
                f.write(str(time.time()))
        except: pass
def _alpha_response_is_rate_limited(j: dict) -> bool:
    if not isinstance(j, dict): return False
    if 'Note' in j and isinstance(j['Note'], str) and 'frequency' in j['Note'].lower():
        return True
    if 'Error Message' in j:
        return True
    return False
def fetch_with_alpha(symbol: str, alpha_key: Optional[str]=None, timeout: int = 10) -> Optional[dict]:
    # alpha_key overrides env ALPHA_KEY for this call
    key = alpha_key or ALPHA_KEY
    if not key:
        return None
    # respect rate-limit
    _respect_rate_limit()
    params = {'function':'GLOBAL_QUOTE','symbol':symbol,'apikey':key}
    try:
        r = requests.get(ALPHA_BASE, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        if _alpha_response_is_rate_limited(j):
            return None
        g = j.get('Global Quote', {}) or {}
        price = _float(g.get('05. price') or g.get('05 price'))
        ts = g.get('07. latest trading day') or None
        return {'symbol':symbol.upper(),'price':price,'raw':j,'provider':'alphavantage','timestamp':ts}
    except Exception:
        return None
def fetch_with_yfinance(symbol: str) -> Optional[dict]:
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        info = t.history(period='1y')
        if info is None or info.empty:
            fast = getattr(t,'fast_info',None)
            price = None
            if fast and isinstance(fast, dict) and 'lastPrice' in fast:
                price = _float(fast['lastPrice'])
            return {'symbol':symbol.upper(),'price':price,'raw':fast,'provider':'yfinance','timestamp':None}
        last_row = info.iloc[-1]
        price = _float(last_row.get('Close'))
        ts = str(last_row.name) if hasattr(last_row,'name') else None
        return {'symbol':symbol.upper(),'price':price,'raw':info.to_dict(),'provider':'yfinance','timestamp':ts, 'history': info.to_dict()}
    except Exception:
        return None
def fetch_quote(symbol: str, preferred: Optional[str]=None, alpha_key: Optional[str]=None) -> dict:
    symbol = symbol.strip().upper()
    tried = []
    order = []
    if preferred: order.append(preferred.lower())
    fallback = os.getenv('FALLBACK_PROVIDER','yfinance') or 'yfinance'
    default_order = [p for p in [fallback.lower(),'alphavantage','yfinance'] if p]
    for p in default_order:
        if p not in order: order.append(p)
    for p in order:
        tried.append(p)
        if p.startswith('alpha') or p=='alphavantage':
            res = fetch_with_alpha(symbol, alpha_key=alpha_key)
            if res and res.get('price') is not None:
                return res
        elif p=='yfinance':
            res = fetch_with_yfinance(symbol)
            if res and res.get('price') is not None:
                return res
        time.sleep(0.05)
    # all failed - return NA-style structured response
    return {'symbol':symbol,'price':None,'raw':None,'provider':None,'timestamp':None,'history':None}
