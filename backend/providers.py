# backend/providers.py
"""
Providers with CSV fallback.

Behavior:
 - IndianAPI for live price (if key provided).
 - Always attempt yfinance history for resolved symbol and common suffixes.
 - If yfinance has no history, try local CSV at backend/historical/<SYMBOL>.csv or <SYMBOL>.NS.csv.
 - Returns dict expected by modeling: symbol, resolved_symbol, current_price, provider, timestamp, raw, history, momentum_pct, fundamentals_score
"""

import os
import time
import requests
from typing import Optional, Dict, Any

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import pandas as pd
except Exception:
    pd = None

INDIANAPI_BASE = os.getenv("INDIANAPI_BASE", "https://stock.indianapi.in")
INDIANAPI_KEY = os.getenv("INDIANAPI_KEY", None)
RATE_LIMIT_CALLS_PER_MIN = int(os.getenv("RATE_LIMIT_CALLS_PER_MIN", "5"))
_LAST_CALL_FILE = ".indianapi_last_call"
SUFFIX_TRIALS = ["", ".NS", ".BO"]
HISTORICAL_DIR = os.path.join(os.path.dirname(__file__), "historical")  # backend/historical

# helpers
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _respect_rate_limit():
    interval = 60.0 / max(1, RATE_LIMIT_CALLS_PER_MIN)
    try:
        ts = 0.0
        if os.path.exists(_LAST_CALL_FILE):
            try:
                with open(_LAST_CALL_FILE, "r") as f:
                    ts = float(f.read().strip() or "0")
            except Exception:
                ts = 0.0
        elapsed = time.time() - ts
        if elapsed < interval:
            time.sleep(interval - elapsed)
    finally:
        try:
            with open(_LAST_CALL_FILE, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

# IndianAPI price fetch
def _indianapi_fetch(symbol: str, api_key: Optional[str] = None, timeout: int = 8) -> Optional[Dict[str, Any]]:
    key = api_key or INDIANAPI_KEY
    if not key:
        return None
    _respect_rate_limit()
    base = INDIANAPI_BASE.rstrip("/")
    url = f"{base}/stock"
    params = {"name": symbol}
    headers = {"x-api-key": key}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            params["api_key"] = key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                return None
        j = resp.json()
        price = None
        ts = None
        raw = j
        if isinstance(j, dict):
            cp = j.get("currentPrice") or j.get("price") or j.get("data") or None
            if isinstance(cp, dict):
                price = _safe_float(cp.get("NSE") or cp.get("BSE") or cp.get("price"))
            else:
                price = _safe_float(j.get("current_price") or j.get("price") or j.get("lastPrice"))
            ts = j.get("timestamp") or j.get("updatedAt")
        return {
            "resolved_symbol": symbol,
            "current_price": price,
            "provider": "indianapi",
            "timestamp": ts,
            "raw": raw,
            "history": None
        }
    except Exception:
        return None

# yfinance fetch and conversion
def _convert_hist_df_to_dict(hist_df):
    try:
        d = {}
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in hist_df.columns:
                series = hist_df[col]
                col_map = {}
                for idx, val in series.iteritems():
                    try:
                        k = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                    except Exception:
                        k = str(idx)
                    try:
                        col_map[k] = float(val) if val is not None and not (isinstance(val, float) and (val != val)) else None
                    except Exception:
                        col_map[k] = None
                d[col] = col_map
        return d if d else None
    except Exception:
        return None

def _yfinance_fetch(symbol: str, period: str = "2y", max_rows: int = 2000) -> Optional[Dict[str, Any]]:
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            fast = getattr(t, "fast_info", None)
            price = None
            if isinstance(fast, dict):
                price = _safe_float(fast.get("lastPrice") or fast.get("last_price") or fast.get("last_close"))
            return {
                "resolved_symbol": symbol,
                "current_price": price,
                "provider": "yfinance",
                "timestamp": None,
                "raw": fast,
                "history": None
            }
        if hist.shape[0] > max_rows:
            hist = hist.tail(max_rows)
        hist = hist.copy()
        hist.columns = [c if c in ["Open","High","Low","Close","Volume"] else (c.title() if isinstance(c, str) else c) for c in hist.columns]
        if "Date" in hist.columns:
            hist = hist.set_index("Date")
        for want in ["Open","High","Low","Close","Volume"]:
            if want not in hist.columns and want.lower() in hist.columns:
                hist[want] = hist[want.lower()]
        hist_dict = _convert_hist_df_to_dict(hist)
        cp = None
        try:
            last_row = hist.tail(1)
            if "Close" in last_row.columns:
                cp_val = list(last_row["Close"].values)[0]
                cp = _safe_float(cp_val)
        except Exception:
            cp = None
        return {
            "resolved_symbol": symbol,
            "current_price": cp,
            "provider": "yfinance",
            "timestamp": None,
            "raw": None,
            "history": hist_dict
        }
    except Exception:
        return None

# CSV fallback loader
def _load_local_csv_history(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Look for backend/historical/<symbol>.csv (symbol may include .NS).
    CSV format: Date,Open,High,Low,Close,Volume  (Date: YYYY-MM-DD)
    Returns history dict like yfinance conversion or None
    """
    if pd is None:
        return None
    try:
        fname = os.path.join(HISTORICAL_DIR, f"{symbol}.csv")
        if not os.path.exists(fname):
            return None
        df = pd.read_csv(fname, parse_dates=["Date"])
        if df is None or df.empty:
            return None
        # ensure expected columns present
        cols = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
        if "Close" not in df.columns:
            return None
        # set index to Date
        if "Date" in df.columns:
            df = df.set_index("Date")
        # convert to dict
        return _convert_hist_df_to_dict(df)
    except Exception:
        return None

# simple analytics
def _compute_momentum_from_history(history_dict):
    try:
        if not history_dict or "Close" not in history_dict:
            return None
        closes_map = history_dict["Close"]
        if not closes_map:
            return None
        items = sorted(closes_map.items(), key=lambda kv: kv[0])
        if len(items) < 2:
            return None
        first = items[0][1]
        last = items[-1][1]
        if first is None or last is None or first == 0:
            return None
        return round((last - first) / first * 100.0, 3)
    except Exception:
        return None

def _compute_fundamentals_score_from_yf(symbol: str):
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        info = {}
        try:
            info = t.info if hasattr(t, "info") else {}
        except Exception:
            info = {}
        pe = info.get("trailingPE") or info.get("trailing_pe") or info.get("peRatio") or None
        pb = info.get("priceToBook") or info.get("price_to_book") or None
        div = info.get("dividendYield") or info.get("dividend_yield") or None
        scores = []
        if pe and isinstance(pe, (int, float)) and pe > 0:
            s = max(0.0, min(1.0, 1.0 / (pe / 15.0)))
            scores.append(s)
        if pb and isinstance(pb, (int, float)) and pb > 0:
            s = max(0.0, min(1.0, 1.0 - ((pb - 2.0) / 10.0)))
            scores.append(s)
        if div and isinstance(div, (int, float)) and div > 0:
            s = max(0.0, min(1.0, div * 5.0))
            scores.append(s)
        if not scores:
            return None
        avg = sum(scores) / len(scores)
        return int(round(avg * 100.0))
    except Exception:
        return None

# main public fetch
def fetch_data(user_symbol: str, indianapi_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not user_symbol or not str(user_symbol).strip():
        return None
    user_symbol = str(user_symbol).strip().upper()
    has_suffix = ("." in user_symbol) or (":" in user_symbol)
    tried_symbols = []

    def _try_variants(fetch_fn):
        to_try = [user_symbol] if has_suffix else [user_symbol + s for s in SUFFIX_TRIALS]
        for sym in to_try:
            if sym in tried_symbols:
                continue
            tried_symbols.append(sym)
            try:
                res = fetch_fn(sym)
            except Exception:
                res = None
            if res and res.get("current_price") is not None:
                return res
        return None

    # 1) indianapi (price) first
    indian_res = None
    try:
        indian_res = _try_variants(lambda s: _indianapi_fetch(s, api_key=indianapi_key))
    except Exception:
        indian_res = None

    # 2) if indian_res found, try to add yfinance history
    if indian_res:
        resolved = indian_res.get("resolved_symbol") or user_symbol
        try:
            y_try = _yfinance_fetch(resolved)
            if y_try and y_try.get("history"):
                indian_res["history"] = y_try.get("history")
        except Exception:
            pass
        if not indian_res.get("history"):
            try:
                if not resolved.endswith(".NS"):
                    y_try_ns = _yfinance_fetch(resolved + ".NS")
                    if y_try_ns and y_try_ns.get("history"):
                        indian_res["history"] = y_try_ns.get("history")
                        indian_res["resolved_symbol"] = resolved + ".NS"
            except Exception:
                pass
        if not indian_res.get("history"):
            for sfx in SUFFIX_TRIALS:
                try:
                    candidate = (user_symbol if has_suffix else user_symbol + sfx)
                    y_try2 = _yfinance_fetch(candidate)
                    if y_try2 and y_try2.get("history"):
                        indian_res["history"] = y_try2.get("history")
                        indian_res["resolved_symbol"] = candidate
                        break
                except Exception:
                    continue

    # 3) fallback to yfinance only if indian_res missing
    yf_res = None
    if not indian_res:
        try:
            yf_res = _try_variants(lambda s: _yfinance_fetch(s))
        except Exception:
            yf_res = None

    chosen = indian_res or yf_res

    # If chosen is missing or no current_price, try forced yfinance .NS for history and price
    if not chosen or chosen.get("current_price") is None:
        # try forcing .NS via yfinance
        forced_history = None
        forced_resolved = user_symbol
        try:
            if not user_symbol.endswith(".NS"):
                forced = _yfinance_fetch(user_symbol + ".NS")
                if fo
::contentReference[oaicite:0]{index=0}
