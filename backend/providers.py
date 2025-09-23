# backend/providers.py
"""
Providers: IndianAPI (primary for current price) + yfinance fallback/supplement for history.

Returns dict with:
  - symbol (user requested)
  - resolved_symbol (actual symbol used for provider calls)
  - current_price (float) or None
  - provider (string: 'indianapi' or 'yfinance')
  - timestamp
  - raw (provider raw payload)
  - history (dict with keys "Open","High","Low","Close","Volume" -> {date_str: value})
  - momentum_pct (float or None)
  - fundamentals_score (int 0-100 or None)
"""

import os
import time
import requests
from typing import Optional, Dict, Any

try:
    import yfinance as yf
except Exception:
    yf = None

INDIANAPI_BASE = os.getenv("INDIANAPI_BASE", "https://stock.indianapi.in")
INDIANAPI_KEY = os.getenv("INDIANAPI_KEY", None)
RATE_LIMIT_CALLS_PER_MIN = int(os.getenv("RATE_LIMIT_CALLS_PER_MIN", "5"))
_LAST_CALL_FILE = ".indianapi_last_call"

SUFFIX_TRIALS = ["", ".NS", ".BO"]

# helper: safe float
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

# simple rate-limit for indianapi (not perfect but prevents bursts)
def _respect_rate_limit():
    interval = 60.0 / max(1, RATE_LIMIT_CALLS_PER_MIN)
    try:
        ts = 0.0
        if os.path.exists(_LAST_CALL_FILE):
            with open(_LAST_CALL_FILE, "r") as f:
                try:
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

def _indianapi_fetch(symbol: str, api_key: Optional[str] = None, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """
    Try to call IndianAPI for a stock symbol's current price.
    Note: IndianAPI sometimes returns only price/no history. We treat it as price provider.
    """
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
            # try passing api_key as param (some instances)
            params["api_key"] = key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                return None
        j = resp.json()
        # IndianAPI returns nested structure; try to get price from known locations
        price = None
        ts = None
        raw = j
        if isinstance(j, dict):
            cp = j.get("currentPrice") or j.get("price") or j.get("data") or None
            if isinstance(cp, dict):
                # try NSE/BSE keys
                price = _safe_float(cp.get("NSE") or cp.get("BSE") or cp.get("price"))
            else:
                # direct price
                price = _safe_float(j.get("current_price") or j.get("price") or j.get("lastPrice"))
            ts = j.get("timestamp") or j.get("updatedAt")
        return {
            "resolved_symbol": symbol,
            "current_price": price,
            "provider": "indianapi",
            "timestamp": ts,
            "raw": raw,
            "history": None  # indianapi typically doesn't provide full OHLCV in a stable dict form
        }
    except Exception:
        return None

def _convert_hist_df_to_dict(hist_df):
    """
    Convert a pandas DataFrame (from yfinance.history) to a dict with keys
    "Open","High","Low","Close","Volume" -> {date_str: value}
    Date strings use YYYY-MM-DD formatting.
    """
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
    """
    Use yfinance to fetch recent history and current price info.
    Returns dict with provider='yfinance', current_price, history as dict (Open/High/Low/Close/Volume).
    """
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            # try to get fast_info price fallback
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
        # trim rows
        if hist.shape[0] > max_rows:
            hist = hist.tail(max_rows)
        # convert index to Date if needed, ensure columns capitalized
        hist = hist.copy()
        # rename lower-case to title-case if necessary
        hist.columns = [c if c in ["Open","High","Low","Close","Volume"] else (c.title() if isinstance(c, str) else c) for c in hist.columns]
        if "Date" in hist.columns:
            hist = hist.set_index("Date")
        # ensure required columns exist by copying lower-case ones
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

def _compute_momentum_from_history(history_dict):
    """
    Compute simple momentum percent from history's Close series (first->last).
    """
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
    """
    Lightweight score (0-100) from yfinance info (PE, PB, dividend). Returns integer 0-100 or None.
    """
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

def fetch_data(user_symbol: str, indianapi_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Public function used by main.py.
    Tries IndianAPI first for price; always attempts to supply a 'history' dict (from yfinance) if possible.
    """
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

    # Try IndianAPI first (for price)
    indian_res = None
    try:
        indian_res = _try_variants(lambda s: _indianapi_fetch(s, api_key=indianapi_key))
    except Exception:
        indian_res = None

    # If indian_res exists, always attempt to supplement with yfinance history.
    if indian_res:
        resolved = indian_res.get("resolved_symbol") or user_symbol

        # 1) Try yfinance with the resolved symbol
        try:
            y_try = _yfinance_fetch(resolved)
            if y_try and y_try.get("history"):
                indian_res["history"] = y_try.get("history")
        except Exception:
            pass

        # 2) If still no history, try explicitly with .NS suffix (common for Indian tickers)
        if not indian_res.get("history"):
            try:
                if not resolved.endswith(".NS"):
                    y_try_ns = _yfinance_fetch(resolved + ".NS")
                    if y_try_ns and y_try_ns.get("history"):
                        indian_res["history"] = y_try_ns.get("history")
                        indian_res["resolved_symbol"] = resolved + ".NS"
            except Exception:
                pass

        # 3) If still no history, try other suffixes as fallback
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

    # If indian_res missing, fallback to yfinance entirely
    yf_res = None
    if not indian_res:
        try:
            yf_res = _try_variants(lambda s: _yfinance_fetch(s))
        except Exception:
            yf_res = None

    chosen = indian_res or yf_res
    if not chosen or chosen.get("current_price") is None:
        return None

    resolved_symbol = chosen.get("resolved_symbol") or user_symbol
    history = chosen.get("history", None)

    # compute momentum and fundamentals
    momentum = None
    fundamentals_score = None
    try:
        if history:
            momentum = _compute_momentum_from_history(history)
    except Exception:
        momentum = None
    try:
        yf_sym = resolved_symbol
        fundamentals_score = _compute_fundamentals_score_from_yf(yf_sym) if yf is not None else None
    except Exception:
        fundamentals_score = None

    result = {
        "symbol": user_symbol,
        "resolved_symbol": resolved_symbol,
        "current_price": _safe_float(chosen.get("current_price")),
        "provider": chosen.get("provider"),
        "timestamp": chosen.get("timestamp"),
        "raw": chosen.get("raw"),
        "history": history,
        "momentum_pct": momentum,
        "fundamentals_score": fundamentals_score,
    }
    return result
