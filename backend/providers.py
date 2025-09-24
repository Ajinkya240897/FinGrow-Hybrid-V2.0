# backend/providers.py
"""
Final providers module.

Behavior:
 - Use IndianAPI (if key provided) to get current_price (fast).
 - Always attempt to supplement IndianAPI with yfinance history (OHLCV).
 - Try symbol variants: SYMBOL, SYMBOL.NS, SYMBOL.BO.
 - If still no history, as a last resort force SYMBOL.NS via yfinance.
 - Return a dict with keys:
     symbol, resolved_symbol, current_price, provider, timestamp, raw, history,
     momentum_pct, fundamentals_score
 - history is a dict: {"Open":{date:val}, "High":{...}, "Low":{...}, "Close":{...}, "Volume":{...}}
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

# Try these suffixes (empty means original input)
SUFFIX_TRIALS = ["", ".NS", ".BO"]

# ---- helpers ----
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _respect_rate_limit():
    """Primitive rate limit to avoid rapid indianapi calls."""
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

# ---- IndianAPI fetch ----
def _indianapi_fetch(symbol: str, api_key: Optional[str] = None, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """
    Returns dict:
      resolved_symbol, current_price, provider='indianapi', timestamp, raw, history=None
    IndianAPI often lacks historical OHLCV; we only use it for current price primarily.
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
            # fallback: try passing api_key param
            params["api_key"] = key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                return None
        j = resp.json()
        price = None
        ts = None
        raw = j
        if isinstance(j, dict):
            # try several shapes IndianAPI might return
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

# ---- yfinance helpers ----
def _convert_hist_df_to_dict(hist_df):
    """
    Convert yfinance DataFrame to dict of columns mapping date->value.
    Date formatted as YYYY-MM-DD.
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
    Uses yfinance to fetch history and last close.
    Returns dict: resolved_symbol, current_price, provider='yfinance', timestamp=None, raw, history
    """
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            # no history, try fast_info fallback for price
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
        # limit rows
        if hist.shape[0] > max_rows:
            hist = hist.tail(max_rows)
        # normalize columns (title-case)
        hist = hist.copy()
        hist.columns = [c if c in ["Open","High","Low","Close","Volume"] else (c.title() if isinstance(c, str) else c) for c in hist.columns]
        # if Date is a column set as index
        if "Date" in hist.columns:
            hist = hist.set_index("Date")
        # ensure expected columns exist (lowercase fallback)
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

# ---- simple analytics ----
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
    # lightweight heuristic using yfinance info (PE/PB/dividend)
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

# ---- public function ----
def fetch_data(user_symbol: str, indianapi_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Public entry used by main.py.

    Steps:
      1) Normalize user input symbol (uppercase).
      2) Try IndianAPI variants to get current_price.
      3) If IndianAPI provided price, try to obtain history via yfinance using resolved symbol, then .NS, then other suffixes.
      4) If IndianAPI did not provide price, fallback to yfinance (variants).
      5) As last resort, force yfinance with .NS.
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

    # 1) Try IndianAPI (price) first
    indian_res = None
    try:
        indian_res = _try_variants(lambda s: _indianapi_fetch(s, api_key=indianapi_key))
    except Exception:
        indian_res = None

    # 2) If indian_res, attempt to add history via yfinance (resolved -> resolved.NS -> other suffixes)
    if indian_res:
        resolved = indian_res.get("resolved_symbol") or user_symbol

        # Try with resolved symbol (yfinance)
        try:
            y_try = _yfinance_fetch(resolved)
            if y_try and y_try.get("history"):
                indian_res["history"] = y_try.get("history")
        except Exception:
            pass

        # If still no history, try resolved + ".NS"
        if not indian_res.get("history"):
            try:
                if not resolved.endswith(".NS"):
                    y_try_ns = _yfinance_fetch(resolved + ".NS")
                    if y_try_ns and y_try_ns.get("history"):
                        indian_res["history"] = y_try_ns.get("history")
                        indian_res["resolved_symbol"] = resolved + ".NS"
            except Exception:
                pass

        # If still no history, try other suffixes (user_symbol + suffix)
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

    # 3) If indian_res missing or indian didn't provide price, fallback to yfinance completely
    yf_res = None
    if not indian_res:
        try:
            yf_res = _try_variants(lambda s: _yfinance_fetch(s))
        except Exception:
            yf_res = None

    chosen = indian_res or yf_res
    if not chosen or chosen.get("current_price") is None:
        # As an extra last resort: force yfinance with .NS if not already tried
        forced_history = None
        forced_resolved = user_symbol
        try:
            if not user_symbol.endswith(".NS"):
                forced = _yfinance_fetch(user_symbol + ".NS")
                if forced and forced.get("history"):
                    forced_history = forced.get("history")
                    forced_resolved = user_symbol + ".NS"
                    # use fast close price if available
                    chosen = chosen or {}
                    chosen["current_price"] = chosen.get("current_price") or forced.get("current_price")
        except Exception:
            forced_history = None

        if forced_history:
            # build result using forced history, maybe no indianapi price
            momentum = _compute_momentum_from_history(forced_history)
            fundamentals = _compute_fundamentals_score_from_yf(forced_resolved) if yf is not None else None
            return {
                "symbol": user_symbol,
                "resolved_symbol": forced_resolved,
                "current_price": _safe_float(chosen.get("current_price")) if chosen else None,
                "provider": "yfinance",
                "timestamp": None,
                "raw": chosen.get("raw") if chosen else None,
                "history": forced_history,
                "momentum_pct": momentum,
                "fundamentals_score": fundamentals,
            }
        # if forced_history also not found, return None (caller must show NA)
        return None

    # Build final result
    resolved_symbol = chosen.get("resolved_symbol") or user_symbol
    history = chosen.get("history", None)

    # Extra last-resort check: if we have no history yet, force .NS once more (ensures heavy coverage)
    if (not history) and (not resolved_symbol.endswith(".NS")):
        try:
            forced = _yfinance_fetch(resolved_symbol + ".NS")
            if forced and forced.get("history"):
                history = forced.get("history")
                resolved_symbol = resolved_symbol + ".NS"
        except Exception:
            pass

    momentum = None
    fundamentals_score = None
    try:
        if history:
            momentum = _compute_momentum_from_history(history)
    except Exception:
        momentum = None
    try:
        fundamentals_score = _compute_fundamentals_score_from_yf(resolved_symbol) if yf is not None else None
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
