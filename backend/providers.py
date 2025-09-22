"""
providers.py

Primary: IndianAPI
Fallback: yfinance

Exposes:
    fetch_data(user_symbol: str, indianapi_key: Optional[str] = None) -> dict|None

Returns dict with keys:
    symbol, resolved_symbol, current_price, provider, timestamp, raw, history, momentum_pct, fundamentals_score
or None (so main.py returns NA).
"""

import os
import time
import requests
from typing import Optional, Dict, Any

try:
    import yfinance as yf
except Exception:
    yf = None

# IndianAPI defaults & env var name
INDIANAPI_BASE = os.getenv("INDIANAPI_BASE", "https://stock.indianapi.in")
INDIANAPI_KEY = os.getenv("INDIANAPI_KEY") or os.getenv("ISE_API_KEY") or None

# Rate-limiter config (small file-based limiter to avoid bursts)
RATE_LIMIT_CALLS_PER_MIN = int(os.getenv("RATE_LIMIT_CALLS_PER_MIN", "5"))
_LAST_CALL_FILE = ".indianapi_last_call"

# Suffixes to try for plain tickers (helps Indian tickers)
SUFFIX_TRIALS = ["", ".NS", ".BO"]


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


def _indianapi_fetch(symbol: str, api_key: Optional[str] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch using IndianAPI. Tries header x-api-key and query param 'api_key'.
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
            params["api_key"] = key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                return None
        j = resp.json()
        # prefer NSE -> BSE -> generic price keys
        price = None
        ts = None
        if isinstance(j, dict) and "currentPrice" in j:
            cp = j.get("currentPrice") or {}
            price = _safe_float(cp.get("NSE") or cp.get("BSE") or cp.get("price"))
            ts = j.get("timestamp") or j.get("updatedAt")
        else:
            if isinstance(j, dict):
                price = _safe_float(j.get("price") or j.get("current_price") or j.get("close") or j.get("lastPrice"))
                ts = j.get("timestamp")
            elif isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict):
                first = j[0]
                price = _safe_float(first.get("price") or first.get("close"))
                ts = first.get("timestamp") or first.get("date")
        return {
            "resolved_symbol": symbol,
            "current_price": price,
            "provider": "indianapi",
            "timestamp": ts,
            "raw": j,
            "history": None,
        }
    except Exception:
        return None


def _yfinance_fetch(symbol: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y")
        if hist is None or hist.empty:
            fast = getattr(t, "fast_info", None)
            price = None
            if isinstance(fast, dict) and "lastPrice" in fast:
                price = _safe_float(fast.get("lastPrice"))
            return {
                "resolved_symbol": symbol,
                "current_price": price,
                "provider": "yfinance",
                "timestamp": None,
                "raw": fast,
                "history": hist.to_dict() if hist is not None else None,
            }
        last_row = hist.iloc[-1]
        price = _safe_float(last_row.get("Close"))
        ts = str(last_row.name) if hasattr(last_row, "name") else None
        return {
            "resolved_symbol": symbol,
            "current_price": price,
            "provider": "yfinance",
            "timestamp": ts,
            "raw": hist.tail(5).to_dict(),
            "history": hist.to_dict(),
        }
    except Exception:
        return None


def _compute_momentum_from_history(history_dict):
    try:
        if not history_dict:
            return None
        if "Close" in history_dict:
            closes = list(history_dict["Close"].values())
            if len(closes) >= 2:
                first = closes[0]
                last = closes[-1]
                if first and first != 0:
                    return round((last - first) / first * 100.0, 3)
    except Exception:
        pass
    return None


def _compute_fundamentals_score_from_yf(symbol: str):
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, "info") else {}
        pe = info.get("trailingPE") or info.get("trailing_pe") or None
        pb = info.get("priceToBook") or info.get("price_to_book") or None
        div = info.get("dividendYield") or info.get("dividend_yield") or None
        scores = []
        if pe and pe > 0:
            s = max(0.0, min(1.0, 1.0 / (pe / 15.0)))
            scores.append(s)
        if pb and pb > 0:
            s = max(0.0, min(1.0, 1.0 - ((pb - 2.0) / 10.0)))
            scores.append(s)
        if div and div > 0:
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
    Public entry: fetch_data(user_symbol, indianapi_key=None)
    If indianapi_key provided, it will be used for IndianAPI calls; otherwise INDIANAPI_KEY env var is used.
    """
    if not user_symbol or not str(user_symbol).strip():
        return None
    user_symbol = str(user_symbol).strip().upper()
    has_suffix = ("." in user_symbol) or (":" in user_symbol)
    tried_symbols = []

    def _try_symbol_variants(fetcher):
        to_try = [user_symbol] if has_suffix else [user_symbol + s for s in SUFFIX_TRIALS]
        for sym in to_try:
            if sym in tried_symbols:
                continue
            tried_symbols.append(sym)
            res = fetcher(sym)
            if res and res.get("current_price") is not None:
                res["symbol"] = user_symbol
                return res
        return None

    # 1) Try IndianAPI first
    indian_res = None
    try:
        indian_res = _try_symbol_variants(lambda s: _indianapi_fetch(s, api_key=indianapi_key))
    except Exception:
        indian_res = None

    # 2) Fallback to yfinance
    yf_res = None
    if indian_res is None:
        try:
            yf_res = _try_symbol_variants(lambda s: _yfinance_fetch(s))
        except Exception:
            yf_res = None

    chosen = indian_res or yf_res
    if not chosen or chosen.get("current_price") is None:
        return None

    history = chosen.get("history", None)
    momentum = None
    fundamentals_score = None
    resolved_symbol = chosen.get("resolved_symbol", None)
    try:
        if history:
            momentum = _compute_momentum_from_history(history)
    except Exception:
        momentum = None
    try:
        if yf is not None and resolved_symbol:
            fundamentals_score = _compute_fundamentals_score_from_yf(resolved_symbol)
    except Exception:
        fundamentals_score = None

    result = {
        "symbol": user_symbol,
        "resolved_symbol": resolved_symbol,
        "current_price": float(chosen.get("current_price")) if _safe_float(chosen.get("current_price")) is not None else None,
        "provider": chosen.get("provider"),
        "timestamp": chosen.get("timestamp"),
        "raw": chosen.get("raw"),
        "history": history,
        "momentum_pct": momentum,
        "fundamentals_score": fundamentals_score,
    }
    return result
