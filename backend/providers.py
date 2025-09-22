"""
providers.py

Alpha Vantage + yfinance adapter with:
- simple rate limiting for Alpha Vantage
- fallback to yfinance
- auto-try common Indian suffixes (.NS, .BO) when user provides ticker without suffix
- returns a consistent dict used by main.py

Returns:
- dict with keys:
    - symbol (requested symbol uppercase)
    - resolved_symbol (symbol actually used to fetch, e.g. 'RELIANCE.NS')
    - current_price (float)
    - provider (str: 'alphavantage' or 'yfinance')
    - timestamp (str or None)
    - raw (raw provider response or None)
    - history (optional dict/time-series or None)
    - momentum_pct (float or None)
    - fundamentals_score (int 0-100 or None)
- OR None if all providers failed (so main will return NA)
"""

import os
import time
import requests
from typing import Optional, Dict, Any

try:
    import yfinance as yf
except Exception:
    yf = None

ALPHA_BASE = os.getenv("ALPHAVANTAGE_BASE", "https://www.alphavantage.co/query")
ALPHA_KEY = os.getenv("ALPHAVANTAGE_KEY", "") or None
ALPHA_CALLS_PER_MIN = int(os.getenv("ALPHAVANTAGE_CALLS_PER_MIN", "5"))
_LAST_CALL_FILE = ".alpha_last_call"

# Suffixes to try for plain tickers (helps Indian tickers)
SUFFIX_TRIALS = ["", ".NS", ".BO"]


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _respect_rate_limit():
    """
    Very small file-based limiter to respect ALPHAVANTAGE_CALLS_PER_MIN across processes on the same disk.
    Not perfect for multi-server, but helps avoid quick bursts in single-instance deployments.
    """
    if not ALPHA_KEY:
        return
    interval = 60.0 / max(1, ALPHA_CALLS_PER_MIN)
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


def _alpha_fetch(symbol: str, alpha_key: Optional[str] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch price using Alpha Vantage GLOBAL_QUOTE.
    Returns dict similar to yfinance wrapper or None on failure/rate-limit.
    """
    key = alpha_key or ALPHA_KEY
    if not key:
        return None

    _respect_rate_limit()
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key}
    try:
        resp = requests.get(ALPHA_BASE, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None
        j = resp.json()
        # handle rate limit / error
        if isinstance(j, dict) and ("Note" in j or "Error Message" in j):
            return None
        g = j.get("Global Quote") or {}
        # Keys sometimes have spaces or different formats
        price = _safe_float(g.get("05. price") or g.get("05 price") or g.get("price"))
        ts = g.get("07. latest trading day") or g.get("07 latest trading day") or None
        return {
            "resolved_symbol": symbol,
            "current_price": price,
            "provider": "alphavantage",
            "timestamp": ts,
            "raw": j,
            "history": None,
        }
    except Exception:
        return None


def _yfinance_fetch(symbol: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch using yfinance. Returns dict or None on failure.
    """
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        # try 1y history to allow momentum calc
        hist = t.history(period="1y")
        # If history empty, still try fast_info
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
            "raw": hist.tail(5).to_dict(),  # small raw footprint
            "history": hist.to_dict(),
        }
    except Exception:
        return None


def _compute_momentum_from_history(history_dict: Dict[str, Any]) -> Optional[float]:
    """
    Compute simple momentum % using earliest vs latest close values in history dict.
    Expects history dict in the form produced by pandas.DataFrame.to_dict() for yfinance.
    """
    try:
        if history_dict is None:
            return None
        # history_dict example: {'Open': {ts: val, ...}, 'Close': {...}, ...}
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


def _compute_fundamentals_score_from_yf(symbol: str) -> Optional[int]:
    """
    Pull basic fundamentals via yfinance and calculate a heuristic score 0-100.
    Uses trailing PE, price-to-book, dividend yield where available.
    """
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
            # ideal PE ~15
            s = max(0.0, min(1.0, 1.0 / (pe / 15.0)))
            scores.append(s)
        if pb and pb > 0:
            # ideal PB <=2
            s = max(0.0, min(1.0, 1.0 - ((pb - 2.0) / 10.0)))
            scores.append(s)
        if div and div > 0:
            s = max(0.0, min(1.0, div * 5.0))  # scaled
            scores.append(s)
        if not scores:
            return None
        avg = sum(scores) / len(scores)
        return int(round(avg * 100.0))
    except Exception:
        return None


def fetch_data(user_symbol: str, alpha_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Public entry point used by main.py.

    - user_symbol: ticker entered by user (no suffix expected)
    - alpha_key: optional per-request Alpha Vantage API key (overrides env key)
    Returns structured dict or None if all attempts failed.
    """
    if not user_symbol or not str(user_symbol).strip():
        return None

    user_symbol = str(user_symbol).strip().upper()

    # If user already included a suffix (contains '.' or ':'), only try that exact symbol
    has_suffix = ("." in user_symbol) or (":" in user_symbol)

    # Try providers and suffix trials in order:
    # 1) Alpha Vantage (exact symbol or with trials)
    # 2) yfinance (exact symbol or with trials)
    # On first successful fetch that contains a price (not None), we proceed to enrich and return.

    tried_symbols = []

    def _try_symbol_variants(fetcher):
        # fetcher is function that accepts resolved_symbol and returns dict or None
        # iterate SUFFIX_TRIALS but if user already had suffix, only try exact once
        to_try = [user_symbol] if has_suffix else [user_symbol + s for s in SUFFIX_TRIALS]
        for sym in to_try:
            if sym in tried_symbols:
                continue
            tried_symbols.append(sym)
            res = fetcher(sym)
            if res and res.get("current_price") is not None:
                # successful fetch
                res["symbol"] = user_symbol  # keep original user-provided symbol normalized
                return res
        return None

    # 1) Try Alpha Vantage first (if key available)
    alpha_res = None
    if ALPHA_KEY or alpha_key:
        try:
            alpha_res = _try_symbol_variants(lambda s: _alpha_fetch(s, alpha_key=alpha_key))
        except Exception:
            alpha_res = None

    # 2) If Alpha Vantage failed, try yfinance
    yf_res = None
    if alpha_res is None:
        try:
            yf_res = _try_symbol_variants(lambda s: _yfinance_fetch(s))
        except Exception:
            yf_res = None

    chosen = alpha_res or yf_res
    if not chosen or chosen.get("current_price") is None:
        # all providers failed -> return None so main will output NA
        return None

    # Enrich: momentum and fundamentals (prefer yfinance history/info)
    momentum = None
    fundamentals_score = None
    history = chosen.get("history", None)
    resolved_symbol = chosen.get("resolved_symbol", None)

    # momentum from available history
    try:
        if history:
            momentum = _compute_momentum_from_history(history)
    except Exception:
        momentum = None

    # fundamentals from yfinance resolved_symbol (best effort)
    try:
        # prefer using resolved_symbol (e.g. RELIANCE.NS) for fundamentals lookup
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
