# Fingrow-Hybrid V2.0

This release implements the full spec you provided: inputs are ticker (no suffix), Alpha Vantage API key (optional input), and time interval. Outputs include Current Share Price, Predicted Price, Implied Return %, Confidence %, Momentum %, Fundamentals Score, and a Buy/Hold/Sell recommendation with beginner-friendly explanation.

## Quick start
1. Backend
   - cd backend
   - copy .env.example to .env and set ALPHAVANTAGE_KEY if you want server-default key
   - python -m venv .venv; source .venv/bin/activate
   - pip install -r requirements.txt
   - python train_all.py    # trains models for all horizons
   - uvicorn main:app --reload --port 8000

2. Frontend
   - cd frontend
   - npm install
   - npm run dev

3. Use the UI
   - Enter ticker (no suffix), choose interval, optionally paste Alpha Vantage key, press Get Output.
   - If data fetch fails, all numeric outputs show as NA to avoid misleading values.

## Notes
- Models are trained per horizon: 3-15d (10d), 1-3m (60d), 3-6m (120d), 1-3y (365d).
- Confidence is a heuristic computed from RandomForest tree dispersion; treat as guidance not absolute certainty.
- Fundamentals score is derived from yfinance fields if available; otherwise 'NA'.
