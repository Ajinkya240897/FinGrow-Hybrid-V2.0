Fingrow-Hybrid backend v2.0
Endpoints:
- POST /model/predict {symbol, interval, alpha_key?}
- Run `python train_all.py` to train models for all horizons (saves models to models/)
Notes:
- If data fetch fails, outputs are NA to avoid misleading values.
