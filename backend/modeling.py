import os, joblib, math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
DATA_PATH = os.getenv('HISTORICAL_CSV','data/historical.csv')
MODEL_DIR = os.getenv('MODEL_OUT','models')
HORIZON_MAP = {'3-15d':10, '1-3m':60, '3-6m':120, '1-3y':365}
def prepare(df):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'],errors='coerce')
    df = df.dropna(subset=['close'])
    df['ret1'] = df['close'].pct_change(1)
    df['ret3'] = df['close'].pct_change(3)
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df = df.dropna()
    return df
def train_all(random_search=False):
    df = pd.read_csv(DATA_PATH)
    df = prepare(df)
    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, days in HORIZON_MAP.items():
        df_temp = df.copy()
        df_temp['target'] = df_temp['close'].shift(-days)
        df_temp = df_temp.dropna()
        X = df_temp[['ret1','ret3','ma5','ma10']]
        y = df_temp['target']
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if random_search:
            param_dist = {'n_estimators':[100,200],'max_depth':[5,10,15,None],'min_samples_split':[2,5,10]}
            tscv = TimeSeriesSplit(n_splits=3)
            rs = RandomizedSearchCV(model,param_distributions=param_dist,n_iter=4,cv=tscv,scoring='neg_mean_absolute_error',n_jobs=1,random_state=42)
            rs.fit(X,y)
            model = rs.best_estimator_
        else:
            model.fit(X,y)
        path = os.path.join(MODEL_DIR, f'model_{days}.joblib')
        joblib.dump(model, path)
        preds = model.predict(X)
        mae = mean_absolute_error(y,preds)
        rmse = mean_squared_error(y,preds,squared=False)
        results[name] = {'days':days,'mae':float(mae),'rmse':float(rmse),'model_path':path}
    return results
def load_model_for_horizon(days):
    path = os.path.join(MODEL_DIR, f'model_{days}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)
def predict_for_horizon(features, days):
    model = load_model_for_horizon(days)
    # ensemble predictions from trees for confidence estimation
    preds = np.array([est.predict([features])[0] for est in model.estimators_])
    mean_pred = float(np.mean(preds))
    std = float(np.std(preds))
    return mean_pred, std
