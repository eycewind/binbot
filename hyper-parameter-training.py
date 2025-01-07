# Import the class from the Python file (module)
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from BinanceClient import BinanceClient
import numpy as np
from typing import Final
import joblib
from BatchFeatures import BatchFeatures
from datetime import datetime, timedelta

# Initialize Binance client with your API credentials
# dotenv_path = Path('.env-secret')
# load_dotenv(dotenv_path=dotenv_path)
api_secret = os.getenv("BINANCE_SECRET_KEY")
api_key = os.getenv("BINANCE_API_KEY")

# Create Binance client & initialize it
pair = "BTCUSDT"
time_delta = 12
db_name = pair + "_1min_" + str(time_delta) + "weeks.db"
db_name = "BTCUSDT_1min_dry_run.db"             # For dry run testing
binance_client = BinanceClient(db_name)
binance_client.set_interval("1m")
batch_feature = BatchFeatures()

#Fetch data from db
df = binance_client.fetch_data_from_db(pair)

# Feature Engineering (mind the order since some features are depended on others)
bf = BatchFeatures()
bf.calculate_sma(df)
bf.calculate_ema(df)
bf.calculate_rsi(df)
bf.calculate_macd(df)
bf.calculate_bollinger_bands(df)
bf.calculate_atr(df)
bf.calculate_volume_features(df)
bf.calculate_roc(df)
bf.calculate_lagged_features(df)
bf.calculate_candle_features(df)
bf.calculate_stochastic_oscillator(df)
bf.calculate_williams_r(df)
bf.calculate_moving_average_crossover(df)
bf.calculate_historical_volatility(df)
bf.calculate_on_balance_volume(df)
bf.calculate_money_flow_index(df)
bf.calculate_croc(df)

# drop NaNs
df.dropna(inplace=True)


df['target'] = (df['ema_5'].shift(-10) - df['ema_5']) / df['ema_5'] * 100
df.dropna(inplace=True)


# df is sorted by time and indexed by a datetime index
n = len(df)
train_end = int(n * 0.8) # 80% data used for training
train_df = df.iloc[:train_end]
test_df = df.iloc[train_end:]

# Separate features and target
X_train = train_df.drop(['target'], axis=1)
y_train = train_df['target']

X_test = test_df.drop(['target'], axis=1)
y_test = test_df['target']


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Define Random Forest
pipeline_rf = Pipeline([
    ('regressor', RandomForestRegressor())
])

# Parameter grid for Random Forest
param_grid_rf = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [3, 5, 10, 20],
    'regressor__max_features': ['sqrt', 'log2', None],
    'regressor__bootstrap': [True, False]
}

# Define XGBoost
pipeline_xgb = Pipeline([
    ('regressor', XGBRegressor(eval_metric='rmse', use_label_encoder=False))
])

# Parameter grid for XGBoost
param_grid_xgb = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [3, 5, 10],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}

# Time series split
time_series_split = TimeSeriesSplit(n_splits=5)

# Perform grid search for Random Forest
grid_search_rf = GridSearchCV(
    pipeline_rf,
    param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=time_series_split,
    return_train_score=True,
    verbose=3
)
grid_search_rf.fit(X_train, y_train)

# Perform grid search for XGBoost
grid_search_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid_xgb,
    scoring='neg_mean_squared_error',
    cv=time_series_split,
    return_train_score=True,
    verbose=3
)
grid_search_xgb.fit(X_train, y_train)

# Output results
print("Random Forest Best Params:", grid_search_rf.best_params_)
print("Random Forest Best Score:", grid_search_rf.best_score_)
print("XGBoost Best Params:", grid_search_xgb.best_params_)
print("XGBoost Best Score:", grid_search_xgb.best_score_)


# Save the GridSearchCV object
joblib.dump(grid_search_rf, 'grid_rf_search_results_1min_ema5_nn10.pkl')
joblib.dump(grid_search_xgb, 'grid_xgb_search_results_1min_ema5_nn10.pkl')