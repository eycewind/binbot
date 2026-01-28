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


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor


def hybrid_search_narrow(X_train, y_train, best_random_params):
    """
    Perform a narrow fine grid search based on the best parameters from random search.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    # Define a narrow grid around the best parameters
    param_grid = {
        'n_estimators': [
            max(10, best_random_params['n_estimators'] - 30), 
            best_random_params['n_estimators'], 
            best_random_params['n_estimators'] + 30
        ],
        'max_depth': [
            max(1, best_random_params['max_depth'] - 1), 
            best_random_params['max_depth'], 
            best_random_params['max_depth'] + 1
        ],
        'max_features': [
            best_random_params['max_features']
        ],
        'bootstrap': [
            best_random_params['bootstrap']
        ]
    }

    # Time series split for cross-validation
    time_series_split = TimeSeriesSplit(n_splits=3)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=time_series_split,
        verbose=3,
        n_jobs=8,
        pre_dispatch='2*n_jobs',
        return_train_score=True
    )

    # Perform the fine grid search
    grid_search.fit(X_train, y_train)

    # Output the best parameters and score
    print("Best Parameters (Narrow):", grid_search.best_params_)
    print("Best Score (Narrow):", grid_search.best_score_)

    return grid_search.best_params_, grid_search.best_score_

def hybrid_search_moderate(X_train, y_train, best_random_params):
    """
    Perform a moderate fine grid search based on the best parameters from random search.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    # Define a moderate grid around the best parameters
    param_grid = {
        'n_estimators': [
            max(10, best_random_params['n_estimators'] - 100), 
            max(10, best_random_params['n_estimators'] - 50), 
            best_random_params['n_estimators'], 
            best_random_params['n_estimators'] + 50, 
            best_random_params['n_estimators'] + 100
        ],
        'max_depth': [
            max(1, best_random_params['max_depth'] - 2), 
            best_random_params['max_depth'] - 1, 
            best_random_params['max_depth'], 
            best_random_params['max_depth'] + 1, 
            best_random_params['max_depth'] + 2
        ],
        'max_features': [
            best_random_params['max_features']
        ],
        'bootstrap': [
            best_random_params['bootstrap']
        ]
    }

    # Time series split for cross-validation
    time_series_split = TimeSeriesSplit(n_splits=3)  # Reduce splits to speed up

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=time_series_split,
        verbose=3,
        n_jobs=8,  # Use 8 cores
        pre_dispatch='2*n_jobs',
        return_train_score=True
    )

    # Perform the fine grid search
    grid_search.fit(X_train, y_train)

    # Output the best parameters and score
    print("Best Parameters (Moderate):", grid_search.best_params_)
    print("Best Score (Moderate):", grid_search.best_score_)

    return grid_search.best_params_, grid_search.best_score_

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

# Randon search hypter-parameter tuning

# Time series split for cross-validation
time_series_split = TimeSeriesSplit(n_splits=5)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42  # Fixed seed for reproducibility
)
param_distributions = {
    'n_estimators': [100, 150, 200],
    'max_depth': [4, 6, 8],
    'max_features': ['sqrt', 'log2', 'None'],  # Remove `None`
    'bootstrap': [True, False]        # Fix bootstrap to reduce variability
}


random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=time_series_split,
    verbose=3,
    random_state=42,
    n_jobs=1,  # Use 8 cores
    pre_dispatch='2*n_jobs'
)
random_search.fit(X_train, y_train)
# Fit random search
random_search.fit(X_train, y_train)

# Print the best parameters from random search
best_random_params = random_search.best_params_
print("Best Parameters from Random Search:", best_random_params)

# Narrow Fine grid search
best_narrow_params, best_narrow_score = hybrid_search_narrow(X_train, y_train, best_random_params)

# Moderate fine grid search
# best_moderate_params, best_moderate_score = hybrid_search_moderate(X_train, y_train, best_random_params)


# Save the GridSearchCV object
joblib.dump(best_narrow_params, 'grid_rf_search_results_1min_ema5_nn10_narrow.pkl')