# ============================================================
# CAPSTONE PROJECT — Financial Asset Analysis with LSTM, ARIMA & GARCH
# ============================================================

# --- Core data and math libraries ---
import yfinance as yf          # pulls stock/crypto price data from Yahoo Finance
import pandas as pd            # for working with tables/dataframes
import numpy as np             # for numerical operations and arrays
from sklearn.preprocessing import MinMaxScaler   # scales data between 0 and 1
import matplotlib.pyplot as plt                  # for plotting charts

# --- Deep learning + reproducibility setup ---
import tensorflow as tf
import random

# Fixing random seeds so results stay the same every time we run the code
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================================================
# SECTION 1: DOWNLOAD RAW DATA
# ============================================================

# Tickers for stock indices and cryptocurrencies
indices_tickers = ["^GSPC", "^BVSP", "^AXJO"]         # S&P 500, Ibovespa, ASX 200
crypto_tickers = ["BTC-USD", "ETH-USD", "DOGE-USD"]    # Bitcoin, Ethereum, Dogecoin

# Download closing prices — indices go back to 2000, crypto to 2014
df_indices_raw = yf.download(indices_tickers, start="2000-01-01", end="2024-03-01")['Close']
df_crypto_raw = yf.download(crypto_tickers, start="2014-01-01", end="2024-03-01")['Close']

# ============================================================
# SECTION 2: MISSING VALUE CHECK
# ============================================================

# Print how many NAs exist in each column before cleaning
print("--- Missing Value Audit ---")
print("Indices NAs:\n", df_indices_raw.isna().sum())
print("Crypto NAs:\n", df_crypto_raw.isna().sum())

# --- Visual heatmaps to see where the gaps are ---
import seaborn as sns
import matplotlib.pyplot as plt

# Yellow = missing, Purple = present (viridis colormap)
sns.heatmap(df_indices_raw.isna(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()

sns.heatmap(df_crypto_raw.isna(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()

# Drop any rows that have missing values
df_indices = df_indices_raw.dropna()
df_crypto = df_crypto_raw.dropna()

# Re-check NAs after dropping — should be 0 across the board
print("--- Missing Value Audit ---")
print("Pre-COVID NAs:\n", df_indices_raw.isna().sum())
print("Pre-COVID NAs:\n", df_crypto_raw.isna().sum())

# Rename columns to cleaner labels
df_indices.columns = ['ASX_200', 'Ibovespa', 'S&P_500']
df_crypto.columns = ['BTC', 'DOGE', 'ETH']

# Final check on the cleaned dataframes
print("--- Missing Value Audit ---")
print("Pre-COVID NAs:\n", df_indices.isna().sum())
print("Pre-COVID NAs:\n", df_crypto.isna().sum())

# ============================================================
# SECTION 3: SPLIT INTO PRE-COVID AND COVID PERIODS
# ============================================================

# Indices: pre-2020 vs during COVID years
df_ind_pre = df_indices.loc['2000-01-01':'2019-12-31']
df_ind_covid = df_indices.loc['2020-01-01':'2023-12-31']

# Crypto: pre-2020 vs during COVID years
df_cry_pre = df_crypto.loc['2014-01-01':'2019-12-31']
df_cry_covid = df_crypto.loc['2020-01-01':'2023-12-31']

# ============================================================
# SECTION 4: SCALE THE DATA (0 to 1) FOR CHARTING
# ============================================================

# MinMaxScaler normalises values so different assets are comparable on one chart
def scale_data(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

df_ind_pre_s = scale_data(df_ind_pre)
df_ind_covid_s = scale_data(df_ind_covid)
df_cry_pre_s = scale_data(df_cry_pre)
df_cry_covid_s = scale_data(df_cry_covid)

# ============================================================
# SECTION 5: PLOT — STOCK INDICES
# ============================================================

# Side-by-side charts: pre-COVID and COVID period for indices
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
df_ind_pre_s.plot(ax=axes1[0], grid=True, title='Indices: Pre-COVID (2000-2019)')
df_ind_covid_s.plot(ax=axes1[1], grid=True, title='Indices: COVID Period (2020-2023)')
axes1[0].set_ylabel('Scaled Value')
axes1[1].set_ylabel('Scaled Value')
plt.tight_layout()
plt.show()

# ============================================================
# SECTION 6: PLOT — CRYPTOCURRENCIES
# ============================================================

# Using recognisable colours: gold for BTC, dark for ETH, tan for DOGE
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
cry_colors = ['#F7931A', '#323232', '#C2A633']
df_cry_pre_s.plot(ax=axes2[0], grid=True, title='Crypto: Pre-COVID (2014-2019)', color=cry_colors)
df_cry_covid_s.plot(ax=axes2[1], grid=True, title='Crypto: COVID Period (2020-2023)', color=cry_colors)
axes2[0].set_ylabel('Scaled Value')
axes2[1].set_ylabel('Scaled Value')
plt.tight_layout()
plt.show()


# ==============================================================
# ==============================================================
# MAIN MODEL SECTION — LSTM + ARIMA + GARCH HYBRID
# ==============================================================
# ==============================================================

import os
import warnings
warnings.filterwarnings("ignore")   # suppress noisy warnings during model training

# Re-importing everything cleanly for the modelling section
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Stats models ---
from statsmodels.tsa.arima.model import ARIMA   # traditional time series model
from arch import arch_model                      # GARCH for volatility modelling

# --- Machine learning tools ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Keras layers for building the LSTM network ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

# ============================================================
# STEP 1: SETUP — OUTPUT FOLDER AND TICKERS
# ============================================================

# All charts and files will be saved here
output_dir = "capstone_hybrid_outputs"
os.makedirs(output_dir, exist_ok=True)   # creates the folder if it doesn't exist yet

indices_tickers = ["^GSPC", "^AXJO", "^BVSP"]
crypto_tickers = ["BTC-USD", "ETH-USD", "DOGE-USD"]

# ============================================================
# STEP 2: DOWNLOAD AND CLEAN DATA
# ============================================================

df_indices_raw = yf.download(indices_tickers, start="2000-01-01", end="2024-03-01")['Close']
df_crypto_raw = yf.download(crypto_tickers, start="2014-01-01", end="2024-03-01")['Close']

# Drop rows where any asset has a missing price
df_indices_clean = df_indices_raw.dropna()
df_crypto_clean = df_crypto_raw.dropna()

# Bundle the four time regimes into a dictionary so we can loop over them later
datasets = {
    "Indices_Pre_Covid": df_indices_clean.loc['2000-01-01':'2019-12-31'],
    "Indices_Covid":     df_indices_clean.loc['2020-01-01':'2023-12-31'],
    "Crypto_Pre_Covid":  df_crypto_clean.loc['2014-01-01':'2019-12-31'],
    "Crypto_Covid":      df_crypto_clean.loc['2020-01-01':'2023-12-31']
}

# ============================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================

# Calculates the three error metrics we care about for model evaluation
def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MSE": mse, "R2": r2}

# Splits the series into 80% train and 20% test — keeps time order intact
def chronological_split(series, split_ratio=0.80):
    split_index = int(len(series) * split_ratio)
    return series.iloc[:split_index].copy(), series.iloc[split_index:].copy()

# Turns a 1D price series into (X, y) pairs where X = last 30 days, y = next day
def create_lstm_sequences(values, lookback=30):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i, 0])
        y.append(values[i, 0])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)

# ============================================================
# STEP 4: MODEL FUNCTIONS
# ============================================================

# ARIMA(5,1,0) — a classic econometric model for financial time series
# Walks through the test set one step at a time (rolling forecast)
def run_arima(train, test):
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])   # add real value to history after each step
    return np.array(predictions)

# GARCH(1,1) — models how volatile the asset is, not the price itself
# Output is used to draw uncertainty bands around the ARIMA forecast
def run_garch(train, test):
    returns = train.pct_change().dropna() * 100   # convert prices to % returns
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    forecasts = model_fit.forecast(horizon=len(test))
    return np.sqrt(forecasts.variance.values[-1, :])   # return as std deviation

# LSTM — a neural network that learns patterns across sequences of prices
# Two stacked LSTM layers with dropout to reduce overfitting
def run_lstm(train, test, lookback=30):
    clear_session()   # wipe previous model from memory to avoid conflicts
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale training data to [0, 1]
    train_values = train.values.reshape(-1, 1)
    scaled_train = scaler.fit_transform(train_values)

    # Prepend last 30 days of training data so the first test window is valid
    combined_test = np.concatenate([train_values[-lookback:], test.values.reshape(-1, 1)], axis=0)
    scaled_combined_test = scaler.transform(combined_test)

    # Build input sequences
    X_train, y_train = create_lstm_sequences(scaled_train, lookback)
    X_test, _ = create_lstm_sequences(scaled_combined_test, lookback)

    # Build the LSTM architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),   # first LSTM layer
        Dropout(0.2),                                                   # randomly drops 20% of neurons
        LSTM(32),                                                       # second LSTM layer
        Dropout(0.2),
        Dense(1)                                                        # single output: next day price
    ])
    model.compile(optimizer="adam", loss="mse")

    # Train — early stopping kicks in if validation loss stops improving after 3 epochs
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    # Predict and reverse the scaling to get back actual prices
    predictions_scaled = model.predict(X_test, verbose=0)
    return scaler.inverse_transform(predictions_scaled).flatten()

# ============================================================
# STEP 5: MAIN LOOP — RUN ALL MODELS ON ALL ASSETS
# ============================================================

results = []

for dataset_name, df in datasets.items():
    for asset in df.columns:
        print(f"\n>>> Processing {asset} ({dataset_name})")
        series = df[asset].dropna()
        if len(series) < 100: continue   # skip if not enough data to train on

        train, test = chronological_split(series)

        # Run all three models
        lstm_preds = run_lstm(train, test)
        arima_preds = run_arima(train, test)
        garch_vol = run_garch(train, test)   # this gives us the volatility band width

        # Score each model
        l_metrics = calculate_metrics(test, lstm_preds)
        a_metrics = calculate_metrics(test, arima_preds)

        # --- PLOT: actual vs forecasts + GARCH uncertainty band ---
        plt.figure(figsize=(14, 6))

        plt.plot(test.index, test.values, label="Actual Price", color='black', alpha=0.6)
        plt.plot(test.index, lstm_preds, label="LSTM Forecast", color='orange')
        plt.plot(test.index, arima_preds, label="ARIMA Forecast", color='blue', linestyle='--')

        # Shaded band: ARIMA prediction ± GARCH volatility
        plt.fill_between(test.index,
                         arima_preds - garch_vol[:len(test)],
                         arima_preds + garch_vol[:len(test)],
                         color='gray', alpha=0.2, label='GARCH Volatility Band')

        plt.title(f"Hybrid Model Comparison: {asset} ({dataset_name})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        # Save to disk and display
        plt.savefig(os.path.join(output_dir, f"{asset}_{dataset_name}_hybrid.png"))
        plt.show()

        # Append both model results to the running list
        results.append({"Asset": asset, "Regime": dataset_name, "Model": "ARIMA", **a_metrics})
        results.append({"Asset": asset, "Regime": dataset_name, "Model": "LSTM", **l_metrics})

# ============================================================
# STEP 6: SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Average RMSE, MSE, R2 grouped by model type and market regime
print("\n--- COMPARATIVE PERFORMANCE (RMSE, MSE, R2) ---")
summary = results_df.groupby(['Model', 'Regime'])[['RMSE', 'MSE', 'R2']].mean()
print(summary)

# ============================================================
# STEP 7: EXPORT RESULTS TO EXCEL
# ============================================================

import pandas as pd

# Save the long-format results in the same folder as the script
results_df.to_excel('results_df.xlsx', index=False)
print("results_df.xlsx saved successfully!")

# ============================================================
# STEP 8: WIDE FORMAT — EASIER TO READ IN EXCEL
# ============================================================

# Pivot so each asset/regime is one row, with ARIMA and LSTM side by side
wide_df = results_df.pivot_table(
    index=['Asset', 'Regime'],
    columns='Model',
    values=['RMSE', 'MSE', 'R2']
)

# Flatten the column names: e.g. ('RMSE', 'ARIMA')  'RMSE_ARIMA'
wide_df.columns = [f'{val}_{mod}' for val, mod in wide_df.columns]
wide_df.reset_index(inplace=True)

# Save wide format in the same folder
wide_df.to_excel('wide_results.xlsx', index=False)
print("wide_results.xlsx saved successfully!")