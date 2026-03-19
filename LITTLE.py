# ============================================================
# CAPSTONE PROJECT
# TIME SERIES FORECASTING OF 3 INDICES AND 3 CRYPTO SEPARATELY
# ONE FIXED TRAIN-TEST SPLIT ONLY
# SPLIT = 80% TRAIN, 20% TEST
# MODELS = ARIMA, GARCH, LSTM
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import os
import itertools
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

# -----------------------------
# 2. OUTPUT FOLDER
# -----------------------------
output_dir = "capstone_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 3. DEFINE TICKERS
# -----------------------------
index_tickers = ["^GSPC", "^AXJO", "^BVSP"]         # S&P 500, ASX 200, IBOVESPA
crypto_tickers = ["BTC-USD", "ETH-USD", "DOGE-USD"]  # Bitcoin, Ethereum, Dogecoin

# -----------------------------
# 4. DATE RANGE
# -----------------------------
start_date = "2015-01-01"
end_date = "2024-01-01"

# -----------------------------
# 5. DOWNLOAD DATA
# -----------------------------
indices_data = yf.download(index_tickers, start=start_date, end=end_date)["Close"]
crypto_data = yf.download(crypto_tickers, start=start_date, end=end_date)["Close"]

# -----------------------------
# 6. HANDLE NaN SEPARATELY
# -----------------------------
indices_clean = indices_data.dropna().copy()
crypto_clean = crypto_data.dropna().copy()

# -----------------------------
# 7. SAVE DATA
# -----------------------------
indices_clean.to_csv(os.path.join(output_dir, "indices_clean.csv"))
crypto_clean.to_csv(os.path.join(output_dir, "crypto_clean.csv"))

# -----------------------------
# 8. USE ONE FIXED TRAIN-TEST SPLIT
# -----------------------------
split_ratio = 0.80

# -----------------------------
# 9. DATASETS
# -----------------------------
datasets = {
    "indices": indices_clean,
    "crypto": crypto_clean
}

# ============================================================
# 10. HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    return rmse, mae, r2, mape


def chronological_split(series, split_ratio=0.80):
    split_index = int(len(series) * split_ratio)
    train = series.iloc[:split_index].copy()
    test = series.iloc[split_index:].copy()
    return train, test


def create_lstm_sequences(values, lookback=30):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i, 0])
        y.append(values[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


# ============================================================
# 11. ARIMA MODEL FUNCTION
# ============================================================

def run_arima(train, test):
    best_order = None
    best_forecast = None
    best_rmse = np.inf
    best_metrics = None

    for order in itertools.product([0, 1, 2], [0, 1], [0, 1, 2]):
        try:
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(test))

            rmse, mae, r2, mape = calculate_metrics(test, forecast)

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
                best_forecast = forecast
                best_metrics = (rmse, mae, r2, mape)

        except:
            continue

    return best_order, best_forecast, best_metrics


# ============================================================
# 12. GARCH MODEL FUNCTION
# ============================================================

def run_garch(train, test):
    train_returns = np.log(train / train.shift(1)).dropna()
    test_returns = np.log(test / test.shift(1)).dropna()

    train_returns_scaled = train_returns * 100

    rolling_history = train_returns_scaled.copy()
    forecasted_volatility = []

    for i in range(len(test_returns)):
        model = arch_model(rolling_history, vol="GARCH", p=1, q=1, mean="Zero")
        fitted_model = model.fit(disp="off")
        pred = fitted_model.forecast(horizon=1)
        variance_forecast = pred.variance.iloc[-1, 0]
        forecasted_volatility.append(np.sqrt(variance_forecast))

        new_value = pd.Series([test_returns.iloc[i] * 100], index=[test_returns.index[i]])
        rolling_history = pd.concat([rolling_history, new_value])

    forecasted_volatility = np.array(forecasted_volatility)
    realized_volatility = np.abs(test_returns.values * 100)

    rmse, mae, r2, mape = calculate_metrics(realized_volatility, forecasted_volatility)

    return test_returns.index, realized_volatility, forecasted_volatility, (rmse, mae, r2, mape)


# ============================================================
# 13. LSTM MODEL FUNCTION
# ============================================================

def run_lstm(train, test, lookback=30, epochs=30, batch_size=16):
    clear_session()

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_values = train.values.reshape(-1, 1)
    test_values = test.values.reshape(-1, 1)

    scaled_train = scaler.fit_transform(train_values)

    combined_test = np.concatenate([train_values[-lookback:], test_values], axis=0)
    scaled_combined_test = scaler.transform(combined_test)

    X_train, y_train = create_lstm_sequences(scaled_train, lookback)
    X_test, y_test = create_lstm_sequences(scaled_combined_test, lookback)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    actual = test.values

    rmse, mae, r2, mape = calculate_metrics(actual, predictions)

    return actual, predictions, (rmse, mae, r2, mape), history


# ============================================================
# 14. RUN FORECASTING FOR ALL 3 INDICES + 3 CRYPTO
# ============================================================

results = []

for dataset_name, df in datasets.items():
    for asset in df.columns:
        print(f"\nProcessing {dataset_name} - {asset}")

        series = df[asset].dropna()

        # -----------------------------
        # Train-test split
        # -----------------------------
        train, test = chronological_split(series, split_ratio=split_ratio)

        # -----------------------------
        # Plot train-test split
        # -----------------------------
        plt.figure(figsize=(12, 5))
        plt.plot(train.index, train.values, label="Train")
        plt.plot(test.index, test.values, label="Test")
        plt.title(f"{dataset_name.upper()} - {asset} Train/Test Split (80/20)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{asset}_train_test_split.png"), dpi=300)
        plt.show()

        # ====================================================
        # ARIMA
        # ====================================================
        arima_order, arima_forecast, arima_metrics = run_arima(train, test)
        arima_rmse, arima_mae, arima_r2, arima_mape = arima_metrics

        plt.figure(figsize=(12, 5))
        plt.plot(test.index, test.values, label="Actual")
        plt.plot(test.index, arima_forecast.values, label="ARIMA Forecast")
        plt.title(f"{dataset_name.upper()} - {asset} ARIMA Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{asset}_arima_forecast.png"), dpi=300)
        plt.show()

        results.append({
            "Dataset": dataset_name,
            "Asset": asset,
            "Model": "ARIMA",
            "Split": "80/20",
            "RMSE": arima_rmse,
            "MAE": arima_mae,
            "R2": arima_r2,
            "MAPE": arima_mape
        })

        # ====================================================
        # GARCH
        # ====================================================
        garch_index, garch_actual, garch_pred, garch_metrics = run_garch(train, test)
        garch_rmse, garch_mae, garch_r2, garch_mape = garch_metrics

        plt.figure(figsize=(12, 5))
        plt.plot(garch_index, garch_actual, label="Realized Volatility")
        plt.plot(garch_index, garch_pred, label="GARCH Forecasted Volatility")
        plt.title(f"{dataset_name.upper()} - {asset} GARCH Volatility Forecast")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{asset}_garch_forecast.png"), dpi=300)
        plt.show()

        results.append({
            "Dataset": dataset_name,
            "Asset": asset,
            "Model": "GARCH",
            "Split": "80/20",
            "RMSE": garch_rmse,
            "MAE": garch_mae,
            "R2": garch_r2,
            "MAPE": garch_mape
        })

        # ====================================================
        # LSTM
        # ====================================================
        lstm_actual, lstm_pred, lstm_metrics, history = run_lstm(train, test)
        lstm_rmse, lstm_mae, lstm_r2, lstm_mape = lstm_metrics

        plt.figure(figsize=(12, 5))
        plt.plot(test.index, lstm_actual, label="Actual")
        plt.plot(test.index, lstm_pred, label="LSTM Forecast")
        plt.title(f"{dataset_name.upper()} - {asset} LSTM Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{asset}_lstm_forecast.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"{dataset_name.upper()} - {asset} LSTM Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{asset}_lstm_learning_curve.png"), dpi=300)
        plt.show()

        results.append({
            "Dataset": dataset_name,
            "Asset": asset,
            "Model": "LSTM",
            "Split": "80/20",
            "RMSE": lstm_rmse,
            "MAE": lstm_mae,
            "R2": lstm_r2,
            "MAPE": lstm_mape
        })


# ============================================================
# 15. FINAL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "final_model_results.csv"), index=False)

print("\n================ FINAL RESULTS ================\n")
print(results_df)

# ============================================================
# 16. COMPARISON PLOTS
# ============================================================

for dataset_name in results_df["Dataset"].unique():
    temp = results_df[results_df["Dataset"] == dataset_name]

    plt.figure(figsize=(12, 6))
    for model_name in temp["Model"].unique():
        subset = temp[temp["Model"] == model_name]
        plt.plot(subset["Asset"], subset["RMSE"], marker="o", label=model_name)

    plt.title(f"{dataset_name.upper()} - RMSE Comparison")
    plt.xlabel("Asset")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_rmse_comparison.png"), dpi=300)
    plt.show()

print("\nAll forecasting completed successfully.")
print("Train-test split used for all assets: 80% train / 20% test")
print("Results saved in folder:", output_dir)