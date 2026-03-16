# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:07:36 2026

@author: Kenny
"""


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

# 1. Download Data (Excluding NEPSE)
tickers = ["^GSPC", "^BVSP", "^AXJO"]
raw_data = yf.download(tickers, start="2018-01-01", end="2026-03-01")['Close']

# 2. CREATE DATAFRAME 1: Raw Closing Prices (For LSTM)
# LSTMs look at price levels and long-term trends
df_prices = raw_data.ffill().dropna()
df_prices.columns = ['ASX_200', 'Ibovespa', 'S&P_500']

# 3. CREATE DATAFRAME 2: Log Returns (For ARIMA & GARCH)
# Statistical models require stationary data (no constant upward trend)
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# 4. CREATE DATAFRAME 3: Scaled Data (For LSTM Neural Network)
# LSTMs perform poorly with large numbers; we scale everything between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_prices), 
    columns=df_prices.columns, 
    index=df_prices.index
)

# --- STATIONARITY CHECK (Required for ARIMA) ---
def check_stationarity(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("Result: Stationary (Ready for ARIMA/GARCH)")
    else:
        print("Result: Non-Stationary (Needs differencing)")

print("--- Stationarity Test on S&P 500 Returns ---")
check_stationarity(df_returns['S&P_500'], "S&P 500")

print("\nDataFrames created: df_prices, df_returns, df_scaled")