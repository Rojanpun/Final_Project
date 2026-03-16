# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:09:12 2026

@author: Kenny
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

# 1. Download Data for Top 3 Cryptocurrencies
# BTC = Bitcoin, ETH = Ethereum, SOL = Solana
crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

print("--- Downloading Crypto Data ---")
# Crypto markets trade 24/7, so we get more data points than stocks!
raw_crypto = yf.download(crypto_tickers, start="2020-01-01", end="2026-03-01")['Close']

# 2. CREATE DATAFRAME 1: Raw Closing Prices (For LSTM)
df_prices_crypto = raw_crypto.ffill().dropna()
df_prices_crypto.columns = ['Bitcoin', 'Ethereum', 'Solana']

# 3. CREATE DATAFRAME 2: Log Returns (For ARIMA & GARCH)
# Crypto volatility is much higher, which makes GARCH very interesting here.
df_returns_crypto = np.log(df_prices_crypto / df_prices_crypto.shift(1)).dropna()

# 4. CREATE DATAFRAME 3: Scaled Data (For LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled_crypto = pd.DataFrame(
    scaler.fit_transform(df_prices_crypto), 
    columns=df_prices_crypto.columns, 
    index=df_prices_crypto.index
)

# --- STATIONARITY CHECK ---
def check_stationarity(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print(f"Result for {name}: Stationary (Ready for ARIMA/GARCH)")
    else:
        print(f"Result for {name}: Non-Stationary")

print("\n--- Stationarity Test on Bitcoin Returns ---")
check_stationarity(df_returns_crypto['Bitcoin'], "Bitcoin")

print("\nCrypto DataFrames created: df_prices_crypto, df_returns_crypto, df_scaled_crypto")