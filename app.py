# =============================
# INSTALL REQUIREMENTS
# =============================
# pip install streamlit yfinance pandas numpy matplotlib ccxt

# =============================
# RUN COMMAND
# =============================
# streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("📈 Trading Strategy Dashboard")

# =============================
# SIDEBAR INPUTS
# =============================
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Crypto"])
symbol = st.sidebar.text_input("Symbol", "AAPL" if asset_type=="Stock" else "BTC/USDT")
initial_balance = st.sidebar.number_input("Initial Balance", value=100)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100

# =============================
# DATA FETCH
# =============================
@st.cache_data
def get_stock_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    return df

@st.cache_data
def get_crypto_data(symbol):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp','Open','High','Low','Close','Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if asset_type == "Stock":
    df = get_stock_data(symbol)
else:
    df = get_crypto_data(symbol)

# =============================
# INDICATORS
# =============================
df['EMA9'] = df['Close'].ewm(span=9).mean()
df['EMA21'] = df['Close'].ewm(span=21).mean()

# ATR
df['H-L'] = df['High'] - df['Low']
df['H-C'] = abs(df['High'] - df['Close'].shift())
df['L-C'] = abs(df['Low'] - df['Close'].shift())
df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
df['ATR'] = df['TR'].rolling(14).mean()

# =============================
# STRATEGY (WITH SL/TP)
# =============================
balance = initial_balance
in_position = False
entry_price = 0

equity = []

for i in range(1, len(df)):
    price = df['Close'].iloc[i]
    atr = df['ATR'].iloc[i]
    ema9 = df['EMA9'].iloc[i]
    ema21 = df['EMA21'].iloc[i]

    # Skip if any value is NaN
    if pd.isna(price) or pd.isna(atr) or pd.isna(ema9) or pd.isna(ema21):
        equity.append(balance)
        continue

    # Entry
    if not in_position and ema9 > ema21:
        in_position = True
        entry_price = price

    # Exit
    if in_position:
        stop_loss = entry_price - atr * 1.5
        take_profit = entry_price + atr * 3

        if price <= stop_loss or price >= take_profit or ema9 < ema21:
            trade_return = (price - entry_price) / entry_price
            balance *= (1 + trade_return)
            in_position = False

    equity.append(balance)

# =============================
# METRICS
# =============================
equity_series = pd.Series(equity)
returns = equity_series.pct_change().fillna(0)

max_drawdown = (equity_series / equity_series.cummax() - 1).min()
total_return = equity_series.iloc[-1] / initial_balance - 1

# =============================
# UI DISPLAY
# =============================
st.subheader("📊 Performance Metrics")
st.write(f"Total Return: {total_return:.2%}")
st.write(f"Max Drawdown: {max_drawdown:.2%}")
st.write(f"Final Balance: ${equity_series.iloc[-1]:.2f}")

# Equity Chart
st.subheader("📈 Equity Curve")
fig, ax = plt.subplots()
ax.plot(equity_series.values)
ax.set_title("Equity Growth")
st.pyplot(fig)

# Price + EMA
st.subheader("📉 Price & EMA")
fig2, ax2 = plt.subplots()
ax2.plot(df['Close'], label='Price')
ax2.plot(df['EMA9'], label='EMA9')
ax2.plot(df['EMA21'], label='EMA21')
ax2.legend()
st.pyplot(fig2)

# =============================
# LIVE REFRESH BUTTON
# =============================
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# =============================
# NOTES
# =============================
# This is a learning system, not a guaranteed profitable bot.
# Add real execution APIs (Binance/Alpaca) for live trading.
