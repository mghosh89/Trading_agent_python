# =============================
# AUTONOMOUS TRADING AGENT (TARGET + LOSS PROTECTION)
# =============================
# pip install streamlit yfinance pandas numpy matplotlib
# streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🤖 Autonomous Trading Agent")

# =============================
# USER INPUT
# =============================
symbols_input = st.sidebar.text_input("Symbols", "AAPL,MSFT,TSLA,BTC-USD")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

initial_balance = st.sidebar.number_input("Initial Capital ($)", value=10000)
target_value = st.sidebar.number_input("Target Value ($)", value=15000)
max_loss_pct = st.sidebar.slider("Max Portfolio Loss (%)", 5, 50, 20) / 100
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100

# =============================
# DATA
# =============================
@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    df.dropna(inplace=True)
    return df

# =============================
# INDICATORS
# =============================
def indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    return df

# =============================
# AGENT LOGIC
# =============================
def run_agent(df, capital):
    df = indicators(df)

    balance = capital
    peak = capital

    in_position = False
    entry = 0
    size = 0

    equity = []

    for i in range(20, len(df)):
        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]

        # STOP if target reached
        if balance >= target_value:
            equity.append(target_value)
            continue

        # STOP if loss exceeded
        if balance <= initial_balance * (1 - max_loss_pct):
            equity.append(balance)
            continue

        # ENTRY (trend following)
        if not in_position and df['EMA9'].iloc[i] > df['EMA21'].iloc[i] and atr > 0:
            size = (balance * risk_per_trade) / atr
            entry = price
            in_position = True

        # EXIT
        if in_position:
            sl = entry - atr * 1.5
            tp = entry + atr * 3

            if price <= sl or price >= tp or df['EMA9'].iloc[i] < df['EMA21'].iloc[i]:
                pnl = (price - entry) * size
                balance += pnl
                in_position = False

        # TRACK
        peak = max(peak, balance)
        equity.append(balance)

    return equity

# =============================
# RUN PORTFOLIO
# =============================
if symbols:
    alloc = initial_balance / len(symbols)
    results = {}

    for sym in symbols:
        try:
            df = get_data(sym)
            results[sym] = run_agent(df, alloc)
        except:
            st.warning(f"Error loading {sym}")

    # Combine
    min_len = min(len(v) for v in results.values())
    portfolio = []

    for i in range(min_len):
        total = sum(results[s][i] for s in results)
        portfolio.append(total)

    portfolio = pd.Series(portfolio)

    # Metrics
    ret = portfolio.iloc[-1] / initial_balance - 1
    peak = portfolio.cummax()
    dd = (portfolio - peak) / peak

    # UI
    col1, col2, col3 = st.columns(3)
    col1.metric("Return", f"{ret:.2%}")
    col2.metric("Max Drawdown", f"{dd.min():.2%}")
    col3.metric("Final Value", f"${portfolio.iloc[-1]:,.2f}")

    # Chart
    st.subheader("📈 Portfolio Growth")
    fig, ax = plt.subplots()
    ax.plot(portfolio.values)
    ax.axhline(target_value, linestyle="--")
    st.pyplot(fig)

    # Status
    if portfolio.iloc[-1] >= target_value:
        st.success("🎯 Target Achieved - Agent Stopped Trading")
    elif portfolio.iloc[-1] <= initial_balance * (1 - max_loss_pct):
        st.error("🛑 Max Loss Hit - Agent Stopped Trading")

else:
    st.info("Enter symbols to start")

# =============================
# SUMMARY
# =============================
st.info("""
AGENT LOGIC:
- Allocates capital across assets
- Trades automatically using EMA + ATR
- Stops when target reached
- Stops when max loss hit

This simulates a real autonomous trading system.
""")
