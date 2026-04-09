# =============================
# ADVANCED AUTO PORTFOLIO (TARGET + AUTO STOP + SMART REBALANCE)
# =============================
# pip install streamlit yfinance pandas numpy matplotlib
# streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🤖 Advanced Auto Trading Portfolio")

# =============================
# USER INPUT
# =============================
symbols_input = st.sidebar.text_input("Symbols (comma separated)", "AAPL,MSFT,TSLA,BTC-USD,ETH-USD")
symbols = [s.strip() for s in symbols_input.split(",")]

initial_balance = st.sidebar.number_input("Initial Capital ($)", value=1000)
target_value = st.sidebar.number_input("Target Value ($)", value=1500)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100
rebalance_threshold = st.sidebar.slider("Rebalance Drawdown (%)", 3, 20, 8) / 100

# =============================
# DATA
# =============================
@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    df.dropna(inplace=True)
    return df

# =============================
# STRATEGY
# =============================
def run_strategy(df, capital):
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()

    df['TR'] = df[['High','Low','Close']].max(axis=1) - df[['High','Low','Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    balance = capital
    in_position = False
    entry = 0
    position_size = 0
    equity = []

    for i in range(20, len(df)):
        try:
            price = float(df['Close'].iloc[i])
            atr = float(df['ATR'].iloc[i])
            ema9 = float(df['EMA9'].iloc[i])
            ema21 = float(df['EMA21'].iloc[i])
        except (ValueError, TypeError):
            equity.append(balance)
            continue

        # Skip if any value is invalid (NaN check: NaN != NaN)
        if price != price or atr != atr or ema9 != ema9 or ema21 != ema21 or atr == 0:
            equity.append(balance)
            continue

        # ENTRY
        if not in_position and ema9 > ema21:
            position_size = (balance * risk_per_trade) / atr
            entry = price
            in_position = True

        # EXIT
        if in_position:
            sl = entry - atr * 1.5
            tp = entry + atr * 3

            if price <= sl or price >= tp or ema9 < ema21:
                pnl = (price - entry) * position_size
                balance += pnl
                in_position = False

        equity.append(balance)

    return equity

# =============================
# RUN PORTFOLIO
# =============================
results = {}
alloc = initial_balance / len(symbols)

for sym in symbols:
    try:
        df = get_data(sym)
        results[sym] = run_strategy(df, alloc)
    except Exception as e:
        st.warning(f"Error loading {sym}: {e}")

# =============================
# COMBINE PORTFOLIO
# =============================
if results:
    min_len = min(len(v) for v in results.values())
    portfolio = []

    for i in range(min_len):
        total = sum(results[s][i] for s in results)
        portfolio.append(total)

    portfolio = pd.Series(portfolio)

    # =============================
    # TARGET + AUTO STOP
    # =============================
    stopped = False
    locked_curve = []

    for val in portfolio:
        if val >= target_value:
            stopped = True
        if stopped:
            locked_curve.append(target_value)
        else:
            locked_curve.append(val)

    portfolio = pd.Series(locked_curve)

    # =============================
    # SMART REBALANCE
    # =============================
    peak = portfolio.cummax()
    drawdown = (portfolio - peak) / peak

    if drawdown.iloc[-1] < -rebalance_threshold:
        st.warning("⚠️ Smart Rebalance Triggered")

    # =============================
    # METRICS
    # =============================
    ret = portfolio.iloc[-1] / initial_balance - 1
    max_dd = drawdown.min()

    col1, col2, col3 = st.columns(3)
    col1.metric("Return", f"{ret:.2%}")
    col2.metric("Max Drawdown", f"{max_dd:.2%}")
    col3.metric("Final Value", f"${portfolio.iloc[-1]:.2f}")

    # =============================
    # CHART
    # =============================
    st.subheader("📈 Portfolio Curve")
    fig, ax = plt.subplots()
    ax.plot(portfolio.values)
    st.pyplot(fig)

    # =============================
    # BREAKDOWN
    # =============================
    st.subheader("📊 Asset Breakdown")
    for s in results:
        st.write(s)
else:
    st.error("No data could be loaded for the specified symbols.")

# =============================
# NOTES
# =============================
st.info("""
SYSTEM FEATURES:
- Multi-asset trading
- Auto execution logic (simulated)
- Target-based stop
- Smart rebalance on drawdown

NEXT STEP:
→ Connect broker API for live trading
→ Add real-time data
""")
