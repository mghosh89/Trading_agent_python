# =============================
# AUTONOMOUS TRADING AGENT (TARGET + LOSS PROTECTION)
# =============================
# pip install streamlit yfinance pandas numpy matplotlib alpaca-trade-api
# streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import time

st.set_page_config(layout="wide")
st.title("🤖 Autonomous Trading Agent")

# =============================
# USER INPUT
# =============================
symbols_input = st.sidebar.text_input("Symbols", "AAPL,MSFT,TSLA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# Live Trading Settings
st.sidebar.subheader("Live Trading (Alpaca)")
use_live_trading = st.sidebar.checkbox("Enable Live Trading")
API_KEY = st.sidebar.text_input("Alpaca API Key", "YOUR_API_KEY")
SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", "YOUR_SECRET_KEY")
BASE_URL = st.sidebar.selectbox("Alpaca Base URL", ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"])

initial_balance = st.sidebar.number_input("Initial Capital ($)", value=10000)
target_value = st.sidebar.number_input("Target Value ($)", value=15000)
max_loss_pct = st.sidebar.slider("Max Portfolio Loss (%)", 5, 50, 20) / 100
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100

# Initialize session state
if 'run_trading' not in st.session_state:
    st.session_state.run_trading = False

if 'live_trading_active' not in st.session_state:
    st.session_state.live_trading_active = False

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
# SIMULATION AGENT LOGIC
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
# LIVE TRADING FUNCTIONS
# =============================
def get_live_data(symbol):
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    df.dropna(inplace=True)

    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    return df

def has_position(api, symbol):
    positions = api.list_positions()
    for p in positions:
        if p.symbol == symbol:
            return True
    return False

def trade_symbol_live(api, symbol):
    df = get_live_data(symbol)
    latest = df.iloc[-1]

    price = latest['Close']
    atr = latest['ATR']

    if atr == 0 or np.isnan(atr):
        return

    account = api.get_account()
    balance = float(account.cash)

    position_size = (balance * risk_per_trade) / atr

    # ENTRY
    if latest['EMA9'] > latest['EMA21'] and not has_position(api, symbol):
        qty = int(position_size)

        if qty > 0:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            st.success(f"BUY {symbol} {qty} shares at ${price:.2f}")

    # EXIT
    if has_position(api, symbol):
        position = api.get_position(symbol)
        entry_price = float(position.avg_entry_price)

        stop_loss = entry_price - atr * 1.5
        take_profit = entry_price + atr * 3

        if price <= stop_loss or price >= take_profit or latest['EMA9'] < latest['EMA21']:
            api.submit_order(
                symbol=symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            st.warning(f"SELL {symbol} at ${price:.2f}")

def run_live_trading(api, symbols):
    account = api.get_account()
    initial_balance_live = float(account.cash)
    equity = float(account.equity)

    st.info(f"Starting live trading | Cash: ${initial_balance_live:.2f} | Equity: ${equity:.2f}")

    status_container = st.empty()
    progress_bar = st.progress(0)

    iteration = 0
    max_iterations = 100  # Safety limit

    while st.session_state.live_trading_active and iteration < max_iterations:
        account = api.get_account()
        equity = float(account.equity)

        status_container.info(f"**Portfolio Value:** ${equity:.2f} | **Iteration:** {iteration + 1}")

        # STOP CONDITIONS
        if equity >= target_value:
            status_container.success("🎯 Target reached. Stopping trading.")
            st.session_state.live_trading_active = False
            break

        if equity <= initial_balance_live * (1 - max_loss_pct):
            status_container.error("🛑 Max loss hit. Stopping trading.")
            st.session_state.live_trading_active = False
            break

        for sym in symbols:
            try:
                with st.expander(f"Trading {sym}", expanded=False):
                    trade_symbol_live(api, sym)
            except Exception as e:
                st.error(f"Error with {sym}: {e}")

        progress_bar.progress(min((iteration + 1) / 20, 1.0))
        iteration += 1
        time.sleep(5)  # Run every 5 seconds for demo (change to 60*15 for real 15-min intervals)

    progress_bar.empty()

# =============================
# RUN SIMULATION
# =============================
if symbols and st.session_state.run_trading and not use_live_trading:
    alloc = initial_balance / len(symbols)
    results = {}

    with st.spinner("Running backtest simulation..."):
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
        ax.axhline(target_value, linestyle="--", label=f"Target: ${target_value}")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        st.pyplot(fig)

        # Status
        if portfolio.iloc[-1] >= target_value:
            st.success("🎯 Target Achieved - Agent Stopped Trading")
        elif portfolio.iloc[-1] <= initial_balance * (1 - max_loss_pct):
            st.error("🛑 Max Loss Hit - Agent Stopped Trading")

elif symbols and st.session_state.run_trading and use_live_trading:
    if API_KEY == "YOUR_API_KEY" or SECRET_KEY == "YOUR_SECRET_KEY":
        st.error("Please enter your Alpaca API credentials to enable live trading")
    else:
        try:
            api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
            # Test connection
            api.get_account()
            st.success("Connected to Alpaca API")

            # Start live trading in a thread-like manner using st.empty()
            st.session_state.live_trading_active = True
            run_live_trading(api, symbols)

        except Exception as e:
            st.error(f"Failed to connect to Alpaca: {e}")

else:
    st.info("Configure settings in the sidebar and click 'Start Trading Agent' below")

# =============================
# SUBMIT BUTTON
# =============================
def start_trading():
    st.session_state.run_trading = True

def stop_trading():
    st.session_state.run_trading = False
    st.session_state.live_trading_active = False

col1, col2 = st.columns(2)

with col1:
    st.button("🚀 Start Trading Agent", type="primary", use_container_width=True, on_click=start_trading)

with col2:
    st.button("🛑 Stop Trading", type="secondary", use_container_width=True, on_click=stop_trading)

# =============================
# SUMMARY
# =============================
st.info("""
**AGENT LOGIC:**
- Allocates capital across assets
- Trades automatically using EMA + ATR
- Stops when target reached
- Stops when max loss hit
- Supports both simulation and live trading via Alpaca

This simulates a real autonomous trading system.
""")
