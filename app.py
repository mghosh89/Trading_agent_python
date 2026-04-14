# =============================
# AUTONOMOUS TRADING AGENT (FIXED)
# =============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# =============================
# SAFE ALPACA IMPORT
# =============================
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

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

API_KEY = st.sidebar.text_input("Alpaca API Key", "")
SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", "")
BASE_URL = st.sidebar.selectbox(
    "Alpaca Base URL",
    ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"]
)

initial_balance = st.sidebar.number_input("Initial Capital ($)", value=10000)
target_value = st.sidebar.number_input("Target Value ($)", value=15000)
max_loss_pct = st.sidebar.slider("Max Portfolio Loss (%)", 5, 50, 20) / 100
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100

# =============================
# SESSION STATE
# =============================
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
# SIMULATION LOGIC
# =============================
def run_agent(df, capital):
    df = indicators(df)

    balance = capital
    in_position = False
    entry = 0
    size = 0

    equity = []

    for i in range(20, len(df)):
        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]

        # STOP CONDITIONS
        if balance >= target_value or balance <= initial_balance * (1 - max_loss_pct):
            equity.append(balance)
            continue

        # ENTRY
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

        equity.append(balance)

    return equity

# =============================
# LIVE TRADING
# =============================
def get_live_data(symbol):
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    df.dropna(inplace=True)
    return indicators(df)

def has_position(api, symbol):
    try:
        api.get_position(symbol)
        return True
    except:
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

    size = (balance * risk_per_trade) / atr
    qty = int(size)

    # ENTRY
    if latest['EMA9'] > latest['EMA21'] and not has_position(api, symbol):
        if qty > 0:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            st.success(f"BUY {symbol} {qty}")

    # EXIT
    if has_position(api, symbol):
        position = api.get_position(symbol)
        entry_price = float(position.avg_entry_price)

        sl = entry_price - atr * 1.5
        tp = entry_price + atr * 3

        if price <= sl or price >= tp or latest['EMA9'] < latest['EMA21']:
            api.submit_order(
                symbol=symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            st.warning(f"SELL {symbol}")

def run_live_trading(api, symbols):
    st.session_state.live_trading_active = True

    for i in range(50):  # safety loop
        if not st.session_state.live_trading_active:
            break

        account = api.get_account()
        equity = float(account.equity)

        st.info(f"Portfolio: ${equity:.2f}")

        if equity >= target_value or equity <= initial_balance * (1 - max_loss_pct):
            st.session_state.live_trading_active = False
            break

        for sym in symbols:
            try:
                trade_symbol_live(api, sym)
            except Exception as e:
                st.error(f"{sym}: {e}")

        time.sleep(5)

# =============================
# RUN
# =============================
if symbols and st.session_state.run_trading:

    # LIVE TRADING
    if use_live_trading:
        if not ALPACA_AVAILABLE:
            st.error("Alpaca not installed. Add to requirements.txt")
        elif not API_KEY or not SECRET_KEY:
            st.error("Enter Alpaca API keys")
        else:
            try:
                api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
                api.get_account()
                st.success("Connected to Alpaca")
                run_live_trading(api, symbols)
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # SIMULATION
    else:
        results = {}
        alloc = initial_balance / len(symbols)

        with st.spinner("Running simulation..."):
            for sym in symbols:
                try:
                    df = get_data(sym)
                    results[sym] = run_agent(df, alloc)
                except:
                    st.warning(f"Error with {sym}")

        if not results:
            st.error("No data available.")
            st.stop()

        min_len = min(len(v) for v in results.values())

        portfolio = [
            sum(results[s][i] for s in results)
            for i in range(min_len)
        ]

        portfolio = pd.Series(portfolio)

        # METRICS
        ret = portfolio.iloc[-1] / initial_balance - 1
        peak = portfolio.cummax()
        dd = (portfolio - peak) / peak

        col1, col2, col3 = st.columns(3)
        col1.metric("Return", f"{ret:.2%}")
        col2.metric("Max Drawdown", f"{dd.min():.2%}")
        col3.metric("Final Value", f"${portfolio.iloc[-1]:,.2f}")

        # CHART
        fig, ax = plt.subplots()
        ax.plot(portfolio.values)
        ax.axhline(target_value, linestyle="--")
        st.pyplot(fig)

# =============================
# BUTTONS
# =============================
def start():
    st.session_state.run_trading = True

def stop():
    st.session_state.run_trading = False
    st.session_state.live_trading_active = False

col1, col2 = st.columns(2)
col1.button("🚀 Start", on_click=start, use_container_width=True)
col2.button("🛑 Stop", on_click=stop, use_container_width=True)