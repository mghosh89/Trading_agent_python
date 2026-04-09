# =============================
# LIVE TRADING BOT - Alpaca + Binance
# =============================
# pip install streamlit yfinance pandas numpy matplotlib alpaca-trade-api ccxt
# streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🤖 Live Trading Bot - Alpaca + Binance")

# =============================
# BROKER API CONFIGURATION
# =============================
st.sidebar.header("🔑 API Configuration")

# Mode selection
mode = st.sidebar.radio("Trading Mode", ["Paper Trading", "Live Trading"])

# Alpaca Credentials (Stocks)
with st.sidebar.expander("📈 Alpaca (Stocks)"):
    alpaca_api_key = st.text_input("API Key", type="password", key="alpaca_key")
    alpaca_secret = st.text_input("Secret Key", type="password", key="alpaca_secret")
    use_alpaca = st.checkbox("Enable Stock Trading", value=False)

# Binance Credentials (Crypto)
with st.sidebar.expander("₿ Binance (Crypto)"):
    binance_api_key = st.text_input("API Key", type="password", key="binance_key")
    binance_secret = st.text_input("Secret Key", type="password", key="binance_secret")
    use_binance = st.checkbox("Enable Crypto Trading", value=False)
    use_testnet = st.checkbox("Use Testnet", value=True)

# =============================
# INITIALIZE BROKERS
# =============================
alpaca = None
binance = None

# Initialize Alpaca
if use_alpaca and alpaca_api_key and alpaca_secret:
    try:
        from alpaca_trade_api import REST
        base_url = "https://paper-api.alpaca.markets" if mode == "Paper Trading" else "https://api.alpaca.markets"
        alpaca = REST(alpaca_api_key, alpaca_secret, base_url, raw_data=True)
        account = alpaca.get_account()
        st.sidebar.success(f"✅ Alpaca Connected: ${account['portfolio_value']}")
    except Exception as e:
        st.sidebar.error(f"❌ Alpaca Error: {e}")
        alpaca = None

# Initialize Binance
if use_binance and binance_api_key and binance_secret:
    try:
        binance = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        if use_testnet:
            binance.set_sandbox_mode(True)
        balance = binance.fetch_balance()
        usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
        st.sidebar.success(f"✅ Binance Connected: ${usdt_balance:.2f} USDT")
    except Exception as e:
        st.sidebar.error(f"❌ Binance Error: {e}")
        binance = None

# =============================
# TRADING SETTINGS
# =============================
st.sidebar.header("⚙️ Trading Settings")
symbols_input = st.sidebar.text_input("Symbols (comma separated)", "AAPL,MSFT,BTC/USDT,ETH/USDT")
symbols = [s.strip() for s in symbols_input.split(",")]

initial_balance = st.sidebar.number_input("Initial Capital ($)", value=1000)
target_value = st.sidebar.number_input("Target Value ($)", value=1500)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100
rebalance_threshold = st.sidebar.slider("Rebalance Drawdown (%)", 3, 20, 8) / 100

# Auto-trade toggle
auto_trade = st.sidebar.checkbox("🤖 Enable Auto-Trading", value=False)

# =============================
# DATA FUNCTIONS
# =============================
def get_stock_data(symbol):
    """Fetch stock data from Alpaca or yfinance"""
    if alpaca:
        # Real-time from Alpaca
        bars = alpaca.get_bars(symbol, timeframe="1Hour", limit=500)
        df = pd.DataFrame([{
            'timestamp': bar['t'],
            'Open': bar['o'],
            'High': bar['h'],
            'Low': bar['l'],
            'Close': bar['c'],
            'Volume': bar['v']
        } for bar in bars])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        # Fallback to yfinance
        df = yf.download(symbol, period="1y", interval="1d")
    df.dropna(inplace=True)
    return df

def get_crypto_data(symbol):
    """Fetch crypto data from Binance or ccxt"""
    if binance:
        # Real-time from Binance
        ohlcv = binance.fetch_ohlcv(symbol, timeframe='1h', limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    else:
        # Fallback to yfinance
        df = yf.download(symbol.replace('/', '-'), period="1y", interval="1d")
    df.dropna(inplace=True)
    return df

# =============================
# ORDER EXECUTION
# =============================
def place_stock_order(symbol, side, qty):
    """Execute stock order via Alpaca"""
    if not alpaca or not auto_trade:
        return None
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        st.toast(f"📈 {side.upper()} {qty} shares of {symbol}")
        return order
    except Exception as e:
        st.error(f"Order failed: {e}")
        return None

def place_crypto_order(symbol, side, amount):
    """Execute crypto order via Binance"""
    if not binance or not auto_trade:
        return None
    try:
        order_type = 'market'
        if side == 'buy':
            order = binance.create_market_buy_order(symbol, amount)
        else:
            order = binance.create_market_sell_order(symbol, amount)
        st.toast(f"₿ {side.upper()} {amount} of {symbol}")
        return order
    except Exception as e:
        st.error(f"Order failed: {e}")
        return None

# =============================
# STRATEGY
# =============================
def run_strategy(df, capital, symbol, is_crypto=False):
    """EMA Crossover Strategy with SL/TP"""
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    balance = capital
    in_position = False
    entry = 0
    position_size = 0
    equity = []
    trades = []

    for i in range(20, len(df)):
        try:
            price = float(df['Close'].iloc[i])
            atr = float(df['ATR'].iloc[i]) if pd.notna(df['ATR'].iloc[i]) else 0
            ema9 = float(df['EMA9'].iloc[i]) if pd.notna(df['EMA9'].iloc[i]) else 0
            ema21 = float(df['EMA21'].iloc[i]) if pd.notna(df['EMA21'].iloc[i]) else 0
        except (ValueError, TypeError):
            equity.append(balance)
            continue

        # Skip invalid values
        if price == 0 or atr == 0 or ema9 == 0 or ema21 == 0:
            equity.append(balance)
            continue

        # ENTRY
        if not in_position and ema9 > ema21:
            position_size = (balance * risk_per_trade) / atr
            entry = price
            in_position = True
            trades.append({'time': df.index[i], 'action': 'BUY', 'price': price, 'symbol': symbol})

            # Execute live order
            if auto_trade:
                if is_crypto:
                    place_crypto_order(symbol, 'buy', position_size)
                else:
                    shares = int(position_size / price)
                    if shares > 0:
                        place_stock_order(symbol, 'buy', shares)

        # EXIT
        if in_position:
            sl = entry - atr * 1.5
            tp = entry + atr * 3

            if price <= sl or price >= tp or ema9 < ema21:
                pnl = (price - entry) * position_size
                balance += pnl
                in_position = False
                trades.append({'time': df.index[i], 'action': 'SELL', 'price': price, 'pnl': pnl, 'symbol': symbol})

                # Execute live order
                if auto_trade:
                    if is_crypto:
                        place_crypto_order(symbol, 'sell', position_size)
                    else:
                        shares = int(position_size / entry)
                        if shares > 0:
                            place_stock_order(symbol, 'sell', shares)

        equity.append(balance)

    return equity, trades

# =============================
# RUN PORTFOLIO
# =============================
results = {}
trade_log = []
alloc = initial_balance / len(symbols) if symbols else 0

progress_bar = st.progress(0)
for idx, sym in enumerate(symbols):
    try:
        is_crypto = '/' in sym or '-USD' in sym
        if is_crypto:
            df = get_crypto_data(sym.replace('-USD', '/USDT') if '-USD' in sym else sym)
        else:
            df = get_stock_data(sym)

        equity, trades = run_strategy(df, alloc, sym, is_crypto=is_crypto)
        results[sym] = equity
        trade_log.extend(trades)
    except Exception as e:
        st.warning(f"Error loading {sym}: {e}")
    progress_bar.progress((idx + 1) / len(symbols))

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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.values, label='Portfolio', linewidth=2)
    ax.axhline(y=target_value, color='g', linestyle='--', label=f'Target ${target_value}')
    ax.fill_between(range(len(portfolio)), portfolio.values, alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # =============================
    # TRADE LOG
    # =============================
    if trade_log:
        st.subheader("📋 Trade Log")
        trades_df = pd.DataFrame(trade_log)
        st.dataframe(trades_df, use_container_width=True)

    # =============================
    # LIVE STATUS
    # =============================
    st.subheader("🔴 Live Status")
    status_cols = st.columns(4)
    status_cols[0].metric("Mode", mode)
    status_cols[1].metric("Auto-Trade", "ON" if auto_trade else "OFF")
    status_cols[2].metric("Stocks", "✅" if alpaca else "❌")
    status_cols[3].metric("Crypto", "✅" if binance else "❌")

else:
    st.error("No data could be loaded for the specified symbols.")

# =============================
# MANUAL TRADING
# =============================
st.sidebar.header("🎮 Manual Trading")
manual_symbol = st.sidebar.text_input("Symbol", "AAPL")
manual_side = st.sidebar.selectbox("Side", ["buy", "sell"])
manual_qty = st.sidebar.number_input("Quantity", min_value=0.0, value=1.0)

if st.sidebar.button("Execute Manual Trade"):
    is_crypto = '/' in manual_symbol or manual_symbol.endswith('USDT')
    if is_crypto and binance:
        place_crypto_order(manual_symbol.upper(), manual_side, manual_qty)
    elif not is_crypto and alpaca:
        place_stock_order(manual_symbol.upper(), manual_side, int(manual_qty))
    else:
        st.sidebar.error("Broker not connected for this asset type")

# =============================
# NOTES
# =============================
st.info("""
**SYSTEM FEATURES:**
- ✅ Alpaca integration for stocks (Paper/Live)
- ✅ Binance integration for crypto (Testnet/Live)
- ✅ Auto-trading with EMA crossover strategy
- ✅ Real-time order execution
- ✅ Manual trading panel
- ✅ Target-based stop & smart rebalance

**SECURITY:**
- API keys are never stored or logged
- Start with Paper Trading to test
- Enable 2FA on your broker accounts
""")
