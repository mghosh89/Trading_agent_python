# =============================
# FREE TRADING BOT - Open Data Sources
# =============================
# Uses free data: Yahoo Finance, Alpha Vantage, Finnhub
# No broker API required - paper trading simulation
# pip install streamlit yfinance pandas numpy matplotlib requests
# streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🤖 Free Trading Bot - Open Data Sources")

# =============================
# FREE DATA SOURCE CONFIG
# =============================
st.sidebar.header("📡 Data Sources")
data_source = st.sidebar.selectbox(
    "Primary Data Source",
    ["Yahoo Finance (Free)", "Alpha Vantage (Free API)", "Finnhub (Free API)"]
)

# API Keys for free data sources
if "Alpha Vantage" in data_source:
    alpha_vantage_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
elif "Finnhub" in data_source:
    finnhub_key = st.sidebar.text_input("Finnhub API Key", type="password")

# =============================
# TRADING SETTINGS
# =============================
st.sidebar.header("⚙️ Trading Settings")
symbols_input = st.sidebar.text_input(
    "Symbols (comma separated)",
    "AAPL,MSFT,GOOGL,TSLA,AMZN,JPM"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

initial_balance = st.sidebar.number_input("Initial Capital ($)", value=10000)
target_value = st.sidebar.number_input("Target Value ($)", value=15000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100
rebalance_threshold = st.sidebar.slider("Rebalance Drawdown (%)", 3, 20, 8) / 100

# Strategy selection
strategy = st.sidebar.selectbox(
    "Trading Strategy",
    ["EMA Crossover", "RSI + MACD", "Bollinger Bands", "Combined"]
)

# =============================
# FREE DATA FETCH FUNCTIONS
# =============================

def get_yahoo_data(symbol, period="1y", interval="1d"):
    """Fetch data from Yahoo Finance (free, no API key)"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.warning(f"Yahoo Finance error for {symbol}: {e}")
        return None

def get_alpha_vantage_data(symbol, api_key):
    """Fetch data from Alpha Vantage (free tier: 25 calls/day)"""
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": "full"
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "Time Series (Daily)" not in data:
            return None

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        st.warning(f"Alpha Vantage error for {symbol}: {e}")
        return None

def get_finnhub_data(symbol, api_key):
    """Fetch data from Finnhub (free tier: 60 calls/minute)"""
    try:
        # Get candle data
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (365 * 24 * 60 * 60)  # 1 year back

        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": start_time,
            "to": end_time,
            "token": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"]
        }, index=pd.to_datetime(data["t"], unit="s"))
        return df
    except Exception as e:
        st.warning(f"Finnhub error for {symbol}: {e}")
        return None

def get_stock_data(symbol, data_source_choice, **kwargs):
    """Unified data fetcher"""
    if data_source_choice == "Yahoo Finance (Free)":
        return get_yahoo_data(symbol)
    elif data_source_choice == "Alpha Vantage (Free API)":
        return get_alpha_vantage_data(symbol, kwargs.get("api_key"))
    elif data_source_choice == "Finnhub (Free API)":
        return get_finnhub_data(symbol, kwargs.get("api_key"))
    return None

# =============================
# TECHNICAL INDICATORS
# =============================
def calculate_indicators(df):
    """Calculate technical indicators"""
    # EMA
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # ATR
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift())
    df["L-C"] = abs(df["Low"] - df["Close"].shift())
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    return df

# =============================
# TRADING STRATEGIES
# =============================
def ema_crossover_strategy(df, i):
    """EMA 9/21 Crossover"""
    ema9 = float(df["EMA9"].iloc[i])
    ema21 = float(df["EMA21"].iloc[i])
    if ema9 == 0 or ema21 == 0:
        return False, False
    return ema9 > ema21, ema9 < ema21  # buy, sell

def rsi_macd_strategy(df, i):
    """RSI + MACD Strategy"""
    rsi = float(df["RSI"].iloc[i])
    macd = float(df["MACD"].iloc[i])
    macd_signal = float(df["MACD_Signal"].iloc[i])

    buy = rsi < 70 and macd > macd_signal
    sell = rsi > 70 or macd < macd_signal
    return buy, sell

def bollinger_strategy(df, i):
    """Bollinger Bands Strategy"""
    price = float(df["Close"].iloc[i])
    upper = float(df["BB_Upper"].iloc[i])
    lower = float(df["BB_Lower"].iloc[i])

    buy = price < lower  # Price below lower band
    sell = price > upper  # Price above upper band
    return buy, sell

def combined_strategy(df, i):
    """Combine multiple strategies"""
    ema_buy, ema_sell = ema_crossover_strategy(df, i)
    rsi_buy, rsi_sell = rsi_macd_strategy(df, i)
    bb_buy, bb_sell = bollinger_strategy(df, i)

    # Require at least 2 signals to agree
    buy_signals = sum([ema_buy, rsi_buy, bb_buy])
    sell_signals = sum([ema_sell, rsi_sell, bb_sell])

    return buy_signals >= 2, sell_signals >= 2

def get_strategy_signals(df, i, strategy_name):
    """Get buy/sell signals based on strategy"""
    if strategy_name == "EMA Crossover":
        return ema_crossover_strategy(df, i)
    elif strategy_name == "RSI + MACD":
        return rsi_macd_strategy(df, i)
    elif strategy_name == "Bollinger Bands":
        return bollinger_strategy(df, i)
    else:  # Combined
        return combined_strategy(df, i)

# =============================
# BACKTEST ENGINE
# =============================
def run_backtest(df, capital, strategy_name):
    """Run strategy backtest"""
    df = calculate_indicators(df)

    balance = capital
    in_position = False
    entry = 0
    position_size = 0
    equity = []
    trades = []

    for i in range(30, len(df)):  # Start after indicators warm-up
        try:
            price = float(df["Close"].iloc[i])
            atr = float(df["ATR"].iloc[i]) if pd.notna(df["ATR"].iloc[i]) else 0
        except (ValueError, TypeError):
            equity.append(balance)
            continue

        if price == 0 or price != price:  # NaN check
            equity.append(balance)
            continue

        buy_signal, sell_signal = get_strategy_signals(df, i, strategy_name)

        # ENTRY
        if not in_position and buy_signal and atr > 0:
            position_size = (balance * risk_per_trade) / atr
            entry = price
            in_position = True
            trades.append({
                "time": df.index[i],
                "action": "BUY",
                "price": price,
                "size": position_size
            })

        # EXIT
        if in_position:
            sl = entry - atr * 1.5
            tp = entry + atr * 3

            if price <= sl or price >= tp or sell_signal:
                pnl = (price - entry) * position_size
                balance += pnl
                in_position = False
                trades.append({
                    "time": df.index[i],
                    "action": "SELL",
                    "price": price,
                    "pnl": pnl,
                    "return_pct": (price - entry) / entry * 100
                })

        equity.append(balance)

    return equity, trades, df

# =============================
# RUN PORTFOLIO
# =============================
api_key = None
if "Alpha Vantage" in data_source:
    api_key = alpha_vantage_key if "alpha_vantage_key" in locals() else None
elif "Finnhub" in data_source:
    api_key = finnhub_key if "finnhub_key" in locals() else None

results = {}
trade_log = []
all_data = {}

if symbols:
    alloc = initial_balance / len(symbols)
    progress_bar = st.progress(0)

    for idx, sym in enumerate(symbols):
        try:
            df = get_stock_data(sym, data_source, api_key=api_key)
            if df is not None and len(df) > 30:
                equity, trades, df = run_backtest(df, alloc, strategy)
                results[sym] = equity
                trade_log.extend([{**t, "symbol": sym} for t in trades])
                all_data[sym] = df
            else:
                st.warning(f"Insufficient data for {sym}")
        except Exception as e:
            st.warning(f"Error processing {sym}: {e}")

        progress_bar.progress((idx + 1) / len(symbols))

# =============================
# DISPLAY RESULTS
# =============================
if results:
    min_len = min(len(v) for v in results.values())
    portfolio = []

    for i in range(min_len):
        total = sum(results[s][i] for s in results)
        portfolio.append(total)

    portfolio = pd.Series(portfolio)

    # Target + Auto Stop
    stopped = False
    locked_curve = []
    for val in portfolio:
        if val >= target_value:
            stopped = True
        locked_curve.append(target_value if stopped else val)
    portfolio = pd.Series(locked_curve)

    # Metrics
    peak = portfolio.cummax()
    drawdown = (portfolio - peak) / peak
    ret = portfolio.iloc[-1] / initial_balance - 1
    max_dd = drawdown.min()
    sharpe = np.sqrt(252) * (portfolio.pct_change().mean() / portfolio.pct_change().std()) if portfolio.pct_change().std() != 0 else 0

    # Dashboard
    st.subheader("📊 Performance Dashboard")
    cols = st.columns(4)
    cols[0].metric("Total Return", f"{ret:.2%}")
    cols[1].metric("Max Drawdown", f"{max_dd:.2%}")
    cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")
    cols[3].metric("Final Value", f"${portfolio.iloc[-1]:,.2f}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Equity Curve")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(portfolio.values, label="Portfolio", linewidth=2, color="blue")
        ax.axhline(y=target_value, color="green", linestyle="--", label=f"Target ${target_value:,}")
        ax.fill_between(range(len(portfolio)), portfolio.values, alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("📉 Drawdown")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.fill_between(range(len(drawdown)), drawdown.values * 100, alpha=0.5, color="red")
        ax2.plot(drawdown.values * 100, color="darkred", linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # Individual Asset Performance
    st.subheader("📊 Asset Performance")
    asset_metrics = []
    for sym in results:
        eq = pd.Series(results[sym])
        asset_ret = eq.iloc[-1] / alloc - 1
        asset_dd = (eq / eq.cummax() - 1).min()
        asset_metrics.append({
            "Symbol": sym,
            "Return": f"{asset_ret:.2%}",
            "Max DD": f"{asset_dd:.2%}",
            "Final Value": f"${eq.iloc[-1]:,.2f}"
        })
    st.dataframe(pd.DataFrame(asset_metrics), use_container_width=True)

    # Trade Log
    if trade_log:
        st.subheader("📋 Trade History")
        trades_df = pd.DataFrame(trade_log)
        st.dataframe(trades_df, use_container_width=True)

        # Win Rate
        closed_trades = [t for t in trade_log if t["action"] == "SELL"]
        if closed_trades:
            wins = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
            win_rate = wins / len(closed_trades) * 100
            st.metric("Win Rate", f"{win_rate:.1f}% ({wins}/{len(closed_trades)})")

    # Price Chart with Indicators
    st.subheader("📉 Price Charts with Signals")
    chart_symbol = st.selectbox("Select symbol to view", list(all_data.keys()))
    if chart_symbol in all_data:
        df = all_data[chart_symbol]
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(df.index, df["Close"], label="Price", linewidth=1)

        if strategy in ["EMA Crossover", "Combined"]:
            ax3.plot(df.index, df["EMA9"], label="EMA9", alpha=0.7)
            ax3.plot(df.index, df["EMA21"], label="EMA21", alpha=0.7)

        if strategy in ["Bollinger Bands", "Combined"]:
            ax3.plot(df.index, df["BB_Upper"], label="BB Upper", linestyle="--", alpha=0.5)
            ax3.plot(df.index, df["BB_Lower"], label="BB Lower", linestyle="--", alpha=0.5)
            ax3.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.1)

        # Mark trades
        symbol_trades = [t for t in trade_log if t["symbol"] == chart_symbol]
        for t in symbol_trades:
            if t["action"] == "BUY":
                ax3.scatter(t["time"], t["price"], color="green", marker="^", s=100, zorder=5)
            else:
                ax3.scatter(t["time"], t["price"], color="red", marker="v", s=100, zorder=5)

        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

else:
    st.info("Enter stock symbols above to start backtesting")

# =============================
# DATA SOURCE INFO
# =============================
with st.sidebar.expander("ℹ️ About Data Sources"):
    st.write("""
    **Yahoo Finance (Free)**
    - No API key required
    - Real-time for some markets
    - 15-min delayed for NYSE/NASDAQ
    - Unlimited requests

    **Alpha Vantage (Free)**
    - Free tier: 25 API calls/day
    - Get key: alphavantage.co/support
    - Real-time US stocks

    **Finnhub (Free)**
    - Free tier: 60 calls/minute
    - Get key: finnhub.io
    - WebSocket available
    """)

# =============================
# STRATEGY INFO
# =============================
st.sidebar.markdown("---")
st.sidebar.info("""
**Available Strategies:**
- EMA Crossover: 9/21 EMA crossover
- RSI + MACD: Momentum + trend
- Bollinger Bands: Mean reversion
- Combined: Multi-signal confirmation

**Risk Management:**
- ATR-based position sizing
- 1.5x ATR stop loss
- 3x ATR take profit
- Target-based exit
""")
