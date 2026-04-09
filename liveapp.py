import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import time

# =============================
# CONFIG
# =============================
API_KEY = "YOUR_API_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"  # paper trading first

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

symbols = ["AAPL", "MSFT", "TSLA"]
risk_per_trade = 0.02
target_value = 15000
max_loss_pct = 0.2

initial_balance = float(api.get_account().cash)

# =============================
# INDICATORS
# =============================
def get_data(symbol):
    df = yf.download(symbol, period="3mo", interval="1d")
    df.dropna(inplace=True)

    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    return df

# =============================
# POSITION CHECK
# =============================
def has_position(symbol):
    positions = api.list_positions()
    for p in positions:
        if p.symbol == symbol:
            return True
    return False

# =============================
# TRADING LOGIC
# =============================
def trade_symbol(symbol):
    df = get_data(symbol)
    latest = df.iloc[-1]

    price = latest['Close']
    atr = latest['ATR']

    if atr == 0 or np.isnan(atr):
        return

    account = api.get_account()
    balance = float(account.cash)

    position_size = (balance * risk_per_trade) / atr

    # ENTRY
    if latest['EMA9'] > latest['EMA21'] and not has_position(symbol):
        qty = int(position_size)

        if qty > 0:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print(f"BUY {symbol} {qty}")

    # EXIT
    if has_position(symbol):
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
            print(f"SELL {symbol}")

# =============================
# MAIN AGENT LOOP
# =============================
while True:
    account = api.get_account()
    equity = float(account.equity)

    print("Portfolio Value:", equity)

    # STOP CONDITIONS
    if equity >= target_value:
        print("🎯 Target reached. Stopping trading.")
        break

    if equity <= initial_balance * (1 - max_loss_pct):
        print("🛑 Max loss hit. Stopping trading.")
        break

    for sym in symbols:
        try:
            trade_symbol(sym)
        except Exception as e:
            print(f"Error with {sym}: {e}")

    time.sleep(60 * 15)  # run every 15 minutes
    #BASE_URL = "https://paper-api.alpaca.markets"
    #pip install alpaca-trade-api