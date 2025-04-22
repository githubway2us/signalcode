import time
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from datetime import datetime
import pytz
from flask import Flask, render_template

app = Flask(__name__)

# ฟังก์ชันดึงข้อมูลจาก Binance
def get_binance_data(symbol, interval='1m', limit=200):
    try:
        url = f'https://api.binance.com/api/v1/klines'
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params)
        response.raise_for_status()  # ตรวจสอบสถานะ HTTP
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        tz = pytz.timezone('Asia/Bangkok')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ฟังก์ชันคำนวณตัวชี้วัด
def compute_indicators(df):
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ฟังก์ชันตรวจจับสัญญาณ
def detect_advanced_signals(df):
    signals = []
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Close'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    if y_pred[-1] > df['Close'].iloc[-1]:
        signals.append("Uptrend Signal (Polynomial Regression)")
    elif y_pred[-1] < df['Close'].iloc[-1]:
        signals.append("Downtrend Signal (Polynomial Regression)")
    
    sentiment_score = random.uniform(-1, 1)
    if sentiment_score > 0.5:
        signals.append("Positive Sentiment")
    elif sentiment_score < -0.5:
        signals.append("Negative Sentiment")
    
    return signals

# ฟังก์ชันคำนวณจุดเข้า
def calculate_entry_price(df, signals):
    row = df.iloc[-1]
    entry_price = None
    stop_loss = None
    take_profit = None
    
    if 'Uptrend Signal (Polynomial Regression)' in signals:
        entry_price = row['Low']
        stop_loss = row['Low'] * 0.99
        take_profit = row['High'] * 1.02
    elif 'Downtrend Signal (Polynomial Regression)' in signals:
        entry_price = row['High']
        stop_loss = row['High'] * 1.01
        take_profit = row['Low'] * 0.98
    
    return entry_price, stop_loss, take_profit

# ฟังก์ชันดึงข้อมูลทั้งหมด
def fetch_data():
    symbols = ['BTCUSDT', 'ETHUSDT']
    results = []
    
    for symbol in symbols:
        try:
            df = get_binance_data(symbol=symbol, interval='1m', limit=200)
            if df is None:
                raise Exception("Failed to fetch data")
            df = compute_indicators(df)
            signals = detect_advanced_signals(df)
            entry_price, stop_loss, take_profit = calculate_entry_price(df, signals)
            
            result = {
                'symbol': symbol,
                'price': df.iloc[-1]['Close'],
                'timestamp': df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'signals': signals if signals else ['No signals'],
                'entry_price': f"{entry_price:.2f}" if entry_price else 'N/A',
                'stop_loss': f"{stop_loss:.2f}" if stop_loss else 'N/A',
                'take_profit': f"{take_profit:.2f}" if take_profit else 'N/A'
            }
            results.append(result)
        except Exception as e:
            results.append({
                'symbol': symbol,
                'price': 'Error',
                'timestamp': datetime.now(pytz.timezone('Asia/Bangkok')).strftime('%Y-%m-%d %H:%M:%S'),
                'signals': [f"Error: {e}"],
                'entry_price': 'N/A',
                'stop_loss': 'N/A',
                'take_profit': 'N/A'
            })
    
    return results

# เส้นทางหลักของ Flask
@app.route('/')
def index():
    data = fetch_data()
    return render_template('index.html', data=data)

# รัน Flask ในโหมด background
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    import threading
    def run_flask():
        app.run(debug=False, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # อัปเดตทุก 30 นาที
    while True:
        print("Updating data...")
        time.sleep(1800)  # 30 นาที
    