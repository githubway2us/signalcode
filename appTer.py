import time
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random  # สำหรับการจำลอง Sentiment Analysis
from datetime import datetime
import pytz  # สำหรับการแปลงเวลา

# ฟังก์ชันดึงข้อมูลจาก Binance
def get_binance_data(symbol, interval='1m', limit=200):
    url = f'https://api.binance.com/api/v1/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
    
    # แปลง timestamp เป็นเวลา
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # ตั้งโซนเวลาเป็น "Asia/Bangkok" (ประเทศไทย)
    tz = pytz.timezone('Asia/Bangkok')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)
    
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    return df

# ฟังก์ชันคำนวณตัวชี้วัดต่างๆ
def compute_indicators(df):
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ฟังก์ชันตรวจจับสัญญาณขั้นสูงของ BlackRock
def detect_advanced_signals(df):
    signals = []
    
    # ตรวจสอบการวิเคราะห์ราคาโดยใช้ Linear Regression (แบบ Polynomial)
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Close'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # คำนวณราคาคาดการณ์
    y_pred = model.predict(X_poly)
    
    # หากราคาคาดการณ์สูงขึ้น, นั่นคือสัญญาณขาขึ้น
    if y_pred[-1] > df['Close'].iloc[-1]:
        signals.append("Uptrend Signal (Polynomial Regression)")
    
    # ถ้าราคาคาดการณ์ต่ำลง, นั่นคือสัญญาณขาลง
    elif y_pred[-1] < df['Close'].iloc[-1]:
        signals.append("Downtrend Signal (Polynomial Regression)")

    # สัญญาณจากการวิเคราะห์ความรู้สึก (Sentiment Analysis)
    sentiment_score = random.uniform(-1, 1)  # จำลอง Sentiment Score ระหว่าง -1 (negative) ถึง 1 (positive)
    if sentiment_score > 0.5:
        signals.append("Positive Sentiment")
    elif sentiment_score < -0.5:
        signals.append("Negative Sentiment")
    
    return signals

# ฟังก์ชันคำนวณจุดเข้า
def calculate_entry_price(df, signals):
    # ตรวจสอบค่าสุดท้ายของข้อมูล (ล่าสุด)
    row = df.iloc[-1]
    
    entry_price = None
    stop_loss = None
    take_profit = None
    
    # ตรวจสอบว่ามีสัญญาณขาขึ้น (Uptrend Signal)
    if 'Uptrend Signal (Polynomial Regression)' in signals:
        entry_price = row['Low']  # ใช้ราคา Low เพื่อจับจุดที่ราคาต่ำ
        stop_loss = row['Low'] * 0.99  # กำหนด Stop Loss 1% ต่ำกว่า
        take_profit = row['High'] * 1.02  # กำหนด Take Profit 2% สูงกว่า
        print(f"📈 สัญญาณขาขึ้น: ซื้อที่ {entry_price:.2f}")
        blink_signal(f"🚨 สัญญาณขาขึ้น 🚨", duration=3, signal_type="uptrend")

    # ตรวจสอบว่ามีสัญญาณขาลง (Downtrend Signal)
    elif 'Downtrend Signal (Polynomial Regression)' in signals:
        entry_price = row['High']  # ใช้ราคา High หากมีสัญญาณขาลง
        stop_loss = row['High'] * 1.01  # กำหนด Stop Loss 1% สูงกว่า
        take_profit = row['Low'] * 0.98  # กำหนด Take Profit 2% ต่ำกว่า
        print(f"📉 สัญญาณขาลง: ขายที่ {entry_price:.2f}")
        blink_signal(f"🚨 สัญญาณขาลง 🚨", duration=3, signal_type="downtrend")
    
    if entry_price:
        return entry_price, stop_loss, take_profit
    else:
        return None, None, None

# ฟังก์ชันกระพริบสัญญาณ (เช่นการแจ้งเตือน)
def blink_signal(message, duration=1, signal_type="neutral"):
    if signal_type == "uptrend":
        color = "\033[32m"  # สีเขียว
    elif signal_type == "downtrend":
        color = "\033[31m"  # สีแดง
    else:
        color = "\033[0m"  # ปกติ (ถ้าไม่ได้ระบุเป็น uptrend หรือ downtrend)

    # เพิ่มการกระพริบสัญญาณที่หน้าจอ
    print(f"{color}{message}\033[0m")
    time.sleep(duration)
    print("\033[0m")  # ลบสี

# 🔁 วนลูปเหรียญที่ต้องการจับสัญญาณ
symbols = ['BTCUSDT', 'ETHUSDT']

while True:
    print(f"\n🕒 กำลังอัปเดตข้อมูลรอบใหม่...\n")
    for symbol in symbols:
        try:
            df = get_binance_data(symbol=symbol, interval='1m', limit=200)
            df = compute_indicators(df)
            signals = detect_advanced_signals(df)  # เรียกฟังก์ชัน detect_advanced_signals เพื่อหาสัญญาณขั้นสูง

            print(f"[{symbol}] เวลา: {df.iloc[-1]['timestamp']} 📈 ราคา: {df.iloc[-1]['Close']:.2f}")
            if signals:
                for sig in signals:
                    print(f"📢 สัญญาณ: {sig}")
                    blink_signal(f"🚨 {sig} 🚨", duration=3)  # กระพริบสัญญาณใน 3 วินาที
                # คำนวณจุดเข้า
                entry_price, stop_loss, take_profit = calculate_entry_price(df, signals)
                if entry_price is not None:
                    print(f"📍 จุดเข้า: {entry_price:.2f}, จุด Stop Loss: {stop_loss:.2f}, จุด Take Profit: {take_profit:.2f}")
            else:
                print("ไม่มีสัญญาณในรอบนี้")
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดกับ {symbol}: {e}")
    
    time.sleep(60)
