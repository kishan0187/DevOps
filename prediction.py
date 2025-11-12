import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# Load pre-trained LSTM model
MODEL_PATH = "lstm_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Pre-trained model not found. Please train and save lstm_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

def load_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        print(f"❌ Error: No data found for {ticker}")
        return None, None
    print(f"✅ Data fetched successfully for {ticker}")
    return df['Close'].values.reshape(-1, 1), df.index

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def prepare_lstm_input(data):
    X = []
    for i in range(60, len(data)):
        X.append(data[i-60:i, 0])
    return np.array(X).reshape(-1, 60, 1)

def predict_price(ticker):
    data, _ = load_stock_data(ticker)
    if data is None:
        return "Invalid Stock Symbol!"
    
    scaled_data, scaler = preprocess_data(data)
    X_input = prepare_lstm_input(scaled_data)
    predicted_price = model.predict(X_input[-1].reshape(1, 60, 1))
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
    
    return predicted_price[0][0]