from flask import Flask, render_template, request
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

app = Flask(__name__, static_folder='static')

def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        return None, None
    return df['Close'].values.reshape(-1, 1), df.index

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def prepare_lstm_input(data):
    X = []
    for i in range(60, len(data)):
        X.append(data[i-60:i, 0])
    X = np.array(X).reshape(-1, 60, 1)
    return X

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form.get('stock_symbol').upper()
    data, dates = fetch_stock_data(stock_symbol)
    if data is None:
        return render_template('index.html', error="Invalid Stock Symbol!")
    
    scaled_data, scaler = preprocess_data(data)
    X_input = prepare_lstm_input(scaled_data)
    predicted_price = model.predict(X_input[-1].reshape(1, 60, 1))
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))[0][0]
    
    return render_template('result.html', stock=stock_symbol, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
