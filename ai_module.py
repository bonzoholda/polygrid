# ai_module.py
import numpy as np
import pandas as pd
import logging
import requests
from sklearn.ensemble import RandomForestClassifier

class MLSignalGeneratorOKX:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    def fetch_ohlcv(self, symbol="POL-USDT", window="1h", limit=200):
        url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={window}&limit={limit}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                logging.warning("⚠️ No candle data returned from OKX.")
                return None
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol"])
            df = df.astype(float)
            df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            logging.error(f"Failed fetching OHLCV from OKX: {e}")
            return None

    def add_indicators(self, df):
        df["sma_fast"] = df["close"].rolling(5).mean()
        df["sma_slow"] = df["close"].rolling(20).mean()
        df["rsi"] = self.compute_rsi(df["close"], 14)
        df["signal"] = np.where(df["sma_fast"] > df["sma_slow"], 1, 0)
        df.dropna(inplace=True)
        return df

    def compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def train_model(self, df):
        features = df[["sma_fast", "sma_slow", "rsi"]]
        target = df["signal"]
        self.model.fit(features, target)
        self.trained = True

    def generate_signal(self):
        df = self.fetch_ohlcv()
        if df is None:
            return None
        df = self.add_indicators(df)
        if not self.trained:
            self.train_model(df)
        latest = df[["sma_fast", "sma_slow", "rsi"]].iloc[-1:]
        pred = self.model.predict(latest)[0]
        return "BUY" if pred == 1 else "SELL"
