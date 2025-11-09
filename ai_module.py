# ai_module.py
import numpy as np
import pandas as pd
import logging
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier


class MLSignalGeneratorOKX:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        )
        self.trained = False
        self.last_train_time = None

    # ============================================================
    # FETCH PRICE HISTORY (OKX)
    # ============================================================
    def fetch_ohlcv(self, symbol="POL-USDT", days=3, interval="1H"):
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": interval, "limit": str(days * 24)}

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()

                if data.get("code") != "0":
                    logging.warning(f"⚠️ OKX API returned code={data.get('code')}, msg={data.get('msg')}")

                if "data" not in data or not data["data"]:
                    logging.error("⚠️ No candle data returned from OKX.")
                    return None

                df = pd.DataFrame(
                    data["data"],
                    columns=[
                        "ts", "o", "h", "l", "c", "vol", "volCcy",
                        "volCcyQuote", "confirm"
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
                df["price"] = df["c"].astype(float)
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df[["timestamp", "price"]]

            except requests.exceptions.RequestException as e:
                logging.warning(f"⚠️ Attempt {attempt+1}/3 failed fetching OKX data: {e}")
                time.sleep(2)

        logging.error("❌ Failed to fetch price history after 3 attempts.")
        return None

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================
    def add_indicators(self, df):
        df["sma_fast"] = df["price"].rolling(5).mean()
        df["sma_slow"] = df["price"].rolling(20).mean()
        df["rsi"] = self.compute_rsi(df["price"], 14)
        df["momentum"] = df["price"].pct_change(3)
        df["volatility"] = df["price"].pct_change().rolling(10).std()
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

    # ============================================================
    # TRAIN MODEL (Predictive labels)
    # ============================================================
    def train_model(self, df, lookahead=6):
        """
        Label based on future price movement.
        lookahead = number of candles ahead to check (6 = 6 hours for 1H data)
        """
        df["future_return"] = df["price"].shift(-lookahead) / df["price"] - 1
        df["signal"] = np.where(df["future_return"] > 0, 1, 0)
        df.dropna(inplace=True)

        features = df[["sma_fast", "sma_slow", "rsi", "momentum", "volatility"]]
        target = df["signal"]

        self.model.fit(features, target)
        self.trained = True
        self.last_train_time = datetime.now()
        logging.info(f"✅ Model retrained on {len(df)} samples | lookahead={lookahead}")

    # ============================================================
    # SIGNAL GENERATION (AI PREDICTION)
    # ============================================================
    def generate_signal(self, symbol="POL-USDT"):
        df = self.fetch_ohlcv(symbol=symbol)
        if df is None or df.empty:
            logging.error("❌ No price data from OKX. Cannot generate AI signal.")
            return None

        df = self.add_indicators(df)

        # Retrain every 6 hours (optimum for hourly data)
        if (not self.trained) or (
            self.last_train_time is None
            or datetime.now() - self.last_train_time > timedelta(hours=6)
        ):
            self.train_model(df)

        latest = df[["sma_fast", "sma_slow", "rsi", "momentum", "volatility"]].iloc[-1:].values
        proba = self.model.predict_proba(latest)[0][1]
        confidence = round(proba, 3)

        momentum = df["momentum"].iloc[-1]
        rsi = df["rsi"].iloc[-1]

        # Dynamic confidence threshold (higher when RSI is hot)
        base_threshold = 0.55
        if rsi > 65:
            base_threshold += 0.05
        elif rsi < 35:
            base_threshold -= 0.05

        signal = (confidence > base_threshold) and (momentum > 0) and (rsi < 70)

        logging.info(
            f"🤖 AI Predictive Signal | Conf={confidence:.3f} | Thresh={base_threshold:.2f} | "
            f"Momentum={momentum:.4f} | RSI={rsi:.2f} | Signal={signal}"
        )

        return "BUY" if signal else "SELL"
