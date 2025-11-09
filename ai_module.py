# ai_module.py
import numpy as np
import pandas as pd
import logging
import requests
import time
from sklearn.ensemble import RandomForestClassifier


class MLSignalGeneratorOKX:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    # ============================================================
    # FETCH PRICE HISTORY (stable version for OKX)
    # ============================================================
    def fetch_ohlcv(self, symbol="POL-USDT", days=3, interval="hourly"):
        """
        Fetch POL/USDT historical data from OKX public API.
        OKX granularity supports 1h, 4h, 1d, etc.
        """
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": "1H", "limit": str(days * 24)}

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()

                # Log full response if API code != 0
                if data.get("code") != "0":
                    logging.warning(f"⚠️ OKX API returned code={data.get('code')}, msg={data.get('msg')}")

                if "data" not in data or not data["data"]:
                    logging.error("⚠️ No candle data returned from OKX.")
                    return None

                df = pd.DataFrame(
                    data["data"],
                    columns=[
                        "ts",
                        "o",
                        "h",
                        "l",
                        "c",
                        "vol",
                        "volCcy",
                        "volCcyQuote",
                        "confirm",
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
                df["price"] = df["c"].astype(float)
                df = df.sort_values("timestamp").reset_index(drop=True)
                logging.info(f"✅ OKX price history fetched: {len(df)} records for {symbol}.")
                return df[["timestamp", "price"]]

            except requests.exceptions.RequestException as e:
                logging.warning(f"⚠️ Attempt {attempt+1}/3 failed fetching OKX data: {e}")
                time.sleep(2)

        logging.error("❌ Failed to fetch price history after 3 attempts.")
        return None

    # ============================================================
    # FEATURE ENGINEERING + INDICATORS
    # ============================================================
    def add_indicators(self, df):
        df["sma_fast"] = df["price"].rolling(5).mean()
        df["sma_slow"] = df["price"].rolling(20).mean()
        df["rsi"] = self.compute_rsi(df["price"], 14)
        df["momentum"] = df["price"].pct_change(3)
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

    # ============================================================
    # MODEL TRAINING
    # ============================================================
    def train_model(self, df):
        features = df[["sma_fast", "sma_slow", "rsi", "momentum"]]
        target = df["signal"]
        self.model.fit(features, target)
        self.trained = True

    # ============================================================
    # SIGNAL GENERATION (AI + Momentum + Confidence)
    # ============================================================
    def generate_signal(self):
        df = self.fetch_ohlcv()
        if df is None or df.empty:
            logging.error("❌ No price data from OKX. Cannot generate AI signal.")
            return None

        df = self.add_indicators(df)
        if not self.trained:
            self.train_model(df)

        latest = df[["sma_fast", "sma_slow", "rsi", "momentum"]].iloc[-1:].values
        proba = self.model.predict_proba(latest)[0][1]
        confidence = round(proba, 3)
        momentum = df["momentum"].iloc[-1]
        rsi = df["rsi"].iloc[-1]

        base_threshold = 0.55
        momentum_boost = 0.1 if momentum > 0.002 else 0
        threshold = base_threshold - momentum_boost
        signal = (confidence > threshold) and (momentum > 0) and (rsi < 70)

        logging.info(
            f"🤖 AI Signal | Conf={confidence:.3f} | Thresh={threshold:.2f} | "
            f"Momentum={momentum:.4f} | RSI={rsi:.2f} | Signal={signal}"
        )

        return "BUY" if signal else "SELL"
