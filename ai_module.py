# ai_module.py
import numpy as np
import pandas as pd
import logging
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class MLSignalGeneratorOKX:
    def __init__(self, seed=42):
        # Base ensemble members
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
        self.gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=seed)
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, random_state=seed, max_iter=1000))

        # Calibrated classifiers (wrap after training)
        self.calibrated = None  # will hold list of calibrated classifiers later
        self.ensemble_weights = {"rf": 0.45, "gb": 0.35, "lr": 0.20}

        self.trained = False
        self.last_train_time = None
        self.seed = seed

    # ============================================================
    # FETCH PRICE HISTORY (OKX)
    # ============================================================
    def fetch_ohlcv(self, symbol="POL-USDT", days=3, interval="15m"):
        """
        Fetch approx days * (24*60 / interval_minutes) candles.
        For 15m candles, days * 24 * 4 = rows. OKX 'limit' max is API-dependent; simple approach here.
        """
        url = "https://www.okx.com/api/v5/market/candles"
        # limit param: approximate number of candles to request
        limit = days * 24 * (60 // int(interval.replace("m", "")))
        params = {"instId": symbol, "bar": interval, "limit": str(limit)}

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()

                if data.get("code") not in (None, "0"):
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
                return df[["timestamp", "price", "o", "h", "l", "c", "vol"]]
            except requests.exceptions.RequestException as e:
                logging.warning(f"⚠️ Attempt {attempt+1}/3 failed fetching OKX data: {e}")
                time.sleep(2)

        logging.error("❌ Failed to fetch price history after 3 attempts.")
        return None

    # ============================================================
    # INDICATORS
    # ============================================================
    def add_indicators(self, df):
        # Price columns expected: price or c
        if "price" not in df.columns:
            df["price"] = df["c"].astype(float)

        # moving averages
        df["sma_fast"] = df["price"].rolling(5).mean()
        df["sma_slow"] = df["price"].rolling(20).mean()

        # ema for macd
        df["ema12"] = df["price"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["price"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # rsi
        df["rsi"] = self.compute_rsi(df["price"], 14)

        # momentum & volatility
        df["momentum"] = df["price"].pct_change(3)
        df["volatility"] = df["price"].pct_change().rolling(10).std()
        df["atr"] = self.compute_atr(df)  # average true range proxy using high-low if available

        # slope of sma_slow as simple trend slope
        df["sma_slow_slope"] = df["sma_slow"].diff()

        # drop rows with NA
        df = df.dropna().reset_index(drop=True)
        return df

    def compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def compute_atr(self, df):
        # If high/low present use them, else approximate via pct_change*price
        if {"h", "l", "c"}.issubset(df.columns):
            high = df["h"].astype(float)
            low = df["l"].astype(float)
            close = df["c"].astype(float)
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            return atr
        else:
            # fallback proxy
            return (df["price"].pct_change().abs() * df["price"]).rolling(14).mean()

    # ============================================================
    # REGIME DETECTION
    # ============================================================
    def detect_regime(self, df):
        """
        Simple regime classifier:
        - Uptrend if sma_slow slope positive and price > sma_slow
        - Downtrend if sma_slow slope negative and price < sma_slow
        - Sideways otherwise
        """
        latest = df.iloc[-1]
        slope = latest["sma_slow_slope"]
        price = latest["price"]
        sma_slow = latest["sma_slow"]
        vol_series = df["volatility"]
        vol = vol_series.iloc[-1]  # current volatility

        # Compare current volatility to historical quantile
        high_vol_threshold = vol_series.quantile(0.75)

        if slope > 0 and price > sma_slow and vol < high_vol_threshold:
            return "up"
        elif slope < 0 and price < sma_slow and vol > high_vol_threshold:
            return "down"
        else:
            return "sideways"

    # ============================================================
    # TRAIN (ensemble + calibration)
    # ============================================================
    def train_model(self, df, lookahead=6):
        """
        Labels: future_return > threshold => 1 else 0
        lookahead in number of candles (6 for 15m = 90 minutes)
        """
        df = df.copy()
        df["future_return"] = df["price"].shift(-lookahead) / df["price"] - 1
        # use modest threshold to avoid labeling noise; tune later
        ret_thresh = 0.002  # 0.2% for 15m lookahead; adjust per pair/timeframe
        df["signal"] = np.where(df["future_return"] > ret_thresh, 1, 0)
        df = df.dropna().reset_index(drop=True)

        features = df[["sma_fast", "sma_slow", "rsi", "momentum", "volatility", "macd_hist", "sma_slow_slope", "atr"]]
        target = df["signal"]

        if len(df) < 200:
            logging.warning("⚠️ Not enough samples to train robust models. Training with available data.")

        # simple train/test split for calibration
        X_train, X_cal, y_train, y_cal = train_test_split(features, target, test_size=0.2, random_state=self.seed, stratify=target if len(np.unique(target))>1 else None)

        # fit base models
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.lr.fit(X_train, y_train)

        # calibrate probabilities using CalibratedClassifierCV on held-out X_cal
        calibrated_models = {}
        try:
            # use 'sigmoid' (Platt scaling) for small data
            rf_cal = CalibratedClassifierCV(self.rf, cv="prefit", method="sigmoid")
            gb_cal = CalibratedClassifierCV(self.gb, cv="prefit", method="sigmoid")
            lr_cal = CalibratedClassifierCV(self.lr, cv="prefit", method="sigmoid")

            rf_cal.fit(X_cal, y_cal)
            gb_cal.fit(X_cal, y_cal)
            lr_cal.fit(X_cal, y_cal)

            calibrated_models["rf"] = rf_cal
            calibrated_models["gb"] = gb_cal
            calibrated_models["lr"] = lr_cal
            self.calibrated = calibrated_models
        except Exception as e:
            # fallback: use raw models if calibration fails
            logging.warning(f"⚠️ Calibration failed: {e}. Using raw model probabilities.")
            self.calibrated = {"rf": self.rf, "gb": self.gb, "lr": self.lr}

        self.trained = True
        self.last_train_time = datetime.now()
        logging.info(f"✅ Ensemble trained on {len(df)} samples | lookahead={lookahead}")

    # ============================================================
    # ENSEMBLE PROBABILITY
    # ============================================================
    def ensemble_proba(self, X):
        # X should be DataFrame with same features
        probs = {}
        try:
            probs["rf"] = self.calibrated["rf"].predict_proba(X)[0][1]
            probs["gb"] = self.calibrated["gb"].predict_proba(X)[0][1]
            probs["lr"] = self.calibrated["lr"].predict_proba(X)[0][1]
        except Exception:
            # fallback if any single model fails
            probs["rf"] = self.rf.predict_proba(X)[0][1]
            probs["gb"] = self.gb.predict_proba(X)[0][1]
            probs["lr"] = self.lr.predict_proba(X)[0][1]

        # weighted average
        w = self.ensemble_weights
        ensemble_score = probs["rf"] * w["rf"] + probs["gb"] * w["gb"] + probs["lr"] * w["lr"]
        return ensemble_score, probs

    # ============================================================
    # SIGNAL GENERATION
    # ============================================================
    def generate_signal(self, symbol="POL-USDT"):
        df = self.fetch_ohlcv(symbol=symbol)
        if df is None or df.empty:
            logging.error("❌ No price data from OKX. Cannot generate AI signal.")
            return None

        df = self.add_indicators(df)

        # retrain every 6 hours
        if (not self.trained) or (self.last_train_time is None or datetime.now() - self.last_train_time > timedelta(hours=6)):
            try:
                self.train_model(df)
            except Exception as e:
                logging.error(f"❌ Training failed: {e}")

        # latest feature row
        features_cols = ["sma_fast", "sma_slow", "rsi", "momentum", "volatility", "macd_hist", "sma_slow_slope", "atr"]
        latest_features = df[features_cols].iloc[-1:].copy()

        # ensemble probability & per-model probs
        ensemble_score, per_model = self.ensemble_proba(latest_features)

        # compute directional indicators
        momentum = df["momentum"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        macd_hist = df["macd_hist"].iloc[-1]
        sma_slope = df["sma_slow_slope"].iloc[-1]
        vol = df["volatility"].iloc[-1]
        atr = df["atr"].iloc[-1]

        # regime detection (up/down/sideways)
        regime = self.detect_regime(df)

        # dynamic thresholding:
        base_threshold = 0.55  # baseline
        # increase requirement when volatility high (avoid fake breakouts)
        if not np.isnan(vol):
            # relative vol scaling: more volatile -> require higher confidence
            vol_med = df["volatility"].median() if "volatility" in df else vol
            if vol > (1.2 * vol_med):
                base_threshold += 0.08
            elif vol < (0.8 * vol_med):
                base_threshold -= 0.03

        # regime adjustments
        if regime == "up":
            # favor buys: lower buy threshold modestly
            buy_adj = -0.03
            sell_adj = +0.05
        elif regime == "down":
            buy_adj = +0.05
            sell_adj = -0.03
        else:  # sideways
            buy_adj = +0.04
            sell_adj = +0.04

        buy_thresh = base_threshold + buy_adj
        sell_thresh = 1 - (base_threshold + sell_adj)  # symmetric

        # directional validation for buy / sell
        buy_conditions = [
            ensemble_score >= buy_thresh,
            momentum > 0,
            macd_hist > 0,
            rsi < 70,
            (regime == "up" or regime == "sideways")
        ]
        sell_conditions = [
            ensemble_score <= sell_thresh,
            momentum < 0,
            macd_hist < 0,
            rsi > 30,
            (regime == "down" or regime == "sideways")
        ]

        buy_signal = all(buy_conditions)
        sell_signal = all(sell_conditions)

        logging.info(
            f"🤖 Ensemble | score={ensemble_score:.3f} | per_model={per_model} | "
            f"buy_thresh={buy_thresh:.3f} | sell_thresh={sell_thresh:.3f} | regime={regime} | "
            f"mom={momentum:.4f} | macd_hist={macd_hist:.6f} | rsi={rsi:.2f} | buy={buy_signal} | sell={sell_signal}"
        )

        if buy_signal:
            return "BUY"
        if sell_signal:
            return "SELL"
        return "HOLD"
