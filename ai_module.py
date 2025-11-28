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

        # Calibrated classifiers
        self.calibrated = None
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
        """
        url = "https://www.okx.com/api/v5/market/candles"
        limit = days * 24 * (60 // int(interval.replace("m", "")))
        params = {"instId": symbol, "bar": interval, "limit": str(limit)}

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()

                if data.get("code") not in (None, "0"):
                    logging.warning(f"⚠️ OKX API returned code={data.get('code')} msg={data.get('msg')}")

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
        if "price" not in df.columns:
            df["price"] = df["c"].astype(float)

        df["sma_fast"] = df["price"].rolling(5).mean()
        df["sma_slow"] = df["price"].rolling(20).mean()

        # MACD
        df["ema12"] = df["price"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["price"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
        df["rsi"] = self.compute_rsi(df["price"], 14)

        # Momentum, Volatility, ATR
        df["momentum"] = df["price"].pct_change(3)
        df["volatility"] = df["price"].pct_change().rolling(10).std()
        df["atr"] = self.compute_atr(df)

        df["sma_slow_slope"] = df["sma_slow"].diff()

        df = df.dropna().reset_index(drop=True)
        return df

    def compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def compute_atr(self, df):
        if {"h", "l", "c"}.issubset(df.columns):
            high = df["h"].astype(float)
            low = df["l"].astype(float)
            close = df["c"].astype(float)
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(14).mean()
        else:
            return (df["price"].pct_change().abs() * df["price"]).rolling(14).mean()

    # ============================================================
    # REGIME DETECTION
    # ============================================================
    def detect_regime(self, df):
        latest = df.iloc[-1]
        slope = latest["sma_slow_slope"]
        price = latest["price"]
        sma_slow = latest["sma_slow"]
        vol_series = df["volatility"]
        vol = vol_series.iloc[-1]
        high_vol_threshold = vol_series.quantile(0.75)

        if slope > 0 and price > sma_slow and vol < high_vol_threshold:
            return "up"
        elif slope < 0 and price < sma_slow and vol > high_vol_threshold:
            return "down"
        else:
            return "sideways"

# ========================= END OF PART 1 / 3 =========================

    # -----------------------------------------
    # SMART LOCAL HIGH/LOW DETECTION
    # -----------------------------------------
    def detect_local_extrema(self, df, window=5):
        """
        Detect local highs and lows using rolling window comparison.
        Returns two boolean Series: local_highs, local_lows
        """

        # Handle short datasets
        if len(df) < window * 2 + 1:
            df["local_high"] = False
            df["local_low"] = False
            return df

        highs = []
        lows = []

        for i in range(len(df)):
            if i < window or i >= len(df) - window:
                highs.append(False)
                lows.append(False)
                continue

            window_slice = df["close"].iloc[i - window:i + window + 1]
            center_price = df["close"].iloc[i]

            highs.append(center_price == window_slice.max())
            lows.append(center_price == window_slice.min())

        df["local_high"] = highs
        df["local_low"] = lows

        return df

    # -----------------------------------------
    # SMART REGIME-BOUNDED EXTREMA
    # -----------------------------------------
    def detect_regime_extrema(self, df, regime):
        """
        Extract local highs/lows *within the current regime context*.
        E.g., in uptrend → track lows for Buy timing
              in downtrend → track highs for Sell timing
              in sideways → track both for range-trading
        """

        if "local_high" not in df.columns or "local_low" not in df.columns:
            return None, None

        recent = df.tail(30).copy()  # last 30 candles for noise reduction
        if recent.empty:
            return None, None

        last_high = recent[recent["local_high"]].tail(1)
        last_low = recent[recent["local_low"]].tail(1)

        high_price = float(last_high["close"].iloc[0]) if len(last_high) > 0 else None
        low_price = float(last_low["close"].iloc[0]) if len(last_low) > 0 else None

        return high_price, low_price

    # -----------------------------------------
    # SIGNAL GENERATION
    # -----------------------------------------
    def generate_signal(self):
        """
        Main entry for AI signal generation.
        Integrates:
        - ML classifier
        - Trend regime detection
        - Smart price-action local high/low
        - Volatility & filter conditions
        """

        df = self.get_recent_ohlcv()
        if df is None or len(df) < 50:
            return "HOLD"

        df = self.compute_indicators(df)

        regime = self.detect_regime(df)

        # Detect local extreme points
        df = self.detect_local_extrema(df)
        regime_high, regime_low = self.detect_regime_extrema(df, regime)

        current_price = float(df["close"].iloc[-1])

        # ------------------------------------------------------------
        # ML CLASSIFICATION SIGNAL
        # ------------------------------------------------------------
        try:
            X = df[["rsi", "sma_fast", "sma_slow", "macd", "macd_signal"]].iloc[-1:].values
            ml_pred = self.model.predict(X)[0]
        except Exception:
            ml_pred = 0  # safe fallback

        # ------------------------------------------------------------
        # BASE DECISION FROM ML
        # ------------------------------------------------------------
        base_signal = "HOLD"
        if ml_pred == 1:
            base_signal = "BUY"
        elif ml_pred == -1:
            base_signal = "SELL"

        # ------------------------------------------------------------
        # SMART PRICE-ACTION ADJUSTMENTS
        # ------------------------------------------------------------
        adjusted_signal = base_signal

        if regime == "uptrend":
            # BUY locally cheap → near regime lows
            if regime_low and current_price <= regime_low * 1.005:
                adjusted_signal = "BUY"
            # AVOID selling inside an uptrend unless ML is very strong
            if base_signal == "SELL" and (regime_high is None or current_price < regime_high):
                adjusted_signal = "HOLD"

        elif regime == "downtrend":
            # SELL locally expensive → near regime highs
            if regime_high and current_price >= regime_high * 0.995:
                adjusted_signal = "SELL"
            # AVOID buying inside a downtrend unless very cheap
            if base_signal == "BUY" and (regime_low is None or current_price > regime_low):
                adjusted_signal = "HOLD"

        elif regime == "sideways":
            # Range trading logic
            if regime_low and current_price <= regime_low * 1.01:
                adjusted_signal = "BUY"
            elif regime_high and current_price >= regime_high * 0.99:
                adjusted_signal = "SELL"

        # ------------------------------------------------------------
        # FINAL SAFETY CHECK — NO SIGNAL→ revert to HOLD
        # ------------------------------------------------------------
        if adjusted_signal not in ["BUY", "SELL"]:
            adjusted_signal = "HOLD"

        return adjusted_signal


    # ============================================================
    # TRAIN (ensemble + calibration)
    # ============================================================
    def train_model(self, df, lookahead=6):
        """
        Labels: future_return > threshold => 1 else 0
        lookahead in number of candles (6 for 15m = 90 minutes)
        """
        df = df.copy()
        # ensure price/close exist
        if "price" not in df.columns and "c" in df.columns:
            df["price"] = df["c"].astype(float)
        df["future_return"] = df["price"].shift(-lookahead) / df["price"] - 1
        ret_thresh = 0.002  # 0.2% for 15m lookahead; adjust per pair/timeframe
        df["signal"] = np.where(df["future_return"] > ret_thresh, 1, 0)
        df = df.dropna().reset_index(drop=True)

        features = df[["sma_fast", "sma_slow", "rsi", "momentum", "volatility", "macd_hist", "sma_slow_slope", "atr"]]
        target = df["signal"]

        if len(df) < 200:
            logging.warning("⚠️ Not enough samples to train robust models. Training with available data.")

        # simple train/test split for calibration
        strat = target if len(np.unique(target)) > 1 else None
        X_train, X_cal, y_train, y_cal = train_test_split(features, target, test_size=0.2, random_state=self.seed, stratify=strat)

        # fit base models
        try:
            self.rf.fit(X_train, y_train)
            self.gb.fit(X_train, y_train)
            self.lr.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"❌ Error fitting base models: {e}")
            # fallback: mark untrained
            self.trained = False
            return

        # calibrate probabilities using CalibratedClassifierCV on held-out X_cal
        calibrated_models = {}
        try:
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
            logging.warning(f"⚠️ Calibration failed: {e}. Using raw model probabilities.")
            self.calibrated = {"rf": self.rf, "gb": self.gb, "lr": self.lr}

        self.trained = True
        self.last_train_time = datetime.now()
        logging.info(f"✅ Ensemble trained on {len(df)} samples | lookahead={lookahead}")

    # ============================================================
    # ENSEMBLE PROBABILITY
    # ============================================================
    def ensemble_proba(self, X):
        """
        X: DataFrame with same feature columns as training
        Returns: ensemble_score (weighted), per-model probs dict
        """
        probs = {}
        try:
            probs["rf"] = float(self.calibrated["rf"].predict_proba(X)[0][1])
            probs["gb"] = float(self.calibrated["gb"].predict_proba(X)[0][1])
            probs["lr"] = float(self.calibrated["lr"].predict_proba(X)[0][1])
        except Exception:
            # fallback
            probs["rf"] = float(self.rf.predict_proba(X)[0][1])
            probs["gb"] = float(self.gb.predict_proba(X)[0][1])
            probs["lr"] = float(self.lr.predict_proba(X)[0][1])

        w = self.ensemble_weights
        ensemble_score = probs["rf"] * w["rf"] + probs["gb"] * w["gb"] + probs["lr"] * w["lr"]
        return ensemble_score, probs

    # ============================================================
    # Helper: unify/bridge function names used in Part2
    # ============================================================
    def get_recent_ohlcv(self, symbol="POL-USDT", days=3, interval="15m"):
        """
        Bridge function used in Part2. Returns DataFrame with columns expected by the module.
        """
        df = self.fetch_ohlcv(symbol=symbol, days=days, interval=interval)
        if df is None:
            return None

        # ensure 'close' & 'price' & 'c' exist for downstream code
        if "c" not in df.columns and "price" in df.columns:
            df["c"] = df["price"]
        if "close" not in df.columns:
            # prefer 'c' if exists
            if "c" in df.columns:
                df["close"] = df["c"].astype(float)
            else:
                df["close"] = df["price"].astype(float)
        # ensure numeric types
        for col in ["c", "close", "price", "o", "h", "l", "vol"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def compute_indicators(self, df):
        """
        Bridge to Part1 add_indicators but also ensure columns used by detect_local_extrema exist.
        """
        # If original add_indicators exists, prefer it
        try:
            df2 = self.add_indicators(df.copy())
            # ensure 'close' exists for extrema detection
            if "close" not in df2.columns:
                if "c" in df2.columns:
                    df2["close"] = df2["c"].astype(float)
                elif "price" in df2.columns:
                    df2["close"] = df2["price"].astype(float)
            return df2
        except Exception as e:
            logging.warning(f"⚠️ add_indicators failed: {e}. Falling back to inline indicators.")
            # fallback minimal indicators
            if "price" not in df.columns and "c" in df.columns:
                df["price"] = df["c"].astype(float)
            df["sma_fast"] = df["price"].rolling(5).mean()
            df["sma_slow"] = df["price"].rolling(20).mean()
            df["rsi"] = self.compute_rsi(df["price"], 14)
            df["momentum"] = df["price"].pct_change(3)
            df["volatility"] = df["price"].pct_change().rolling(10).std()
            df["ema12"] = df["price"].ewm(span=12, adjust=False).mean()
            df["ema26"] = df["price"].ewm(span=26, adjust=False).mean()
            df["macd"] = df["ema12"] - df["ema26"]
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
            df["atr"] = self.compute_atr(df)
            df["sma_slow_slope"] = df["sma_slow"].diff()
            if "c" in df.columns:
                df["close"] = df["c"].astype(float)
            elif "price" in df.columns:
                df["close"] = df["price"].astype(float)
            df = df.dropna().reset_index(drop=True)
            return df

    # ============================================================
    # FINAL SIGNAL GENERATOR (integrates ML + Smart Extrema)
    # ============================================================
    def generate_signal(self, symbol="POL-USDT"):
        """
        Public method kept with the original signature.
        Combines:
        - data fetch
        - indicator computation
        - periodic retrain (6 hours)
        - ensemble scoring
        - regime detection
        - local extrema within regime for BUY/SELL timing
        Returns "BUY" / "SELL" / "HOLD"
        """
        df = self.get_recent_ohlcv(symbol=symbol)
        if df is None or len(df) < 50:
            logging.warning("Not enough data to generate signal. Returning HOLD.")
            return "HOLD"

        df = self.compute_indicators(df)

        # retrain every 6 hours if needed
        if (not self.trained) or (self.last_train_time is None or datetime.now() - self.last_train_time > timedelta(hours=6)):
            try:
                self.train_model(df)
            except Exception as e:
                logging.error(f"❌ Training failed: {e}")

        # prepare features for ensemble_proba
        feat_cols = ["sma_fast", "sma_slow", "rsi", "momentum", "volatility", "macd_hist", "sma_slow_slope", "atr"]
        latest_features = df[feat_cols].iloc[-1:].copy()

        # ensemble probability & per-model
        try:
            ensemble_score, per_model = self.ensemble_proba(latest_features)
        except Exception as e:
            logging.error(f"Ensemble proba failed: {e}")
            ensemble_score, per_model = 0.5, {"rf": 0.5, "gb": 0.5, "lr": 0.5}

        # directional indicators
        momentum = float(df["momentum"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
        macd_hist = float(df["macd_hist"].iloc[-1])
        sma_slope = float(df["sma_slow_slope"].iloc[-1])
        vol = float(df["volatility"].iloc[-1]) if "volatility" in df.columns else np.nan
        price = float(df["close"].iloc[-1])

        # regime detection
        regime = self.detect_regime(df)

        # detect local extrema & regime-bounded extremes
        df = self.detect_local_extrema(df, window=5)
        regime_high, regime_low = self.detect_regime_extrema(df, regime)

        # dynamic thresholding base
        base_threshold = 0.55
        try:
            vol_med = df["volatility"].median() if "volatility" in df.columns else vol
            if not np.isnan(vol) and vol > (1.2 * vol_med):
                base_threshold += 0.08
            elif not np.isnan(vol) and vol < (0.8 * vol_med):
                base_threshold -= 0.03
        except Exception:
            pass

        # regime adjustment
        if regime == "up":
            buy_adj = -0.03
            sell_adj = +0.05
        elif regime == "down":
            buy_adj = +0.05
            sell_adj = -0.03
        else:
            buy_adj = +0.04
            sell_adj = +0.04

        buy_thresh = base_threshold + buy_adj
        sell_thresh = 1 - (base_threshold + sell_adj)

        # Base ML directional suggestion (from ensemble score + thresholds)
        ml_buy = ensemble_score >= buy_thresh
        ml_sell = ensemble_score <= sell_thresh

        # initial signals from ML + directional filters
        buy_conditions = [ml_buy, momentum > 0, macd_hist > 0, rsi < 70]
        sell_conditions = [ml_sell, momentum < 0, macd_hist < 0, rsi > 30]

        buy_ok = all(buy_conditions)
        sell_ok = all(sell_conditions)

        # Smart extrema confirmations
        buy_ext_ok = False
        sell_ext_ok = False

        # If regime_low exists and current price near regime low -> buy_ext_ok
        if regime_low is not None and regime_low > 0:
            if price <= regime_low * 1.005:
                buy_ext_ok = True

        # If regime_high exists and current price near regime high -> sell_ext_ok
        if regime_high is not None and regime_high > 0:
            if price >= regime_high * 0.995:
                sell_ext_ok = True

        # Final decision integrating ML + extrema + regime rules
        final_decision = "HOLD"

        if regime == "up":
            # prioritize buys near local lows; avoid selling unless clear extreme
            if buy_ok and buy_ext_ok:
                final_decision = "BUY"
            elif sell_ok and sell_ext_ok and ensemble_score > 0.90:
                # allow aggressive exit if ensemble extremely confident and price at regime high
                final_decision = "SELL"
            else:
                final_decision = "HOLD"

        elif regime == "down":
            # prioritize sells near local highs
            if sell_ok and sell_ext_ok:
                final_decision = "SELL"
            elif buy_ok and buy_ext_ok and ensemble_score > 0.90:
                final_decision = "BUY"
            else:
                final_decision = "HOLD"

        else:  # sideways
            # range-bound: require both ml_ok and extrema confirmation
            if buy_ok and buy_ext_ok:
                final_decision = "BUY"
            elif sell_ok and sell_ext_ok:
                final_decision = "SELL"
            else:
                final_decision = "HOLD"

        logging.info(
            f"🤖 SmartSignal | price={price:.6f} | regime={regime} | ensemble={ensemble_score:.3f} | "
            f"per_model={per_model} | buy_ok={buy_ok} | sell_ok={sell_ok} | buy_ext={buy_ext_ok} | sell_ext={sell_ext_ok} | final={final_decision}"
        )

        return final_decision

