import json
import os
import platform
import threading
import time
import warnings

import numpy as np
import pandas as pd
import requests
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from plyer import notification
from tkinter import ttk

# ===== GPU 加速：PyTorch =====
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前计算设备:", device)

# ==============================
# 配置
# ==============================
FUTURES_SYMBOL = "BTCUSDT"  # 永续
SPOT_SYMBOL = "BTCUSDT"  # 现货
FUTURES_BASE_URL = "https://fapi.binance.com"
SPOT_BASE_URL = "https://api.binance.com"
OKX_BASE_URL = "https://www.okx.com"
BYBIT_BASE_URL = "https://api.bybit.com"
FNG_API_URL = "https://api.alternative.me/fng/"

VOL_WINDOW = 20
VOL_MIN = 25
VOL_MAX = 400
BIG_TRADE_QTY = 60
IMB_THRESH = 0.28
FAKE_BREAK_DIST = 8
VWAP_NEAR = 40
MIN_SCORE = 5  # 中线均衡：信号更严格

LOG_DIR = os.path.join(os.path.expanduser("~"), "btc_terminal_logs")
os.makedirs(LOG_DIR, exist_ok=True)
SIGNAL_LOG_FILE = os.path.join(LOG_DIR, "signals_log.jsonl")
BEST_PARAM_FILE = os.path.join(LOG_DIR, "best_params.json")

DEFAULT_PARAMS = {
    "fee_rate": 0.0004,
    "slippage": 0.0002,
    "leverage": 3.0,
    "funding_rate": 0.0001,
    "strategy_min_score": 3,
}
CURRENT_PARAMS = DEFAULT_PARAMS.copy()

_last_depth_snapshot = None
_win_rate_cache = {}


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 60 * 60_000
    if unit == "d":
        return value * 24 * 60 * 60_000
    raise ValueError(f"未知周期: {interval}")


def normalize_klines_to_frame(klines):
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_vol",
            "trades",
            "tb_base_av",
            "tb_quote_av",
            "ignore",
        ],
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def fill_missing_klines(klines, interval):
    if not klines:
        return klines
    df = normalize_klines_to_frame(klines)
    if df.empty:
        return klines
    step = interval_to_ms(interval)
    df = df.set_index("open_time").sort_index()
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f"{step}ms")
    df = df.reindex(expected_index)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()
    df["open_time"] = (df.index.view("int64") // 1_000_000).astype(int)
    df["close_time"] = df["open_time"] + step - 1
    df["quote_asset_vol"] = df["volume"]
    df["trades"] = df["trades"].fillna(0).astype(int)
    df["tb_base_av"] = df["tb_base_av"].fillna(0)
    df["tb_quote_av"] = df["tb_quote_av"].fillna(0)
    df["ignore"] = 0
    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_vol",
            "trades",
            "tb_base_av",
            "tb_quote_av",
            "ignore",
        ]
    ].values.tolist()


def load_best_params():
    if not os.path.exists(BEST_PARAM_FILE):
        return DEFAULT_PARAMS.copy()
    try:
        with open(BEST_PARAM_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged = DEFAULT_PARAMS.copy()
            merged.update(data)
            return merged
    except Exception:
        return DEFAULT_PARAMS.copy()


def save_best_params(params):
    with open(BEST_PARAM_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


class DataFeed:
    def __init__(self, cache_dir=None, cache_ttl=30):
        self.cache_dir = cache_dir or os.path.join(LOG_DIR, "cache")
        self.cache_ttl = cache_ttl
        self._cache_lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key):
        safe_key = key.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def _load_cache(self, key):
        path = self._cache_path(key)
        with self._cache_lock:
            if not os.path.exists(path):
                return None
            if time.time() - os.path.getmtime(path) > self.cache_ttl:
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                try:
                    os.remove(path)
                except OSError:
                    pass
                return None

    def _save_cache(self, key, payload):
        path = self._cache_path(key)
        tmp_path = f"{path}.{threading.get_ident()}.tmp"
        with self._cache_lock:
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                for attempt in range(3):
                    try:
                        os.replace(tmp_path, path)
                        return
                    except OSError:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                        time.sleep(0.1 * (attempt + 1))
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                except OSError:
                    return
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def fetch_klines(self, exchange, symbol, interval, limit=200, fill_missing=True):
        cache_key = f"{exchange}_{symbol}_{interval}_{limit}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached
        if exchange == "binance_futures":
            klines = self._fetch_binance_klines(FUTURES_BASE_URL, symbol, interval, limit)
        elif exchange == "binance_spot":
            klines = self._fetch_binance_klines(SPOT_BASE_URL, symbol, interval, limit)
        elif exchange == "okx":
            klines = self._fetch_okx_klines(symbol, interval, limit)
        elif exchange == "bybit":
            klines = self._fetch_bybit_klines(symbol, interval, limit)
        else:
            raise ValueError(f"未知交易所: {exchange}")
        if fill_missing:
            klines = fill_missing_klines(klines, interval)
        self._save_cache(cache_key, klines)
        return klines

    def _fetch_binance_klines(self, base_url, symbol, interval, limit):
        url = f"{base_url}/api/v3/klines" if base_url == SPOT_BASE_URL else f"{base_url}/fapi/v1/klines"
        response = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_okx_klines(self, symbol, interval, limit):
        okx_interval = interval.replace("m", "m").replace("h", "H").replace("d", "D")
        response = requests.get(
            f"{OKX_BASE_URL}/api/v5/market/candles",
            params={"instId": f"{symbol}-SWAP", "bar": okx_interval, "limit": limit},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        klines = []
        for row in reversed(data):
            ts, open_, high, low, close, vol, _, _, _ = row + [0] * (9 - len(row))
            ts = int(ts)
            klines.append([ts, open_, high, low, close, vol, ts, vol, 0, 0, 0, 0])
        return klines

    def _fetch_bybit_klines(self, symbol, interval, limit):
        bybit_interval = interval.replace("m", "").replace("h", "60").replace("d", "D")
        response = requests.get(
            f"{BYBIT_BASE_URL}/v5/market/kline",
            params={"category": "linear", "symbol": symbol, "interval": bybit_interval, "limit": limit},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json().get("result", {}).get("list", [])
        klines = []
        for row in reversed(data):
            ts, open_, high, low, close, vol, *_ = row
            ts = int(ts)
            klines.append([ts, open_, high, low, close, vol, ts, vol, 0, 0, 0, 0])
        return klines

    def fetch_order_book(self, exchange, symbol, limit=50):
        if exchange == "binance_futures":
            url = f"{FUTURES_BASE_URL}/fapi/v1/depth"
        elif exchange == "binance_spot":
            url = f"{SPOT_BASE_URL}/api/v3/depth"
        elif exchange == "okx":
            url = f"{OKX_BASE_URL}/api/v5/market/books"
        elif exchange == "bybit":
            url = f"{BYBIT_BASE_URL}/v5/market/orderbook"
        else:
            raise ValueError(f"未知交易所: {exchange}")
        params = {"symbol": symbol, "limit": limit}
        if exchange == "okx":
            params = {"instId": f"{symbol}-SWAP", "sz": limit}
        if exchange == "bybit":
            params = {"category": "linear", "symbol": symbol, "limit": limit}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if exchange == "okx":
            book = payload.get("data", [{}])[0]
            return {"bids": book.get("bids", []), "asks": book.get("asks", [])}
        if exchange == "bybit":
            book = payload.get("result", {})
            return {"bids": book.get("b", []), "asks": book.get("a", [])}
        return payload


class TAEngine:
    def __init__(self):
        self.indicators_cache = {}

    def compute_all(self, klines, cache_key=None):
        if cache_key and cache_key in self.indicators_cache:
            return self.indicators_cache[cache_key]
        df = normalize_klines_to_frame(klines)
        if df.empty:
            return {}
        closes = df["close"].tolist()
        highs = df["high"].tolist()
        lows = df["low"].tolist()
        ema_fast = pd.Series(closes).ewm(span=12, adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(closes).ewm(span=26, adjust=False).mean().iloc[-1]
        sma_20 = pd.Series(closes).rolling(20).mean().iloc[-1]
        macd_line = ema_fast - ema_slow
        macd_signal = pd.Series(closes).ewm(span=9, adjust=False).mean().iloc[-1]
        rsi = calc_rsi(closes)[-1] if closes else 50
        bb = bollinger(closes)
        atr = calc_atr_from_klines(klines)
        vwap = vwap_from_klines(klines)
        ichi = self.ichimoku(highs, lows)
        indicators = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "sma_20": sma_20,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "rsi": rsi,
            "bb": bb,
            "atr": atr,
            "vwap": vwap,
            "ichimoku": ichi,
        }
        if cache_key:
            self.indicators_cache[cache_key] = indicators
        return indicators

    def ichimoku(self, highs, lows, tenkan=9, kijun=26, senkou=52):
        if len(highs) < senkou:
            return {"tenkan": 0, "kijun": 0, "span_a": 0, "span_b": 0}
        tenkan_sen = (max(highs[-tenkan:]) + min(lows[-tenkan:])) / 2
        kijun_sen = (max(highs[-kijun:]) + min(lows[-kijun:])) / 2
        span_a = (tenkan_sen + kijun_sen) / 2
        span_b = (max(highs[-senkou:]) + min(lows[-senkou:])) / 2
        return {"tenkan": tenkan_sen, "kijun": kijun_sen, "span_a": span_a, "span_b": span_b}

    def signal_triggers(self, indicators):
        triggers = []
        bb = indicators.get("bb")
        if bb:
            upper, mid, lower = bb
            triggers.append({"name": "bb_upper", "value": upper})
            triggers.append({"name": "bb_lower", "value": lower})
        triggers.append({"name": "rsi_overbought", "value": indicators.get("rsi", 50) > 70})
        triggers.append({"name": "rsi_oversold", "value": indicators.get("rsi", 50) < 30})
        return triggers


class StrategyEngine:
    def __init__(self, ta_engine):
        self.ta_engine = ta_engine

    def generate_signal(self, klines, regime="未知"):
        indicators = self.ta_engine.compute_all(klines)
        if not indicators:
            return {"direction": None, "score": 0, "reason": "无指标"}
        score_long = 0
        score_short = 0
        rsi = indicators.get("rsi", 50)
        bb = indicators.get("bb")
        vwap = indicators.get("vwap")
        close = float(klines[-1][4]) if klines else 0
        if rsi < 30:
            score_long += 2
        if rsi > 70:
            score_short += 2
        if bb:
            upper, mid, lower = bb
            if close < lower:
                score_long += 1
            if close > upper:
                score_short += 1
        if vwap:
            if close > vwap:
                score_long += 1
            if close < vwap:
                score_short += 1
        if regime == "趋势期":
            score_long += 1
            score_short += 1
        min_score = CURRENT_PARAMS.get("strategy_min_score", 3)
        if score_long > score_short and score_long >= min_score:
            return {"direction": "long", "score": score_long, "reason": "策略引擎信号"}
        if score_short > score_long and score_short >= min_score:
            return {"direction": "short", "score": score_short, "reason": "策略引擎信号"}
        return {"direction": None, "score": max(score_long, score_short), "reason": "观望"}


def multi_factor_ai_strategy(features, sentiment_score, macro_score):
    if features is None or len(features) == 0:
        return {"direction": None, "score": 0, "reason": "无特征"}
    last = features[-1]
    open_, high, low, close, volume = last
    score = 0
    if close > open_:
        score += 1
    if volume > np.mean(features[:, 4]):
        score += 1
    if sentiment_score > 0.2:
        score += 1
    if macro_score > 0.2:
        score += 1
    if score >= 3:
        return {"direction": "long", "score": score, "reason": "多因子 AI 策略"}
    if score <= 1 and sentiment_score < -0.2:
        return {"direction": "short", "score": score, "reason": "多因子 AI 策略"}
    return {"direction": None, "score": score, "reason": "观望"}


def fused_ai_direction(extra):
    votes = {"long": 0, "short": 0}
    ml_pred = extra.get("ml_pred_price")
    transformer_pred = extra.get("transformer_pred_price")
    ensemble_pred = extra.get("ensemble_pred_price")
    market_data = extra.get("market_data") or {}
    close = market_data.get("close")
    if close is not None:
        if ml_pred and ml_pred > close:
            votes["long"] += 1
        elif ml_pred and ml_pred < close:
            votes["short"] += 1
        if transformer_pred and transformer_pred > close:
            votes["long"] += 1
        elif transformer_pred and transformer_pred < close:
            votes["short"] += 1
        if ensemble_pred and ensemble_pred > close:
            votes["long"] += 2
        elif ensemble_pred and ensemble_pred < close:
            votes["short"] += 2
    ai_signal = extra.get("ai_signal") or {}
    if ai_signal.get("direction") == "long":
        votes["long"] += 2
    elif ai_signal.get("direction") == "short":
        votes["short"] += 2
    strategy_signal = extra.get("strategy_signal") or {}
    if strategy_signal.get("direction") == "long":
        votes["long"] += 1
    elif strategy_signal.get("direction") == "short":
        votes["short"] += 1
    sentiment_score = extra.get("sentiment_score", 0)
    macro_score = extra.get("macro_score", 0)
    if sentiment_score > 0.2:
        votes["long"] += 1
    elif sentiment_score < -0.2:
        votes["short"] += 1
    if macro_score > 0.2:
        votes["long"] += 1
    elif macro_score < -0.2:
        votes["short"] += 1
    if votes["long"] > votes["short"]:
        return {"direction": "long", "score": votes["long"], "votes": votes}
    if votes["short"] > votes["long"]:
        return {"direction": "short", "score": votes["short"], "votes": votes}
    return {"direction": None, "score": 0, "votes": votes}


def multi_factor_score(extra):
    score_long = 0
    score_short = 0
    sentiment_score = extra.get("sentiment_score", 0)
    macro_score = extra.get("macro_score", 0)
    taker_ratio = extra.get("taker_ratio", 0.5)
    imbalance = extra.get("imbalance", 0)
    indicators = extra.get("indicators") or {}
    rsi = indicators.get("rsi", 50)
    bb = indicators.get("bb")
    market_data = extra.get("market_data") or {}
    close = market_data.get("close")
    vwap = indicators.get("vwap")
    if sentiment_score > 0.2:
        score_long += 1
    elif sentiment_score < -0.2:
        score_short += 1
    if macro_score > 0.2:
        score_long += 1
    elif macro_score < -0.2:
        score_short += 1
    if taker_ratio > 0.57:
        score_long += 1
    elif taker_ratio < 0.43:
        score_short += 1
    if imbalance < -IMB_THRESH:
        score_long += 1
    elif imbalance > IMB_THRESH:
        score_short += 1
    if rsi < 30:
        score_long += 1
    elif rsi > 70:
        score_short += 1
    if bb and close is not None:
        upper, mid, lower = bb
        if close < lower:
            score_long += 1
        elif close > upper:
            score_short += 1
    if vwap and close is not None:
        if close > vwap:
            score_long += 1
        elif close < vwap:
            score_short += 1
    ai_signal = extra.get("ai_signal") or {}
    if ai_signal.get("direction") == "long":
        score_long += 2
    elif ai_signal.get("direction") == "short":
        score_short += 2
    strategy_signal = extra.get("strategy_signal") or {}
    if strategy_signal.get("direction") == "long":
        score_long += 1
    elif strategy_signal.get("direction") == "short":
        score_short += 1
    if score_long > score_short:
        direction = "long"
    elif score_short > score_long:
        direction = "short"
    else:
        direction = None
    return {"direction": direction, "long_score": score_long, "short_score": score_short}


def combined_direction(extra):
    multi_score = extra.get("multi_factor_score") or {}
    if multi_score.get("direction"):
        return multi_score.get("direction")
    fused_ai = extra.get("fused_ai") or {}
    if fused_ai.get("direction"):
        return fused_ai.get("direction")
    return None


class Backtester:
    def __init__(self, fee_rate=0.0004, slippage=0.0002, leverage=3.0, funding_rate=0.0001):
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.leverage = leverage
        self.funding_rate = funding_rate

    def run(self, klines, strategy_engine):
        trades = []
        position = None
        entry_price = 0
        equity = 1.0
        for i in range(60, len(klines)):
            window = klines[:i]
            signal = strategy_engine.generate_signal(window)
            price = float(klines[i][4])
            if position is None and signal["direction"]:
                position = signal["direction"]
                entry_price = price * (1 + self.slippage if position == "long" else 1 - self.slippage)
                continue
            if position == "long":
                if signal["direction"] == "short":
                    exit_price = price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    pnl = pnl * self.leverage
                    pnl -= self.fee_rate * 2
                    pnl -= self.funding_rate
                    equity *= 1 + pnl
                    trades.append(pnl)
                    position = None
            elif position == "short":
                if signal["direction"] == "long":
                    exit_price = price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) / entry_price
                    pnl = pnl * self.leverage
                    pnl -= self.fee_rate * 2
                    pnl -= self.funding_rate
                    equity *= 1 + pnl
                    trades.append(pnl)
                    position = None
        return self.report(trades, equity)

    def report(self, trades, equity):
        if not trades:
            return {"trades": 0, "win_rate": 0, "equity": equity, "max_drawdown": 0, "sharpe": 0}
        wins = [t for t in trades if t > 0]
        returns = pd.Series(trades)
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        return {
            "trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "equity": equity,
            "max_drawdown": abs(drawdown.min()),
            "sharpe": sharpe,
        }


def optimize_params():
    candidates = [
        {"fee_rate": 0.0004, "slippage": 0.0002, "leverage": 3.0, "funding_rate": 0.0001},
        {"fee_rate": 0.0006, "slippage": 0.0003, "leverage": 5.0, "funding_rate": 0.0002},
        {"fee_rate": 0.0003, "slippage": 0.0001, "leverage": 2.0, "funding_rate": 0.00005},
    ]
    best = None
    best_score = -1e9
    klines = get_futures_klines("1m", 600)
    for params in candidates:
        tester = Backtester(
            fee_rate=params["fee_rate"],
            slippage=params["slippage"],
            leverage=params["leverage"],
            funding_rate=params["funding_rate"],
        )
        report = tester.run(klines, strategy_engine)
        score = report["equity"] - report["max_drawdown"]
        if score > best_score:
            best_score = score
            best = params
    if best:
        best["strategy_min_score"] = CURRENT_PARAMS.get("strategy_min_score", 3)
        save_best_params(best)
        CURRENT_PARAMS.update(best)
    return best


def run_auto_pipeline():
    train_ml_predictor(get_futures_klines("1m", 500), epochs=1)
    train_transformer_predictor(get_futures_klines("1m", 500), epochs=1)
    best = optimize_params()
    klines = get_futures_klines("1m", 1000)
    report = backtester.run(klines, strategy_engine)
    return best, report


class MLPricePredictor(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


data_feed = DataFeed()
ta_engine = TAEngine()
strategy_engine = StrategyEngine(ta_engine)
backtester = Backtester(
    fee_rate=CURRENT_PARAMS["fee_rate"],
    slippage=CURRENT_PARAMS["slippage"],
    leverage=CURRENT_PARAMS["leverage"],
    funding_rate=CURRENT_PARAMS["funding_rate"],
)
ml_predictor = MLPricePredictor().to(device)
ml_predictor.eval()
ml_trained = False


class TransformerPredictor(torch.nn.Module):
    def __init__(self, input_size=5, model_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, model_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


transformer_predictor = TransformerPredictor().to(device)
transformer_predictor.eval()
transformer_trained = False


def _build_feature_matrix(klines):
    features = []
    for row in klines:
        features.append(
            [
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ]
        )
    return np.array(features, dtype=np.float32)


def _baseline_next_close(klines, window=60):
    if len(klines) < 5:
        return None
    closes = np.array([float(k[4]) for k in klines[-window:]], dtype=np.float32)
    x = np.arange(len(closes), dtype=np.float32)
    slope, intercept = np.polyfit(x, closes, 1)
    raw = intercept + slope * (len(closes))
    atr = calc_atr_from_klines(klines)
    last_close = closes[-1]
    if atr and atr > 0:
        low = last_close - 2.5 * atr
        high = last_close + 2.5 * atr
        raw = float(min(max(raw, low), high))
    return float(raw)


def _blend_prediction(pred, baseline, last_close, min_delta=0.001):
    if baseline is None:
        return pred
    if last_close <= 0:
        return pred
    if abs(pred - last_close) / last_close < min_delta:
        return baseline
    return 0.7 * pred + 0.3 * baseline


def train_ml_predictor(klines, epochs=1):
    global ml_trained
    if len(klines) < 80:
        return
    features = _build_feature_matrix(klines)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    scaled = (features - mean) / std
    window = 60
    X = []
    y = []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i])
        y.append(scaled[i, 3])
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
    optimizer = torch.optim.Adam(ml_predictor.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    ml_predictor.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = ml_predictor(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    ml_predictor.eval()
    ml_trained = True


def predict_next_price(klines):
    if len(klines) < 60:
        return None
    features = _build_feature_matrix(klines)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    scaled = (features - mean) / std
    window = scaled[-60:]
    x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    if not ml_trained:
        return _baseline_next_close(klines)
    with torch.no_grad():
        pred = ml_predictor(x).cpu().numpy().flatten()[0]
    denorm = float(pred * std[3] + mean[3])
    baseline = _baseline_next_close(klines)
    last_close = float(features[-1, 3])
    return float(_blend_prediction(denorm, baseline, last_close))


def train_transformer_predictor(klines, epochs=1):
    global transformer_trained
    if len(klines) < 80:
        return
    features = _build_feature_matrix(klines)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    scaled = (features - mean) / std
    window = 60
    X = []
    y = []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i])
        y.append(scaled[i, 3])
    X = torch.tensor(X, dtype=torch.float32, device=device).clone().contiguous()
    y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1).clone().contiguous()
    optimizer = torch.optim.Adam(transformer_predictor.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    transformer_predictor.train()
    torch.autograd.set_detect_anomaly(True)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = transformer_predictor(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    transformer_predictor.eval()
    transformer_trained = True


def predict_next_price_transformer(klines):
    if len(klines) < 60:
        return None
    features = _build_feature_matrix(klines)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    scaled = (features - mean) / std
    window = scaled[-60:]
    x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    if not transformer_trained:
        return _baseline_next_close(klines)
    with torch.no_grad():
        pred = transformer_predictor(x).cpu().numpy().flatten()[0]
    denorm = float(pred * std[3] + mean[3])
    baseline = _baseline_next_close(klines)
    last_close = float(features[-1, 3])
    return float(_blend_prediction(denorm, baseline, last_close))

# ==============================
# Windows 声音
# ==============================
if platform.system() == "Windows":
    import winsound


# ==============================
# GPU 版预期胜率模型
# ==============================
class WinRateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


winrate_model = WinRateModel().to(device)
winrate_model.eval()  # 推理模式


def winrate_from_gpu_model(
    shendu_budui,
    taker_zhanbi,
    duanqi_duo,
    duanqi_kong,
    duo_qiangdu,
    kong_qiangdu,
    fake_break_flag,
    duotou_baocang,
    kongtou_baocang,
    jiacha_hexian,
):
    with torch.no_grad():
        x = torch.tensor(
            [
                shendu_budui,
                taker_zhanbi,
                float(duanqi_duo),
                float(duanqi_kong),
                float(duo_qiangdu),
                float(kong_qiangdu),
                float(fake_break_flag),
                float(duotou_baocang),
                float(kongtou_baocang),
                float(jiacha_hexian),
            ],
            dtype=torch.float32,
            device=device,
        )
        y = winrate_model(x)
        return float(y.item())


# ==============================
# 行情获取
# ==============================

def get_futures_klines(interval="1m", limit=200):
    return data_feed.fetch_klines("binance_futures", FUTURES_SYMBOL, interval, limit)


def get_spot_klines(interval="1m", limit=200):
    return data_feed.fetch_klines("binance_spot", SPOT_SYMBOL, interval, limit)


def get_multi_exchange_klines(interval="1m", limit=200):
    sources = [
        data_feed.fetch_klines("binance_spot", SPOT_SYMBOL, interval, limit),
        data_feed.fetch_klines("okx", SPOT_SYMBOL, interval, limit),
        data_feed.fetch_klines("bybit", SPOT_SYMBOL, interval, limit),
    ]
    if any(len(src) == 0 for src in sources):
        return []
    merged = []
    for idx in range(min(len(src) for src in sources)):
        rows = [src[idx] for src in sources]
        open_time = rows[0][0]
        open_ = np.mean([float(r[1]) for r in rows])
        high = max(float(r[2]) for r in rows)
        low = min(float(r[3]) for r in rows)
        close = np.mean([float(r[4]) for r in rows])
        volume = np.mean([float(r[5]) for r in rows])
        merged.append([open_time, open_, high, low, close, volume, open_time, volume, 0, 0, 0, 0])
    return merged


def ema_trend_direction(klines, span_short=20, span_long=60):
    if not klines or len(klines) < span_long + 5:
        return 0
    closes = [float(k[4]) for k in klines]
    series = pd.Series(closes)
    ema_s = series.ewm(span=span_short, adjust=False).mean().iloc[-1]
    ema_l = series.ewm(span=span_long, adjust=False).mean().iloc[-1]
    if ema_s > ema_l:
        return 1
    if ema_s < ema_l:
        return -1
    return 0


def multi_tf_regime_snapshot():
    intervals = ["1m", "5m", "1h", "4h", "1d"]
    trends = {}
    for interval in intervals:
        kl = get_futures_klines(interval, 200)
        trends[interval] = ema_trend_direction(kl, 20, 60)
    long_count = sum(1 for v in trends.values() if v == 1)
    short_count = sum(1 for v in trends.values() if v == -1)
    return trends, long_count, short_count


def get_futures_depth(limit=50):
    global _last_depth_snapshot
    try:
        depth = data_feed.fetch_order_book("binance_futures", FUTURES_SYMBOL, limit)
    except Exception:
        depth = None
    if depth and depth.get("bids") and depth.get("asks"):
        _last_depth_snapshot = depth
        return depth
    if _last_depth_snapshot:
        return _last_depth_snapshot
    return {"bids": [], "asks": []}


def get_futures_trades(limit=100):
    url = f"{FUTURES_BASE_URL}/fapi/v1/trades"
    response = requests.get(url, params={"symbol": FUTURES_SYMBOL, "limit": limit}, timeout=10)
    response.raise_for_status()
    return response.json()


# ==============================
# 技术指标
# ==============================

def bollinger(closes, n=20, k=2):
    if len(closes) < n:
        return None
    ma = np.mean(closes[-n:])
    std = np.std(closes[-n:])
    return ma + k * std, ma, ma - k * std


def taker_ratio(trades):
    buy = sum(float(t["qty"]) for t in trades if not t["isBuyerMaker"])
    sell = sum(float(t["qty"]) for t in trades if t["isBuyerMaker"])
    total = buy + sell
    return buy / total if total > 0 else 0.5


def big_trade_side(trades, threshold=2):
    da_mairu = any(float(t["qty"]) >= threshold and not t["isBuyerMaker"] for t in trades)
    da_maichu = any(float(t["qty"]) >= threshold and t["isBuyerMaker"] for t in trades)
    return da_mairu, da_maichu


def volatility(closes, window=20):
    if len(closes) < window + 1:
        return 0
    return np.std(np.diff(closes[-(window + 1) :]))


def vwap_from_klines(klines):
    prices = [(float(k[2]) + float(k[3]) + float(k[4])) / 3 for k in klines]
    vols = [float(k[5]) for k in klines]
    return sum(p * v for p, v in zip(prices, vols)) / sum(vols) if sum(vols) > 0 else prices[-1]


def find_support_resistance(closes, lookback=80):
    if len(closes) < lookback:
        return None, None
    window = closes[-lookback:]
    highs = [window[i] for i in range(1, len(window) - 1) if window[i] > window[i - 1] and window[i] > window[i + 1]]
    lows = [window[i] for i in range(1, len(window) - 1) if window[i] < window[i - 1] and window[i] < window[i + 1]]
    if not highs or not lows:
        return None, None
    return np.median(lows), np.median(highs)


def advanced_support_resistance(klines, lookback=120, pivot=2):
    if not klines or len(klines) < lookback:
        return None, None
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    piv_highs = []
    piv_lows = []
    for i in range(pivot, len(highs) - pivot):
        if all(highs[i] > highs[i - j] for j in range(1, pivot + 1)) and all(
            highs[i] > highs[i + j] for j in range(1, pivot + 1)
        ):
            piv_highs.append(highs[i])
        if all(lows[i] < lows[i - j] for j in range(1, pivot + 1)) and all(
            lows[i] < lows[i + j] for j in range(1, pivot + 1)
        ):
            piv_lows.append(lows[i])
    atr = calc_atr_from_klines(klines)
    if piv_lows:
        support = np.median(piv_lows[-10:])
    else:
        support = np.percentile(closes[-lookback:], 20)
    if piv_highs:
        resistance = np.median(piv_highs[-10:])
    else:
        resistance = np.percentile(closes[-lookback:], 80)
    if atr and atr > 0:
        support -= 0.15 * atr
        resistance += 0.15 * atr
    return support, resistance


def fake_breakout(closes, upper, lower):
    if len(closes) < 3:
        return None
    c1, c2 = closes[-2], closes[-1]
    if c1 > upper and abs(c2 - upper) < FAKE_BREAK_DIST:
        return "fake_up"
    if c1 < lower and abs(c2 - lower) < FAKE_BREAK_DIST:
        return "fake_down"
    return None


def trend_strength(closes):
    if len(closes) < 40:
        return 0, 0
    t1 = closes[-1] - closes[-10]
    t2 = closes[-1] - closes[-20]
    t3 = closes[-1] - closes[-40]
    duo = (t1 > 0) + (t2 > 0) + (t3 > 0)
    kong = (t1 < 0) + (t2 < 0) + (t3 < 0)
    return duo, kong


def calc_macd(closes, fast=12, slow=26, signal=9):
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd.values, signal_line.values, hist.values


def calc_rsi(closes, period=14):
    closes = np.array(closes)
    diff = np.diff(closes)
    up = np.where(diff > 0, diff, 0)
    down = np.where(diff < 0, -diff, 0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50] * (len(closes) - len(rsi)), rsi])
    return rsi


def calc_atr_from_klines(klines, period=14):
    if len(klines) < period + 1:
        return 0
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    closes = np.array([float(k[4]) for k in klines])
    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    tr1 = highs - lows
    tr2 = np.abs(highs - prev_close)
    tr3 = np.abs(lows - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(period).mean().iloc[-1]
    return float(atr)


def feature_mining_score(indicators, closes):
    if not indicators or not closes:
        return 0.0
    rsi = indicators.get("rsi", 50)
    macd = indicators.get("macd", 0)
    macd_signal = indicators.get("macd_signal", 0)
    atr = indicators.get("atr", 0)
    trend_up = float(macd > macd_signal)
    trend_down = float(macd < macd_signal)
    momentum = (rsi - 50) / 50.0
    vol_score = 0.0
    if atr and atr > 0:
        vol_score = min(1.0, atr / (np.mean(closes[-20:]) + 1e-6))
    score = 0.4 * momentum + 0.3 * (trend_up - trend_down) + 0.3 * vol_score
    return float(max(-1.0, min(1.0, score)))


def sentiment_factor_score(news_sentiment, sentiment_score, twitter_sentiment):
    raw = 0.5 * sentiment_score + 0.3 * news_sentiment + 0.2 * twitter_sentiment
    return float(max(-1.0, min(1.0, raw)))


def strategy_adaptivity_score(closes):
    if len(closes) < 10:
        return 0.0
    returns = np.diff(closes[-20:]) / (np.array(closes[-20:-1]) + 1e-6)
    volatility_score = np.std(returns)
    return float(max(0.0, min(1.0, volatility_score * 100)))


# ==============================
# 深度分析
# ==============================

def analyze_depth(depth):
    global _last_depth_snapshot

    bids_raw = depth.get("bids", []) if depth else []
    asks_raw = depth.get("asks", []) if depth else []
    if not bids_raw or not asks_raw:
        return {
            "jiacha": 0.0,
            "shendu_budui": 0.0,
            "maipan_liang": 0.0,
            "maichu_liang": 0.0,
            "mai_qiang": False,
            "jia_mai_qiang": False,
        }

    bids = [(float(p), float(q)) for p, q in bids_raw[:20]]
    asks = [(float(p), float(q)) for p, q in asks_raw[:20]]

    best_bid, _ = bids[0]
    best_ask, _ = asks[0]

    jiacha = best_ask - best_bid

    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    shendu_budui = (ask_vol - bid_vol) / (ask_vol + bid_vol + 1e-9)

    ask_qtys = [q for _, q in asks[:5]]
    avg_ask = sum(ask_qtys) / len(ask_qtys)
    maiqiang = any(q > avg_ask * 3 for q in ask_qtys)

    jia_maiqiang = False
    if _last_depth_snapshot is not None:
        prev_asks = [(float(p), float(q)) for p, q in _last_depth_snapshot["asks"][:5]]
        prev_qtys = [q for _, q in prev_asks]
        if prev_qtys:
            prev_avg = sum(prev_qtys) / len(prev_qtys)
            prev_mai = any(q > prev_avg * 3 for q in prev_qtys)
        else:
            prev_mai = False
        if prev_mai and not maiqiang:
            jia_maiqiang = True

    _last_depth_snapshot = depth

    return {
        "jiacha": jiacha,
        "shendu_budui": shendu_budui,
        "mai_qiang": maiqiang,
        "jia_mai_qiang": jia_maiqiang,
        "maipan_liang": bid_vol,
        "maichu_liang": ask_vol,
    }


# ==============================
# 多周期趋势（15m + 1h + 4h）
# ==============================

def multi_tf_trend_midterm():
    k15 = get_futures_klines("15m", 200)
    k1h = get_futures_klines("1h", 200)
    k4h = get_futures_klines("4h", 200)

    def ema_trend(kl, span_short=20, span_long=60):
        closes = [float(k[4]) for k in kl]
        if len(closes) < span_long + 5:
            return 0
        s = pd.Series(closes)
        ema_s = s.ewm(span=span_short, adjust=False).mean()
        ema_l = s.ewm(span=span_long, adjust=False).mean()
        diff = ema_s - ema_l
        return np.sign(diff.iloc[-1])

    t15 = ema_trend(k15, 20, 60)
    t1h = ema_trend(k1h, 20, 60)
    t4h = ema_trend(k4h, 20, 60)

    duo = (t15 == 1) + (t1h == 1) + (t4h == 1)
    kong = (t15 == -1) + (t1h == -1) + (t4h == -1)

    duo_qiangdu = duo / 3.0
    kong_qiangdu = kong / 3.0

    return duo_qiangdu, kong_qiangdu, k15, k1h, k4h


# ==============================
# 行情状态识别
# ==============================

def detect_market_regime():
    k1h = get_futures_klines("1h", 200)
    closes = [float(k[4]) for k in k1h]
    if len(closes) < 60:
        return "未知", 0

    atr_1h = calc_atr_from_klines(k1h, period=14)
    duan_qi_duo, duan_qi_kong = trend_strength(closes)
    trend_score = max(duan_qi_duo, duan_qi_kong)

    if atr_1h > 150 and trend_score >= 2:
        return "趋势期", atr_1h
    if atr_1h < 80 and trend_score <= 1:
        return "震荡期", atr_1h
    return "过渡期", atr_1h


# ==============================
# 爆仓压力识别
# ==============================

def detect_liquidation_pressure(klines_1m, lookback=30):
    if len(klines_1m) < lookback + 5:
        return 0.0, 0.0

    ks = klines_1m[-lookback:]
    highs = np.array([float(k[2]) for k in ks])
    lows = np.array([float(k[3]) for k in ks])
    opens = np.array([float(k[1]) for k in ks])
    closes = np.array([float(k[4]) for k in ks])
    vols = np.array([float(k[5]) for k in ks])

    avg_vol = np.mean(vols)
    if avg_vol <= 0:
        return 0.0, 0.0

    duotou_baocang_score = 0.0
    kongtou_baocang_score = 0.0

    for h, l, o, c, v in zip(highs, lows, opens, closes, vols):
        body = abs(c - o)
        range_ = h - l
        if range_ <= 0:
            continue
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if c < o and lower_wick > body * 1.5 and v > avg_vol * 1.8:
            duotou_baocang_score += 1.0

        if c > o and upper_wick > body * 1.5 and v > avg_vol * 1.8:
            kongtou_baocang_score += 1.0

    duotou_baocang_score = min(3.0, duotou_baocang_score)
    kongtou_baocang_score = min(3.0, kongtou_baocang_score)

    return duotou_baocang_score, kongtou_baocang_score


# ==============================
# 国际宏观因子（美股 / 黄金 / 新闻情绪占位）
# ==============================

def get_macro_context():
    """
    简化版：
    - 用 ETHUSDT 代表风险资产（类美股）
    - 用 XAUUSDT（若有）代表黄金避险
    - 新闻情绪暂用 0，占位，后续可接入新闻 API
    """
    risk_on_ret = 0.0
    risk_off_ret = 0.0
    news_sentiment = 0.0
    macro_score = 0.0

    # 风险资产：ETH
    try:
        eth_kl = requests.get(
            f"{SPOT_BASE_URL}/api/v3/klines",
            params={"symbol": "ETHUSDT", "interval": "1h", "limit": 2},
        ).json()
        if len(eth_kl) >= 2:
            p0 = float(eth_kl[0][4])
            p1 = float(eth_kl[1][4])
            risk_on_ret = (p1 - p0) / p0
    except Exception:
        risk_on_ret = 0.0

    # 黄金：XAUUSDT（部分交易所支持，若失败则视为中性）
    try:
        xau_kl = requests.get(
            f"{SPOT_BASE_URL}/api/v3/klines",
            params={"symbol": "XAUUSDT", "interval": "1h", "limit": 2},
        ).json()
        if isinstance(xau_kl, list) and len(xau_kl) >= 2:
            g0 = float(xau_kl[0][4])
            g1 = float(xau_kl[1][4])
            risk_off_ret = (g1 - g0) / g0
    except Exception:
        risk_off_ret = 0.0

    # 归一化成 -1 ~ 1
    risk_on_score = max(-1.0, min(1.0, risk_on_ret * 50))  # 2% ≈ 1
    risk_off_score = max(-1.0, min(1.0, risk_off_ret * 50))
    macro_score = max(-1.0, min(1.0, (risk_on_score - risk_off_score) / 2))

    return {
        "risk_on_score": risk_on_score,
        "risk_off_score": risk_off_score,
        "news_sentiment": news_sentiment,
        "macro_score": macro_score,
    }


def get_market_sentiment():
    """使用外部情绪指标（如恐惧贪婪指数）作为 NLP/情绪因子的占位。"""
    try:
        response = requests.get(FNG_API_URL, params={"limit": 1, "format": "json"}, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return 0.0
        value = float(data[0].get("value", 50))
        return max(-1.0, min(1.0, (value - 50) / 50))
    except Exception:
        return 0.0


def get_twitter_sentiment():
    return 0.0


def get_news_sentiment():
    return 0.0


def get_macro_factors():
    """宏观经济/政策因素占位：返回组合分数，便于策略叠加。"""
    context = get_macro_context()
    sentiment = get_market_sentiment()
    twitter_sentiment = get_twitter_sentiment()
    news_sentiment = get_news_sentiment()
    macro_bias = context["macro_score"]
    return {
        "macro_score": macro_bias,
        "sentiment_score": sentiment,
        "twitter_sentiment": twitter_sentiment,
        "news_sentiment_extra": news_sentiment,
        "risk_on_score": context["risk_on_score"],
        "risk_off_score": context["risk_off_score"],
        "news_sentiment": context["news_sentiment"],
    }


# ==============================
# 自动标注 + 日志
# ==============================

def auto_label_signal(entry_price, direction, future_closes):
    if direction == "long":
        zhi_ying = entry_price + 30
        zhi_sun = entry_price - 25
        for p in future_closes:
            if p >= zhi_ying:
                return "good"
            if p <= zhi_sun:
                return "bad"
    else:
        zhi_ying = entry_price - 30
        zhi_sun = entry_price + 25
        for p in future_closes:
            if p <= zhi_ying:
                return "good"
            if p >= zhi_sun:
                return "bad"
    return "neutral"


def log_signal(record):
    try:
        with open(SIGNAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("写入日志失败：", e)


def load_signals():
    if not os.path.exists(SIGNAL_LOG_FILE):
        return []
    data = []
    with open(SIGNAL_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data


def build_win_rate_table():
    global _win_rate_cache
    data = load_signals()
    if not data:
        _win_rate_cache = {}
        return

    buckets = {}
    for s in data:
        label = s.get("label")
        if label not in ("good", "bad"):
            continue

        imb = s.get("imbalance", 0)
        tr = s.get("taker_ratio", 0.5)
        tl = s.get("trend_long", 0)
        ts = s.get("trend_short", 0)

        imb_bucket = int(imb * 10)
        tr_bucket = int((tr - 0.5) * 20)
        trend_bucket = tl - ts

        key = (imb_bucket, tr_bucket, trend_bucket)
        if key not in buckets:
            buckets[key] = {"good": 0, "bad": 0}
        buckets[key][label] += 1

    _win_rate_cache = {}
    for k, v in buckets.items():
        total = v["good"] + v["bad"]
        if total == 0:
            continue
        _win_rate_cache[k] = v["good"] / total


def estimate_win_rate(imb, tr, tl, ts):
    if not _win_rate_cache:
        build_win_rate_table()
    imb_bucket = int(imb * 10)
    tr_bucket = int((tr - 0.5) * 20)
    trend_bucket = tl - ts
    key = (imb_bucket, tr_bucket, trend_bucket)
    return _win_rate_cache.get(key, 0.5)


# ==============================
# 声音 & 桌面提醒
# ==============================

def play_ding():
    if platform.system() == "Windows":
        winsound.MessageBeep()


def desktop_notify(title, message):
    try:
        notification.notify(title=title, message=message, timeout=3)
    except Exception:
        pass


# ==============================
# 多策略并行 + 投票系统
# ==============================

def strategy_trend_follow(price, duo_qiangdu, kong_qiangdu, regime):
    duo = 0
    kong = 0
    if regime == "趋势期":
        if duo_qiangdu > 0.66:
            duo += 3
        if kong_qiangdu > 0.66:
            kong += 3
    else:
        if duo_qiangdu > 0.66:
            duo += 2
        if kong_qiangdu > 0.66:
            kong += 2
    return duo, kong


def strategy_pullback(closes_15m, duo_qiangdu, kong_qiangdu):
    duo = 0
    kong = 0
    if len(closes_15m) < 40:
        return duo, kong
    bb_15 = bollinger(closes_15m)
    if not bb_15:
        return duo, kong
    shang_15, zhong_15, xia_15 = bb_15
    jia_15 = closes_15m[-1]
    if duo_qiangdu > 0.66 and abs(jia_15 - zhong_15) < (shang_15 - zhong_15) * 0.4:
        duo += 2
    if kong_qiangdu > 0.66 and abs(jia_15 - zhong_15) < (zhong_15 - xia_15) * 0.4:
        kong += 2
    return duo, kong


def strategy_fake_breakout(closes_1m, shanggui, xiagui):
    duo = 0
    kong = 0
    fb = fake_breakout(closes_1m, shanggui, xiagui)
    if fb == "fake_up":
        kong += 2
    if fb == "fake_down":
        duo += 2
    return duo, kong, fb


def strategy_rsi_long(closes, rsi_threshold=30):
    rsi_series = calc_rsi(closes)
    if rsi_series[-1] < rsi_threshold:
        return {"action": "buy", "reason": "RSI 低于30买入"}
    if rsi_series[-1] > 70:
        return {"action": "sell", "reason": "RSI 超过70卖出"}
    return {"action": None, "reason": "观望"}


def strategy_ema_macd_long(closes, ema_short=12, ema_long=26):
    s = pd.Series(closes)
    ema_s = s.ewm(span=ema_short, adjust=False).mean()
    ema_l = s.ewm(span=ema_long, adjust=False).mean()
    macd, signal_line, _ = calc_macd(closes)
    if ema_s.iloc[-1] > ema_l.iloc[-1] and macd[-1] > signal_line[-1]:
        return {"action": "buy", "reason": "EMA 上穿且 MACD 在线上方"}
    if ema_s.iloc[-1] < ema_l.iloc[-1]:
        return {"action": "sell", "reason": "EMA 下穿卖出"}
    return {"action": None, "reason": "观望"}


def strategy_support_resistance_long(closes):
    support, resistance = find_support_resistance(closes)
    price = closes[-1]
    if support and abs(price - support) < 15 and price > support:
        return {"action": "buy", "reason": "接近支撑位反弹买入"}
    if resistance and abs(price - resistance) < 15:
        return {"action": "sell", "reason": "接近阻力位卖出"}
    return {"action": None, "reason": "观望"}


def strategy_ema_trend_long(closes, ema_short=50, ema_long=200):
    s = pd.Series(closes)
    ema_s = s.ewm(span=ema_short, adjust=False).mean()
    ema_l = s.ewm(span=ema_long, adjust=False).mean()
    if ema_s.iloc[-1] > ema_l.iloc[-1]:
        return {"action": "buy", "reason": "50EMA 上穿200EMA 买入"}
    if ema_s.iloc[-1] < ema_l.iloc[-1]:
        return {"action": "sell", "reason": "50EMA 下穿200EMA 卖出"}
    return {"action": None, "reason": "观望"}


def strategy_breakout_long(closes, lookback=20):
    if len(closes) < lookback + 1:
        return {"action": None, "reason": "数据不足"}
    price = closes[-1]
    resistance = max(closes[-(lookback + 1) : -1])
    support = min(closes[-(lookback + 1) : -1])
    if price > resistance:
        return {"action": "buy", "reason": "突破阻力位买入"}
    if price < support:
        return {"action": "sell", "reason": "跌破支撑位卖出"}
    return {"action": None, "reason": "观望"}


def strategy_bollinger_long(closes):
    bb = bollinger(closes)
    if not bb:
        return {"action": None, "reason": "数据不足"}
    upper, mid, lower = bb
    price = closes[-1]
    if price < lower:
        return {"action": "buy", "reason": "触及下轨反弹买入"}
    if price > upper:
        return {"action": "sell", "reason": "触及上轨卖出"}
    return {"action": None, "reason": "观望"}


def strategy_volume_breakout_long(closes, volumes, lookback=20):
    if len(closes) < lookback + 1:
        return {"action": None, "reason": "数据不足"}
    price = closes[-1]
    prev_high = max(closes[-(lookback + 1) : -1])
    avg_vol = np.mean(volumes[-lookback:])
    if volumes[-1] > avg_vol * 1.5 and price > prev_high:
        return {"action": "buy", "reason": "放量突破前高买入"}
    return {"action": None, "reason": "观望"}


def strategy_combo_long(closes):
    rsi_series = calc_rsi(closes)
    macd, signal_line, _ = calc_macd(closes)
    s = pd.Series(closes)
    ema_s = s.ewm(span=12, adjust=False).mean()
    ema_l = s.ewm(span=26, adjust=False).mean()
    if rsi_series[-1] < 30 and macd[-1] > signal_line[-1] and ema_s.iloc[-1] > ema_l.iloc[-1]:
        return {"action": "buy", "reason": "RSI+MACD+EMA 同向做多"}
    return {"action": None, "reason": "观望"}


def strategy_atr_risk_control(closes, klines, risk_ratio=0.02):
    atr = calc_atr_from_klines(klines)
    price = closes[-1]
    tp = price + atr * 2
    sl = price - atr * risk_ratio * price
    return {"tp": tp, "sl": sl, "reason": "ATR 风险控制"}


def strategy_rsi_short(closes, rsi_threshold=70):
    rsi_series = calc_rsi(closes)
    if rsi_series[-1] > rsi_threshold:
        return {"action": "sell", "reason": "RSI 高于70开空"}
    if rsi_series[-1] < 30:
        return {"action": "cover", "reason": "RSI 低于30平空"}
    return {"action": None, "reason": "观望"}


def strategy_ema_macd_short(closes, ema_short=12, ema_long=26):
    s = pd.Series(closes)
    ema_s = s.ewm(span=ema_short, adjust=False).mean()
    ema_l = s.ewm(span=ema_long, adjust=False).mean()
    macd, signal_line, _ = calc_macd(closes)
    if ema_s.iloc[-1] < ema_l.iloc[-1] and macd[-1] < signal_line[-1]:
        return {"action": "sell", "reason": "EMA 下穿且 MACD 在线下方"}
    if ema_s.iloc[-1] > ema_l.iloc[-1]:
        return {"action": "cover", "reason": "EMA 上穿平空"}
    return {"action": None, "reason": "观望"}


def strategy_support_resistance_short(closes):
    support, resistance = find_support_resistance(closes)
    price = closes[-1]
    if resistance and abs(price - resistance) < 15 and price < resistance:
        return {"action": "sell", "reason": "接近阻力位反转开空"}
    if support and abs(price - support) < 15:
        return {"action": "cover", "reason": "接近支撑位平空"}
    return {"action": None, "reason": "观望"}


def strategy_ema_trend_short(closes, ema_short=50, ema_long=200):
    s = pd.Series(closes)
    ema_s = s.ewm(span=ema_short, adjust=False).mean()
    ema_l = s.ewm(span=ema_long, adjust=False).mean()
    if ema_s.iloc[-1] < ema_l.iloc[-1]:
        return {"action": "sell", "reason": "50EMA 下穿200EMA 开空"}
    if ema_s.iloc[-1] > ema_l.iloc[-1]:
        return {"action": "cover", "reason": "50EMA 上穿200EMA 平空"}
    return {"action": None, "reason": "观望"}


def strategy_bollinger_short(closes):
    bb = bollinger(closes)
    if not bb:
        return {"action": None, "reason": "数据不足"}
    upper, mid, lower = bb
    price = closes[-1]
    if price > upper:
        return {"action": "sell", "reason": "突破上轨回调开空"}
    if price < lower:
        return {"action": "cover", "reason": "触及下轨平空"}
    return {"action": None, "reason": "观望"}


# ==============================
# 核心：中短线止盈逻辑 + 宏观因子
# ==============================

def compute_signal_realtime_core(state):
    has_position = state.get("has_position", False)

    k4h = get_futures_klines("4h", 200)
    closes_4h = [float(k[4]) for k in k4h]
    if not closes_4h:
        return 0, "观望", 0, 0, 0, None, None, None, None
    price = closes_4h[-1]
    last_kline = k4h[-1] if k4h else None
    market_data = None
    if last_kline:
        market_data = {
            "open": float(last_kline[1]),
            "high": float(last_kline[2]),
            "low": float(last_kline[3]),
            "close": float(last_kline[4]),
            "volume": float(last_kline[5]),
        }

    indicators = ta_engine.compute_all(k4h)
    bb = bollinger(closes_4h)
    if not bb:
        return price, "观望", 0, 0, 0, None, None, None, None
    shanggui, zhongxian, xiagui = bb

    jiaquan_junjia = vwap_from_klines(k4h)
    atr_4h = indicators.get("atr", 0)
    if atr_4h <= 0:
        atr_4h = max(1.0, volatility(closes_4h, VOL_WINDOW))

    if atr_4h < VOL_MIN:
        return price, "波动率过低", 0, 0, jiaquan_junjia, None, None, None, None

    zhicheng, yali = advanced_support_resistance(k4h)

    ema_fast = indicators.get("ema_fast")
    ema_slow = indicators.get("ema_slow")
    if ema_fast and ema_slow:
        shichang_zhuangtai = "趋势期" if ema_fast > ema_slow else "震荡期"
    else:
        shichang_zhuangtai = "过渡期"

    tf_trends, tf_long, tf_short = multi_tf_regime_snapshot()
    shichang_zhuangtai = f"多周期 {tf_long}/{tf_short} (1m/5m/1h/4h/1d)"

    strategy_signal = strategy_engine.generate_signal(k4h, shichang_zhuangtai)
    duanqi_duo, duanqi_kong = trend_strength(closes_4h)
    jia_tupo = fake_breakout(closes_4h, shanggui, xiagui)

    taker_zhanbi = 0.5
    da_mairu, da_maichu = False, False
    shendu_info = {
        "shendu_budui": 0.0,
        "mai_qiang": False,
        "jia_mai_qiang": False,
        "maipan_liang": 0.0,
        "maichu_liang": 0.0,
        "jiacha": 0.0,
    }
    shendu_budui = shendu_info["shendu_budui"]
    maiqiang = shendu_info["mai_qiang"]
    jia_maiqiang = shendu_info["jia_mai_qiang"]
    jiacha_hexian = 0.0
    duotou_baocang, kongtou_baocang = 0.0, 0.0

    macro = get_macro_factors()
    risk_on_score = macro["risk_on_score"]
    risk_off_score = macro["risk_off_score"]
    news_sentiment = macro["news_sentiment"]
    sentiment_score = macro["sentiment_score"]
    twitter_sentiment = macro["twitter_sentiment"]
    news_sentiment_extra = macro["news_sentiment_extra"]
    macro_score = macro["macro_score"]
    ml_pred_price = predict_next_price(k4h)
    transformer_pred_price = predict_next_price_transformer(k4h)
    if ml_pred_price is not None and transformer_pred_price is not None:
        ensemble_pred_price = (ml_pred_price + transformer_pred_price) / 2
    else:
        ensemble_pred_price = ml_pred_price or transformer_pred_price

    duo_qiangdu = 0.0
    kong_qiangdu = 0.0
    s1_duo, s1_kong = strategy_trend_follow(price, duanqi_duo, duanqi_kong, shichang_zhuangtai)
    s2_duo, s2_kong = strategy_pullback(closes_4h, duanqi_duo, duanqi_kong)
    s3_duo, s3_kong, fb = strategy_fake_breakout(closes_4h, shanggui, xiagui)
    ai_signal = multi_factor_ai_strategy(_build_feature_matrix(k4h), sentiment_score, macro_score)

    macro_bias = (
        0.3 * risk_on_score
        - 0.3 * risk_off_score
        + 0.2 * news_sentiment
        + 0.2 * sentiment_score
        + 0.2 * macro_score
        + 0.15 * twitter_sentiment
        + 0.15 * news_sentiment_extra
    )
    macro_bias = max(-1.0, min(1.0, macro_bias))

    ensemble_delta = None
    if ensemble_pred_price is not None and price:
        ensemble_delta = (ensemble_pred_price - price) / price
    factor_mining = feature_mining_score(indicators, closes_4h)
    sentiment_factor = sentiment_factor_score(news_sentiment, sentiment_score, twitter_sentiment)
    adaptivity_score = strategy_adaptivity_score(closes_4h)

    duo_defen = 0.0
    kong_defen = 0.0

    duo_defen += s1_duo * 1.5
    kong_defen += s1_kong * 1.5
    duo_defen += s2_duo * 1.2
    kong_defen += s2_kong * 1.2
    duo_defen += s3_duo * 1.0
    kong_defen += s3_kong * 1.0

    if shendu_budui < -IMB_THRESH:
        duo_defen += 1.2
    elif shendu_budui > IMB_THRESH:
        kong_defen += 1.2

    if taker_zhanbi > 0.57:
        duo_defen += 1.0
    elif taker_zhanbi < 0.43:
        kong_defen += 1.0

    if da_mairu:
        duo_defen += 0.8
    if da_maichu:
        kong_defen += 0.8

    if maiqiang:
        kong_defen += 0.6
    if jia_maiqiang:
        kong_defen += 0.6

    if jiacha_hexian > 30:
        kong_defen += 1.0
    elif jiacha_hexian < -30:
        duo_defen += 1.0

    if duanqi_duo > duanqi_kong:
        duo_defen += 1.1
    elif duanqi_kong > duanqi_duo:
        kong_defen += 1.1

    if duotou_baocang > 0:
        kong_defen += 0.7 * float(duotou_baocang)
    if kongtou_baocang > 0:
        duo_defen += 0.7 * float(kongtou_baocang)

    if macro_bias > 0.15:
        duo_defen += 1.1
    elif macro_bias < -0.15:
        kong_defen += 1.1

    if factor_mining > 0.2:
        duo_defen += 0.8
    elif factor_mining < -0.2:
        kong_defen += 0.8

    if sentiment_factor > 0.2:
        duo_defen += 0.7
    elif sentiment_factor < -0.2:
        kong_defen += 0.7

    if adaptivity_score > 0.6:
        if duanqi_duo > duanqi_kong:
            duo_defen += 0.5
        elif duanqi_kong > duanqi_duo:
            kong_defen += 0.5

    if strategy_signal["direction"] == "long":
        duo_defen += 1.4
    elif strategy_signal["direction"] == "short":
        kong_defen += 1.4

    if ai_signal["direction"] == "long":
        duo_defen += 1.6
    elif ai_signal["direction"] == "short":
        kong_defen += 1.6

    if ensemble_delta is not None:
        if ensemble_delta > 0.002:
            duo_defen += 1.6
        elif ensemble_delta < -0.002:
            kong_defen += 1.6
        elif ensemble_delta > 0.0005:
            duo_defen += 0.7
        elif ensemble_delta < -0.0005:
            kong_defen += 0.7

    if ml_pred_price is not None and price:
        ml_delta = (ml_pred_price - price) / price
        if ml_delta > 0.001:
            duo_defen += 0.6
        elif ml_delta < -0.001:
            kong_defen += 0.6

    if transformer_pred_price is not None and price:
        tf_delta = (transformer_pred_price - price) / price
        if tf_delta > 0.001:
            duo_defen += 0.6
        elif tf_delta < -0.001:
            kong_defen += 0.6

    pianxiang = "观望"
    fangxiang = None
    score_gap = abs(duo_defen - kong_defen)

    if duo_defen >= MIN_SCORE and duo_defen > kong_defen and score_gap >= 0.8:
        pianxiang = "偏多"
        fangxiang = "long"
    elif kong_defen >= MIN_SCORE and kong_defen > duo_defen and score_gap >= 0.8:
        pianxiang = "偏空"
        fangxiang = "short"

    extra = {
        "market_regime": shichang_zhuangtai,
        "atr_4h": atr_4h,
        "mt_long": duanqi_duo,
        "mt_short": duanqi_kong,
        "fake_break": fb,
        "basis": jiacha_hexian,
        "taker_ratio": taker_zhanbi,
        "imbalance": shendu_budui,
        "depth_profile": shendu_info,
        "trend_long": duanqi_duo,
        "trend_short": duanqi_kong,
        "long_liq": duotou_baocang,
        "short_liq": kongtou_baocang,
        "risk_on": risk_on_score,
        "risk_off": risk_off_score,
        "news_sentiment": news_sentiment,
        "sentiment_score": sentiment_score,
        "twitter_sentiment": twitter_sentiment,
        "news_sentiment_extra": news_sentiment_extra,
        "macro_score": macro_score,
        "feature_mining_score": factor_mining,
        "sentiment_factor": sentiment_factor,
        "strategy_adaptivity": adaptivity_score,
        "tf_trends": tf_trends,
        "ai_signal": ai_signal,
        "strategy_signal": strategy_signal,
        "indicators": indicators,
        "ml_pred_price": ml_pred_price,
        "transformer_pred_price": transformer_pred_price,
        "ensemble_pred_price": ensemble_pred_price,
        "market_data": market_data,
    }
    fused_ai = fused_ai_direction(extra)
    extra["fused_ai"] = fused_ai
    multi_score = multi_factor_score(extra)
    extra["multi_factor_score"] = multi_score
    if fangxiang is None:
        combined_dir = combined_direction(extra)
        if combined_dir == "long":
            pianxiang = "偏多"
            fangxiang = "long"
        elif combined_dir == "short":
            pianxiang = "偏空"
            fangxiang = "short"

    if has_position or fangxiang is None:
        return (
            price,
            pianxiang,
            round(duo_defen, 2),
            round(kong_defen, 2),
            jiaquan_junjia,
            zhicheng,
            yali,
            None,
            extra,
        )

    # ========= 新止盈 / 止损逻辑（中短线单笔价差） =========

    if atr_4h <= 0:
        atr_4h = 150

    # 止损：基于波动，限制在 250~600
    jichu_zhi_sun = atr_4h * 1.2
    jichu_zhi_sun = max(250, jichu_zhi_sun)
    jichu_zhi_sun = min(600, jichu_zhi_sun)

    # 止盈：波段级，至少 400，上不封顶，但不超过 5000（中短线）
    jichu_zhi_ying = atr_4h * 3.0
    jichu_zhi_ying = max(400, jichu_zhi_ying)
    jichu_zhi_ying = min(5000, jichu_zhi_ying)

    if fb == "fake_up":
        fake_flag = 1
    elif fb == "fake_down":
        fake_flag = -1
    else:
        fake_flag = 0

    try:
        yq_shengl = winrate_from_gpu_model(
            shendu_budui,
            taker_zhanbi,
            duanqi_duo,
            duanqi_kong,
            duo_qiangdu,
            kong_qiangdu,
            fake_flag,
            duotou_baocang,
            kongtou_baocang,
            jiacha_hexian,
        )
    except Exception:
        yq_shengl = estimate_win_rate(shendu_budui, taker_zhanbi, duanqi_duo, duanqi_kong)

    if fangxiang == "long":
        qushi_qiangdu = duo_qiangdu
        baocang_qiangdu = kongtou_baocang / 3.0
    else:
        qushi_qiangdu = kong_qiangdu
        baocang_qiangdu = duotou_baocang / 3.0

    baocang_qiangdu = max(0.0, min(1.0, baocang_qiangdu))

    # 宏观偏向：风险偏好 - 避险 + 新闻
    zonghe_qiangdu = (
        0.45 * qushi_qiangdu + 0.35 * yq_shengl + 0.1 * baocang_qiangdu + 0.1 * (macro_bias + 1) / 2
    )

    mubiao_yingkui = 1.5 + zonghe_qiangdu * 2.0
    mubiao_yingkui = max(1.5, min(3.5, mubiao_yingkui))

    zhi_sun_juli = jichu_zhi_sun

    if fb == "fake_up" and fangxiang == "short":
        zhi_sun_juli *= 0.7
    elif fb == "fake_down" and fangxiang == "long":
        zhi_sun_juli *= 0.7

    zhi_ying_juli = max(jichu_zhi_ying, zhi_sun_juli * mubiao_yingkui)
    zhi_ying_juli = min(5000, zhi_ying_juli)  # 单笔价差上限，保持中短线

    if fangxiang == "long":
        zhi_ying = price + zhi_ying_juli
        zhi_sun = price - zhi_sun_juli
    else:
        zhi_ying = price - zhi_ying_juli
        zhi_sun = price + zhi_sun_juli

    xin_signal = {
        "ts": time.time(),
        "price": price,
        "direction": fangxiang,
        "tp": zhi_ying,
        "sl": zhi_sun,
        "long_score": round(duo_defen, 2),
        "short_score": round(kong_defen, 2),
        "vwap": jiaquan_junjia,
        "support": zhicheng,
        "resistance": yali,
        "taker_ratio": taker_zhanbi,
        "imbalance": shendu_budui,
        "depth_profile": shendu_info,
        "fake_break": fb,
        "trend_long": duanqi_duo,
        "trend_short": duanqi_kong,
        "mt_long": duo_qiangdu,
        "mt_short": kong_qiangdu,
        "basis": jiacha_hexian,
        "expected_win": yq_shengl,
        "market_regime": shichang_zhuangtai,
        "atr_4h": atr_4h,
        "long_liq": duotou_baocang,
        "short_liq": kongtou_baocang,
        "risk_on": risk_on_score,
        "risk_off": risk_off_score,
        "news_sentiment": news_sentiment,
        "sentiment_score": sentiment_score,
        "twitter_sentiment": twitter_sentiment,
        "news_sentiment_extra": news_sentiment_extra,
        "macro_score": macro_score,
        "ai_signal": ai_signal,
        "strategy_signal": strategy_signal,
        "ml_pred_price": ml_pred_price,
        "transformer_pred_price": transformer_pred_price,
        "ensemble_pred_price": ensemble_pred_price,
        "market_data": market_data,
        "fused_ai": fused_ai,
        "extra": extra,
        "label": None,
    }

    return price, pianxiang, duo_defen, kong_defen, jiaquan_junjia, zhicheng, yali, xin_signal, extra


# ==============================
# 回测（简化版）
# ==============================

def compute_signal_on_window(klines_window):
    closes = [float(k[4]) for k in klines_window]
    if len(closes) < 60:
        return closes[-1], "观望", 0, 0, 0, None, None, None

    price = closes[-1]
    bb = bollinger(closes)
    if not bb:
        return price, "观望", 0, 0, 0, None, None, None

    shanggui, zhongxian, xiagui = bb
    jiaquan_junjia = vwap_from_klines(klines_window)
    duanqi_bodong = volatility(closes, VOL_WINDOW)
    if duanqi_bodong < VOL_MIN:
        return price, "波动率过低", 0, 0, jiaquan_junjia, None, None, None

    zhicheng, yali = advanced_support_resistance(klines_window)

    tr = 0.5
    da_mairu = False
    da_maichu = False
    imb = 0
    prof = {"bids_5": 1, "asks_5": 1, "top_bid_ratio": 0.2, "top_ask_ratio": 0.2}
    duanqi_duo, duanqi_kong = trend_strength(closes)
    jia_tupo = fake_breakout(closes, shanggui, xiagui)

    duo_defen = 0
    kong_defen = 0

    if price <= xiagui + 5:
        duo_defen += 1
    if tr > 0.57:
        duo_defen += 1
    if imb < -IMB_THRESH:
        duo_defen += 1
    if prof["bids_5"] > prof["asks_5"] * 1.3:
        duo_defen += 1
    if prof["top_bid_ratio"] > 0.35:
        duo_defen += 1
    if duanqi_duo > duanqi_kong:
        duo_defen += 1
    if da_mairu:
        duo_defen += 1
    if price < jiaquan_junjia and (jiaquan_junjia - price) < VWAP_NEAR:
        duo_defen += 1
    if jia_tupo == "fake_down":
        duo_defen += 1
    if zhicheng and abs(price - zhicheng) < 15:
        duo_defen += 1

    if price >= shanggui - 5:
        kong_defen += 1
    if tr < 0.43:
        kong_defen += 1
    if imb > IMB_THRESH:
        kong_defen += 1
    if prof["asks_5"] > prof["bids_5"] * 1.3:
        kong_defen += 1
    if prof["top_ask_ratio"] > 0.35:
        kong_defen += 1
    if duanqi_kong > duanqi_duo:
        kong_defen += 1
    if da_maichu:
        kong_defen += 1
    if price > jiaquan_junjia and (price - jiaquan_junjia) < VWAP_NEAR:
        kong_defen += 1
    if jia_tupo == "fake_up":
        kong_defen += 1
    if yali and abs(price - yali) < 15:
        kong_defen += 1

    pianxiang = "观望"
    fangxiang = None
    if duo_defen >= MIN_SCORE and duo_defen > kong_defen:
        pianxiang = "偏多"
        fangxiang = "long"
    elif kong_defen >= MIN_SCORE and kong_defen > duo_defen:
        pianxiang = "偏空"
        fangxiang = "short"

    signal_record = None
    if fangxiang:
        signal_record = {
            "price": price,
            "direction": fangxiang,
            "long_score": duo_defen,
            "short_score": kong_defen,
            "vwap": jiaquan_junjia,
            "support": zhicheng,
            "resistance": yali,
        }

    return price, pianxiang, duo_defen, kong_defen, jiaquan_junjia, zhicheng, yali, signal_record


def backtest_and_plot():
    print("开始回测（约 1000 根 K 线）...")
    klines = get_futures_klines("1m", 1000)
    closes = [float(k[4]) for k in klines]
    report = backtester.run(klines, strategy_engine)
    print(
        "回测报告:",
        f"交易次数 {report['trades']},",
        f"胜率 {report['win_rate'] * 100:.1f}%,",
        f"最大回撤 {report['max_drawdown'] * 100:.2f}%,",
        f"夏普 {report['sharpe']:.2f},",
        f"权益 {report['equity']:.2f}",
    )

    good_idx = []
    bad_idx = []
    good_price = []
    bad_price = []

    for i in range(60, len(closes) - 15):
        window_klines = klines[:i]
        (
            price,
            pianxiang,
            duo_defen,
            kong_defen,
            jiaquan_junjia,
            zhicheng,
            yali,
            signal_record,
        ) = compute_signal_on_window(window_klines)
        if not signal_record:
            continue
        future = closes[i : i + 10]
        label = auto_label_signal(price, signal_record["direction"], future)
        if label == "good":
            good_idx.append(i)
            good_price.append(price)
        elif label == "bad":
            bad_idx.append(i)
            bad_price.append(price)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(closes, color="white")
    plt.scatter(good_idx, good_price, color="lime", marker="^", s=60, label="真实信号")
    plt.scatter(bad_idx, bad_price, color="red", marker="v", s=60, label="假信号")
    plt.legend()
    plt.title("回测：真实/假信号标记")
    plt.grid(True)
    plt.show()


def auto_pipeline_thread():
    best, report = run_auto_pipeline()
    if best:
        print("自动调参完成，最佳参数:", best)
    print("自动回测报告:", report)


def plot_equity_curve(trades):
    if not trades:
        return
    returns = pd.Series(trades)
    equity = (1 + returns).cumprod()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(equity, color="cyan")
    plt.title("收益曲线")
    plt.grid(True)
    plt.show()


def plot_param_radar(params):
    labels = list(params.keys())
    values = list(params.values())
    if not labels:
        return
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, color="orange")
    ax.fill(angles, values, color="orange", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("参数雷达图")
    plt.show()


# ==============================
# UI 终端 + 信号详情面板 + 模拟仓 + 事件驱动
# ==============================
class TradingTerminal:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BTC 永续中线均衡量化终端（GPU 胜率 + 多策略投票 + 宏观因子）")
        self.root.geometry("1320x780")

        top = tk.Frame(self.root)
        top.pack(fill="both", expand=True)

        left_top = tk.Frame(top)
        mid_top = tk.Frame(top)
        right_top = tk.Frame(top)

        left_top.pack(side="left", fill="both", expand=True)
        mid_top.pack(side="left", fill="both", expand=True)
        right_top.pack(side="right", fill="both", expand=True)

        self.fig_k = Figure(figsize=(5.8, 4.0), dpi=110)
        self.ax_price = self.fig_k.add_subplot(311)
        self.ax_macd = self.fig_k.add_subplot(312)
        self.ax_rsi = self.fig_k.add_subplot(313)
        self.canvas_k = FigureCanvasTkAgg(self.fig_k, master=left_top)
        self.canvas_k.get_tk_widget().pack(fill="both", expand=True)

        interval_frame = tk.Frame(left_top)
        interval_frame.pack(fill="x")
        tk.Label(interval_frame, text="K线周期：").pack(side="left")
        self.k_interval_var = tk.StringVar(value="15m")
        self.k_interval_box = ttk.Combobox(
            interval_frame,
            textvariable=self.k_interval_var,
            values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            width=6,
        )
        self.k_interval_box.pack(side="left")
        self.k_interval_box.bind("<<ComboboxSelected>>", lambda e: None)

        self.fig_d = Figure(figsize=(4.2, 3.2), dpi=110)
        self.ax_d = self.fig_d.add_subplot(111)
        self.canvas_d = FigureCanvasTkAgg(self.fig_d, master=mid_top)
        self.canvas_d.get_tk_widget().pack(fill="both", expand=True)

        detail_frame = tk.Frame(right_top)
        detail_frame.pack(fill="both", expand=True)

        tk.Label(detail_frame, text="信号详情面板", font=("Arial", 14, "bold")).pack(pady=5)

        self.text_detail = tk.Text(detail_frame, height=25, width=50)
        self.text_detail.pack(fill="both", expand=True)

        mid = tk.Frame(self.root)
        mid.pack(fill="x")

        self.label_price = tk.Label(mid, text="价格: --", font=("Arial", 16))
        self.label_price.pack()

        self.label_bias = tk.Label(mid, text="偏向: --", font=("Arial", 14))
        self.label_bias.pack()

        self.label_scores = tk.Label(mid, text="多头得分: -- | 空头得分: --", font=("Arial", 12))
        self.label_scores.pack()

        self.label_tp_sl = tk.Label(mid, text="方向: -- | 止盈: -- | 止损: --", font=("Arial", 12))
        self.label_tp_sl.pack()

        self.label_regime = tk.Label(mid, text="行情状态: -- | 四小时波动范围: --", font=("Arial", 12))
        self.label_regime.pack()

        self.label_trend_warn = tk.Label(mid, text="趋势预警: --", font=("Arial", 12), fg="orange")
        self.label_trend_warn.pack()

        sim_frame = tk.Frame(mid)
        sim_frame.pack(fill="x", pady=5)

        self.label_sim_balance = tk.Label(sim_frame, text="模拟资金: 100000.00 USDT", font=("Arial", 12))
        self.label_sim_balance.pack()

        self.label_sim_position = tk.Label(sim_frame, text="模拟仓位: 无持仓", font=("Arial", 12))
        self.label_sim_position.pack()

        self.label_sim_pnl = tk.Label(sim_frame, text="浮动盈亏: 0.00 USDT", font=("Arial", 12))
        self.label_sim_pnl.pack()

        bottom = tk.Frame(self.root)
        bottom.pack(fill="both", expand=True)

        self.text_log = tk.Text(bottom, height=10)
        self.text_log.pack(side="left", fill="both", expand=True)

        btn_frame = tk.Frame(bottom)
        btn_frame.pack(side="right", fill="y")

        self.btn_backtest = tk.Button(btn_frame, text="运行回测并画图", command=self.run_backtest_thread)
        self.btn_backtest.pack(pady=10)
        self.btn_auto_pipeline = tk.Button(btn_frame, text="自动训练/回测/调参", command=self.run_auto_pipeline_thread)
        self.btn_auto_pipeline.pack(pady=10)
        self.btn_load_params = tk.Button(btn_frame, text="加载最优参数", command=self.load_best_params)
        self.btn_load_params.pack(pady=10)
        self.btn_show_metrics = tk.Button(btn_frame, text="显示高级指标", command=self.show_advanced_metrics)
        self.btn_show_metrics.pack(pady=10)
        self.btn_refresh = tk.Button(btn_frame, text="手动刷新数据", command=self.refresh_once)
        self.btn_refresh.pack(pady=10)

        self.last_signal = None
        self.last_signal_index = None
        self.last_extra = None

        self.sim_balance = 100000.0
        self.sim_position = 0.0
        self.sim_entry_price = None
        self.sim_trade_size = 0.02
        self.sim_realized_pnl = 0.0

        self.last_trend_dir = None

        threading.Thread(target=self.update_loop, daemon=True).start()

        self.root.mainloop()

    def sim_update_pnl_label(self, last_price):
        if self.sim_position == 0 or self.sim_entry_price is None:
            float_pnl = 0.0
        else:
            if self.sim_position > 0:
                float_pnl = (last_price - self.sim_entry_price) * abs(self.sim_position)
            else:
                float_pnl = (self.sim_entry_price - last_price) * abs(self.sim_position)
        total_equity = self.sim_balance + float_pnl
        self.label_sim_balance.config(text=f"模拟资金: {self.sim_balance:.2f} USDT（已实现）")
        if self.sim_position == 0:
            self.label_sim_position.config(text="模拟仓位: 无持仓")
        else:
            direction_cn = "多仓" if self.sim_position > 0 else "空仓"
            self.label_sim_position.config(
                text=f"模拟仓位: {direction_cn} {abs(self.sim_position):.4f} BTC @ {self.sim_entry_price:.2f}"
            )
        self.label_sim_pnl.config(text=f"浮动盈亏: {float_pnl:.2f} USDT | 总权益: {total_equity:.2f} USDT")

    def sim_open_on_signal(self, signal_record, last_price):
        fangxiang = signal_record["direction"]
        if self.sim_position != 0:
            return
        size = self.sim_trade_size
        if fangxiang == "long":
            self.sim_position = size
        else:
            self.sim_position = -size
        self.sim_entry_price = last_price
        self.text_log.insert("end", f"模拟开仓：{'做多' if fangxiang == 'long' else '做空'} {size:.4f} BTC @ {last_price:.2f}\n")
        self.text_log.see("end")

    def sim_close_position(self, reason, last_price):
        if self.sim_position == 0 or self.sim_entry_price is None:
            return
        if self.sim_position > 0:
            pnl = (last_price - self.sim_entry_price) * abs(self.sim_position)
        else:
            pnl = (self.sim_entry_price - last_price) * abs(self.sim_position)
        self.sim_balance += pnl
        self.sim_realized_pnl += pnl
        self.text_log.insert("end", f"模拟平仓（{reason}）：盈亏 {pnl:.2f} USDT\n")
        self.text_log.see("end")
        self.sim_position = 0.0
        self.sim_entry_price = None

    def run_backtest_thread(self):
        threading.Thread(target=backtest_and_plot, daemon=True).start()

    def run_auto_pipeline_thread(self):
        threading.Thread(target=auto_pipeline_thread, daemon=True).start()

    def load_best_params(self):
        params = load_best_params()
        CURRENT_PARAMS.update(params)
        global backtester
        backtester = Backtester(
            fee_rate=CURRENT_PARAMS["fee_rate"],
            slippage=CURRENT_PARAMS["slippage"],
            leverage=CURRENT_PARAMS["leverage"],
            funding_rate=CURRENT_PARAMS["funding_rate"],
        )
        plot_param_radar(CURRENT_PARAMS)
        self.text_log.insert("end", f"已加载最优参数: {CURRENT_PARAMS}\n")
        self.text_log.see("end")

    def update_loop(self):
        while True:
            try:
                self.update_kline()
                self.update_depth()
                self.update_signal_event_driven()
            except Exception as e:
                self.text_log.insert("end", f"错误: {e}\n")
            time.sleep(5)

    def refresh_once(self):
        try:
            self.update_kline()
            self.update_depth()
            self.update_signal_event_driven()
            self.text_log.insert("end", "手动刷新完成。\n")
            self.text_log.see("end")
        except Exception as e:
            self.text_log.insert("end", f"手动刷新失败: {e}\n")
            self.text_log.see("end")

    def update_kline(self):
        interval = self.k_interval_var.get()
        klines = get_futures_klines(interval, 200)
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]

        self.fig_k.clear()
        self.ax_price = self.fig_k.add_subplot(311)
        self.ax_macd = self.fig_k.add_subplot(312)
        self.ax_rsi = self.fig_k.add_subplot(313)

        jiaquan_junjia = vwap_from_klines(klines)
        self.ax_price.plot(closes, color="#e8e8e8", linewidth=1.4, label="收盘价")
        self.ax_price.axhline(jiaquan_junjia, color="#f7d060", linestyle="--", linewidth=1.0, label="加权均价")
        self.ax_price.fill_between(range(len(highs)), lows, highs, color="#4b4b4b", alpha=0.25)

        if self.last_signal_index is not None and self.last_signal_index < len(closes) and self.last_signal:
            idx = self.last_signal_index
            sig = self.last_signal
            if sig["direction"] == "long":
                self.ax_price.scatter(idx, closes[idx], color="lime", marker="^", s=80)
            else:
                self.ax_price.scatter(idx, closes[idx], color="red", marker="v", s=80)
            if sig.get("tp") is not None:
                self.ax_price.axhline(sig["tp"], color="green", linestyle="--", alpha=0.7)
            if sig.get("sl") is not None:
                self.ax_price.axhline(sig["sl"], color="red", linestyle="--", alpha=0.7)
        self.ax_price.set_facecolor("#0f1116")
        self.ax_price.tick_params(colors="#c7c7c7")
        self.ax_price.grid(color="#2a2f3a", linestyle="--", linewidth=0.6, alpha=0.6)
        self.ax_price.legend(facecolor="#0f1116", edgecolor="#2a2f3a", labelcolor="#e8e8e8")

        macd, sig_line, hist = calc_macd(closes)
        colors = ["green" if h >= 0 else "red" for h in hist]
        self.ax_macd.bar(range(len(hist)), hist, color=colors, alpha=0.7)
        self.ax_macd.plot(macd, color="#e8e8e8", linewidth=1.2, label="MACD")
        self.ax_macd.plot(sig_line, color="#f7d060", linewidth=1.0, label="信号线")
        self.ax_macd.set_facecolor("#0f1116")
        self.ax_macd.tick_params(colors="#c7c7c7")
        self.ax_macd.grid(color="#2a2f3a", linestyle="--", linewidth=0.6, alpha=0.6)
        self.ax_macd.legend(facecolor="#0f1116", edgecolor="#2a2f3a", labelcolor="#e8e8e8")

        rsi = calc_rsi(closes)
        self.ax_rsi.plot(rsi, color="#4fc3f7", linewidth=1.2, label="相对强弱指标")
        self.ax_rsi.axhline(70, color="#ef5350", linestyle="--", linewidth=0.9)
        self.ax_rsi.axhline(30, color="#66bb6a", linestyle="--", linewidth=0.9)
        self.ax_rsi.set_facecolor("#0f1116")
        self.ax_rsi.tick_params(colors="#c7c7c7")
        self.ax_rsi.grid(color="#2a2f3a", linestyle="--", linewidth=0.6, alpha=0.6)
        self.ax_rsi.legend(facecolor="#0f1116", edgecolor="#2a2f3a", labelcolor="#e8e8e8")

        self.fig_k.tight_layout()
        self.canvas_k.draw()

        if closes:
            self.sim_update_pnl_label(closes[-1])

    def update_depth(self):
        try:
            depth = get_futures_depth(50)
            bids = depth.get("bids", [])
            asks = depth.get("asks", [])
        except Exception:
            bids = []
            asks = []

        self.ax_d.clear()
        if not bids or not asks:
            self.ax_d.set_title("深度图（暂无数据）", color="#e8e8e8")
            self.ax_d.set_facecolor("#0f1116")
            self.ax_d.tick_params(colors="#c7c7c7")
            self.ax_d.grid(color="#2a2f3a", linestyle="--", linewidth=0.6, alpha=0.6)
            self.canvas_d.draw()
            return

        bid_prices = [float(b[0]) for b in bids]
        bid_amount = np.cumsum([float(b[1]) for b in bids])

        ask_prices = [float(a[0]) for a in asks]
        ask_amount = np.cumsum([float(a[1]) for a in asks])
        total_bid = sum(float(b[1]) for b in bids) if bids else 0.0
        total_ask = sum(float(a[1]) for a in asks) if asks else 0.0

        if bid_prices and bid_amount.size:
            self.ax_d.plot(bid_prices, bid_amount, color="#66bb6a", linewidth=1.4, label="买盘累积")
        if ask_prices and ask_amount.size:
            self.ax_d.plot(ask_prices, ask_amount, color="#ef5350", linewidth=1.4, label="卖盘累积")
        self.ax_d.set_title("深度图（买卖盘累积量）", color="#e8e8e8")
        self.ax_d.set_facecolor("#0f1116")
        self.ax_d.tick_params(colors="#c7c7c7")
        self.ax_d.grid(color="#2a2f3a", linestyle="--", linewidth=0.6, alpha=0.6)
        self.ax_d.legend(facecolor="#0f1116", edgecolor="#2a2f3a", labelcolor="#e8e8e8")
        ax_bar = self.ax_d.twinx()
        ax_bar.bar(["买单总量", "卖单总量"], [total_bid, total_ask], color=["#66bb6a", "#ef5350"], alpha=0.25)
        ax_bar.set_ylabel("订单对比", color="#c7c7c7")
        ax_bar.tick_params(colors="#c7c7c7")
        self.canvas_d.draw()

    def update_signal_event_driven(self):
        k4h = get_futures_klines("4h", 5)
        closes = [float(k[4]) for k in k4h]
        if not closes:
            return
        price = closes[-1]

        if self.sim_position != 0 and self.last_signal is not None:
            sig = self.last_signal
            if self.sim_position > 0:
                if price >= sig["tp"]:
                    self.sim_close_position("止盈", price)
                elif price <= sig["sl"]:
                    self.sim_close_position("止损", price)
            else:
                if price <= sig["tp"]:
                    self.sim_close_position("止盈", price)
                elif price >= sig["sl"]:
                    self.sim_close_position("止损", price)

        self.update_trend_warning()

        has_position = self.sim_position != 0

        (
            price,
            pianxiang,
            duo_defen,
            kong_defen,
            jiaquan_junjia,
            zhicheng,
            yali,
            signal_record,
            extra,
        ) = compute_signal_realtime_core(
            {
                "has_position": has_position,
                "last_direction": "long"
                if self.sim_position > 0
                else ("short" if self.sim_position < 0 else None),
            }
        )

        self.label_price.config(text=f"价格: {price:.2f}")
        self.label_bias.config(text=f"偏向: {pianxiang}")
        self.label_scores.config(text=f"多头得分: {duo_defen} | 空头得分: {kong_defen}")

        if extra:
            self.label_regime.config(text=f"行情状态: {extra['market_regime']} | 四小时波动范围: {extra['atr_4h']:.2f}")
            self.last_extra = extra
        if self.last_signal and self.last_signal.get("label") is None:
            future_klines = get_futures_klines("4h", 30)
            future_closes = [float(k[4]) for k in future_klines]
            label = auto_label_signal(self.last_signal["price"], self.last_signal["direction"], future_closes)
            self.last_signal["label"] = label
            log_signal(self.last_signal)
            self.text_log.insert("end", f"自动标注上一信号：{label}\n")
            self.text_log.see("end")

        if not has_position and signal_record:
            self.last_signal = signal_record
            self.last_signal_index = -1
            fangxiang_cn = "做多" if signal_record["direction"] == "long" else "做空"
            zhi_ying = signal_record["tp"]
            zhi_sun = signal_record["sl"]
            self.label_tp_sl.config(text=f"方向: {fangxiang_cn} | 止盈: {zhi_ying:.2f} | 止损: {zhi_sun:.2f}")

            msg = (
                f"新信号：{fangxiang_cn} @ {price:.2f} | "
                f"止盈 {zhi_ying:.2f} | 止损 {zhi_sun:.2f} | "
                f"预期胜率 {signal_record['expected_win'] * 100:.1f}% | "
                f"行情状态 {signal_record['market_regime']}"
            )
            self.text_log.insert("end", msg + "\n")
            self.text_log.see("end")
            play_ding()
            desktop_notify("交易信号", msg)

            self.sim_open_on_signal(signal_record, price)
            self.update_signal_detail_panel(signal_record)

    def update_trend_warning(self):
        k4h = get_futures_klines("4h", 100)
        closes = [float(k[4]) for k in k4h]
        if not closes:
            return
        duo_qiangdu, kong_qiangdu = trend_strength(closes)
        current_dir = None
        if duo_qiangdu > 0.66 and duo_qiangdu > kong_qiangdu:
            current_dir = "long"
        elif kong_qiangdu > 0.66 and kong_qiangdu > duo_qiangdu:
            current_dir = "short"

        warn_text = "趋势预警: 无明显信号"

        if self.last_trend_dir is None:
            self.last_trend_dir = current_dir
        else:
            if self.last_trend_dir == "long" and current_dir == "short":
                warn_text = "趋势预警: 多头趋势可能反转为空头"
                self.text_log.insert("end", "趋势预警：多头 → 空头 反转迹象\n")
                self.text_log.see("end")
            elif self.last_trend_dir == "short" and current_dir == "long":
                warn_text = "趋势预警: 空头趋势可能反转为多头"
                self.text_log.insert("end", "趋势预警：空头 → 多头 反转迹象\n")
                self.text_log.see("end")
            self.last_trend_dir = current_dir

        self.label_trend_warn.config(text=warn_text)

    def update_signal_detail_panel(self, sig):
        self.text_detail.delete("1.0", "end")
        fangxiang_cn = "做多" if sig["direction"] == "long" else "做空"

        lines = []
        lines.append(f"方向: {fangxiang_cn}")
        lines.append(f"价格: {sig['price']:.2f}")
        lines.append(f"止盈: {sig['tp']:.2f}")
        lines.append(f"止损: {sig['sl']:.2f}")
        rr = abs(sig["tp"] - sig["price"]) / max(1e-6, abs(sig["price"] - sig["sl"]))
        lines.append(f"盈亏比: {rr:.2f}")
        lines.append(f"预期胜率: {sig['expected_win'] * 100:.1f}%")
        lines.append(f"行情状态: {sig['market_regime']} | 四小时波动范围: {sig['atr_4h']:.2f}")
        lines.append("")
        lines.append(f"多头得分: {sig['long_score']} | 空头得分: {sig['short_score']}")
        lines.append(f"加权均价: {sig['vwap']:.2f}")
        if sig["support"]:
            lines.append(f"支撑位: {sig['support']:.2f}")
        else:
            lines.append("支撑位: 无")
        if sig["resistance"]:
            lines.append(f"压力位: {sig['resistance']:.2f}")
        else:
            lines.append("压力位: 无")
        lines.append("")
        lines.append(f"Taker 占比: {sig['taker_ratio']:.3f}")
        lines.append(f"深度不对称: {sig['imbalance']:.3f}")
        dp = sig["depth_profile"]
        lines.append(f"买盘量: {dp['maipan_liang']:.1f}")
        lines.append(f"卖盘量: {dp['maichu_liang']:.1f}")
        lines.append(f"买卖价差: {dp['jiacha']:.2f}")
        maiqiang_cn = "是" if dp["mai_qiang"] else "否"
        jia_maiqiang_cn = "是" if dp["jia_mai_qiang"] else "否"
        lines.append(f"卖墙: {maiqiang_cn} | 假卖墙: {jia_maiqiang_cn}")
        lines.append("")
        lines.append(f"4小时趋势强度: 多 {sig['trend_long']} / 空 {sig['trend_short']}")
        lines.append(f"4小时趋势评分: 多 {sig['mt_long']:.2f} / 空 {sig['mt_short']:.2f}")
        lines.append(f"合约-现货价差: {sig['basis']:.2f}")
        market_data = sig.get("market_data")
        if market_data:
            lines.append(
                "市场数据: "
                f"开 {market_data['open']:.2f} "
                f"高 {market_data['high']:.2f} "
                f"低 {market_data['low']:.2f} "
                f"收 {market_data['close']:.2f} "
                f"量 {market_data['volume']:.2f}"
            )
        fb = sig["fake_break"]
        if fb == "fake_up":
            fb_cn = "假向上突破"
        elif fb == "fake_down":
            fb_cn = "假向下突破"
        else:
            fb_cn = "无"
        lines.append(f"假突破结构: {fb_cn}")
        lines.append("")
        lines.append(f"多头爆仓压力: {sig.get('long_liq', 0):.1f}")
        lines.append(f"空头爆仓压力: {sig.get('short_liq', 0):.1f}")
        lines.append("")
        extra = sig.get("extra") or {}
        if extra:
            lines.append("AI 因子评估:")
            lines.append(f"自动因子挖掘评分: {extra.get('feature_mining_score', 0):.2f}")
            lines.append(f"情绪因子评分: {extra.get('sentiment_factor', 0):.2f}")
            lines.append(f"策略自适应评分: {extra.get('strategy_adaptivity', 0):.2f}")
            tf_trends = extra.get("tf_trends")
            if tf_trends:
                tf_desc = " | ".join(f"{k}:{'多' if v == 1 else ('空' if v == -1 else '平')}" for k, v in tf_trends.items())
                lines.append(f"多周期趋势: {tf_desc}")
        self.text_detail.insert("end", "\n".join(lines))

    def show_advanced_metrics(self):
        extra = self.last_extra or {}
        window = tk.Toplevel(self.root)
        window.title("高级指标面板")
        window.geometry("600x600")
        text = tk.Text(window)
        text.pack(fill="both", expand=True)
        lines = []
        lines.append(f"风险偏好得分: {extra.get('risk_on', 0):.2f}")
        lines.append(f"避险得分: {extra.get('risk_off', 0):.2f}")
        lines.append(f"新闻情绪(宏观): {extra.get('news_sentiment', 0):.2f}")
        lines.append(f"市场情绪: {extra.get('sentiment_score', 0):.2f}")
        lines.append(f"Twitter 情绪: {extra.get('twitter_sentiment', 0):.2f}")
        lines.append(f"新闻情绪(媒体): {extra.get('news_sentiment_extra', 0):.2f}")
        lines.append(f"宏观因子: {extra.get('macro_score', 0):.2f}")
        lines.append(f"自动因子挖掘评分: {extra.get('feature_mining_score', 0):.2f}")
        lines.append(f"情绪因子评分: {extra.get('sentiment_factor', 0):.2f}")
        lines.append(f"策略自适应评分: {extra.get('strategy_adaptivity', 0):.2f}")
        tf_trends = extra.get("tf_trends")
        if tf_trends:
            tf_desc = " | ".join(f"{k}:{'多' if v == 1 else ('空' if v == -1 else '平')}" for k, v in tf_trends.items())
            lines.append(f"多周期趋势: {tf_desc}")
        lines.append(f"AI 预测价格: {extra.get('ml_pred_price', 0):.2f}")
        lines.append(f"Transformer 预测价格: {extra.get('transformer_pred_price', 0):.2f}")
        lines.append(f"集成预测价格: {extra.get('ensemble_pred_price', 0):.2f}")
        ai_sig = extra.get("ai_signal") or {}
        ai_dir = "观望" if ai_sig.get("direction") is None else ("做多" if ai_sig.get("direction") == "long" else "做空")
        lines.append(f"AI 多因子方向: {ai_dir} | 得分: {ai_sig.get('score', 0)}")
        multi_score = extra.get("multi_factor_score") or {}
        multi_dir = "观望" if multi_score.get("direction") is None else ("做多" if multi_score.get("direction") == "long" else "做空")
        lines.append(
            f"多因子评分方向: {multi_dir} | 多头 {multi_score.get('long_score', 0)} | 空头 {multi_score.get('short_score', 0)}"
        )
        fused_ai = extra.get("fused_ai") or {}
        fused_dir = "观望" if fused_ai.get("direction") is None else ("做多" if fused_ai.get("direction") == "long" else "做空")
        lines.append(f"AI 综合方向: {fused_dir} | 票数 {fused_ai.get('score', 0)} | 投票 {fused_ai.get('votes')}")
        text.insert("end", "\n".join(lines))


# ==============================
# 启动
# ==============================
if __name__ == "__main__":
    TradingTerminal()
