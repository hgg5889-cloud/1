import time
import warnings
from datetime import datetime
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    Button,
    Entry,
    Frame,
    Label,
    Listbox,
    StringVar,
    Text,
    Tk,
    ttk,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from tensorflow.keras.layers import Dense, GRU, Input, LSTM
from tensorflow.keras.models import Sequential

# 忽略 tkinter 字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

# 设置API
BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_FUTURES_BASE = "https://fapi.binance.com/fapi/v1"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"  # 4小时数据
DEFAULT_REFRESH_SECONDS = 5
PLOT_POINTS = 200
SUPPORTED_INTERVALS = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "12h",
    "1d",
]
MODEL_OPTIONS = ["LSTM", "GRU"]
FLOW_INTERVAL_OPTIONS = ["15m", "30m", "1h", "2h", "4h", "1d"]
SR_VERSION_OPTIONS = ["版本1", "版本2", "版本3", "版本4"]
DEFAULT_SR_VERSION = "版本2"
STRATEGY_RISK_CONFIG = {
    "aggressive": {"risk_pct": 0.02, "rr": 2.0},
    "balanced": {"risk_pct": 0.01, "rr": 1.8},
    "conservative": {"risk_pct": 0.005, "rr": 1.5},
}


# 获取实时价格数据
def fetch_klines(symbol, interval="4h", limit=500):
    url = f"{BINANCE_BASE}/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 检查API请求是否成功
        data = response.json()

        if not data:
            print("获取数据失败或API返回空数据。")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.columns = [
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

        # 确保所有需要计算的数据列都是浮动类型
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")

        return df

    except requests.exceptions.RequestException as error:
        print(f"API 请求失败: {error}")
        return pd.DataFrame()  # 返回空数据框


# 获取外部经济数据
def fetch_external_data(max_retries=3, backoff_seconds=2):
    for attempt in range(1, max_retries + 1):
        try:
            tickers = ["^GSPC", "DX-Y.NYB", "GC=F", "^VIX", "^TNX"]
            data = yf.download(
                tickers,
                start="2022-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                group_by="ticker",
                progress=False,
            )
            return data
        except Exception as error:
            print(f"获取外部数据失败(第{attempt}次): {error}")
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    return pd.DataFrame()


# 获取市场深度数据
def get_order_book(symbol="BTCUSDT", depth_limit=1000):
    url = f"{BINANCE_BASE}/depth?symbol={symbol}&limit={depth_limit}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 检查API请求是否成功
        data = response.json()

        buy_depth = data.get("bids", [])
        sell_depth = data.get("asks", [])

        return buy_depth, sell_depth
    except requests.exceptions.RequestException as error:
        print(f"API 请求失败: {error}")
        return [], []  # 如果请求失败，返回空列表


def compute_order_book_signal(buy_depth, sell_depth, mid_price=None):
    if not buy_depth or not sell_depth:
        return None
    try:
        bid_prices = np.array([float(item[0]) for item in buy_depth])
        bid_qty = np.array([float(item[1]) for item in buy_depth])
        ask_prices = np.array([float(item[0]) for item in sell_depth])
        ask_qty = np.array([float(item[1]) for item in sell_depth])
    except (ValueError, TypeError):
        return None

    best_bid = np.max(bid_prices)
    best_ask = np.min(ask_prices)
    mid = (best_bid + best_ask) / 2
    ref_mid = mid_price or mid

    bid_value = np.sum(bid_prices * bid_qty)
    ask_value = np.sum(ask_prices * ask_qty)
    total_value = bid_value + ask_value
    if total_value <= 0:
        return None

    bid_vwap = bid_value / max(np.sum(bid_qty), 1e-9)
    ask_vwap = ask_value / max(np.sum(ask_qty), 1e-9)
    weighted_mid = (bid_vwap + ask_vwap) / 2

    distance_bid = np.abs(bid_prices - ref_mid) / max(ref_mid, 1e-9)
    distance_ask = np.abs(ask_prices - ref_mid) / max(ref_mid, 1e-9)
    bid_weights = 1 / (1 + distance_bid * 100)
    ask_weights = 1 / (1 + distance_ask * 100)
    bid_pressure = np.sum(bid_qty * bid_weights)
    ask_pressure = np.sum(ask_qty * ask_weights)
    pressure_total = bid_pressure + ask_pressure
    if pressure_total <= 0:
        pressure = 0.0
    else:
        pressure = (bid_pressure - ask_pressure) / pressure_total

    imbalance = (bid_value - ask_value) / total_value
    top_n = min(5, len(bid_qty), len(ask_qty))
    top_bid_qty = np.sum(bid_qty[:top_n])
    top_ask_qty = np.sum(ask_qty[:top_n])
    micro_price = (best_ask * top_bid_qty + best_bid * top_ask_qty) / max(
        top_bid_qty + top_ask_qty, 1e-9
    )

    return {
        "imbalance": imbalance,
        "weighted_mid": weighted_mid,
        "micro_price": micro_price,
        "pressure": pressure,
    }


def compute_macro_bias(external_df):
    if external_df.empty:
        return 0.0

    def latest_pct_change(ticker):
        if isinstance(external_df.columns, pd.MultiIndex):
            series = external_df.get(ticker)
            if series is None:
                return None
            close = series["Close"].dropna()
        else:
            if "Close" not in external_df.columns:
                return None
            close = external_df["Close"].dropna()
        if len(close) < 2:
            return None
        return (close.iloc[-1] / close.iloc[-2] - 1) * 100

    sp500 = latest_pct_change("^GSPC")
    dxy = latest_pct_change("DX-Y.NYB")
    gold = latest_pct_change("GC=F")
    vix = latest_pct_change("^VIX")
    us10y = latest_pct_change("^TNX")

    if sp500 is None or dxy is None:
        base_bias = 0.0
    elif sp500 > 0 and dxy < 0:
        base_bias = 0.2
    elif sp500 < 0 and dxy > 0:
        base_bias = -0.2
    else:
        base_bias = 0.0

    extra_bias = 0.0
    if gold is not None:
        extra_bias += gold * 0.01
    if vix is not None:
        extra_bias -= vix * 0.02
    if us10y is not None:
        extra_bias -= us10y * 0.01

    return base_bias + extra_bias


def compute_feature_forecast(df, depth_signal=None, macro_bias=0.0, window=200):
    if df.empty or len(df) < 60:
        return None

    data = df.tail(window).copy()
    data["return"] = data["close"].pct_change()
    data["rsi"] = RSIIndicator(data["close"], window=14).rsi()
    macd = MACD(data["close"])
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    bands = BollingerBands(data["close"], window=20, window_dev=2)
    band_width = (bands.bollinger_hband() - bands.bollinger_lband()).replace(0, np.nan)
    data["band_pos"] = (data["close"] - bands.bollinger_lband()) / band_width
    data["mfi"] = MFIIndicator(
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        window=14,
    ).money_flow_index()
    obv = OnBalanceVolumeIndicator(close=data["close"], volume=data["volume"]).on_balance_volume()
    data["obv_delta"] = obv.diff()
    data["vol_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
    data["momentum_5"] = data["close"].pct_change(5)
    data["momentum_10"] = data["close"].pct_change(10)

    external_bias = 0.0
    for column in ("sp500_close", "dxy_close", "gold_close", "vix_close", "us10y_close"):
        if column in data.columns:
            pct = data[column].pct_change()
            if column == "sp500_close":
                external_bias += pct * 0.03
            elif column == "dxy_close":
                external_bias -= pct * 0.03
            elif column == "gold_close":
                external_bias += pct * 0.01
            elif column == "vix_close":
                external_bias -= pct * 0.02
            elif column == "us10y_close":
                external_bias -= pct * 0.01
    data["external_bias"] = external_bias
    data["macro_bias"] = macro_bias
    data["depth_pressure"] = depth_signal["pressure"] if depth_signal else 0.0
    data["depth_imbalance"] = depth_signal["imbalance"] if depth_signal else 0.0

    feature_columns = [
        "rsi",
        "macd",
        "macd_signal",
        "band_pos",
        "mfi",
        "obv_delta",
        "vol_ratio",
        "momentum_5",
        "momentum_10",
        "external_bias",
        "macro_bias",
        "depth_pressure",
        "depth_imbalance",
    ]
    data["target"] = data["return"].shift(-1)
    model_data = data[feature_columns + ["target"]].dropna()
    if len(model_data) < 50:
        return None

    X = model_data[feature_columns].values
    y = model_data["target"].values
    model = LinearRegression()
    model.fit(X, y)
    latest_features = data[feature_columns].iloc[-1].values.reshape(1, -1)
    predicted_return = float(model.predict(latest_features)[0])
    volatility = data["return"].rolling(20).std().iloc[-1]
    if pd.notna(volatility) and volatility > 0:
        cap = volatility * 2.5
        predicted_return = float(np.clip(predicted_return, -cap, cap))
    return predicted_return


def compute_prediction_metrics(
    df,
    price,
    predicted_price_raw,
    rsi,
    macd_line,
    signal,
    short_ema,
    long_ema,
    upper_band=None,
    lower_band=None,
    support=None,
    resistance=None,
    depth_signal=None,
    macro_bias=0.0,
):
    base_prediction = predicted_price_raw
    depth_bias = 0.0
    if depth_signal:
        base_prediction = (
            0.55 * predicted_price_raw
            + 0.25 * depth_signal["weighted_mid"]
            + 0.20 * depth_signal["micro_price"]
        )
        depth_bias = 0.6 * depth_signal["pressure"] + 0.4 * depth_signal["imbalance"]

    feature_return = compute_feature_forecast(
        df,
        depth_signal=depth_signal,
        macro_bias=macro_bias,
    )
    if feature_return is not None:
        base_prediction = base_prediction * 0.7 + price * (1 + feature_return) * 0.3

    ema_trend = (short_ema - long_ema) / price if price else 0.0
    macd_trend = (macd_line - signal) / price if price else 0.0
    rsi_bias = (rsi - 50) / 50
    if len(df) >= 6:
        price_bias = (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6]
    else:
        price_bias = 0.0
    if len(df) >= 20:
        recent_volume = df["volume"].tail(5).mean()
        avg_volume = df["volume"].tail(20).mean()
        volume_bias = (recent_volume / avg_volume - 1) if avg_volume else 0.0
    else:
        volume_bias = 0.0

    flow_bias = 0.0
    if len(df) >= 15:
        mfi = MFIIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            window=14,
        ).money_flow_index()
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
        if len(mfi.dropna()) >= 1:
            flow_bias += (mfi.iloc[-1] - 50) / 50
        if len(obv) >= 2:
            obv_change = obv.iloc[-1] - obv.iloc[-2]
            obv_scale = df["volume"].tail(20).mean() if len(df) >= 20 else df["volume"].mean()
            if obv_scale:
                flow_bias += obv_change / obv_scale * 0.01

    band_bias = 0.0
    if upper_band is not None and lower_band is not None and pd.notna(upper_band) and pd.notna(lower_band):
        band_width = upper_band - lower_band
        if band_width > 0:
            band_bias = ((price - lower_band) / band_width - 0.5) * 2

    range_bias = 0.0
    if support is not None and resistance is not None and pd.notna(support) and pd.notna(resistance):
        range_width = resistance - support
        if range_width > 0:
            range_bias = ((price - support) / range_width - 0.5) * 2

    momentum_bias = 0.0
    if len(df) >= 11:
        roc5 = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if df["close"].iloc[-6] else 0.0
        roc10 = (df["close"].iloc[-1] / df["close"].iloc[-11] - 1) if df["close"].iloc[-11] else 0.0
        momentum_bias = (roc5 + roc10) / 2

    external_bias = 0.0
    for column in ("sp500_close", "dxy_close", "gold_close", "vix_close", "us10y_close"):
        if column in df.columns and len(df[column].dropna()) >= 2:
            series = df[column].dropna()
            pct = (series.iloc[-1] / series.iloc[-2] - 1) * 100
            if column == "sp500_close":
                external_bias += pct * 0.03
            elif column == "dxy_close":
                external_bias -= pct * 0.03
            elif column == "gold_close":
                external_bias += pct * 0.01
            elif column == "vix_close":
                external_bias -= pct * 0.02
            elif column == "us10y_close":
                external_bias -= pct * 0.01

    composite = (
        0.35 * depth_bias
        + 0.2 * ema_trend
        + 0.15 * macd_trend
        + 0.1 * rsi_bias
        + 0.1 * price_bias
        + 0.05 * volume_bias
        + 0.05 * flow_bias
        + 0.05 * band_bias
        + 0.05 * range_bias
        + 0.05 * momentum_bias
        + 0.05 * external_bias
        + 0.1 * macro_bias
    )
    volatility = df["close"].pct_change().tail(20).std() if len(df) >= 20 else 0.0
    vol_scale = 1 - min(0.5, volatility * 8) if volatility else 1.0
    vol_scale = float(np.clip(vol_scale, 0.3, 1.0))
    composite *= vol_scale
    cap = 0.04 + min(0.03, volatility * 5) if volatility else 0.04
    composite = float(np.clip(composite, -cap, cap))
    predicted_price = base_prediction * (1 + composite)

    delta = predicted_price - price
    pct = (delta / price * 100) if price else 0.0
    outlook = "看涨" if delta >= 0 else "看跌"
    direction = "上涨" if delta >= 0 else "下跌"
    prediction_text = f"预测{direction}: {delta:+.2f} ({pct:+.2f}%)"
    return predicted_price, outlook, prediction_text


def fetch_futures_klines(symbol, interval="1h", limit=500):
    url = (
        f"{BINANCE_FUTURES_BASE}/continuousKlines"
        f"?pair={symbol}&contractType=PERPETUAL&interval={interval}&limit={limit}"
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.columns = [
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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        return df
    except requests.exceptions.RequestException as error:
        print(f"期货API 请求失败: {error}")
        return pd.DataFrame()


# 计算RSI、MACD和布林带
def compute_indicators(df):
    rsi = RSIIndicator(df["close"], window=14).rsi().iloc[-1]
    macd_line = MACD(df["close"]).macd().iloc[-1]
    signal = MACD(df["close"]).macd_signal().iloc[-1]
    bollinger = BollingerBands(df["close"], window=20, window_dev=2)
    upper_band = bollinger.bollinger_hband().iloc[-1]
    lower_band = bollinger.bollinger_lband().iloc[-1]
    short_ema = EMAIndicator(df["close"], window=50).ema_indicator().iloc[-1]
    long_ema = EMAIndicator(df["close"], window=200).ema_indicator().iloc[-1]
    return rsi, macd_line, signal, upper_band, lower_band, short_ema, long_ema


def compute_atr(df, window=14):
    if df.empty or len(df) < window + 1:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    latest_atr = atr.iloc[-1]
    return float(latest_atr) if pd.notna(latest_atr) else None


# 数据标准化
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler


# LSTM模型训练
def train_lstm(df, model_type="LSTM"):
    df_close = df["close"].values.reshape(-1, 1)
    data_scaled, scaler = scale_data(df_close)

    if len(data_scaled) <= 60:
        return float(df["close"].iloc[-1])

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i - 60 : i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    if model_type == "GRU":
        model = Sequential(
            [
                Input(shape=(X.shape[1], 1)),
                GRU(units=64, return_sequences=True),
                GRU(units=32, return_sequences=False),
                Dense(units=1),
            ]
        )
    else:
        model = Sequential(
            [
                Input(shape=(X.shape[1], 1)),
                LSTM(units=50, return_sequences=True),
                LSTM(units=50, return_sequences=False),
                Dense(units=1),
            ]
        )

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)

    predicted_price = model.predict(X[-1].reshape(1, 60, 1))
    predicted_price = predicted_price[0][0]

    # 反标准化预测价格
    predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

    return predicted_price


# 支撑位与阻力位计算
def _parse_depth_levels(levels, limit=200):
    parsed = []
    for price, qty in levels[:limit]:
        parsed.append((float(price), float(qty)))
    return parsed


def _calculate_sr_version1(df, window=200):
    recent = df.tail(window) if len(df) > window else df
    highs = recent["high"].dropna().values
    lows = recent["low"].dropna().values
    if highs.size == 0 or lows.size == 0:
        return float("nan"), float("nan")
    support = float(np.nanquantile(lows, 0.2))
    resistance = float(np.nanquantile(highs, 0.8))
    return support, resistance


def _calculate_sr_version2(df, buy_depth=None, sell_depth=None, window=200):
    recent = df.tail(window) if len(df) > window else df
    highs = recent["high"].dropna().values
    lows = recent["low"].dropna().values
    price = recent["close"].iloc[-1] if not recent.empty else float("nan")

    if highs.size == 0 or lows.size == 0:
        return float("nan"), float("nan")

    support_candidates = [float(np.nanquantile(lows, q)) for q in (0.1, 0.2, 0.3, 0.4)]
    resistance_candidates = [float(np.nanquantile(highs, q)) for q in (0.6, 0.7, 0.8, 0.9)]

    support_below = [level for level in support_candidates if level <= price]
    resistance_above = [level for level in resistance_candidates if level >= price]
    support = max(support_below) if support_below else float(np.nanquantile(lows, 0.05))
    resistance = min(resistance_above) if resistance_above else float(np.nanquantile(highs, 0.95))

    if buy_depth and sell_depth:
        bids = _parse_depth_levels(buy_depth)
        asks = _parse_depth_levels(sell_depth)

        if bids:
            bid_prices, bid_qtys = zip(*bids)
            weighted_bid = np.average(bid_prices, weights=bid_qtys)
            support_candidates.append(weighted_bid)

        if asks:
            ask_prices, ask_qtys = zip(*asks)
            weighted_ask = np.average(ask_prices, weights=ask_qtys)
            resistance_candidates.append(weighted_ask)

    support_below = [level for level in support_candidates if level <= price]
    resistance_above = [level for level in resistance_candidates if level >= price]
    support = max(support_below) if support_below else min(support_candidates)
    resistance = min(resistance_above) if resistance_above else max(resistance_candidates)

    return support, resistance


def _calculate_sr_version3(df, window=200):
    recent = df.tail(window) if len(df) > window else df
    highs = recent["high"].dropna()
    lows = recent["low"].dropna()
    if highs.empty or lows.empty:
        return float("nan"), float("nan")
    atr = compute_atr(recent)
    if atr is None:
        atr = (highs.max() - lows.min()) / max(len(recent), 1)
    high_level = highs.max()
    low_level = lows.min()
    return float(low_level + atr * 0.5), float(high_level - atr * 0.5)


def _calculate_sr_version4(df, window=200):
    recent = df.tail(window) if len(df) > window else df
    close = recent["close"].dropna()
    if close.empty or len(close) < 20:
        return float("nan"), float("nan")
    ma = close.rolling(window=20).mean().iloc[-1]
    std = close.rolling(window=20).std().iloc[-1]
    if pd.isna(ma) or pd.isna(std):
        return float("nan"), float("nan")
    return float(ma - 2 * std), float(ma + 2 * std)


def calculate_support_resistance(df, buy_depth=None, sell_depth=None, window=200, version="版本2"):
    if version == "版本1":
        return _calculate_sr_version1(df, window=window)
    if version == "版本3":
        return _calculate_sr_version3(df, window=window)
    if version == "版本4":
        return _calculate_sr_version4(df, window=window)
    return _calculate_sr_version2(df, buy_depth=buy_depth, sell_depth=sell_depth, window=window)


# 数据整合：获取所有相关数据
def merge_market_data(btc_df, external_df, resample_rule="4H"):
    if btc_df.empty:
        return btc_df
    merged = btc_df.copy()
    merged["timestamp"] = pd.to_datetime(merged["open_time"], unit="ms")
    merged = merged.set_index("timestamp")
    if external_df.empty:
        return merged
    if isinstance(external_df.columns, pd.MultiIndex):
        external_close = external_df.xs("Close", axis=1, level=1)
    else:
        external_close = external_df[["Close"]].rename(columns={"Close": "^GSPC"})
    external_close = external_close.rename(
        columns={
            "^GSPC": "sp500_close",
            "DX-Y.NYB": "dxy_close",
            "GC=F": "gold_close",
            "^VIX": "vix_close",
            "^TNX": "us10y_close",
        }
    )
    external_close.index = pd.to_datetime(external_close.index)
    external_close = external_close.resample(resample_rule).ffill()
    merged = merged.join(external_close, how="left")
    return merged


def interval_to_resample_rule(interval):
    mapping = {
        "1m": "1T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "12h": "12H",
        "1d": "1D",
    }
    return mapping.get(interval, "4H")


def collect_data(interval=INTERVAL):
    df_btc = fetch_klines(SYMBOL, interval)
    df_ext = fetch_external_data()  # 获取外部经济数据
    resample_rule = interval_to_resample_rule(interval)
    merged = merge_market_data(df_btc, df_ext, resample_rule=resample_rule)
    return merged, df_ext


# 生成最终报告
def build_report(
    df,
    predicted_price,
    price,
    rsi,
    macd_line,
    signal,
    upper_band,
    lower_band,
    short_ema,
    long_ema,
    support,
    resistance,
    buy_depth=None,
    sell_depth=None,
    trend_info=None,
    macro_info=None,
    price_change_info=None,
    prediction_info=None,
    strategy_info=None,
):
    def fmt(value):
        return f"{value:.2f}" if pd.notna(value) else "N/A"

    lines = [
        f"价格: {fmt(price)}",
        f"预测价格: {fmt(predicted_price)}",
        f"支撑位: {fmt(support)} 阻力位: {fmt(resistance)}",
    ]
    if price_change_info:
        lines.append(price_change_info)
    if prediction_info:
        lines.append(prediction_info)
    if buy_depth and sell_depth:
        lines.append(f"深度: 买盘{len(buy_depth)} 档 / 卖盘{len(sell_depth)} 档")
    lines.append(f"RSI: {rsi:.2f}  MACD: {macd_line:.4f}/{signal:.4f}")
    lines.append(f"布林带: 上轨: {fmt(upper_band)} 下轨: {fmt(lower_band)}")
    lines.append(f"EMA: 短期: {fmt(short_ema)} 长期: {fmt(long_ema)}")
    trend = "看涨" if predicted_price > price else "看跌"
    lines.append(f"趋势: {trend}")
    if trend_info:
        lines.append(trend_info)
    if macro_info:
        lines.append(macro_info)
    if strategy_info:
        lines.append(strategy_info)
    return "\n".join(lines)


def generate_report(*args, **kwargs):
    report = build_report(*args, **kwargs)
    print(report)
    return report


def analyze_trend(df, timeframe_label):
    if df.empty:
        return "趋势: 数据不足"
    close = df["close"]
    volume = df["volume"]
    short_ma = close.rolling(window=20).mean().iloc[-1]
    long_ma = close.rolling(window=50).mean().iloc[-1]
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    vol_mean = volume.rolling(window=20).mean().iloc[-1]
    latest_price = close.iloc[-1]
    latest_volume = volume.iloc[-1]

    if pd.isna(short_ma) or pd.isna(long_ma) or pd.isna(rsi) or pd.isna(vol_mean):
        return f"{timeframe_label}趋势判断: 数据不足"

    trend = "震荡"
    if latest_price > long_ma and short_ma > long_ma:
        trend = "上行趋势"
    elif latest_price < long_ma and short_ma < long_ma:
        trend = "下行趋势"

    momentum = "多头" if rsi > 55 else "空头" if rsi < 45 else "中性"
    vol_state = "放量" if latest_volume > vol_mean else "缩量"

    signal = "观望"
    if trend == "上行趋势" and momentum == "多头" and vol_state == "放量":
        signal = "短线偏多"
    elif trend == "下行趋势" and momentum == "空头" and vol_state == "放量":
        signal = "短线偏空"

    return (
        f"{timeframe_label}趋势判断: {trend} | RSI: {rsi:.2f} | "
        f"均线: MA20 {short_ma:.2f} / MA50 {long_ma:.2f} | "
        f"成交量: {vol_state} | 信号: {signal}"
    )


def summarize_flows(df, timeframe_label):
    if df.empty or len(df) < 2:
        return f"{timeframe_label}资金流向: 数据不足"
    mfi = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=14,
    ).money_flow_index()
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    mfi_value = mfi.iloc[-1]
    obv_change = obv.iloc[-1] - obv.iloc[-2]
    flow_state = "流入" if obv_change > 0 else "流出" if obv_change < 0 else "平衡"
    return (
        f"{timeframe_label}资金流向: MFI {format_level(mfi_value)} | "
        f"OBV变化 {format_level(obv_change)} ({flow_state})"
    )


def summarize_flow_intervals(intervals):
    summaries = []
    for interval in intervals:
        df_interval = fetch_futures_klines(SYMBOL, interval=interval)
        label = interval.upper()
        summaries.append(summarize_flows(df_interval, label))
    return "\n".join(summaries)


def build_macro_summary(external_df, selected_tickers=None):
    if external_df.empty:
        return "宏观指标: 数据不足 (ai自动选择源)"

    def latest_pct_change(ticker, label):
        if isinstance(external_df.columns, pd.MultiIndex):
            series = external_df.get(ticker)
            if series is None:
                return f"{label}: N/A"
            close = series["Close"].dropna()
        else:
            if "Close" not in external_df.columns:
                return f"{label}: N/A"
            close = external_df["Close"].dropna()
        if len(close) < 2:
            return f"{label}: N/A"
        pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        return f"{label}: {pct:.2f}%"

    ticker_map = {
        "^GSPC": ("标普500", latest_pct_change("^GSPC", "标普500")),
        "DX-Y.NYB": ("美元指数", latest_pct_change("DX-Y.NYB", "美元指数")),
        "GC=F": ("黄金", latest_pct_change("GC=F", "黄金")),
        "^VIX": ("VIX", latest_pct_change("^VIX", "VIX")),
        "^TNX": ("美债10Y", latest_pct_change("^TNX", "美债10Y")),
    }
    selected = selected_tickers or list(ticker_map.keys())
    selected_lines = [ticker_map[key][1] for key in selected if key in ticker_map]
    if not selected_lines:
        selected_lines = [value[1] for value in ticker_map.values()]

    sp500 = ticker_map["^GSPC"][1]
    dxy = ticker_map["DX-Y.NYB"][1]

    macro_signal = "风险中性"
    if "标普500" in sp500 and "美元指数" in dxy:
        try:
            sp500_val = float(sp500.split(":")[1].replace("%", ""))
            dxy_val = float(dxy.split(":")[1].replace("%", ""))
            if sp500_val > 0 and dxy_val < 0:
                macro_signal = "风险偏好"
            elif sp500_val < 0 and dxy_val > 0:
                macro_signal = "风险规避"
        except ValueError:
            macro_signal = "风险中性"

    policy_note = "货币政策: 需接入利率/央行数据"
    return (
        "宏观指标: "
        f"{' | '.join(selected_lines)}\n"
        f"宏观信号: {macro_signal} | {policy_note}"
    )


def build_depth_insights():
    return (
        "衍生品/资金指标:\n"
        "- 杠杆借贷存量增速: 24h N/A | 30天 N/A\n"
        "- 杠杆多空比: 24h N/A | 30天 N/A\n"
        "- 5 x 24小时大单净流入(BTC): N/A\n"
        "- 5日主力净流入: N/A\n"
        "- 场内持仓集中度(BTC): 24h N/A | 30天 N/A\n"
        "- 逐仓借贷比: 24h N/A | 30天 N/A\n"
        "- 24小时资金净流入(BTC): N/A\n"
        "_________________________________________\n"
        "分时/图表:\n"
        "- 1秒 | 15分钟 | 1小时 | 4小时 | 1日 | 1周 | 基本版 | Trading View | 深度图\n"
        "主图指标:\n"
        "- MA | EMA | WMA | BOLL | VWAP | AVL | TRIX | SAR | SUPER\n"
        "副图指标:\n"
        "- VOL | MACD | RSI | MFI | KDJ | OBV | CCI | StochRSI | WR | DMI | MTM | EMV\n"
        "合约数据:\n"
        "- 合约持仓量(双边) | 大户账户数多空比 | 大户持仓量多空比 | 多空账户数比\n"
        "- 合约主动买卖量 | 基差 | 资金费率: 0.007503% (最近40次) | 未平仓量与市值比率"
    )


class TradeStrategy:
    def __init__(self, risk_profile="balanced", config_override=None):
        self.risk_profile = risk_profile
        base_config = STRATEGY_RISK_CONFIG.get(risk_profile, STRATEGY_RISK_CONFIG["balanced"])
        override = config_override or {}
        self.config = {
            "risk_pct": override.get("risk_pct", base_config["risk_pct"]),
            "rr": override.get("rr", base_config["rr"]),
        }

    def _position_size(self, account_balance, entry, stop):
        risk_amount = account_balance * self.config["risk_pct"]
        risk_per_unit = max(abs(entry - stop), 1e-6)
        return max(risk_amount / risk_per_unit, 0)

    def short_term_strategy(self, price, support, resistance, rsi, macd_line, signal, atr=None):
        entry_long = support * 1.01
        entry_short = resistance * 0.99
        if atr:
            stop_long = entry_long - atr * 1.2
            stop_short = entry_short + atr * 1.2
        else:
            stop_long = support * 0.98
            stop_short = resistance * 1.02
        take_long = entry_long + (entry_long - stop_long) * self.config["rr"]
        take_short = entry_short - (stop_short - entry_short) * self.config["rr"]
        long_ok = price <= entry_long and rsi < 45 and macd_line > signal
        short_ok = price >= entry_short and rsi > 55 and macd_line < signal
        return (
            "短线策略:\n"
            f"多单条件: 价格<= {entry_long:.2f} & RSI<45 & MACD金叉\n"
            f"空单条件: 价格>= {entry_short:.2f} & RSI>55 & MACD死叉\n"
            f"止损/止盈: 多 {stop_long:.2f}/{take_long:.2f} | 空 {stop_short:.2f}/{take_short:.2f}\n"
            f"信号: {'多单可尝试' if long_ok else '空单可尝试' if short_ok else '观望'}"
        )

    def range_strategy(self, price, support, resistance, atr=None):
        buy_zone = support * 1.01
        sell_zone = resistance * 0.99
        if atr:
            stop_break = buy_zone - atr * 1.1
            stop_fail = sell_zone + atr * 1.1
        else:
            stop_break = support * 0.98
            stop_fail = resistance * 1.02
        in_range = support < price < resistance
        return (
            "区间震荡策略:\n"
            f"低吸区: {buy_zone:.2f} 附近，止损 {stop_break:.2f}\n"
            f"高抛区: {sell_zone:.2f} 附近，止损 {stop_fail:.2f}\n"
            f"破位应急: 跌破支撑减仓/止损；突破压力追随并设移动止损\n"
            f"状态: {'区间内执行' if in_range else '区间外等待确认'}"
        )

    def conservative_strategy(self, price, support, resistance, short_ema, long_ema, atr=None):
        trend = "上行" if short_ema > long_ema else "下行"
        entry = resistance * 1.005 if trend == "上行" else support * 0.995
        if atr:
            stop = entry - atr * 1.5 if trend == "上行" else entry + atr * 1.5
        else:
            stop = support * 0.985 if trend == "上行" else resistance * 1.015
        take = entry + (entry - stop) * self.config["rr"] if trend == "上行" else entry - (stop - entry) * self.config["rr"]
        return (
            "稳健合约策略:\n"
            f"趋势: {trend} | 建仓价 {entry:.2f}\n"
            f"止损/止盈: {stop:.2f}/{take:.2f}\n"
            "加仓/减仓: 价格站稳均线分批加仓，跌破均线分批减仓"
        )

    def spike_response(self, price, support, resistance, atr=None):
        spike_up = resistance * 1.02 if not atr else resistance + atr * 1.5
        spike_down = support * 0.98 if not atr else support - atr * 1.5
        if price > spike_up:
            return "极端波动应对: 大涨，分批止盈或上移止损，必要时锁仓保护利润"
        if price < spike_down:
            return "极端波动应对: 大跌，快速止损或对冲锁仓，等待企稳再评估"
        return "极端波动应对: 价格正常，维持原策略或减小仓位"


def build_strategy_report(
    price,
    support,
    resistance,
    rsi,
    macd_line,
    signal,
    short_ema,
    long_ema,
    atr=None,
    strategy=None,
):
    strategy = strategy or TradeStrategy()
    return "\n".join(
        [
            strategy.short_term_strategy(price, support, resistance, rsi, macd_line, signal, atr=atr),
            strategy.range_strategy(price, support, resistance, atr=atr),
            strategy.conservative_strategy(price, support, resistance, short_ema, long_ema, atr=atr),
            strategy.spike_response(price, support, resistance, atr=atr),
        ]
    )


def describe_market_state(price, support, resistance):
    if pd.isna(support) or pd.isna(resistance):
        return "行情结构: 数据不足"
    range_width = resistance - support
    if range_width <= 0:
        return "行情结构: 区间异常"
    if price > resistance:
        return f"行情结构: 突破上沿，强弱分界 {resistance:.2f}"
    if price < support:
        return f"行情结构: 跌破下沿，强弱分界 {support:.2f}"
    if (price - support) / range_width < 0.2 or (resistance - price) / range_width < 0.2:
        return f"行情结构: 震荡偏强/弱，强弱分界 {support:.2f}-{resistance:.2f}"
    return f"行情结构: 区间震荡，强弱分界 {support:.2f}-{resistance:.2f}"


def format_level(value):
    return f"{value:.2f}" if pd.notna(value) else "N/A"


# 可视化价格与预测结果
def plot_graph(ax, df, predicted_price, levels=None, trend_outlook=None, prediction_text=None):
    ax.clear()
    ax.set_facecolor("#0f1117")
    view = df.tail(PLOT_POINTS)
    ax.plot(view["close"], label="实际价格", color="#58a6ff", linewidth=1.5)
    ax.axvline(x=len(view), color="r", linestyle="--", label="预测点")
    ax.scatter(len(view), predicted_price, color="#2ea043", label="预测价格", zorder=5)
    if levels:
        x_pos = len(view) - 1
        for label, value, color in levels:
            if pd.isna(value):
                continue
            ax.axhline(y=value, color=color, linestyle=":", linewidth=1)
            ax.text(
                x_pos,
                value,
                f"{label}:{value:.2f}",
                color=color,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
            )
    if trend_outlook:
        ax.text(
            0.02,
            0.95,
            f"预测趋势: {trend_outlook}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    if prediction_text:
        ax.text(
            0.02,
            0.9,
            prediction_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    ax.legend(loc="best")
    ax.tick_params(axis="x", colors="#8b949e")
    ax.tick_params(axis="y", colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.set_title("BTC价格与预测结果", color="#e6edf3")


def run_gui():
    root = Tk()
    root.title("BTC量化交易工具")
    root.geometry("1100x700")
    root.configure(bg="#0f1117")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TFrame", background="#0f1117")
    style.configure("TLabel", background="#0f1117", foreground="#e6edf3", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10), padding=(8, 4))
    style.configure("TCombobox", padding=(6, 2))
    style.configure(
        "Treeview",
        background="#0f1117",
        fieldbackground="#0f1117",
        foreground="#e6edf3",
        rowheight=24,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Treeview.Heading",
        background="#161b22",
        foreground="#e6edf3",
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "Treeview",
        background=[("selected", "#1f6feb")],
        foreground=[("selected", "#ffffff")],
    )

    interval_var = StringVar(value=INTERVAL)
    refresh_seconds = StringVar(value=str(DEFAULT_REFRESH_SECONDS))
    model_var = StringVar(value=MODEL_OPTIONS[0])
    sr_version_var = StringVar(value=DEFAULT_SR_VERSION)
    risk_profile_var = StringVar(value="balanced")
    risk_pct_var = StringVar(value=str(STRATEGY_RISK_CONFIG["balanced"]["risk_pct"]))
    rr_var = StringVar(value=str(STRATEGY_RISK_CONFIG["balanced"]["rr"]))
    macro_sources = [
        ("标普500", "^GSPC"),
        ("美元指数", "DX-Y.NYB"),
        ("黄金", "GC=F"),
        ("VIX", "^VIX"),
        ("美债10Y", "^TNX"),
    ]
    flow_sources = [(label, label) for label in FLOW_INTERVAL_OPTIONS]

    def sync_risk_profile():
        config = STRATEGY_RISK_CONFIG.get(risk_profile_var.get(), STRATEGY_RISK_CONFIG["balanced"])
        risk_pct_var.set(str(config["risk_pct"]))
        rr_var.set(str(config["rr"]))

    def parse_float(value, default):
        try:
            return float(value)
        except ValueError:
            return default

    def get_strategy_config():
        profile = risk_profile_var.get()
        base = STRATEGY_RISK_CONFIG.get(profile, STRATEGY_RISK_CONFIG["balanced"])
        risk_pct = parse_float(risk_pct_var.get(), base["risk_pct"])
        rr = parse_float(rr_var.get(), base["rr"])
        return profile, {"risk_pct": risk_pct, "rr": rr}

    control_frame = ttk.Labelframe(root, text="控制面板", padding=8)
    control_frame.pack(fill=BOTH, padx=10, pady=8)

    Label(control_frame, text="数据周期:").pack(side=LEFT, padx=(4, 2))
    interval_select = ttk.Combobox(
        control_frame,
        textvariable=interval_var,
        values=SUPPORTED_INTERVALS,
        width=6,
        state="readonly",
    )
    interval_select.pack(side=LEFT, padx=6)

    Label(control_frame, text="刷新(秒):").pack(side=LEFT, padx=(8, 2))
    refresh_select = ttk.Combobox(
        control_frame,
        textvariable=refresh_seconds,
        values=["3", "5", "10", "15", "30"],
        width=6,
        state="readonly",
    )
    refresh_select.pack(side=LEFT, padx=6)

    Label(control_frame, text="模型:").pack(side=LEFT, padx=(8, 2))
    model_select = ttk.Combobox(
        control_frame,
        textvariable=model_var,
        values=MODEL_OPTIONS,
        width=6,
        state="readonly",
    )
    model_select.pack(side=LEFT, padx=6)
    Label(control_frame, text="支撑/压力版本:").pack(side=LEFT, padx=(8, 2))
    sr_select = ttk.Combobox(
        control_frame,
        textvariable=sr_version_var,
        values=SR_VERSION_OPTIONS,
        width=6,
        state="readonly",
    )
    sr_select.pack(side=LEFT, padx=6)

    Label(control_frame, text="宏观指标:").pack(side=LEFT, padx=(8, 2))
    macro_listbox = Listbox(control_frame, selectmode="multiple", height=3, exportselection=False)
    for label, _ in macro_sources:
        macro_listbox.insert(END, label)
    macro_listbox.pack(side=LEFT, padx=6)
    macro_listbox.select_set(0, END)

    Label(control_frame, text="资金流向:").pack(side=LEFT, padx=(8, 2))
    flow_listbox = Listbox(control_frame, selectmode="multiple", height=3, exportselection=False)
    for label, _ in flow_sources:
        flow_listbox.insert(END, label)
    flow_listbox.pack(side=LEFT, padx=6)
    flow_listbox.select_set(0, END)

    status_label = Label(control_frame, text="状态: 手动刷新", font=("Segoe UI", 10, "bold"))
    status_label.pack(side=RIGHT)

    strategy_config_frame = ttk.Labelframe(root, text="策略定制", padding=8)
    strategy_config_frame.pack(fill=BOTH, padx=10, pady=6)
    Label(strategy_config_frame, text="风险偏好:").pack(side=LEFT, padx=4)
    risk_profile_select = ttk.Combobox(
        strategy_config_frame,
        textvariable=risk_profile_var,
        values=list(STRATEGY_RISK_CONFIG.keys()),
        width=12,
        state="readonly",
    )
    risk_profile_select.pack(side=LEFT, padx=4)
    risk_profile_select.bind("<<ComboboxSelected>>", lambda _event: sync_risk_profile())
    Label(strategy_config_frame, text="风险占比:").pack(side=LEFT, padx=4)
    risk_pct_entry = Entry(strategy_config_frame, textvariable=risk_pct_var, width=8)
    risk_pct_entry.pack(side=LEFT, padx=4)
    Label(strategy_config_frame, text="盈亏比:").pack(side=LEFT, padx=4)
    rr_entry = Entry(strategy_config_frame, textvariable=rr_var, width=8)
    rr_entry.pack(side=LEFT, padx=4)

    figure, ax = plt.subplots(figsize=(6, 4), facecolor="#0f1117")
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=8, pady=6)

    metrics_frame = ttk.Labelframe(root, text="关键指标", padding=8)
    metrics_frame.pack(fill=BOTH, padx=10, pady=6)
    metrics_table = ttk.Treeview(
        metrics_frame,
        columns=("label", "value"),
        show="headings",
        height=8,
    )
    metrics_table.heading("label", text="指标")
    metrics_table.heading("value", text="数值")
    metrics_table.column("label", width=140, anchor="w")
    metrics_table.column("value", width=180, anchor="center")
    metrics_table.pack(side=LEFT, padx=6)
    metrics_rows = [
        ("price", "当前价"),
        ("predicted", "预测价"),
        ("prediction_move", "预测涨跌"),
        ("trend", "看涨/看跌"),
        ("delta", "涨跌幅"),
        ("support", "支撑位"),
        ("resistance", "压力位"),
        ("support_1h", "1H支撑"),
        ("resistance_1h", "1H压力"),
        ("support_4h", "4H支撑"),
        ("resistance_4h", "4H压力"),
    ]
    for row_id, label in metrics_rows:
        metrics_table.insert("", "end", iid=row_id, values=(label, "—"))

    report_frame = ttk.Labelframe(root, text="分类输出", padding=8)
    report_frame.pack(fill=BOTH, padx=10, pady=6)
    report_table = ttk.Treeview(
        report_frame,
        columns=("category", "content"),
        show="headings",
        height=6,
    )
    report_table.heading("category", text="分类")
    report_table.heading("content", text="内容")
    report_table.column("category", width=120, anchor="w")
    report_table.column("content", width=700, anchor="w")
    report_table.pack(side=LEFT, padx=6)
    report_rows = [
        ("market", "行情概览"),
        ("trend", "趋势/资金流向"),
        ("macro", "宏观/衍生品"),
        ("strategy", "交易策略"),
    ]
    for row_id, label in report_rows:
        report_table.insert("", "end", iid=row_id, values=(label, "—"))

    strategy_frame = ttk.Labelframe(root, text="策略表", padding=8)
    strategy_frame.pack(fill=BOTH, padx=10, pady=6)
    strategy_table = ttk.Treeview(
        strategy_frame,
        columns=("side", "entry", "stop", "take", "amount"),
        show="headings",
        height=4,
    )
    strategy_table.heading("side", text="方向")
    strategy_table.heading("entry", text="入场")
    strategy_table.heading("stop", text="止损")
    strategy_table.heading("take", text="止盈")
    strategy_table.heading("amount", text="金额")
    strategy_table.column("side", width=80, anchor="center")
    strategy_table.column("entry", width=120, anchor="center")
    strategy_table.column("stop", width=120, anchor="center")
    strategy_table.column("take", width=120, anchor="center")
    strategy_table.column("amount", width=120, anchor="center")
    strategy_table.pack(side=LEFT, padx=6)
    for row_id, label in [("long", "做多"), ("short", "做空")]:
        strategy_table.insert("", "end", iid=row_id, values=(label, "—", "—", "—", "—"))

    account_frame = ttk.Labelframe(root, text="币安模拟仓", padding=8)
    account_frame.pack(fill=BOTH, padx=10, pady=6)
    balance_var = StringVar(value="余额: 10000.00 USDT")
    position_var = StringVar(value="持仓: 0.00 BTC")
    entry_var = StringVar(value="入场价: N/A")
    pnl_var = StringVar(value="未实现盈亏: 0.00 USDT")
    side_var = StringVar(value="方向: N/A")
    Label(account_frame, textvariable=balance_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=position_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=entry_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=pnl_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=side_var).pack(side=LEFT, padx=6)
    amount_var = StringVar(value="1000")
    Label(account_frame, text="下单金额(USDT):").pack(side=LEFT, padx=8)
    amount_entry = Entry(account_frame, textvariable=amount_var, width=8)
    amount_entry.pack(side=LEFT, padx=4)

    account_state = {
        "balance": 10000.0,
        "position": 0.0,
        "entry_price": None,
        "side": None,
        "notional": 0.0,
    }

    running = {"value": False}

    def update_once(force=False):
        if not running["value"] and not force:
            return
        interval = interval_var.get()
        sr_version = sr_version_var.get()
        status_label.config(text="状态: 获取数据中")
        df_btc, df_ext = collect_data(interval=interval)
        df_futures_1h = fetch_futures_klines(SYMBOL, interval="1h")
        df_futures_4h = fetch_futures_klines(SYMBOL, interval="4h")

        if df_btc.empty:
            status_label.config(text="状态: 数据为空")
        else:
            price = df_btc["close"].iloc[-1]
            prev_price = df_btc["close"].iloc[-2] if len(df_btc) > 1 else price
            delta = price - prev_price
            delta_pct = (delta / prev_price * 100) if prev_price else 0
            price_change_info = f"最新价变动: {delta:+.2f} ({delta_pct:+.2f}%)"
            (
                rsi,
                macd_line,
                signal,
                upper_band,
                lower_band,
                short_ema,
                long_ema,
            ) = compute_indicators(df_btc)
            atr = compute_atr(df_btc)
            predicted_price_raw = train_lstm(df_btc, model_type=model_var.get())
            buy_depth, sell_depth = get_order_book(SYMBOL, depth_limit=1000)
            depth_signal = compute_order_book_signal(buy_depth, sell_depth, mid_price=price)
            macro_bias = compute_macro_bias(df_ext)
            support, resistance = calculate_support_resistance(
                df_btc,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                version=sr_version,
            )
            support_1h, resistance_1h = calculate_support_resistance(
                df_futures_1h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                version=sr_version,
            )
            support_4h, resistance_4h = calculate_support_resistance(
                df_futures_4h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                version=sr_version,
            )
            (
                predicted_price,
                trend_outlook,
                prediction_text,
            ) = compute_prediction_metrics(
                df_btc,
                price,
                predicted_price_raw,
                rsi,
                macd_line,
                signal,
                short_ema,
                long_ema,
                upper_band=upper_band,
                lower_band=lower_band,
                support=support,
                resistance=resistance,
                depth_signal=depth_signal,
                macro_bias=macro_bias,
            )
            trend_1h = analyze_trend(df_futures_1h, "1H")
            trend_4h = analyze_trend(df_futures_4h, "4H")
            flow_main = summarize_flows(df_btc, interval.upper())
            flow_1h = summarize_flows(df_futures_1h, "1H")
            flow_4h = summarize_flows(df_futures_4h, "4H")
            selected_flow_indices = flow_listbox.curselection()
            selected_flow_intervals = []
            if selected_flow_indices:
                selected_flow_intervals = [flow_sources[i][1] for i in selected_flow_indices]
            flow_interval_summary = (
                summarize_flow_intervals(selected_flow_intervals) if selected_flow_intervals else None
            )
            market_state = describe_market_state(price, support_4h, resistance_4h)
            selected_indices = macro_listbox.curselection()
            selected_tickers = None
            if selected_indices:
                selected_tickers = [macro_sources[i][1] for i in selected_indices]
            macro_info = build_macro_summary(df_ext, selected_tickers=selected_tickers)

            risk_profile, custom_config = get_strategy_config()
            strategy = TradeStrategy(risk_profile=risk_profile, config_override=custom_config)
            entry_long = support * 1.01
            stop_long = support * 0.98
            take_long = entry_long + (entry_long - stop_long) * strategy.config["rr"]
            entry_short = resistance * 0.99
            stop_short = resistance * 1.02
            take_short = entry_short - (stop_short - entry_short) * strategy.config["rr"]
            if atr:
                stop_long = entry_long - atr * 1.2
                stop_short = entry_short + atr * 1.2
                take_long = entry_long + (entry_long - stop_long) * strategy.config["rr"]
                take_short = entry_short - (stop_short - entry_short) * strategy.config["rr"]
            qty_long = strategy._position_size(account_state["balance"], entry_long, stop_long)
            qty_short = strategy._position_size(account_state["balance"], entry_short, stop_short)
            amount_long = qty_long * entry_long
            amount_short = qty_short * entry_short

            strategy_block = build_strategy_report(
                price,
                support,
                resistance,
                rsi,
                macd_line,
                signal,
                short_ema,
                long_ema,
                atr=atr,
                strategy=strategy,
            )
            trend_block = (
                f"1H支撑/压力: {format_level(support_1h)}/{format_level(resistance_1h)} | "
                f"4H支撑/压力: {format_level(support_4h)}/{format_level(resistance_4h)}\n"
                f"{trend_1h}\n{trend_4h}\n"
                f"{flow_main}\n{flow_1h}\n{flow_4h}"
                + (f"\n{flow_interval_summary}" if flow_interval_summary else "")
                + f"\n解读BTC实时价格{format_level(price)}: {market_state}，注意关键强弱分界。"
            )
            report_table.set(
                "market",
                "content",
                (
                    f"{price_change_info} | 预测价: {predicted_price:.2f} | "
                    f"SR版本: {sr_version} | "
                    f"{prediction_text} | 趋势: {trend_outlook} | 深度: {len(buy_depth)}/{len(sell_depth)}"
                ),
            )
            report_table.set("trend", "content", trend_block)
            report_table.set("macro", "content", f"{macro_info}\n{build_depth_insights()}")
            report_table.set("strategy", "content", strategy_block)
            strategy_table.set(
                "long",
                "entry",
                f"{entry_long:.2f}",
            )
            strategy_table.set("long", "stop", f"{stop_long:.2f}")
            strategy_table.set("long", "take", f"{take_long:.2f}")
            strategy_table.set("long", "amount", f"{amount_long:.2f} USDT")
            strategy_table.set(
                "short",
                "entry",
                f"{entry_short:.2f}",
            )
            strategy_table.set("short", "stop", f"{stop_short:.2f}")
            strategy_table.set("short", "take", f"{take_short:.2f}")
            strategy_table.set("short", "amount", f"{amount_short:.2f} USDT")
            metrics_table.set("price", "value", f"{price:.2f}")
            metrics_table.set("predicted", "value", f"{predicted_price:.2f}")
            metrics_table.set("prediction_move", "value", prediction_text)
            metrics_table.set("trend", "value", trend_outlook)
            metrics_table.set("delta", "value", f"{delta:+.2f} ({delta_pct:+.2f}%)")
            metrics_table.set("support", "value", f"{format_level(support)} ({sr_version})")
            metrics_table.set("resistance", "value", f"{format_level(resistance)} ({sr_version})")
            metrics_table.set("support_1h", "value", f"{format_level(support_1h)} ({sr_version})")
            metrics_table.set("resistance_1h", "value", f"{format_level(resistance_1h)} ({sr_version})")
            metrics_table.set("support_4h", "value", f"{format_level(support_4h)} ({sr_version})")
            metrics_table.set("resistance_4h", "value", f"{format_level(resistance_4h)} ({sr_version})")

            levels = [
                (f"{interval.upper()}支撑({sr_version})", support, "tab:green"),
                (f"{interval.upper()}压力({sr_version})", resistance, "tab:red"),
                (f"1H支撑({sr_version})", support_1h, "tab:blue"),
                (f"1H压力({sr_version})", resistance_1h, "tab:purple"),
                (f"4H支撑({sr_version})", support_4h, "tab:olive"),
                (f"4H压力({sr_version})", resistance_4h, "tab:brown"),
            ]
            plot_graph(
                ax,
                df_btc,
                predicted_price,
                levels=levels,
                trend_outlook=trend_outlook,
                prediction_text=prediction_text,
            )
            canvas.draw()
            status_label.config(text=f"状态: 已更新 {datetime.now().strftime('%H:%M:%S')}")

            if account_state["position"] != 0 and account_state["entry_price"]:
                if account_state["side"] == "short":
                    pnl = (account_state["entry_price"] - price) * account_state["position"]
                else:
                    pnl = (price - account_state["entry_price"]) * account_state["position"]
            else:
                pnl = 0.0
            balance_var.set(f"余额: {account_state['balance']:.2f} USDT")
            position_var.set(f"持仓: {account_state['position']:.4f} BTC")
            entry_var.set(
                "入场价: N/A"
                if account_state["entry_price"] is None
                else f"入场价: {account_state['entry_price']:.2f}"
            )
            pnl_var.set(f"未实现盈亏: {pnl:.2f} USDT")
            side_var.set(f"方向: {account_state['side'] or 'N/A'}")

        if running["value"]:
            refresh_ms = int(refresh_seconds.get()) * 1000
            root.after(refresh_ms, update_once)

    def toggle_running():
        running["value"] = not running["value"]
        state_text = "自动刷新中" if running["value"] else "手动刷新"
        status_label.config(text=f"状态: {state_text}")
        if running["value"]:
            update_once()

    def manual_refresh():
        update_once(force=True)

    def open_long():
        if account_state["position"] != 0:
            return
        interval = interval_var.get()
        df_btc, _ = collect_data(interval=interval)
        if df_btc.empty:
            return
        price = df_btc["close"].iloc[-1]
        try:
            amount = float(amount_var.get())
        except ValueError:
            amount = 0.0
        amount = max(min(amount, account_state["balance"]), 0.0)
        if amount <= 0:
            return
        qty = amount / price
        account_state["position"] = qty
        account_state["balance"] -= amount
        account_state["entry_price"] = price
        account_state["side"] = "long"
        account_state["notional"] = amount
        update_once(force=True)

    def close_position():
        if account_state["position"] == 0 or account_state["entry_price"] is None:
            return
        interval = interval_var.get()
        df_btc, _ = collect_data(interval=interval)
        if df_btc.empty:
            return
        price = df_btc["close"].iloc[-1]
        if account_state["side"] == "short":
            pnl = (account_state["entry_price"] - price) * account_state["position"]
            account_state["balance"] += account_state["notional"] + pnl
        else:
            account_state["balance"] += account_state["position"] * price
        account_state["position"] = 0.0
        account_state["entry_price"] = None
        account_state["side"] = None
        account_state["notional"] = 0.0
        update_once(force=True)

    def open_short():
        if account_state["position"] != 0:
            return
        interval = interval_var.get()
        df_btc, _ = collect_data(interval=interval)
        if df_btc.empty:
            return
        price = df_btc["close"].iloc[-1]
        try:
            amount = float(amount_var.get())
        except ValueError:
            amount = 0.0
        amount = max(min(amount, account_state["balance"]), 0.0)
        if amount <= 0:
            return
        qty = amount / price
        account_state["position"] = qty
        account_state["balance"] -= amount
        account_state["entry_price"] = price
        account_state["side"] = "short"
        account_state["notional"] = amount
        update_once(force=True)

    Button(control_frame, text="自动/手动", command=toggle_running).pack(side=LEFT, padx=8)
    Button(control_frame, text="手动刷新", command=manual_refresh).pack(side=LEFT, padx=6)
    Button(control_frame, text="开多(模拟)", command=open_long).pack(side=LEFT, padx=6)
    Button(control_frame, text="开空(模拟)", command=open_short).pack(side=LEFT, padx=6)
    Button(control_frame, text="平仓(模拟)", command=close_position).pack(side=LEFT, padx=6)

    root.mainloop()


# 主函数
def monitor():
    while True:
        try:
            # 获取数据
            df_btc, df_ext = collect_data()
            df_futures_1h = fetch_futures_klines(SYMBOL, interval="1h")
            df_futures_4h = fetch_futures_klines(SYMBOL, interval="4h")

            if df_btc.empty:
                continue  # 如果没有数据返回，跳过当前循环

            price = df_btc["close"].iloc[-1]
            prev_price = df_btc["close"].iloc[-2] if len(df_btc) > 1 else price
            delta = price - prev_price
            delta_pct = (delta / prev_price * 100) if prev_price else 0
            price_change_info = f"最新价变动: {delta:+.2f} ({delta_pct:+.2f}%)"

            # 计算指标
            (
                rsi,
                macd_line,
                signal,
                upper_band,
                lower_band,
                short_ema,
                long_ema,
            ) = compute_indicators(df_btc)
            atr = compute_atr(df_btc)

            # LSTM预测
            predicted_price_raw = train_lstm(df_btc)

            # 市场深度
            buy_depth, sell_depth = get_order_book(SYMBOL, depth_limit=1000)
            depth_signal = compute_order_book_signal(buy_depth, sell_depth, mid_price=price)
            macro_bias = compute_macro_bias(df_ext)

            sr_version = DEFAULT_SR_VERSION
            # 支撑位和阻力位
            support, resistance = calculate_support_resistance(
                df_btc,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                version=sr_version,
            )
            support_1h, resistance_1h = calculate_support_resistance(
                df_futures_1h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                version=sr_version,
            )
            support_4h, resistance_4h = calculate_support_resistance(
                df_futures_4h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                version=sr_version,
            )
            predicted_price, trend_outlook, prediction_text = compute_prediction_metrics(
                df_btc,
                price,
                predicted_price_raw,
                rsi,
                macd_line,
                signal,
                short_ema,
                long_ema,
                upper_band=upper_band,
                lower_band=lower_band,
                support=support,
                resistance=resistance,
                depth_signal=depth_signal,
                macro_bias=macro_bias,
            )
            trend_1h = analyze_trend(df_futures_1h, "1H")
            trend_4h = analyze_trend(df_futures_4h, "4H")
            flow_main = summarize_flows(df_btc, INTERVAL.upper())
            flow_1h = summarize_flows(df_futures_1h, "1H")
            flow_4h = summarize_flows(df_futures_4h, "4H")
            flow_interval_summary = summarize_flow_intervals(FLOW_INTERVAL_OPTIONS)
            market_state = describe_market_state(price, support_4h, resistance_4h)
            macro_info = build_macro_summary(df_ext)

            # 输出报告
            generate_report(
                df_btc,
                predicted_price,
                price,
                rsi,
                macd_line,
                signal,
                upper_band,
                lower_band,
                short_ema,
                long_ema,
                support,
                resistance,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
                trend_info=(
                    f"1H支撑/压力({sr_version}): {format_level(support_1h)}/{format_level(resistance_1h)} | "
                    f"4H支撑/压力({sr_version}): {format_level(support_4h)}/{format_level(resistance_4h)}\n"
                    f"{trend_1h}\n{trend_4h}\n"
                    f"{flow_main}\n{flow_1h}\n{flow_4h}\n{flow_interval_summary}\n"
                    f"解读BTC实时价格{format_level(price)}: {market_state}，注意关键强弱分界。\n"
                    f"{build_depth_insights()}"
                ),
                macro_info=macro_info,
                price_change_info=price_change_info,
                prediction_info=prediction_text,
                strategy_info=build_strategy_report(
                    price,
                    support,
                    resistance,
                    rsi,
                    macd_line,
                    signal,
                    short_ema,
                    long_ema,
                    atr=atr,
                ),
            )

            # 可视化
            plt.figure(figsize=(10, 5))
            ax = plt.gca()
            levels = [
                (f"{INTERVAL.upper()}支撑", support, "tab:green"),
                (f"{INTERVAL.upper()}压力", resistance, "tab:red"),
                ("1H支撑", support_1h, "tab:blue"),
                ("1H压力", resistance_1h, "tab:purple"),
                ("4H支撑", support_4h, "tab:olive"),
                ("4H压力", resistance_4h, "tab:brown"),
            ]
            plot_graph(
                ax,
                df_btc,
                predicted_price,
                levels=levels,
                trend_outlook=trend_outlook,
                prediction_text=prediction_text,
            )
            plt.show()

            # 每5秒更新一次
            time.sleep(5)

        except Exception as error:
            print("更新失败:", error)


if __name__ == "__main__":
    run_gui()
