import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential

# 忽略 tkinter 字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

# 设置API
BINANCE_BASE = "https://api.binance.com/api/v3"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"  # 4小时数据


# 获取实时价格数据
def fetch_klines(symbol, interval="4h", limit=300):
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
            return yf.download(
                "^GSPC",
                start="2022-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                progress=False,
            )
        except Exception as error:
            print(f"获取外部数据失败(第{attempt}次): {error}")
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    return pd.DataFrame()


# 获取市场深度数据
def get_order_book(symbol="BTCUSDT"):
    url = f"{BINANCE_BASE}/depth?symbol={symbol}&limit=50"
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


# 数据标准化
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler


# LSTM模型训练
def train_lstm(df):
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
def _parse_depth_levels(levels, limit=10):
    parsed = []
    for price, qty in levels[:limit]:
        parsed.append((float(price), float(qty)))
    return parsed


def calculate_support_resistance(df, buy_depth=None, sell_depth=None, window=200):
    recent = df.tail(window) if len(df) > window else df
    highs = recent["high"].dropna().values
    lows = recent["low"].dropna().values

    if highs.size == 0 or lows.size == 0:
        return float("nan"), float("nan")

    # 使用分位数降低极值影响
    support_quantile = float(np.nanquantile(lows, 0.1))
    resistance_quantile = float(np.nanquantile(highs, 0.9))

    support = support_quantile
    resistance = resistance_quantile

    if buy_depth and sell_depth:
        bids = _parse_depth_levels(buy_depth)
        asks = _parse_depth_levels(sell_depth)

        if bids:
            bid_prices, bid_qtys = zip(*bids)
            weighted_bid = np.average(bid_prices, weights=bid_qtys)
            support = (support + weighted_bid) / 2

        if asks:
            ask_prices, ask_qtys = zip(*asks)
            weighted_ask = np.average(ask_prices, weights=ask_qtys)
            resistance = (resistance + weighted_ask) / 2

    return support, resistance


# 数据整合：获取所有相关数据
def collect_data():
    df_btc = fetch_klines(SYMBOL, INTERVAL)
    df_ext = fetch_external_data()  # 获取外部经济数据
    return df_btc, df_ext


# 生成最终报告
def generate_report(
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
):
    def fmt(value):
        return f"{value:.2f}" if pd.notna(value) else "N/A"

    print(f"价格: {fmt(price)}")
    print(f"预测价格: {fmt(predicted_price)}")
    print(f"支撑位: {fmt(support)} 阻力位: {fmt(resistance)}")
    if buy_depth and sell_depth:
        print(f"深度: 买盘{len(buy_depth)} 档 / 卖盘{len(sell_depth)} 档")
    print(f"RSI: {rsi:.2f}  MACD: {macd_line:.4f}/{signal:.4f}")
    print(f"布林带: 上轨: {fmt(upper_band)} 下轨: {fmt(lower_band)}")
    print(f"EMA: 短期: {fmt(short_ema)} 长期: {fmt(long_ema)}")
    trend = "看涨" if predicted_price > price else "看跌"
    print(f"趋势: {trend}")


# 可视化价格与预测结果
def plot_graph(df, predicted_price):
    plt.figure(figsize=(10, 5))
    plt.plot(df["close"], label="实际价格")
    plt.axvline(x=len(df), color="r", linestyle="--", label="预测点")
    plt.scatter(len(df), predicted_price, color="g", label="预测价格", zorder=5)
    plt.legend(loc="best")
    plt.title("BTC价格与预测结果")
    plt.show()


# 主函数
def monitor():
    while True:
        try:
            # 获取数据
            df_btc, df_ext = collect_data()

            if df_btc.empty:
                continue  # 如果没有数据返回，跳过当前循环

            price = df_btc["close"].iloc[-1]

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

            # LSTM预测
            predicted_price = train_lstm(df_btc)

            # 市场深度
            buy_depth, sell_depth = get_order_book(SYMBOL)

            # 支撑位和阻力位
            support, resistance = calculate_support_resistance(
                df_btc,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
            )

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
            )

            # 可视化
            plot_graph(df_btc, predicted_price)

            # 每5秒更新一次
            time.sleep(5)

        except Exception as error:
            print("更新失败:", error)


# 启动监控
monitor()
