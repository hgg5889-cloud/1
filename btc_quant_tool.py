import time
import warnings
from datetime import datetime
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    Button,
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
        "衍生品/资金指标: "
        "杠杆借贷存量增速 N/A | 杠杆多空比 N/A | 大单净流入(BTC) N/A | "
        "主力净流入 N/A | 持仓集中度 N/A | 逐仓借贷比 N/A | 24h资金净流入 N/A"
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
def plot_graph(ax, df, predicted_price, levels=None):
    ax.clear()
    view = df.tail(PLOT_POINTS)
    ax.plot(view["close"], label="实际价格")
    ax.axvline(x=len(view), color="r", linestyle="--", label="预测点")
    ax.scatter(len(view), predicted_price, color="g", label="预测价格", zorder=5)
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
    ax.legend(loc="best")
    ax.set_title("BTC价格与预测结果")


def run_gui():
    root = Tk()
    root.title("BTC量化交易工具")
    root.geometry("1100x700")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    interval_var = StringVar(value=INTERVAL)
    refresh_seconds = StringVar(value=str(DEFAULT_REFRESH_SECONDS))
    model_var = StringVar(value=MODEL_OPTIONS[0])
    macro_sources = [
        ("标普500", "^GSPC"),
        ("美元指数", "DX-Y.NYB"),
        ("黄金", "GC=F"),
        ("VIX", "^VIX"),
        ("美债10Y", "^TNX"),
    ]
    flow_sources = [(label, label) for label in FLOW_INTERVAL_OPTIONS]

    control_frame = Frame(root)
    control_frame.pack(fill=BOTH, padx=8, pady=6)

    Label(control_frame, text="数据周期:").pack(side=LEFT)
    interval_select = ttk.Combobox(
        control_frame,
        textvariable=interval_var,
        values=SUPPORTED_INTERVALS,
        width=6,
        state="readonly",
    )
    interval_select.pack(side=LEFT, padx=6)

    Label(control_frame, text="刷新(秒):").pack(side=LEFT)
    refresh_select = ttk.Combobox(
        control_frame,
        textvariable=refresh_seconds,
        values=["3", "5", "10", "15", "30"],
        width=6,
        state="readonly",
    )
    refresh_select.pack(side=LEFT, padx=6)

    Label(control_frame, text="模型:").pack(side=LEFT, padx=6)
    model_select = ttk.Combobox(
        control_frame,
        textvariable=model_var,
        values=MODEL_OPTIONS,
        width=6,
        state="readonly",
    )
    model_select.pack(side=LEFT, padx=6)

    Label(control_frame, text="宏观指标:").pack(side=LEFT, padx=6)
    macro_listbox = Listbox(control_frame, selectmode="multiple", height=3, exportselection=False)
    for label, _ in macro_sources:
        macro_listbox.insert(END, label)
    macro_listbox.pack(side=LEFT, padx=6)
    macro_listbox.select_set(0, END)

    Label(control_frame, text="资金流向:").pack(side=LEFT, padx=6)
    flow_listbox = Listbox(control_frame, selectmode="multiple", height=3, exportselection=False)
    for label, _ in flow_sources:
        flow_listbox.insert(END, label)
    flow_listbox.pack(side=LEFT, padx=6)

    status_label = Label(control_frame, text="状态: 等待中")
    status_label.pack(side=RIGHT)

    figure, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=8, pady=6)

    report_box = Text(root, height=12)
    report_box.pack(fill=BOTH, padx=8, pady=6)

    account_frame = Frame(root)
    account_frame.pack(fill=BOTH, padx=8, pady=6)
    Label(account_frame, text="币安模拟仓").pack(side=LEFT, padx=6)
    balance_var = StringVar(value="余额: 10000.00 USDT")
    position_var = StringVar(value="持仓: 0.00 BTC")
    entry_var = StringVar(value="入场价: N/A")
    pnl_var = StringVar(value="未实现盈亏: 0.00 USDT")
    Label(account_frame, textvariable=balance_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=position_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=entry_var).pack(side=LEFT, padx=6)
    Label(account_frame, textvariable=pnl_var).pack(side=LEFT, padx=6)

    account_state = {
        "balance": 10000.0,
        "position": 0.0,
        "entry_price": None,
    }

    running = {"value": True}

    def update_once(force=False):
        if not running["value"] and not force:
            return
        interval = interval_var.get()
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
            predicted_price = train_lstm(df_btc, model_type=model_var.get())
            buy_depth, sell_depth = get_order_book(SYMBOL, depth_limit=1000)
            support, resistance = calculate_support_resistance(
                df_btc,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
            )
            support_1h, resistance_1h = calculate_support_resistance(
                df_futures_1h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
            )
            support_4h, resistance_4h = calculate_support_resistance(
                df_futures_4h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
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

            report = build_report(
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
                    f"1H支撑/压力: {format_level(support_1h)}/{format_level(resistance_1h)} | "
                    f"4H支撑/压力: {format_level(support_4h)}/{format_level(resistance_4h)}\n"
                    f"{trend_1h}\n{trend_4h}\n"
                    f"{flow_main}\n{flow_1h}\n{flow_4h}"
                    + (f"\n{flow_interval_summary}" if flow_interval_summary else "")
                    + f"\n解读BTC实时价格{format_level(price)}: {market_state}，注意关键强弱分界。"
                    + f"\n{build_depth_insights()}"
                ),
                macro_info=macro_info,
                price_change_info=price_change_info,
            )
            report_box.delete("1.0", END)
            report_box.insert(END, report)
            levels = [
                (f"{interval.upper()}支撑", support, "tab:green"),
                (f"{interval.upper()}压力", resistance, "tab:red"),
                ("1H支撑", support_1h, "tab:blue"),
                ("1H压力", resistance_1h, "tab:purple"),
                ("4H支撑", support_4h, "tab:olive"),
                ("4H压力", resistance_4h, "tab:brown"),
            ]
            plot_graph(ax, df_btc, predicted_price, levels=levels)
            canvas.draw()
            status_label.config(text=f"状态: 已更新 {datetime.now().strftime('%H:%M:%S')}")

            if account_state["position"] != 0 and account_state["entry_price"]:
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

        refresh_ms = int(refresh_seconds.get()) * 1000
        root.after(refresh_ms, update_once)

    def toggle_running():
        running["value"] = not running["value"]
        state_text = "暂停" if running["value"] else "已暂停"
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
        qty = account_state["balance"] / price
        account_state["position"] = qty
        account_state["balance"] = 0.0
        account_state["entry_price"] = price
        update_once(force=True)

    def close_position():
        if account_state["position"] == 0 or account_state["entry_price"] is None:
            return
        interval = interval_var.get()
        df_btc, _ = collect_data(interval=interval)
        if df_btc.empty:
            return
        price = df_btc["close"].iloc[-1]
        account_state["balance"] = account_state["position"] * price
        account_state["position"] = 0.0
        account_state["entry_price"] = None
        update_once(force=True)

    Button(control_frame, text="开始/暂停", command=toggle_running).pack(side=LEFT, padx=8)
    Button(control_frame, text="手动刷新", command=manual_refresh).pack(side=LEFT, padx=6)
    Button(control_frame, text="开多(模拟)", command=open_long).pack(side=LEFT, padx=6)
    Button(control_frame, text="平仓(模拟)", command=close_position).pack(side=LEFT, padx=6)

    update_once()
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

            # LSTM预测
            predicted_price = train_lstm(df_btc)

            # 市场深度
            buy_depth, sell_depth = get_order_book(SYMBOL, depth_limit=1000)

            # 支撑位和阻力位
            support, resistance = calculate_support_resistance(
                df_btc,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
            )
            support_1h, resistance_1h = calculate_support_resistance(
                df_futures_1h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
            )
            support_4h, resistance_4h = calculate_support_resistance(
                df_futures_4h,
                buy_depth=buy_depth,
                sell_depth=sell_depth,
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
                    f"1H支撑/压力: {format_level(support_1h)}/{format_level(resistance_1h)} | "
                    f"4H支撑/压力: {format_level(support_4h)}/{format_level(resistance_4h)}\n"
                    f"{trend_1h}\n{trend_4h}\n"
                    f"{flow_main}\n{flow_1h}\n{flow_4h}\n{flow_interval_summary}\n"
                    f"解读BTC实时价格{format_level(price)}: {market_state}，注意关键强弱分界。\n"
                    f"{build_depth_insights()}"
                ),
                macro_info=macro_info,
                price_change_info=price_change_info,
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
            plot_graph(ax, df_btc, predicted_price, levels=levels)
            plt.show()

            # 每5秒更新一次
            time.sleep(5)

        except Exception as error:
            print("更新失败:", error)


if __name__ == "__main__":
    run_gui()
