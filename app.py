import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# -------------------- 全局设置 --------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

WINDOW_SIZE = 120
TRAIN_RATIO = 0.90
EPOCHS = 50
BATCH_SIZE = 16

# -------------------- 数据处理函数 --------------------
def load_and_process_data(csv_path):
    """加载 CSV，工程特征，返回训练/测试集和辅助对象"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')

    # 解析日期
    time_cols = ["date", "timestamp", "unix"]
    for col in time_cols:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], unit="s" if col != "date" else None, errors="coerce")
            break
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 清洗数值列
    numeric_cols = ["open", "high", "low", "close", "Volume BTC", "Volume USDT", "tradecount"]
    for c in numeric_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.replace(",", "").str.replace(" ", "")
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 对数变换
    for col in ["open", "high", "low", "close", "Volume BTC"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    df["log_close_diff"] = df["log_close"].diff().fillna(0)

    target_col = "log_close_diff"

    # 移动平均线
    df["sma_7"] = SMAIndicator(close=df[target_col], window=7, fillna=True).sma_indicator()
    df["sma_30"] = SMAIndicator(close=df[target_col], window=30, fillna=True).sma_indicator()
    df["ema_9"] = EMAIndicator(close=df[target_col], window=9, fillna=True).ema_indicator()

    # 动量
    df["rsi"] = RSIIndicator(close=df[target_col], window=14, fillna=True).rsi()
    df["roc"] = ROCIndicator(close=df[target_col], window=12, fillna=True).roc()

    # 趋势
    macd = MACD(close=df[target_col], fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # 波动率
    bb = BollingerBands(close=df[target_col], window=20, window_dev=2, fillna=True)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    atr = AverageTrueRange(high=df["log_high"], low=df["log_low"], close=df["log_close"], window=14, fillna=True)
    df["atr"] = atr.average_true_range()

    # 成交量
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["Volume BTC"], fillna=True)
    df["obv"] = obv.on_balance_volume()

    # 时间特征
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # 特征列表
    features = [
        "log_close_diff", "log_open", "log_high", "log_low", "log_close", "log_Volume BTC",
        "sma_7", "sma_30", "ema_9",
        "rsi", "roc", "macd", "macd_signal",
        "bb_high", "bb_low", "atr", "obv",
        "day_of_week", "month"
    ]
    features = [f for f in features if f in df.columns]

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    split_idx = int(len(df) * TRAIN_RATIO)
    train_data = df.iloc[:split_idx][features].values
    scaler.fit(train_data)
    data_scaled = scaler.transform(df[features].values)

    # 构造序列
    X, y = [], []
    dates = []
    target_idx = features.index("log_close_diff")
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X.append(data_scaled[i - WINDOW_SIZE:i])
        y.append(data_scaled[i, target_idx])
        dates.append(df["date"].iloc[i])

    X, y = np.array(X), np.array(y)
    dates = np.array(dates)

    train_len = split_idx - WINDOW_SIZE
    if train_len <= 0:
        raise ValueError("数据量太少，无法构建窗口")

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]
    dates_test = dates[train_len:]

    return X_train, y_train, X_test, y_test, dates_test, scaler, features, df

# -------------------- 模型构建 --------------------
def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.00001)), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.00001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, kernel_regularizer=l2(0.00001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='huber', metrics=['mae'])
    return model

# -------------------- 反归一化辅助 --------------------
def inverse_predictions(y_scaled, scaler, features, base_log_prices):
    """将归一化后的 log_close_diff 还原为真实价格"""
    target_idx = features.index("log_close_diff")
    temp = np.zeros((len(y_scaled), len(features)))
    temp[:, target_idx] = y_scaled
    diff_real = scaler.inverse_transform(temp)[:, target_idx]

    # 逐日累加 log 价格
    log_prices = np.zeros(len(diff_real))
    log_prices[0] = base_log_prices[0] + diff_real[0]
    for i in range(1, len(diff_real)):
        log_prices[i] = log_prices[i-1] + diff_real[i]

    real_prices = np.expm1(log_prices)  # exp(log) - 1
    return real_prices

# -------------------- 未来预测 --------------------
def predict_future(model, last_window_df, scaler, features, future_steps=30):
    """
    滚动预测未来价格，每步重新计算所有技术指标。
    last_window_df: 最近 WINDOW_SIZE 天的完整数据 DataFrame（必须包含所有原始列和 log 列）
    """
    hist_df = last_window_df.copy()
    target_idx = features.index("log_close_diff")

    future_prices = []

    for _ in range(future_steps):
        missing_cols = set(features) - set(hist_df.columns)
        if missing_cols:
            raise ValueError(f"hist_df 缺少列: {missing_cols}")

        recent = hist_df[features].iloc[-120:].values
        recent_scaled = scaler.transform(recent)
        X_input = recent_scaled.reshape(1, 120, len(features))

        pred_scaled = model.predict(X_input, verbose=0)[0, 0]

        temp = np.zeros((1, len(features)))
        temp[0, target_idx] = pred_scaled
        diff_real = scaler.inverse_transform(temp)[0, target_idx]

        last_log_close = hist_df["log_close"].iloc[-1]
        new_log_close = last_log_close + diff_real
        new_close = np.expm1(new_log_close)
        future_prices.append(new_close)

        last_row = hist_df.iloc[-1]
        next_date = last_row["date"] + pd.Timedelta(days=1)

        new_row = {
            "date": next_date,
            "open": new_close,
            "high": new_close,
            "low": new_close,
            "close": new_close,
            "Volume BTC": last_row["Volume BTC"],
        }
        new_row_df = pd.DataFrame([new_row])
        hist_df = pd.concat([hist_df, new_row_df], ignore_index=True)

        hist_df["log_open"] = np.log1p(hist_df["open"])
        hist_df["log_high"] = np.log1p(hist_df["high"])
        hist_df["log_low"] = np.log1p(hist_df["low"])
        hist_df["log_close"] = np.log1p(hist_df["close"])
        hist_df["log_Volume BTC"] = np.log1p(hist_df["Volume BTC"])

        hist_df["log_close_diff"] = hist_df["log_close"].diff().fillna(0)
        tcol = "log_close_diff"
        hist_df["sma_7"] = SMAIndicator(close=hist_df[tcol], window=7, fillna=True).sma_indicator()
        hist_df["sma_30"] = SMAIndicator(close=hist_df[tcol], window=30, fillna=True).sma_indicator()
        hist_df["ema_9"] = EMAIndicator(close=hist_df[tcol], window=9, fillna=True).ema_indicator()
        hist_df["rsi"] = RSIIndicator(close=hist_df[tcol], window=14, fillna=True).rsi()
        hist_df["roc"] = ROCIndicator(close=hist_df[tcol], window=12, fillna=True).roc()
        macd = MACD(close=hist_df[tcol], fillna=True)
        hist_df["macd"] = macd.macd()
        hist_df["macd_signal"] = macd.macd_signal()
        bb = BollingerBands(close=hist_df[tcol], window=20, window_dev=2, fillna=True)
        hist_df["bb_high"] = bb.bollinger_hband()
        hist_df["bb_low"] = bb.bollinger_lband()
        atr = AverageTrueRange(high=hist_df["log_high"], low=hist_df["log_low"], close=hist_df["log_close"], window=14, fillna=True)
        hist_df["atr"] = atr.average_true_range()
        obv = OnBalanceVolumeIndicator(close=hist_df["close"], volume=hist_df["Volume BTC"], fillna=True)
        hist_df["obv"] = obv.on_balance_volume()
        hist_df["day_of_week"] = hist_df["date"].dt.dayofweek
        hist_df["month"] = hist_df["date"].dt.month

    return np.array(future_prices)

# -------------------- Streamlit 页面 --------------------
st.set_page_config(page_title="比特币价格预测", layout="wide")
st.title("📈 比特币价格预测系统 (LSTM)")

# 侧边栏
st.sidebar.header("⚙️ 参数设置")
csv_path = st.sidebar.text_input("数据文件路径",
                                 value=r"D:\Python-Project\比特币预测\Data\day.csv",
                                 help="请输入 day.csv 的完整路径")
model_path = st.sidebar.text_input("模型权重文件路径",
                                   value="bitcoin_model.weights.h5",
                                   help="训练后保存的权重文件")
future_days = st.sidebar.slider("未来预测天数", min_value=7, max_value=90, value=30, step=1)

col1, col2 = st.sidebar.columns(2)
train_btn = col1.button("🔄 重新训练模型")
predict_btn = col2.button("📊 加载模型并预测")

st.sidebar.markdown("---")
st.sidebar.info(
    "**操作说明：**\n\n"
    "1. 首次使用请点击 **重新训练模型**，等待训练完成。\n"
    "2. 之后可直接点击 **加载模型并预测** 查看结果。\n"
    "3. 所有图表均可悬停查看数据，支持缩放。"
)

# -------------------- 训练逻辑 --------------------
if train_btn:
    st.info("正在加载数据并训练模型，请耐心等待...")
    try:
        X_train, y_train, X_test, y_test, dates_test, scaler, features, df = load_and_process_data(csv_path)
    except Exception as e:
        st.error(f"数据处理失败: {e}")
        st.stop()

    model = build_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    ]

    with st.spinner("模型训练中..."):
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_split=0.2,
                            callbacks=callbacks,
                            verbose=0)

    model.save_weights(model_path)
    st.success(f"训练完成！模型权重已保存至 {model_path}")

    # 展示训练曲线
    st.subheader("训练过程")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(history.history['loss'], label='训练损失')
    ax_loss.plot(history.history['val_loss'], label='验证损失')
    ax_loss.set_title('模型损失变化')
    ax_loss.legend()
    st.pyplot(fig_loss)

# -------------------- 加载与评估逻辑 --------------------
if predict_btn:
    if not os.path.exists(model_path):
        st.error(f"找不到模型文件 {model_path}，请先训练模型。")
        st.stop()

    with st.spinner("正在加载数据..."):
        try:
            X_train, y_train, X_test, y_test, dates_test, scaler, features, df = load_and_process_data(csv_path)
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            st.stop()

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.load_weights(model_path)

    # ---------- 历史价格 ----------
    st.subheader("1️⃣ 历史价格全貌")
    fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
    ax_hist.plot(df['date'], df['close'], color='#1f77b4', lw=1.5)
    ax_hist.fill_between(df['date'], df['close'], alpha=0.1, color='#1f77b4')
    ax_hist.set_title("比特币历史价格")
    ax_hist.set_ylabel("价格 (USD)")
    st.pyplot(fig_hist)

    # ---------- 测试集评估 ----------
    st.subheader("2️⃣ 测试集预测对比")
    y_pred_test = model.predict(X_test, verbose=0).flatten()

    test_start_idx = len(df) - len(y_test)
    base_log = df["log_close"].iloc[test_start_idx - 1: test_start_idx + len(y_test) - 1].values
    base_log = base_log[:len(y_test)]

    y_true_orig = inverse_predictions(y_test, scaler, features, base_log)
    y_pred_orig = inverse_predictions(y_pred_test, scaler, features, base_log)

    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    r2 = r2_score(y_true_orig, y_pred_orig)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE (平均绝对误差)", f"${mae:.2f}")
    col2.metric("RMSE (均方根误差)", f"${rmse:.2f}")
    col3.metric("MAPE (百分比误差)", f"{mape:.2f}%")
    col4.metric("R² (决定系数)", f"{r2:.4f}")

    fig_comp, ax_comp = plt.subplots(figsize=(12, 5))
    ax_comp.plot(dates_test, y_true_orig, label='实际价格', color='#2ca02c', lw=2, alpha=0.8)
    ax_comp.plot(dates_test, y_pred_orig, label='预测价格', color='#ff7f0e', ls='--', lw=2)
    ax_comp.fill_between(dates_test, y_true_orig, y_pred_orig, color='gray', alpha=0.3, label='误差范围')
    ax_comp.set_title("测试集：实际 vs 预测价格")
    ax_comp.legend()
    st.pyplot(fig_comp)

    # ---------- 残差分析 ----------
    st.subheader("3️⃣ 残差分析")
    residuals = y_true_orig - y_pred_orig

    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_pred_orig, residuals, alpha=0.5, color='#2ca02c')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title("残差 vs 预测值")
    ax1.set_xlabel("预测值")
    ax1.set_ylabel("残差")
    ax2.hist(residuals, bins=30, density=True, alpha=0.7, color='#2ca02c')
    sns.kdeplot(residuals, ax=ax2, color='darkblue')
    ax2.set_title("残差分布")
    ax2.set_xlabel("残差")
    st.pyplot(fig_res)

    # ---------- 特征重要性 ----------
    st.subheader("4️⃣ 特征重要性 (Permutation)")
    baseline_pred = model.predict(X_test, verbose=0).flatten()
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    importances = []
    for i in range(X_test.shape[2]):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, :, i])
        perm_pred = model.predict(X_perm, verbose=0).flatten()
        perm_mse = mean_squared_error(y_test, perm_pred)
        importances.append(max(perm_mse - baseline_mse, 0))
    importances = np.array(importances)
    if np.sum(importances) > 0:
        importances = importances / np.sum(importances)
    else:
        importances = np.zeros(len(features))

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_imp = importances[sorted_idx]

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_imp, y=sorted_features, palette="viridis", ax=ax_imp)
    ax_imp.set_title("特征重要性 (MSE 增加比例)")
    st.pyplot(fig_imp)

    # ---------- 未来预测 ----------
    st.subheader(f"5️⃣ 未来 {future_days} 天价格预测")

    last_window_df = df.iloc[-WINDOW_SIZE:].copy()
    future_prices = predict_future(model, last_window_df, scaler, features, future_days)

    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
    )

    fig_fut, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(future_dates, future_prices, marker='o', color='#ff7f0e', lw=2, label='预测价格')
    ax1.set_title("预测价格走势（对数刻度）", fontsize=13)
    ax1.set_ylabel("预测价格 (USD)")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    initial_price = df['close'].iloc[-1]
    cumulative_return = (future_prices / initial_price - 1) * 100
    ax2.fill_between(future_dates, cumulative_return, 0, alpha=0.3, color='#2ca02c')
    ax2.plot(future_dates, cumulative_return, marker='s', color='#2ca02c', lw=2)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title("累计收益率 (% 相对今日)", fontsize=13)
    ax2.set_ylabel("收益率 (%)")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_fut)

    pred_df = pd.DataFrame({
        "日期": future_dates.strftime('%Y-%m-%d'),
        "预测价格 (USD)": future_prices.round(2)
    })
    st.dataframe(pred_df, use_container_width=True)

    csv_data = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载预测结果为 CSV",
        data=csv_data,
        file_name=f'bitcoin_future_{future_days}days.csv',
        mime='text/csv'
    )

    st.success("所有分析完成！")