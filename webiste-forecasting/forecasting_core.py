# forecasting_core.py
import pandas as pd
import numpy as np
import ccxt

from tensorflow import keras
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler


# CONFIG class kamu paste di sini
class CONFIG:
    symbol = "ETH/USDT"
    timeframe = "1h"
    start = "2025-01-01"
    end = None
    seq_len = 168
    horizon = 1
    train_ratio = 0.7
    val_ratio = 0.15
    tcn_epochs = 50
    tcn_batch = 32

def fetch_ohlcv_ccxt(symbol, timeframe, limit=500):
    """
    Versi ringan untuk website:
    - Abaikan start/end
    - Ambil hanya 'limit' candle terakhir dari Binance
    - Cukup untuk membuat window seq_len (168) + buffer
    """
    ex = ccxt.binance()

    # ambil data terakhir
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    if not ohlcv:
        raise ValueError(f"Tidak ada data OHLCV yang diterima untuk {symbol} {timeframe}")

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    return df



def make_supervised_tabular(df, seq_len, horizon):
    data = df.copy()
    data["return"] = data["close"].pct_change() # menghitung return (persentase perubahan harga)
    data = data.dropna()

    X_list, y_list, idx_list = [], [], []

    values = data[["open","high","low","close","volume","return"]].values
    closes = data["close"].values
    index = data.index

    for i in range(seq_len, len(data) - horizon + 1):
        X_list.append(values[i-seq_len:i, :].reshape(-1))
        y_list.append(closes[i + horizon - 1])
        idx_list.append(index[i + horizon - 1])

    return np.array(X_list), np.array(y_list), np.array(idx_list)


def make_supervised_sequence(df, seq_len, horizon):
    data = df.copy()
    data["return"] = data["close"].pct_change() # menghitung return (persentase perubahan harga)
    data = data.dropna()

    # Target: tetap pakai CLOSE asli (tidak dinormalisasi)
    closes = data["close"].values
    index = data.index

    # Fitur yang dinormalisasi untuk TCN
    feat_cols = ["open", "high", "low", "close", "volume", "return"]
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(data[feat_cols])

    X_list, y_list, idx_list = [], [], []

    for i in range(seq_len, len(data) - horizon + 1):
        # pakai fitur yang sudah di-scale sebagai input TCN
        X_list.append(feats_scaled[i-seq_len:i, :])
        # tapi y tetap harga close asli
        y_list.append(closes[i + horizon - 1])
        idx_list.append(index[i + horizon - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    idx = np.array(idx_list)

    return X, y, idx


# paste fungsi fetch_ohlcv_ccxt, make_supervised_tabular, 

def get_candle_hours(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("h"):
        return float(tf[:-1])
    if tf.endswith("m"):
        return float(tf[:-1]) / 60.0
    if tf.endswith("d"):
        return float(tf[:-1]) * 24.0
    raise ValueError(f"Timeframe tidak dikenali: {tf}")

def forecast_next_step(df_input, seq_len, model_lgb, model_tcn, meta_model):
    X_tab_all, _, _ = make_supervised_tabular(df_input, seq_len, horizon=1)
    X_seq_all, _, _ = make_supervised_sequence(df_input, seq_len, horizon=1)

    X_tab_last = X_tab_all[-1:]
    X_seq_last = X_seq_all[-1:]

    y_lgb = model_lgb.predict(X_tab_last)
    y_tcn = model_tcn.predict(X_seq_last).reshape(-1)
    X_meta = np.column_stack([y_lgb, y_tcn])
    y_stack = meta_model.predict(X_meta)

    return {
        "pred_lgbm": float(y_lgb[0]),
        "pred_tcn": float(y_tcn[0]),
        "pred_stack": float(y_stack[0]),
    }

def forecast_multi_steps(df_input, steps, seq_len, model_lgb, model_tcn, meta_model):
    df_tmp = df_input.copy()
    preds = []
    for _ in range(int(steps)):
        out = forecast_next_step(df_tmp, seq_len, model_lgb, model_tcn, meta_model)
        y_hat = out["pred_stack"]
        preds.append(y_hat)

        next_ts = df_tmp.index[-1] + pd.Timedelta(CONFIG.timeframe)
        last_row = df_tmp.iloc[-1].copy()
        last_row["close"] = y_hat
        df_tmp.loc[next_ts] = last_row
    return preds

def forecast_hours(df_input, hours_list, seq_len, model_lgb, model_tcn, meta_model):
    tf_h = get_candle_hours(CONFIG.timeframe)
    results = {}
    for h in hours_list:
        raw_steps = h / tf_h
        steps = max(1, int(round(raw_steps)))
        preds = forecast_multi_steps(df_input, steps, seq_len, model_lgb, model_tcn, meta_model)
        results[f"{h}h"] = {
            "steps_used": steps,
            "pred_stack": float(preds[-1]),
        }
    return results
