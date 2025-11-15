from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
import lightgbm as lgb
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# ==================== CONFIG ====================

class CONFIG:
    symbol = "ETH/USDT"
    timeframe = "1h"
    seq_len = 64        # sama seperti di notebook
    horizon = 1
    train_ratio = 0.7
    val_ratio = 0.15


DATA_PATH = Path("data/eth_ohlcv_1h_2021.csv")

# >>>>>>> SESUAIKAN NAMA FILE MODEL DI SINI <<<<<<
LGB_MODEL_PATH  = Path("models/lgb_splitA.txt")          # atau nama file LGB kamu
TCN_MODEL_PATH  = Path("models/tcn_splitA.h5")           # atau nama file TCN kamu
META_MODEL_PATH = Path("models/meta_stack_splitA.pkl")   # atau meta_stack_ETH_forecast_1hour_2021.pkl


# ==================== APP SETUP ====================

app = FastAPI(title="ETH TCN-LGBM Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== DATA ====================

df_raw = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df_raw = df_raw.sort_index()
# Pastikan kolom standar ada: open, high, low, close, volume
# Kalau header-nya beda (Open, High, dll) sesuaikan di sini:
df = df_raw.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
})


# ==================== FUNGSI SUPERVISED (VERSI DEPLOY) ====================

def make_supervised_tabular(df, seq_len, horizon):
    """
    Versi tabular: flatten window (seq_len x 6) jadi 1D.
    Fitur persis seperti di notebook:
    [open, high, low, close, volume, return]
    """
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data = data.dropna()

    X_list, y_list, idx_list = [], [], []

    values = data[["open", "high", "low", "close", "volume", "return"]].values
    closes = data["close"].values
    index = data.index

    for i in range(seq_len, len(data) - horizon + 1):
        X_list.append(values[i - seq_len:i, :].reshape(-1))
        y_list.append(closes[i + horizon - 1])
        idx_list.append(index[i + horizon - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    idx = np.array(idx_list)

    return X, y, idx


def make_supervised_sequence(df, seq_len, horizon):
    """
    Versi sequence untuk TCN:
    - Hitung 'return'
    - StandardScaler di fitur
    - Output shape: (samples, seq_len, n_features)
    """
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data = data.dropna()

    feats = data[["open", "high", "low", "close", "volume", "return"]].values
    closes = data["close"].values
    index = data.index

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    X_list, y_list, idx_list = [], [], []

    for i in range(seq_len, len(data) - horizon + 1):
        X_list.append(feats_scaled[i - seq_len:i, :])
        y_list.append(closes[i + horizon - 1])
        idx_list.append(index[i + horizon - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    idx = np.array(idx_list)

    return X, y, idx


# ==================== LOAD MODEL ====================

# LightGBM sebagai Booster
lgb_booster = lgb.Booster(model_file=str(LGB_MODEL_PATH))

def lgb_predict_tabular(X):
    return lgb_booster.predict(X)

# TCN keras model
tcn_model = keras.models.load_model(TCN_MODEL_PATH)

# Meta-stacking LinearRegression
meta_model: LinearRegression = joblib.load(META_MODEL_PATH)


# ==================== UTILS FORECAST ====================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def get_candle_hours(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("h"):
        return float(tf[:-1])
    if tf.endswith("m"):
        return float(tf[:-1]) / 60.0
    if tf.endswith("d"):
        return float(tf[:-1]) * 24.0
    raise ValueError(f"Timeframe tidak dikenali: {tf}")

def forecast_next_step(df_input, seq_len):
    """
    1-step ahead forecast pakai:
    - LGBM (tabular)
    - TCN (sequence)
    - Meta stacking (LinearRegression) → gabungan
    """
    X_tab_all, _, _ = make_supervised_tabular(df_input, seq_len, horizon=1)
    X_seq_all, _, _ = make_supervised_sequence(df_input, seq_len, horizon=1)

    X_tab_last = X_tab_all[-1:]
    X_seq_last = X_seq_all[-1:]

    pred_lgb = lgb_predict_tabular(X_tab_last).reshape(-1)
    pred_tcn = tcn_model.predict(X_seq_last, verbose=0).reshape(-1)

    X_meta = np.column_stack([pred_lgb, pred_tcn])
    pred_stack = meta_model.predict(X_meta).reshape(-1)

    return {
        "pred_lgb": float(pred_lgb[-1]),
        "pred_tcn": float(pred_tcn[-1]),
        "pred_stack": float(pred_stack[-1]),
    }


def forecast_multi_steps(df_input, steps, seq_len):
    """
    Multi-step forecast dengan cara iteratif:
    setiap step:
    - prediksi 1-step
    - append ke df sebagai close baru
    """
    df_work = df_input.copy()
    preds = []

    for _ in range(steps):
        pred_dict = forecast_next_step(df_work, seq_len)
        preds.append(pred_dict["pred_stack"])

        # tambahkan baris baru dengan close = pred_stack
        last_row = df_work.iloc[-1].copy()
        new_index = last_row.name + (df_work.index[-1] - df_work.index[-2])
        last_row["close"] = pred_dict["pred_stack"]
        df_work.loc[new_index] = last_row.values

    return np.array(preds, dtype=np.float32)


def forecast_hours(df_input, hours_list, seq_len):
    tf_h = get_candle_hours(CONFIG.timeframe)
    results = {}
    for h in hours_list:
        raw_steps = h / tf_h
        steps = max(1, int(round(raw_steps)))
        preds = forecast_multi_steps(df_input, steps, seq_len)
        results[f"{h}h"] = {
            "steps_used": steps,
            "pred_stack": float(preds[-1]),
        }
    return results


# ==================== PRECOMPUTE METRICS (split A) ====================

def chronological_split_A(N, train_ratio=0.7, val_ratio=0.15):
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    tr_idx = np.arange(0, n_train)
    va_idx = np.arange(n_train, n_train + n_val)
    te_idx = np.arange(n_train + n_val, N)
    return tr_idx, va_idx, te_idx


def compute_stacking_metrics():
    X_tab, y_tab, idx_all = make_supervised_tabular(df, CONFIG.seq_len, CONFIG.horizon)
    X_seq, y_seq, _ = make_supervised_sequence(df, CONFIG.seq_len, CONFIG.horizon)

    N = len(y_tab)
    tr_idx, va_idx, te_idx = chronological_split_A(N, CONFIG.train_ratio, CONFIG.val_ratio)

    Xtr_tab, Xva_tab, Xte_tab = X_tab[tr_idx], X_tab[va_idx], X_tab[te_idx]
    ytr,      yva,      yte   = y_tab[tr_idx], y_tab[va_idx], y_tab[te_idx]

    Xtr_seq, Xva_seq, Xte_seq = X_seq[tr_idx], X_seq[va_idx], X_seq[te_idx]

    # base predictions
    pte_lgb = lgb_predict_tabular(Xte_tab)
    pte_tcn = tcn_model.predict(Xte_seq, verbose=0).reshape(-1)

    Xte_meta = np.column_stack([pte_lgb, pte_tcn])
    pte_stack = meta_model.predict(Xte_meta)

    return {
        "MAE": float(mean_absolute_error(yte, pte_stack)),
        "RMSE": rmse(yte, pte_stack),
        "MAPE": mape(yte, pte_stack),
    }


STACKING_METRICS = compute_stacking_metrics()


# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {"message": "ETH TCN-LGBM Forecast API. Buka /static/index.html untuk UI."}


@app.get("/forecast_hours")
def api_forecast_hours():
    """
    Endpoint utama untuk 1h, 5h, 30h forecasting
    """
    hours_list = [1, 5, 30]
    results = forecast_hours(df, hours_list, CONFIG.seq_len)
    return results


@app.get("/metrics")
def api_metrics():
    """
    Akurasi ensemble stacking (test split A)
    """
    return STACKING_METRICS
