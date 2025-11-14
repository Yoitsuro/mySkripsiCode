from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.staticfiles import StaticFiles


# ---------- CONFIG ----------
MODEL_PATH = Path("models/meta_stack_ETH_forecast_1hour_2021.pkl")
DATA_PATH = Path("data/eth_ohlcv_1h_2021.csv")

TARGET_COL = "close"     # ganti sesuai nama kolom target kamu
TIME_COL = "timestamp"      # ganti sesuai nama kolom waktu
# fitur yang dipakai model (SESUAIKAN dengan notebook kamu)
FEATURE_COLS = [
    # contoh:
    # "close_lag_1", "close_lag_2", "rsi_14", "ema_13", "ema_21"
]

# ---------- APP SETUP ----------
app = FastAPI(title="ETH Forecast API")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # untuk dev bebas dulu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LOAD MODEL & DATA ----------
print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Pastikan urutan waktu
df = df.sort_values(TIME_COL).reset_index(drop=True)

# ---------- UTIL: HITUNG METRIK ----------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

# ---------- SCHEMAS ----------
class ForecastResponse(BaseModel):
    timestamps: list[str]
    y_actual: list[float]
    y_pred: list[float]

class MetricsResponse(BaseModel):
    MAE: float
    RMSE: float
    MAPE: float

# ---------- ENDPOINT: FORECAST ----------
@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(horizon: int = 50):
    """
    Ambil data paling belakang sejumlah `horizon` lalu prediksi.
    Untuk awal, kita pakai data test yg sama dengan training pipeline.
    """
    # ambil window terakhir
    data_tail = df.tail(horizon)

    # kalau model kamu pakai pipeline, cukup X = data_tail[FEATURE_COLS]
    if FEATURE_COLS:
        X = data_tail[FEATURE_COLS]
    else:
        # fallback: coba semua kolom numerik kecuali target & timestamp
        ignore = {TARGET_COL, TIME_COL}
        num_cols = [
            c for c in data_tail.columns
            if c not in ignore and pd.api.types.is_numeric_dtype(data_tail[c])
        ]
        X = data_tail[num_cols]

    y_true = data_tail[TARGET_COL].values
    y_pred = model.predict(X)

    return ForecastResponse(
        timestamps=data_tail[TIME_COL].astype(str).tolist(),
        y_actual=y_true.tolist(),
        y_pred=y_pred.tolist(),
    )

# ---------- ENDPOINT: METRICS ----------
@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Hitung akurasi di seluruh data test (df).
    Bisa kamu ganti supaya cuma pakai test split.
    """
    if FEATURE_COLS:
        X = df[FEATURE_COLS]
    else:
        ignore = {TARGET_COL, TIME_COL}
        num_cols = [
            c for c in df.columns
            if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[num_cols]

    y_true = df[TARGET_COL].values
    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)
    return MetricsResponse(**metrics)

# ---------- ROOT ----------
@app.get("/")
def root():
    return {"message": "ETH Forecast API. Buka /static/index.html untuk UI."}
