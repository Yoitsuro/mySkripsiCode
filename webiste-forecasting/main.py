# main.py
from fastapi import FastAPI, Query
import pandas as pd

from forecasting_core import CONFIG, fetch_ohlcv_ccxt, forecast_hours
from model_loader import load_models

app = FastAPI()

# load model sekali di awal
model_lgb, model_tcn, meta_model = load_models()
metrics_df = pd.read_csv("metrics_eth.csv", index_col=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast")
def get_forecast(
    symbol: str = "ETH/USDT",
    horizons: str = "1,10,24,48,72",   # ini bisa kamu ubah sewaktu-waktu
):
    # parse  "1,10,24,48,72"  -> [1, 10, 24, 48, 72]
    hours_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]

    # ambil data OHLCV terbaru
    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe=CONFIG.timeframe,
        start=CONFIG.start,
        end=CONFIG.end,
        limit=1000,   # bisa disesuaikan
    )

    # panggil fungsi dari notebook yang kamu porting
    results = forecast_hours(
        df_input=df,
        hours_list=hours_list,
        seq_len=CONFIG.seq_len,
        model_lgb=model_lgb,
        model_tcn=model_tcn,
        meta_model=meta_model,
    )

    return {
        "symbol": symbol,
        "timeframe": CONFIG.timeframe,
        "results": results,
    }

@app.get("/metrics")
def get_metrics():
    # kirim tabel akurasi ke frontend
    return metrics_df.to_dict(orient="index")
