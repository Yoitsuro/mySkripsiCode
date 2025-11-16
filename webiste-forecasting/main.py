from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from datetime import datetime, timedelta, timezone
from forecasting_core import CONFIG, fetch_ohlcv_ccxt, forecast_hours
from model_loader import load_models

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk development, boleh semua origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model sekali di awal
model_lgb, model_tcn, meta_model = load_models()

# Coba load metrics; kalau tidak ada, pakai DataFrame kosong
try:
    metrics_df = pd.read_csv("metrics_eth.csv", index_col=0)
except FileNotFoundError:
    metrics_df = pd.DataFrame()

try:
  eval_series_df = pd.read_csv("eval_series_eth.csv", parse_dates=["timestamp"])
except FileNotFoundError:
  eval_series_df = pd.DataFrame()


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/history")
def get_history(
    symbol: str = "ETH/USDT",
    days: int = 1,
):
    """
    Mengembalikan data harga untuk 'days' terakhir dengan interval 1 jam (CONFIG.timeframe).
    days = 1, 7, atau 30.
    """
    if days not in (1, 7, 30):
        raise HTTPException(status_code=400, detail="days harus 1, 7, atau 30")

    candles_needed = days * 24
    limit = candles_needed + 20  # buffer dikit

    try:
        df = fetch_ohlcv_ccxt(
            symbol=symbol,
            timeframe=CONFIG.timeframe,
            limit=limit,
        )
    except Exception as e:
        print("Error saat fetch_ohlcv_ccxt di /history:", repr(e))
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data OHLCV: {e}")

    df = df.tail(candles_needed)

    data = [
        {
            "timestamp": ts.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for ts, row in df.iterrows()
    ]

    return {
        "symbol": symbol,
        "timeframe": CONFIG.timeframe,
        "days": days,
        "data": data,
    }



@app.get("/metrics")
def get_metrics():
    if metrics_df.empty:
        # Supaya jelas kalau file metrics belum ada
        raise HTTPException(status_code=500, detail="metrics_eth.csv tidak ditemukan atau kosong.")
    return metrics_df.to_dict(orient="index")


@app.get("/eval-series")
def get_eval_series(limit: int = 200):
    """
    Mengembalikan deret waktu actual vs prediksi.
    limit = berapa titik terakhir yang mau dikirim (biar chart nggak kebanyakan).
    """
    if eval_series_df.empty:
        raise HTTPException(status_code=500, detail="eval_series_eth.csv tidak ditemukan atau kosong.")

    df = eval_series_df.copy().sort_values("timestamp")
    if limit is not None and limit > 0:
        df = df.tail(limit)

    data = []
    for _, row in df.iterrows():
        data.append({
            "timestamp": row["timestamp"].isoformat(),
            "y_true": float(row["y_true"]),
            "y_lgbm": float(row["y_lgbm"]),
            "y_tcn": float(row["y_tcn"]),
            "y_mean": float(row["y_mean"]),
            "y_wmean": float(row["y_wmean"]),
            "y_stack": float(row["y_stack"]),
        })

    return {"data": data}


@app.get("/forecast")
def get_forecast(
    symbol: str = "ETH/USDT",
    horizons: str = "1,10,24,48,72",
):
    # waktu sekarang UTC, DIBUAT SETIAP REQUEST
    forecast_time = datetime.now(timezone.utc)

    # 1) Parse horizons
    try:
        hours_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Format horizons tidak valid. Contoh: 1,10,24,48,72"
        )

    # 2) Ambil data OHLCV terbaru
    try:
        df = fetch_ohlcv_ccxt(
            symbol=symbol,
            timeframe=CONFIG.timeframe,
            limit=500,  # cukup untuk seq_len=168 + buffer
        )
    except Exception as e:
        print("Error saat fetch_ohlcv_ccxt:", repr(e))
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data OHLCV: {e}")

    # 3) Pastikan panjang data cukup
    if len(df) < CONFIG.seq_len + 5:
        raise HTTPException(
            status_code=500,
            detail=f"Data OHLCV terlalu sedikit ({len(df)} baris). Butuh minimal {CONFIG.seq_len + 5} baris."
        )

    # 4) Hitung forecast
    try:
        results = forecast_hours(
            df_input=df,
            hours_list=hours_list,
            seq_len=CONFIG.seq_len,
            model_lgb=model_lgb,
            model_tcn=model_tcn,
            meta_model=meta_model,
        )

        enhanced_results = {}
        for key, val in results.items():
            h = int(key.replace("h",""))
            target_utc = forecast_time + timedelta(hours=h)
            enhanced_results[key] = {
                **val,
                "target_time_utc": target_utc.isoformat(),
                "target_time_local": target_utc.astimezone().isoformat(),
            }

    except Exception as e:
        print("Error saat forecast_hours:", repr(e))
        raise HTTPException(status_code=500, detail=f"Terjadi error saat menghitung forecast: {e}")

    return {
        "symbol": symbol,
        "timeframe": CONFIG.timeframe,
        "generated_at_utc": forecast_time.isoformat(),
        "generated_at_local": forecast_time.astimezone().isoformat(),
        "results": enhanced_results,
    }

