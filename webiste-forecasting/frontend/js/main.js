const API_BASE = "http://localhost:8000";

let priceChart = null;
let evalChart = null;

document.addEventListener("DOMContentLoaded", () => {
  setupThemeToggle();
  setupNavigation();
  setupHomePage();
  setupMetricsPage();
  setupForecastPage();

  // initial load
  loadHistory(1);
  loadMetrics();
  loadEvalSeries();

  // Tampilkan waktu lokal di halaman
  const localTimeEl = document.getElementById("localTime");
  if (localTimeEl) {
    localTimeEl.textContent = new Date().toLocaleString("id-ID", {
      timeZone: "Asia/Jakarta",
    });
  }
});

function applyTheme(theme) {
  if (theme === "dark") {
    document.body.classList.add("dark-theme");
  } else {
    document.body.classList.remove("dark-theme");
  }
}

function setupThemeToggle() {
  const toggleBtn = document.getElementById("themeToggle");
  if (!toggleBtn) return;

  const savedTheme = localStorage.getItem("theme");
  const prefersDark =
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;

  const initialTheme = savedTheme || (prefersDark ? "dark" : "light");
  applyTheme(initialTheme);

  // icon awal: kalau dark → ☀️ (artinya klik untuk terang), kalau light → 🌙
  toggleBtn.textContent = initialTheme === "dark" ? "☀️" : "🌙";

  toggleBtn.addEventListener("click", () => {
    const isDark = document.body.classList.contains("dark-theme");
    const nextTheme = isDark ? "light" : "dark";
    applyTheme(nextTheme);
    localStorage.setItem("theme", nextTheme);
    toggleBtn.textContent = nextTheme === "dark" ? "☀️" : "🌙";
  });
}

/* ========== NAVIGATION ========== */

function setupNavigation() {
  const navButtons = document.querySelectorAll(".nav-btn");
  const pages = document.querySelectorAll(".page");

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.page;

      navButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      pages.forEach((p) => {
        if (p.id === `page-${target}`) {
          p.classList.add("active");
        } else {
          p.classList.remove("active");
        }
      });
    });
  });
}

/* ========== HOME PAGE ========== */

function setupHomePage() {
  const rangeButtons = document.querySelectorAll(".range-btn");
  rangeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      rangeButtons.forEach((b) => b.classList.remove("primary"));
      btn.classList.add("primary");
      const days = parseInt(btn.dataset.days, 10);
      loadHistory(days);
    });
  });
}

async function loadHistory(days) {
  const statusEl = document.getElementById("homeStatus");
  statusEl.textContent = "Mengambil data harga...";
  if (priceChart) {
    priceChart.destroy();
  }

  try {
    const res = await fetch(`${API_BASE}/history?symbol=ETH/USDT&days=${days}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const series = data.data || [];

    const labels = series.map((d) => d.timestamp);
    const prices = series.map((d) => d.close);

    const ctx = document.getElementById("priceChart").getContext("2d");
    priceChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Close Price (1h)",
            data: prices,
            fill: false,
            borderWidth: 1,
            pointRadius: 2, // kasih titik kecil biar kelihatan
            pointHitRadius: 6, // area hover lebih lebar
          },
        ],
      },
      options: {
        responsive: true,
        interaction: {
          mode: "nearest", // tooltip muncul ke titik terdekat
          intersect: false,
        },
        plugins: {
          tooltip: {
            callbacks: {
              // format tooltip: "Jam - Harga"
              label: function (ctx) {
                const price = ctx.parsed.y.toFixed(2);
                return ` Close: ${price}`;
              },
              title: function (items) {
                // title = timestamp
                if (items.length) {
                  const ts = items[0].label;
                  return ts; // bisa diformat lagi kalau mau
                }
                return "";
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 10,
            },
          },
          y: {
            beginAtZero: false,
          },
        },
      },
    });

    statusEl.textContent = `Menampilkan ${data.symbol} untuk ${data.days} hari terakhir (interval ${data.timeframe}).`;
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      "Gagal mengambil data harga (cek backend & koneksi).";
  }
}

/* ========== METRICS PAGE ========== */

function setupMetricsPage() {
  // kalau nanti mau tombol refresh, bisa di-setup di sini
}

async function loadMetrics() {
  const statusEl = document.getElementById("metricsStatus");
  const tbody = document.querySelector("#metricsTable tbody");
  statusEl.textContent = "Mengambil metrics...";
  tbody.innerHTML = "";

  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    Object.entries(data).forEach(([modelName, metrics]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${modelName}</td>
        <td>${metrics.RMSE !== undefined ? metrics.RMSE.toFixed(4) : "-"}</td>
        <td>${metrics.MSE !== undefined ? metrics.MSE.toFixed(4) : "-"}</td>
        <td>${metrics.MAE !== undefined ? metrics.MAE.toFixed(4) : "-"}</td>
        <td>${metrics.MAPE !== undefined ? metrics.MAPE.toFixed(4) : "-"}</td>
      `;
      tbody.appendChild(tr);
    });

    statusEl.textContent = "Metrics berhasil dimuat.";
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      "Gagal mengambil metrics (cek backend & file metrics_eth.csv).";
  }
}

async function loadEvalSeries() {
  const statusEl = document.getElementById("evalStatus");
  const ctx = document.getElementById("evalChart").getContext("2d");

  statusEl.textContent = "Mengambil data visualisasi prediksi...";
  if (evalChart) {
    evalChart.destroy();
  }

  try {
    const res = await fetch(`${API_BASE}/eval-series?limit=200`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    const series = payload.data || [];

    const labels = series.map((d) => d.timestamp);
    const yTrue = series.map((d) => d.y_true);
    const yStack = series.map((d) => d.y_stack);
    // kalau mau tambah garis lain:
    const yLgbm = series.map((d) => d.y_lgbm);
    const yTcn = series.map((d) => d.y_tcn);

    evalChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Actual",
            data: yTrue,
            borderWidth: 1.2,
            pointRadius: 0,
          },
          {
            label: "Stacking",
            data: yStack,
            borderWidth: 1,
            pointRadius: 0,
          },
          {
            label: "LGBM",
            data: yLgbm,
            borderWidth: 1,
            borderDash: [4, 2],
            pointRadius: 0,
          },
          {
            label: "TCN",
            data: yTcn,
            borderWidth: 1,
            borderDash: [2, 2],
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        interaction: {
          mode: "nearest",
          intersect: false,
        },
        plugins: {
          tooltip: {
            callbacks: {
              title(items) {
                return items[0]?.label || "";
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { maxTicksLimit: 10 },
          },
          y: {
            beginAtZero: false,
          },
        },
      },
    });

    statusEl.textContent = "Visualisasi prediksi berhasil dimuat.";
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      "Gagal memuat visualisasi prediksi (cek endpoint /eval-series).";
  }
}

/* ========== FORECAST PAGE ========== */

function setupForecastPage() {
  const btnForecast = document.getElementById("btnForecast");
  btnForecast.addEventListener("click", doForecast);
}

async function doForecast() {
  const statusEl = document.getElementById("forecastStatus");
  const tbody = document.querySelector("#forecastTable tbody");
  const presetCheckboxes = document.querySelectorAll(".preset-horizon");
  const customInput = document.getElementById("customHorizon");

  statusEl.textContent = "";
  tbody.innerHTML = "";

  // Kumpulkan horizons
  const horizons = new Set();
  presetCheckboxes.forEach((cb) => {
    if (cb.checked) horizons.add(parseInt(cb.value, 10));
  });

  const customVal = customInput.value.trim();
  if (customVal) {
    const n = parseInt(customVal, 10);
    if (isNaN(n) || n < 1 || n > 168) {
      statusEl.textContent = "Custom horizon harus antara 1 hingga 168 jam.";
      return;
    }
    horizons.add(n);
  }

  if (horizons.size === 0) {
    statusEl.textContent = "Pilih minimal satu horizon.";
    return;
  }

  const hoursArr = Array.from(horizons).sort((a, b) => a - b);
  const horizonsStr = hoursArr.join(",");

  statusEl.textContent = "Menghitung forecast...";

  try {
    const url = `${API_BASE}/forecast?symbol=${encodeURIComponent(
      "ETH/USDT"
    )}&horizons=${encodeURIComponent(horizonsStr)}`;
    const res = await fetch(url);
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(`HTTP ${res.status}: ${msg}`);
    }
    const data = await res.json();

    const entries = Object.entries(data.results || {});
    entries.sort((a, b) => {
      const ha = parseInt(a[0].replace("h", ""), 10);
      const hb = parseInt(b[0].replace("h", ""), 10);
      return ha - hb;
    });

    entries.forEach(([key, info]) => {
      // convert ke format yang lebih enak dibaca local user
      const targetLocal = new Date(info.target_time_local).toLocaleString(
        "id-ID",
        {
          timeZone: "Asia/Jakarta",
          hour12: false,
        }
      );

      const tr = document.createElement("tr");
      tr.innerHTML = `
    <td>${key} <br><small>Target: ${targetLocal}</small></td>
    <td>${info.steps_used}</td>
    <td>${Number(info.pred_stack).toFixed(4)}</td>
  `;
      tbody.appendChild(tr);
    });
    document.getElementById("forecastTimeNow").textContent =
      "Forecast dihasilkan pada: " +
      new Date().toLocaleString("id-ID", { timeZone: "Asia/Jakarta" });

    statusEl.textContent = `Forecast selesai. Timeframe: ${data.timeframe}.`;
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      "Terjadi error saat memanggil API forecast. Cek log backend.";
  }
}
