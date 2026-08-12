/**
 * dashboard.js
 * Populates every view with data, preferring the live Flask API (started
 * locally or deployed on Render) and transparently falling back to mockdata.js
 * when the backend isn't reachable. Wires up upload, prediction, and charts.
 */

async function tryApi(fn, fallback) {
  try {
    return await fn();
  } catch (e) {
    return fallback;
  }
}

function fmtPct(x) {
  return `${Math.round((x || 0) * 100)}%`;
}

function getFullUrl(url) {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) {
    return url;
  }
  const base = window.BACKEND_BASE || "";
  return url.startsWith("/") ? `${base}${url}` : `${base}/${url}`;
}

// ---------------------------------------------------------------------
// Overview / dashboard
// ---------------------------------------------------------------------
async function loadOverview() {
  const data = await tryApi(() => Api.getDashboardOverview(), MOCK.overview);
  const grid = document.getElementById("overviewStatGrid");
  const cards = [
    { icon: "history", label: "Total patients", value: data.total_patients ?? 0, delta: "synced dataset", up: true },
    { icon: "scan", label: "Predictions today", value: data.predictions_today ?? 0, delta: "+12 vs yesterday", up: true },
    { icon: "chart", label: "Avg. confidence", value: fmtPct(data.avg_confidence), delta: "stable", up: true },
    { icon: "hospital", label: "Active hospitals", value: data.active_hospitals ?? 4, delta: "all synced", up: true },
  ];
  grid.innerHTML = cards
    .map(
      (c) => `
    <div class="card stat-card">
      <div class="icon-wrap" data-icon="${c.icon}"></div>
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
      <div class="delta ${c.up ? "up" : "down"}">${c.delta}</div>
    </div>`
    )
    .join("");
  mountIcons(grid);

  const preds = await tryApi(() => Api.getPatientHistory(), { items: MOCK.recentPredictions });
  const rows = (preds.items || preds).slice(0, 6);
  const tbody = document.querySelector("#recentPredictionsTable tbody");
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--ink-500);">No predictions logged yet today. Upload an X-ray to test!</td></tr>`;
  } else {
    tbody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td>${r.patient}</td>
        <td><span class="pill pill-green">${r.finding}</span></td>
        <td class="mono">${fmtPct(r.confidence)}</td>
        <td>${r.hospital}</td>
        <td>${r.time || r.date}</td>
      </tr>`
      )
      .join("");
  }

  const fl = await tryApi(() => Api.getFLStatus(), MOCK.flStatus);
  document.getElementById("flRoundTag").textContent = `round ${fl.current_round}/${fl.total_rounds}`;
  const hospitals = await tryApi(() => Api.getHospitalStatus(), { items: MOCK.hospitals });
  document.getElementById("flMiniList").innerHTML = (hospitals.items || hospitals)
    .map(
      (h) => `
    <div class="fl-mini-row">
      <div><div class="hname">${h.name}</div><div class="hmeta">${h.samples} samples · epoch ${h.epoch}</div></div>
      <span class="pill ${h.status === "Synced" ? "pill-green" : "pill-amber"}"><span class="dot ${h.status !== "Synced" ? "blink" : ""}"></span>${h.status}</span>
    </div>`
    )
    .join("");
}

// ---------------------------------------------------------------------
// Upload & predict
// ---------------------------------------------------------------------
function initUpload() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const predictBtn = document.getElementById("predictBtn");
  const predictBtnLabel = document.getElementById("predictBtnLabel");
  let currentFile = null;

  if (!dropzone) return;

  dropzone.addEventListener("click", () => fileInput.click());
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  function handleFile(file) {
    if (!file) return;
    currentFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewWrap.style.display = "block";
    predictBtn.disabled = false;
    predictBtnLabel.textContent = `Analyze ${file.name}`;
  }

  predictBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    predictBtn.disabled = true;
    predictBtnLabel.innerHTML =
      '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Running inference…';

    const meta = {
      name: document.getElementById("patientName").value || "Unnamed patient",
      age: document.getElementById("patientAge").value || "—",
    };

    let result;
    try {
      result = await Api.predict(currentFile, meta);
      showToast("Prediction complete.", "success");
    } catch (e) {
      console.warn("API prediction failed, using fallback:", e);
      result = {
        top_disease: "Pneumonia",
        confidence: 0.91,
        model_version: "2.0-fedavg",
        top_diseases: [
          { name: "Pneumonia", score: 0.91 },
          { name: "Infiltration", score: 0.62 },
          { name: "Effusion", score: 0.24 },
          { name: "No Finding", score: 0.08 },
        ],
        gradcam_overlay_url: previewImg.src,
      };
      showToast("Backend not reachable — showing demo prediction.", "error");
    }

    renderPrediction(result, previewImg.src);
    predictBtn.disabled = false;
    predictBtnLabel.textContent = `Analyze ${currentFile.name}`;

    // Automatically refresh all dashboard stats, history, and disease metrics
    try {
      await Promise.all([
        loadOverview(),
        loadHistory(),
        loadDiseaseStats(),
        loadDatasetStats()
      ]);
    } catch (err) {
      console.warn("Post-prediction view refresh error:", err);
    }
  });
}

function renderPrediction(result, imgSrc) {
  const box = document.getElementById("predictionResult");
  box.style.display = "block";
  document.getElementById("gradcamBase").src = imgSrc;

  const overlayUrl = result.gradcam_overlay_url
    ? getFullUrl(result.gradcam_overlay_url)
    : imgSrc;
  document.getElementById("gradcamOverlay").src = overlayUrl;

  document.getElementById("modelVersionTag").textContent = result.model_version || "2.0-fedavg";
  document.getElementById("topDiseaseName").textContent = result.top_disease;
  document.getElementById("confidenceBar").style.width = `${result.confidence * 100}%`;
  document.getElementById("confidenceValue").textContent = fmtPct(result.confidence);
  document.getElementById("topDiseasesList").innerHTML = (result.top_diseases || [])
    .map(
      (d) => `
    <div class="top-disease-row">
      <div class="name">${d.name}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.score || 0) * 100}%;"></div></div>
      <div class="pct">${fmtPct(d.score)}</div>
    </div>`
    )
    .join("");
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.getElementById("overlaySlider")?.addEventListener("input", (e) => {
  const overlay = document.getElementById("gradcamOverlay");
  if (overlay) overlay.style.opacity = e.target.value / 100;
});

// ---------------------------------------------------------------------
// Patient history
// ---------------------------------------------------------------------
async function loadHistory(query = "") {
  const data = await tryApi(() => Api.getPatientHistory(query), { items: MOCK.history });
  const rows = data.items || data;
  const tbody = document.querySelector("#historyTable tbody");
  if (!rows || rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--ink-500);">No patient records found.</td></tr>`;
  } else {
    tbody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td>${r.patient}</td>
        <td>${r.age}</td>
        <td><span class="pill pill-green">${r.finding}</span></td>
        <td class="mono">${fmtPct(r.confidence)}</td>
        <td>${r.hospital}</td>
        <td>${r.date}</td>
        <td><button class="btn btn-ghost btn-sm">View</button></td>
      </tr>`
      )
      .join("");
  }
}

document.getElementById("historySearch")?.addEventListener("input", (e) => loadHistory(e.target.value));

// ---------------------------------------------------------------------
// Disease statistics
// ---------------------------------------------------------------------
let diseaseBarChart, diseasePieChart, ageHistChart;
async function loadDiseaseStats() {
  const data = await tryApi(() => Api.getDiseaseStatistics(), MOCK.diseaseStats);
  diseaseBarChart?.destroy();
  diseasePieChart?.destroy();
  ageHistChart?.destroy();
  diseaseBarChart = renderDiseaseBar(document.getElementById("chartDiseaseBar"), data.labels, data.counts);
  diseasePieChart = renderDiseasePie(document.getElementById("chartDiseasePie"), data.labels, data.counts);
  ageHistChart = renderAgeHistogram(
    document.getElementById("chartAgeHist"),
    MOCK.ageHistogram.labels,
    MOCK.ageHistogram.counts
  );
  renderHeatmap(document.getElementById("corrHeatmap"), MOCK.correlationLabels, MOCK.correlationMatrix);
}

// ---------------------------------------------------------------------
// Dataset statistics
// ---------------------------------------------------------------------
let classBalanceChart, splitChart;
async function loadDatasetStats() {
  const data = await tryApi(() => Api.getDatasetStatistics(), MOCK.datasetStats);
  const grid = document.getElementById("datasetStatGrid");
  const cards = [
    { icon: "dataset", label: "Total images", value: data.total_images ?? 5606 },
    { icon: "layers", label: "Training set", value: data.train ?? 4030 },
    { icon: "history", label: "Validation set", value: data.val ?? 806 },
    { icon: "stats", label: "Disease classes", value: data.classes ?? 8 },
  ];
  grid.innerHTML = cards
    .map(
      (c) => `
    <div class="card stat-card">
      <div class="icon-wrap" data-icon="${c.icon}"></div>
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>`
    )
    .join("");
  mountIcons(grid);
  classBalanceChart?.destroy();
  splitChart?.destroy();
  const cbLabels = data.labels && data.labels.length > 0 ? data.labels : MOCK.classBalance.labels;
  const cbCounts = data.labels && data.labels.length > 0 ? data.labels.map(l => (data.label_counts && data.label_counts[l]) || 500) : MOCK.classBalance.counts;
  classBalanceChart = renderClassBalance(document.getElementById("chartClassBalance"), cbLabels, cbCounts);
  splitChart = renderSplitDonut(document.getElementById("chartSplit"), data.train || 4030, data.val || 806, data.test || 770);
}

// ---------------------------------------------------------------------
// Visualization gallery
// ---------------------------------------------------------------------
async function loadVizGallery() {
  const grid = document.getElementById("vizGallery");
  const vizData = await tryApi(() => Api.getVisualizations(), { items: [] });
  const items = vizData.items && vizData.items.length > 0 ? vizData.items : MOCK.vizGallery;

  grid.innerHTML = items
    .map((v) => {
      const fullUrl = v.url ? getFullUrl(v.url) : "";
      const imgHtml = fullUrl
        ? `<img src="${fullUrl}" style="width:100%; height:180px; object-fit:cover; border-radius:var(--radius-md); margin-bottom:12px;" alt="${v.title}" onerror="this.onerror=null; this.style.display='none'; this.nextElementSibling.insertAdjacentHTML('beforebegin', '<div class=\\'viz-placeholder\\'>Visualization unavailable</div>');" />`
        : `<div class="viz-placeholder">Visualization unavailable</div>`;
      return `
    <div class="card viz-card">
      ${imgHtml}
      <h4>${v.title}</h4>
      <p>${v.desc}</p>
    </div>`;
    })
    .join("");
}

// ---------------------------------------------------------------------
// FL monitor
// ---------------------------------------------------------------------
let flRoundsChart;
async function loadFlMonitor() {
  const status = await tryApi(() => Api.getFLStatus(), MOCK.flStatus);
  const statusLabel = document.getElementById("flStatusLabel");
  if (statusLabel) {
    statusLabel.textContent = status.status === "complete" ? "Global model synchronized" : "Round in progress";
  }
  const grid = document.getElementById("flStatGrid");
  const cards = [
    { icon: "network", label: "Current round", value: `${status.current_round}/${status.total_rounds}` },
    { icon: "chart", label: "Global AUC", value: (status.global_auc || 0).toFixed(2) },
    { icon: "hospital", label: "Hospitals online", value: status.num_hospitals || 4 },
    { icon: "layers", label: "Aggregation method", value: status.aggregation || "FedAvg" },
  ];
  grid.innerHTML = cards
    .map(
      (c) => `
    <div class="card stat-card">
      <div class="icon-wrap" data-icon="${c.icon}"></div>
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>`
    )
    .join("");
  mountIcons(grid);

  const hospitals = await tryApi(() => Api.getHospitalStatus(), { items: MOCK.hospitals });
  document.querySelector("#hospitalTable tbody").innerHTML = (hospitals.items || hospitals)
    .map(
      (h) => `
    <tr>
      <td>${h.name}</td>
      <td class="mono">${h.samples}</td>
      <td class="mono">${h.epoch}</td>
      <td class="mono">${(h.loss || 0).toFixed(3)}</td>
      <td class="mono">${fmtPct(h.accuracy)}</td>
      <td><span class="pill ${h.status === "Synced" ? "pill-green" : "pill-amber"}"><span class="dot ${h.status !== "Synced" ? "blink" : ""}"></span>${h.status}</span></td>
    </tr>`
    )
    .join("");

  renderFlWorkflow(document.getElementById("flWorkflowDiagram"));
  const curves = await tryApi(() => Api.getTrainingCurves(), MOCK.flRounds);
  flRoundsChart?.destroy();
  flRoundsChart = renderFlRounds(
    document.getElementById("chartFlRounds"),
    curves.labels || MOCK.flRounds.labels,
    curves.accuracy || MOCK.flRounds.accuracy
  );
}

// ---------------------------------------------------------------------
// Model performance
// ---------------------------------------------------------------------
let accLossChart, rocChart, prChart;
async function loadModelPerformance() {
  const data = await tryApi(() => Api.getModelEvaluation(), MOCK.modelEval);
  const grid = document.getElementById("modelStatGrid");
  const cards = [
    { icon: "chart", label: "Accuracy", value: fmtPct(data.accuracy) },
    { icon: "stats", label: "Precision", value: fmtPct(data.precision) },
    { icon: "history", label: "Recall", value: fmtPct(data.recall) },
    { icon: "network", label: "AUC", value: (data.auc || 0).toFixed(2) },
  ];
  grid.innerHTML = cards
    .map(
      (c) => `
    <div class="card stat-card">
      <div class="icon-wrap" data-icon="${c.icon}"></div>
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>`
    )
    .join("");
  mountIcons(grid);

  accLossChart?.destroy();
  rocChart?.destroy();
  prChart?.destroy();
  const accLossData = data.accLossCurve || MOCK.modelEval.accLossCurve;
  const rocData = data.roc || MOCK.modelEval.roc;
  const prData = data.pr || MOCK.modelEval.pr;
  const cmData = data.confusion || MOCK.modelEval.confusion;

  accLossChart = renderAccLoss(document.getElementById("chartAccLoss"), accLossData);
  rocChart = renderRoc(document.getElementById("chartRoc"), rocData.fpr, rocData.tpr);
  prChart = renderPr(document.getElementById("chartPr"), prData.recall, prData.precision);
  renderConfusionMatrix(
    document.getElementById("confusionMatrix"),
    cmData.labels && cmData.labels.length ? cmData.labels : MOCK.modelEval.confusion.labels,
    cmData.matrix && cmData.matrix.length ? cmData.matrix : MOCK.modelEval.confusion.matrix
  );
}

// ---------------------------------------------------------------------
// View-shown router
// ---------------------------------------------------------------------
const viewLoaders = {
  dashboard: loadOverview,
  history: () => loadHistory(),
  "disease-stats": loadDiseaseStats,
  "dataset-stats": loadDatasetStats,
  visualizations: loadVizGallery,
  "fl-monitor": loadFlMonitor,
  "model-performance": loadModelPerformance,
};

document.addEventListener("DOMContentLoaded", () => {
  initUpload();
  loadOverview();
});

document.addEventListener("view:shown", (e) => {
  const loader = viewLoaders[e.detail.name];
  if (loader) loader();
});
