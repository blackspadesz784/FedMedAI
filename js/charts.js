/**
 * charts.js
 * All chart construction lives here so dashboard.js can stay focused on
 * fetching data and wiring up interactions. Uses Chart.js (loaded via CDN
 * in index.html) for standard chart types, plus small custom DOM renderers
 * for the correlation heatmap, confusion matrix, and FedAvg workflow steps.
 */

const CHART_GREEN = "#2f9e68";
const CHART_TEAL = "#6fc8a8";
const CHART_AMBER = "#e8a33d";
const CHART_INK = "#33473f";
const CHART_GRID = "#eaf5ee";

Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = CHART_INK;

function baseGridOptions() {
  return {
    plugins: { legend: { labels: { boxWidth: 10, usePointStyle: true } } },
    scales: {
      x: { grid: { color: CHART_GRID }, ticks: { font: { size: 11 } } },
      y: { grid: { color: CHART_GRID }, ticks: { font: { size: 11 } }, beginAtZero: true },
    },
  };
}

function renderDiseaseBar(ctx, labels, counts) {
  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "Cases", data: counts, backgroundColor: CHART_GREEN, borderRadius: 6, maxBarThickness: 28 }] },
    options: { ...baseGridOptions(), plugins: { legend: { display: false } } },
  });
}

function renderDiseasePie(ctx, labels, counts) {
  const palette = ["#2f9e68", "#6fc8a8", "#8fd3ab", "#1b6e4a", "#e8a33d", "#4db683", "#d64545", "#3f7fd9"];
  return new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data: counts, backgroundColor: palette, borderWidth: 2, borderColor: "#fff" }] },
    options: { plugins: { legend: { position: "right", labels: { boxWidth: 10, font: { size: 11 } } } }, cutout: "58%" },
  });
}

function renderAgeHistogram(ctx, labels, counts) {
  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "Patients", data: counts, backgroundColor: CHART_TEAL, borderRadius: 6 }] },
    options: { ...baseGridOptions(), plugins: { legend: { display: false } } },
  });
}

function renderClassBalance(ctx, labels, counts) {
  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "Images per class", data: counts, backgroundColor: CHART_GREEN, borderRadius: 6 }] },
    options: { ...baseGridOptions(), indexAxis: "y", plugins: { legend: { display: false } } },
  });
}

function renderSplitDonut(ctx, train, val, test) {
  return new Chart(ctx, {
    type: "doughnut",
    data: { labels: ["Train", "Validation", "Test"], datasets: [{ data: [train, val, test], backgroundColor: [CHART_GREEN, CHART_TEAL, CHART_AMBER], borderWidth: 2, borderColor: "#fff" }] },
    options: { cutout: "62%", plugins: { legend: { position: "bottom" } } },
  });
}

function renderFlRounds(ctx, labels, accuracy) {
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets: [{ label: "Global accuracy", data: accuracy, borderColor: CHART_GREEN, backgroundColor: "rgba(47,158,104,0.12)", fill: true, tension: 0.35, pointRadius: 2 }] },
    options: { ...baseGridOptions(), plugins: { legend: { display: false } }, scales: { ...baseGridOptions().scales, y: { ...baseGridOptions().scales.y, max: 1 } } },
  });
}

function renderAccLoss(ctx, data) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        { label: "Train acc", data: data.train_acc, borderColor: CHART_GREEN, tension: 0.3, pointRadius: 0 },
        { label: "Val acc", data: data.val_acc, borderColor: CHART_TEAL, borderDash: [5, 4], tension: 0.3, pointRadius: 0 },
        { label: "Train loss", data: data.train_loss, borderColor: CHART_AMBER, tension: 0.3, pointRadius: 0, yAxisID: "y1" },
        { label: "Val loss", data: data.val_loss, borderColor: "#d64545", borderDash: [5, 4], tension: 0.3, pointRadius: 0, yAxisID: "y1" },
      ],
    },
    options: {
      ...baseGridOptions(),
      scales: {
        x: { grid: { display: false } },
        y: { min: 0, max: 1, grid: { color: CHART_GRID }, title: { display: true, text: "Accuracy" } },
        y1: { min: 0, max: 1.2, position: "right", grid: { display: false }, title: { display: true, text: "Loss" } },
      },
    },
  });
}

function renderRoc(ctx, fpr, tpr) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: fpr,
      datasets: [
        { label: "Global model ROC", data: fpr.map((f, i) => ({ x: f, y: tpr[i] })), borderColor: CHART_GREEN, backgroundColor: "rgba(47,158,104,0.1)", fill: true, tension: 0.25, pointRadius: 0 },
        { label: "Chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#c8d6cf", borderDash: [4, 4], pointRadius: 0 },
      ],
    },
    options: { scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate" } }, y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate" } } } },
  });
}

function renderPr(ctx, recall, precision) {
  return new Chart(ctx, {
    type: "line",
    data: { labels: recall, datasets: [{ label: "Precision-Recall", data: recall.map((r, i) => ({ x: r, y: precision[i] })), borderColor: CHART_TEAL, backgroundColor: "rgba(111,200,168,0.12)", fill: true, tension: 0.25, pointRadius: 0 }] },
    options: { scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall" } }, y: { min: 0, max: 1, title: { display: true, text: "Precision" } } }, plugins: { legend: { display: false } } },
  });
}

/** Renders a small correlation heatmap as a DOM grid (keeps the bundle CDN-light). */
function renderHeatmap(container, labels, matrix) {
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.style.display = "grid";
  wrap.style.gridTemplateColumns = `90px repeat(${labels.length}, 1fr)`;
  wrap.style.gap = "3px";

  wrap.appendChild(document.createElement("div"));
  labels.forEach((l) => {
    const h = document.createElement("div");
    h.className = "heatmap-labels-col";
    h.style.textAlign = "center";
    h.textContent = l.slice(0, 4);
    wrap.appendChild(h);
  });

  matrix.forEach((row, i) => {
    const rowLabel = document.createElement("div");
    rowLabel.className = "heatmap-labels-row";
    rowLabel.style.alignSelf = "center";
    rowLabel.textContent = labels[i];
    wrap.appendChild(rowLabel);

    row.forEach((v) => {
      const cell = document.createElement("div");
      cell.className = "heatmap-cell";
      const alpha = 0.15 + v * 0.75;
      cell.style.background = `rgba(47, 158, 104, ${alpha})`;
      cell.style.color = v > 0.55 ? "#fff" : "#1b6e4a";
      cell.textContent = v.toFixed(2);
      wrap.appendChild(cell);
    });
  });
  container.appendChild(wrap);
}

/** Renders a confusion matrix as a colored grid, shading intensity by cell value. */
function renderConfusionMatrix(container, labels, matrix) {
  container.innerHTML = "";
  const max = Math.max(...matrix.flat());
  const grid = document.createElement("div");
  grid.className = "cm-grid";

  grid.appendChild(document.createElement("div"));
  labels.forEach((l) => {
    const h = document.createElement("div");
    h.className = "cm-head";
    h.textContent = l.split(" ")[0];
    grid.appendChild(h);
  });

  matrix.forEach((row, i) => {
    const rl = document.createElement("div");
    rl.className = "cm-rowlabel";
    rl.textContent = labels[i].split(" ")[0];
    grid.appendChild(rl);
    row.forEach((v, j) => {
      const cell = document.createElement("div");
      cell.className = "cm-cell";
      const isDiag = i === j;
      const intensity = 0.25 + (v / max) * 0.65;
      cell.style.background = isDiag ? `rgba(47,158,104, ${intensity})` : `rgba(214,69,69, ${intensity * 0.7})`;
      cell.textContent = v;
      grid.appendChild(cell);
    });
  });
  container.appendChild(grid);
}

/** Renders the FedAvg step-by-step workflow as a horizontal flow diagram. */
function renderFlWorkflow(container) {
  const steps = ["Local training", "Weight upload", "Global aggregation", "Model broadcast", "Next round"];
  container.innerHTML = "";
  const flow = document.createElement("div");
  flow.className = "fl-flow";
  steps.forEach((s, i) => {
    const step = document.createElement("div");
    step.className = "fl-flow-step";
    step.innerHTML = `<div class="num">STEP ${i + 1}</div><div class="lbl">${s}</div>`;
    flow.appendChild(step);
    if (i < steps.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "fl-flow-arrow";
      arrow.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
      flow.appendChild(arrow);
    }
  });
  container.appendChild(flow);
}
