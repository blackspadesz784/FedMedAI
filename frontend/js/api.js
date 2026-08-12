/**
 * api.js
 * Thin fetch() wrapper around the Flask REST API.
 *
 * The base URL is read from window.BACKEND_URL (set in js/config.js) so that
 * the same frontend code works in local development AND on GitHub Pages against
 * a Render-deployed backend — no hard-coded localhost anywhere.
 */

// config.js must be loaded before this script (see index.html / login.html).
const API_BASE_URL = window.BACKEND_URL || "https://fedmedai.onrender.com/api";

async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`;
  const res = await fetch(url, {
    headers:
      options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

const Api = {
  // ── Auth ───────────────────────────────────────────────────────────────────
  login: (email, password) =>
    request("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  register: (name, email, password) =>
    request("/auth/register", {
      method: "POST",
      body: JSON.stringify({ name, email, password }),
    }),

  // ── Prediction ─────────────────────────────────────────────────────────────
  predict: (file, patientMeta) => {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("patient", JSON.stringify(patientMeta || {}));
    return request("/predict", { method: "POST", body: fd });
  },
  getGradCam: (predictionId) => request(`/gradcam/${predictionId}`),

  // ── Patients ───────────────────────────────────────────────────────────────
  getPatientHistory: (query = "") =>
    request(`/patients/history?q=${encodeURIComponent(query)}`),
  getPatient: (id) => request(`/patients/${id}`),

  // ── Dashboard / stats ──────────────────────────────────────────────────────
  getDashboardOverview: () => request("/dashboard/overview"),
  getDiseaseStatistics: () => request("/dashboard/disease-stats"),
  getDatasetStatistics: () => request("/dashboard/dataset-stats"),

  // ── Visualizations ─────────────────────────────────────────────────────────
  getVisualizations: () => request("/visualizations"),

  // ── Federated learning ─────────────────────────────────────────────────────
  getFLStatus: () => request("/fl/status"),
  getHospitalStatus: () => request("/fl/hospitals"),
  getTrainingCurves: () => request("/fl/training-curves"),
  getModelEvaluation: () => request("/fl/evaluation"),
  getPerClassEvaluation: () => request("/fl/evaluation/per-class"),
};
