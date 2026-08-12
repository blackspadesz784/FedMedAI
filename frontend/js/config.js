/**
 * config.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Single source of truth for the backend API URL.
 *
 * LOCAL DEVELOPMENT
 *   Leave BACKEND_URL as-is — it points to your local Flask server.
 *
 * GITHUB PAGES → RENDER DEPLOYMENT
 *   1. Deploy the backend to Render and note the URL, e.g.:
 *        https://fedmedai.onrender.com
 *   2. Replace the URL below (or set window.BACKEND_URL before this script loads):
 *        window.BACKEND_URL = "https://fedmedai.onrender.com/api";
 *
 * ENVIRONMENT-BASED OVERRIDE
 *   If you use a build step (Vite/Webpack), you can inject this at build time.
 *   For the current vanilla JS setup, just edit the URL below for each deployment.
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ── Production Backend URL ───────────────────────────────────────────────────
// Render: "https://fedmedai.onrender.com/api"
window.BACKEND_URL = window.BACKEND_URL || "https://fedmedai.onrender.com/api";

// Expose the base URL (without /api) for constructing image URLs (Grad-CAM etc.)
window.BACKEND_BASE = window.BACKEND_URL.replace(/\/api$/, "");
