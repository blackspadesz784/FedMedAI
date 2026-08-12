# FedRad API Reference

Base URL: `http://127.0.0.1:5000/api` (started from Section 16 of `FedRad_Backend.ipynb`)

All responses are JSON unless noted. All endpoints are same-origin-relaxed via `flask-cors`
so `frontend/*.html` can call them directly when opened from the filesystem or a static server.

---

## Auth

### `POST /auth/login`
Request body:
```json
{ "email": "dr.mehta@stmarcus-hosp.org", "password": "demo1234" }
```
Response `200`:
```json
{ "doctor": { "name": "Dr. Aanya Mehta", "email": "dr.mehta@stmarcus-hosp.org" } }
```
Response `401`: `{ "error": "Invalid credentials" }`

### `POST /auth/register`
Request body: `{ "name": "...", "email": "...", "password": "..." }`
Response `200`: `{ "doctor": { "name": "...", "email": "..." } }`

---

## Prediction

### `POST /predict`
`multipart/form-data` with fields:
- `image` — the chest X-ray file
- `patient` — JSON string, e.g. `{"name":"R. Kapoor","age":54,"hospital":"Hospital A"}`

Response `200`:
```json
{
  "top_disease": "Pneumonia",
  "confidence": 0.91,
  "model_version": "1.0-fedavg",
  "top_diseases": [{ "name": "Pneumonia", "score": 0.91 }, "..."],
  "gradcam_overlay_url": "/api/gradcam/12"
}
```

### `GET /gradcam/<id>`
Returns the Grad-CAM overlay PNG for a given prediction id (binary image response).

---

## Patients

### `GET /patients/history?q=<search>`
Returns `{ "items": [ { "patient", "age", "finding", "confidence", "hospital", "date" }, ... ] }`,
most recent first. `q` is optional and matches on patient name.

### `GET /patients/<id>`
Returns a single prediction-log entry.

---

## Dashboard

### `GET /dashboard/overview`
```json
{ "total_patients": 1284, "predictions_today": 6, "avg_confidence": 0.87, "active_hospitals": 4 }
```

### `GET /dashboard/disease-stats`
```json
{ "labels": ["No Finding", "Infiltration", "..."], "counts": [412, 231, "..."] }
```

### `GET /dashboard/dataset-stats`
```json
{ "total_images": 5606, "train": 4030, "val": 806, "test": 770, "classes": 8 }
```

---

## Federated learning

### `GET /fl/status`
```json
{ "current_round": 8, "total_rounds": 8, "global_auc": 0.93, "status": "complete" }
```

### `GET /fl/hospitals`
```json
{ "items": [ { "name": "Hospital A", "samples": 1401, "epoch": "2/2", "loss": 0.21, "accuracy": 0.91, "status": "Synced" }, "..." ] }
```

### `GET /fl/training-curves`
```json
{ "labels": ["R1", "R2", "..."], "accuracy": [0.61, 0.67, "..."] }
```

### `GET /fl/evaluation`
Full model evaluation payload: `accuracy`, `precision`, `recall`, `f1`, `auc`, `loss`, plus
`roc`, `pr`, `confusion`, and `accLossCurve` objects consumed directly by the *Model
Performance* view's charts.

---

## Error format

Failures return a JSON body `{ "error": "message" }` with an appropriate HTTP status code
(`400` bad request, `401` unauthorized, `404` not found).
