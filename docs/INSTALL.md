# Installation Guide

## Prerequisites
- Python 3.10+
- pip
- A Kaggle account (for the dataset)
- (Recommended) a CUDA-capable GPU for reasonable federated-training speed

## 1. Get a Kaggle API token
1. Sign in at kaggle.com → click your profile picture → **Account**.
2. Under **API**, click **Create New Token** — this downloads `kaggle.json`.
3. Copy it into `backend/notebooks/kaggle.json`, or to `~/.kaggle/kaggle.json`.

## 2. Install backend dependencies
```bash
cd fl-chest-xray-dashboard/backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```
On a system-managed Python (no virtualenv), use:
```bash
pip install -r requirements.txt --break-system-packages
```

## 3. Run the notebook
```bash
jupyter notebook notebooks/FedRad_Backend.ipynb
```
Run every cell **in order, top to bottom**:
- Section 1–2 install/import everything and set config (edit `FL_ROUNDS`,
  `HOSPITAL_LOCAL_EPOCHS`, `IMG_SIZE` here for a faster smoke test on CPU).
- Section 3 authenticates with Kaggle and downloads the dataset automatically.
- Sections 4–9 clean the data, run EDA, preprocess images, and partition the 4 hospitals.
- Sections 10–12 run the federated training loop and evaluate the final global model.
- Section 13 generates Grad-CAM visualizations.
- Sections 14–16 save the model and start the Flask API in a background thread.

## 4. Open the dashboard
Once Section 16 has printed `Flask API running at http://127.0.0.1:5000/api`, open:
```
fl-chest-xray-dashboard/frontend/login.html
```
directly in a browser (double-click, or `open`/`xdg-open`). Sign in with the pre-filled demo
credentials, or register a new doctor account.

> If you'd rather serve the frontend over HTTP instead of `file://`, run
> `python3 -m http.server 8080` from inside `frontend/` and visit
> `http://localhost:8080/login.html`.

## 5. Troubleshooting
- **"kaggle.json not found"** — re-check step 1; the file must be readable by the notebook's
  working directory or `~/.kaggle/`.
- **Dashboard shows demo/placeholder numbers** — the Flask cell (Section 16) hasn't run yet,
  or the notebook kernel was restarted; re-run the notebook top to bottom.
- **Training is very slow** — you're likely on CPU only; lower `IMG_SIZE` to 128, `FL_ROUNDS`
  to 2–3, and `HOSPITAL_LOCAL_EPOCHS` to 1 for a quick functional test.
- **CORS errors in the browser console** — confirm `flask-cors` installed correctly and that
  Section 15.0's `CORS(app)` line ran without error.
