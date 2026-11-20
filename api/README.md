# FastAPI service

`main.py` loads `model_output/accident_severity_pipeline.joblib` from the project root. The root and `/health` endpoints report service status; `POST /predict` returns a severity estimate and class probabilities.

Run locally from the project root:

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --reload
```

For Render deployment, use the root-level `render.yaml`. It installs this requirements file and starts the API on Render's `PORT`. Add the trained joblib file to the GitHub repository before deploying, then check `/health`, `/docs`, and `/predict` on the Render URL.
