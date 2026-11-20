# Road Accident Severity Project

This project contains the ML notebook, a FastAPI prediction service, and a Render deployment file. GitHub stores the project files; Render builds the service from that repository and provides the public API URL. The Kaggle notebook link, if your mentor also requests one, is a separate deliverable.

## 1. Train and save the ML model

Keep the CSV at `C:\Users\srira\Desktop\1US_Accidents_March23.csv`, or update `LOCAL_CSV_PATH` in the notebook. The notebook also searches `/kaggle/input` if the dataset is attached in Kaggle.

Open a terminal in the folder containing this README and the notebook. Run the notebook and all cells. It reads the CSV in chunks, keeps up to 100,000 rows, compares three algorithms, tunes the leading model, and saves:

```text
model_output/accident_severity_pipeline.joblib
```

Run Jupyter from this project folder so the saved model lands where the API expects it.

## 2. Test FastAPI locally

From the project folder:

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --reload
```

Check `http://127.0.0.1:8000/health` for `"status": "ready"`, then open `http://127.0.0.1:8000/docs`. Import `api/postman_collection.json` into Postman and send the **Predict severity** request. Its `baseUrl` variable defaults to `http://127.0.0.1:8000`.

## 3. Upload the project to GitHub

Create a GitHub repository and upload the project contents, including:

- `api/`
- `render.yaml`
- `.python-version`
- `Road_Accident_Severity_ML.ipynb`
- `model_output/accident_severity_pipeline.joblib` (created by running the notebook)

The trained model file must be in the repository for Render to load it at startup. Keep the repository public or grant Render access to it.

## 4. Deploy from GitHub to Render

In Render, create a **Blueprint** and connect the GitHub repository. Render reads `render.yaml`, installs `api/requirements.txt`, starts FastAPI on Render's assigned port, and checks `/health`. The `.python-version` file keeps Render on Python 3.14.8 to match the pinned model libraries.

After a successful deploy, Render provides an HTTPS address such as `https://your-service.onrender.com`. Check:

- Health: `https://your-service.onrender.com/health`
- API docs: `https://your-service.onrender.com/docs`
- Prediction: `https://your-service.onrender.com/predict`

In Postman, change the collection's `baseUrl` variable to your Render address, then send **Predict severity** again. Share the Render `/docs` URL with your mentor as the live API link. The GitHub repository link is the source-code link.

## 5. Model/API details

The request fields and sample JSON are documented in `/docs` and in the Postman collection. The model returns severity 1–4 and probability estimates. Its output is an investigation aid based on recorded patterns, not a claim that the model prevents accidents.
