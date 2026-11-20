from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent.parent / "model_output" / "accident_severity_pipeline.joblib"
model_bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Road Accident Severity API",
    description="Predicts recorded accident severity classes 1–4 from supplied conditions.",
    version="1.0.0",
    lifespan=lifespan,
)


class AccidentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time: datetime = Field(examples=["2023-03-15T08:30:00"])
    start_lat: float = Field(ge=-90, le=90, examples=[37.7749])
    start_lng: float = Field(ge=-180, le=180, examples=[-122.4194])
    state: str = Field(min_length=2, max_length=2, examples=["CA"])
    weather_condition: str = "Unknown"
    sunrise_sunset: str = "Unknown"
    temperature_f: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_in: Optional[float] = None
    visibility_mi: Optional[float] = None
    wind_speed_mph: Optional[float] = None
    precipitation_in: Optional[float] = None
    amenity: bool = False
    bump: bool = False
    crossing: bool = False
    give_way: bool = False
    junction: bool = False
    no_exit: bool = False
    railway: bool = False
    roundabout: bool = False
    station: bool = False
    stop: bool = False
    traffic_calming: bool = False
    traffic_signal: bool = False
    turning_loop: bool = False


def request_to_features(request: AccidentRequest) -> pd.DataFrame:
    start = request.start_time
    values = {
        "Weather_Condition": request.weather_condition,
        "Sunrise_Sunset": request.sunrise_sunset,
        "State": request.state.upper(),
        "Temperature(F)": request.temperature_f,
        "Humidity(%)": request.humidity_pct,
        "Pressure(in)": request.pressure_in,
        "Visibility(mi)": request.visibility_mi,
        "Wind_Speed(mph)": request.wind_speed_mph,
        "Precipitation(in)": request.precipitation_in,
        "Start_Lat": request.start_lat,
        "Start_Lng": request.start_lng,
        "Hour": start.hour,
        "DayOfWeek": start.weekday(),
        "Month": start.month,
        "Year": start.year,
        "Is_Weekend": int(start.weekday() >= 5),
        "Amenity": int(request.amenity),
        "Bump": int(request.bump),
        "Crossing": int(request.crossing),
        "Give_Way": int(request.give_way),
        "Junction": int(request.junction),
        "No_Exit": int(request.no_exit),
        "Railway": int(request.railway),
        "Roundabout": int(request.roundabout),
        "Station": int(request.station),
        "Stop": int(request.stop),
        "Traffic_Calming": int(request.traffic_calming),
        "Traffic_Signal": int(request.traffic_signal),
        "Turning_Loop": int(request.turning_loop),
    }
    return pd.DataFrame([values])


@app.get("/")
def root():
    return {
        "message": "Road Accident Severity API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "model_missing", "model_file": str(MODEL_PATH)},
        )
    return {"status": "ready", "model_file": str(MODEL_PATH)}


@app.post("/predict")
def predict(request: AccidentRequest):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained model not found. Run all cells in the notebook first; "
                "it should create outputs/model_output/accident_severity_pipeline.joblib."
            ),
        )

    features = request_to_features(request)
    pipeline = model_bundle["pipeline"]
    offset = int(model_bundle.get("prediction_offset", 0))
    prediction_code = int(pipeline.predict(features)[0])
    predicted_severity = prediction_code + offset

    response = {
        "predicted_severity": predicted_severity,
        "model": model_bundle.get("selected_model", "unknown"),
        "interpretation": "Model estimate from recorded patterns; use to prioritize investigation.",
    }
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)[0]
        model_classes = pipeline.named_steps["model"].classes_
        response["severity_probabilities"] = {
            str(int(class_code) + offset): round(float(probability), 4)
            for class_code, probability in zip(model_classes, probabilities)
        }
    return response
