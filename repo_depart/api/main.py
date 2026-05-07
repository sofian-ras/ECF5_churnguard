"""
ChurnGuard API
API de prédiction de churn client

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Testing:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
"""

import mlflow.sklearn
import pandas as pd
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: Annotated[int, Field(ge=0)]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: Annotated[float, Field(ge=0)]
    TotalCharges: Annotated[float, Field(ge=0)]


# Variables globales pour stocker le modèle et sa version
model = None
model_version = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère les ressources à l'arrêt."""
    global model, model_version
    try:
        model = mlflow.sklearn.load_model("models:/churnguard/Production")
        latest = MlflowClient().get_latest_versions("churnguard", stages=["Production"])
        model_version = latest[0].version if latest else "unknown"
        print(f"Modèle chargé depuis MLflow (version {model_version}) !")
    except Exception as exc:
        model = None
        model_version = None
        print(f"Modèle indisponible au démarrage : {exc}")
    yield
    print("API arrêtée.")


app = FastAPI(title="ChurnGuard API", lifespan=lifespan)


# Endpoints :


@app.get("/health")
def health():
    """GET /health - Vérifie que l'API est en ligne."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "model": "churnguard", "version": model_version}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    """POST /predict - Prédit si un client va churner."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        df = pd.DataFrame([customer.model_dump()])
        probability = float(model.predict_proba(df)[0, 1])

        return {"churn": probability >= 0.5, "probability": probability}

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    """POST /predict/batch - Prédit le churn pour une liste de clients (max 100)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="La liste est vide")

    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 clients par requête")

    try:
        df = pd.DataFrame([c.model_dump() for c in customers])
        probabilities = model.predict_proba(df)[:, 1]

        return [
            {"churn": float(p) >= 0.5, "probability": float(p)} for p in probabilities
        ]

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
