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
from typing import List

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# Variables globales pour stocker le modèle et sa version
model = None
model_version = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère les ressources à l'arrêt."""
    global model, model_version
    model = mlflow.sklearn.load_model("models:/churnguard@Production")
    mv = MlflowClient().get_model_version_by_alias("churnguard", "Production")
    model_version = mv.version
    print(f"Modèle chargé depuis MLflow (version {model_version}) !")
    yield
    print("API arrêtée.")


app = FastAPI(title="ChurnGuard API", lifespan=lifespan)


# Endpoints :

@app.get("/health")
def health():
    """GET /health - Vérifie que l'API est en ligne."""
    return {"status": "ok", "model": "churnguard", "version": model_version}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    """POST /predict - Prédit si un client va churner."""
    try:
        df = pd.DataFrame([customer.model_dump()])
        probability = float(model.predict_proba(df)[0, 1])

        return {
            "churn": probability >= 0.5,
            "probability": probability
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: List[CustomerFeatures]):
    """POST /predict/batch - Prédit le churn pour une liste de clients (max 100)."""
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="La liste est vide")

    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 clients par requête")

    try:
        df = pd.DataFrame([c.model_dump() for c in customers])
        probabilities = model.predict_proba(df)[:, 1]

        return [
            {"churn": float(p) >= 0.5, "probability": float(p)}
            for p in probabilities
        ]

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))