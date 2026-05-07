# ChurnGuard MLOps

![CI](https://github.com/sofian-ras/ECF5_churnguard/actions/workflows/ci.yml/badge.svg)

API FastAPI de prédiction de churn + tracking MLflow.

## Quickstart (prof-friendly)

```bash
git clone https://github.com/sofian-ras/ECF5_churnguard.git
cd ECF5_churnguard/repo_depart
docker compose up --build
```

Au premier lancement, l'API initialise automatiquement MLflow (téléchargement dataset + entraînement + promotion en `Production`) si aucun modèle n'existe encore.

- API : http://localhost:8000
- Health : http://localhost:8000/health
- MLflow UI : http://localhost:5000

## Architecture

```text
docker compose up
 ├─ mlflow (5000)
 └─ api (8000)
      └─ charge models:/churnguard/Production
```

## Endpoints

### GET /health

```bash
curl http://localhost:8000/health
```

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Male",
    "SeniorCitizen":0,
    "Partner":"Yes",
    "Dependents":"No",
    "tenure":12,
    "PhoneService":"Yes",
    "MultipleLines":"No",
    "InternetService":"DSL",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"No",
    "StreamingMovies":"No",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":29.85,
    "TotalCharges":358.20
  }'
```

### POST /predict/batch (max 100)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":358.20}]'
```

## Entraînement manuel (optionnel)

```bash
python scripts/train_mlflow.py
```

Bonus CLI demandé :

```bash
python -m churnguard.train --model rf --register
```

## Tests

```bash
pip install -e .
pip install pytest pytest-cov
pytest tests/ --cov=churnguard --cov-fail-under=70 -v
```

## Image Docker publique

```bash
docker pull ghcr.io/sofian-ras/churnguard:v1.0.0
```

## Captures MLflow

![Experiments](data/ui_mlflow_experiments_run.png)
![Production](data/ui_mlflow_version1_alias_prod.png)

## Bonus

- Healthcheck Docker sur `mlflow` et `api`.
- Monitoring drift : `python monitoring/drift.py`.
- Manifests Kubernetes dans `../k8s`.
- Notification Slack sur échec CI (`SLACK_WEBHOOK_URL`).
- Release notes automatiques sur tag `v*.*.*`.

## Licence

MIT
