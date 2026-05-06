# ChurnGuard MLOps

![CI](https://github.com/sofian-ras/ECF5_churnguard/actions/workflows/ci.yml/badge.svg)

API de prédiction de churn client — TelcoFr.

## Architecture

```
git clone → docker compose up → MLflow (port 5000) + API FastAPI (port 8000)
                                      ↓
                              Modèle churnguard@Production
```

## Quickstart

```bash
git clone https://github.com/sofian-ras/ECF5_churnguard.git
cd ECF5_churnguard/repo_depart
docker compose up --build
```

L'API est disponible sur `http://localhost:8000`.
MLflow UI est disponible sur `http://localhost:5000`.

## Entraîner les modèles

```bash
docker run --rm \
  --network repo_depart_default \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/mlruns:/app/mlruns" \
  repo_depart-api \
  python scripts/train_mlflow.py
```

## Endpoints API

### Health check
```bash
curl http://localhost:8000/health
```

### Prédiction unitaire
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":358.20}'
```

## Tests

```bash
cd repo_depart
pip install -e .
pytest tests/ --cov=churnguard -v
```

## Image Docker

```bash
docker pull ghcr.io/sofian-ras/churnguard:v1.0.0
```

## MLflow UI

![Experiments](data/ui_mlflow_experiments_run.png)
![Production](data/ui_mlflow_version1_alias_prod.png)

## Bonus

### Monitoring de drift (Evidently)

Génère un rapport HTML de drift entre les données d'entraînement et un échantillon de production simulé :

```bash
pip install evidently
cd repo_depart
python monitoring/drift.py
# => drift_report.html
```

### Déploiement Kubernetes (k3s / kind / k3d)

Les manifests sont dans `k8s/` :

```bash
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```

L'API est accessible sur le port `30800` du nœud.

### Notification Slack sur échec CI

Un job `notify` dans `.github/workflows/ci.yml` envoie une alerte Slack si un job échoue.
Pour l'activer, je dois ajouter un secret `SLACK_WEBHOOK_URL` dans **Settings → Secrets → Actions** du repo GitHub.

## Licence

MIT.