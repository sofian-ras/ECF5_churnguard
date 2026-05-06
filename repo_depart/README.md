# ChurnGuard MLOps

![CI](https://github.com/sofian-ras/ECF5_churnguard/actions/workflows/ci.yml/badge.svg)

API de prediction de churn client — TelcoFr.

## Architecture

```
git clone → docker compose up → MLflow (port 5000) + API FastAPI (port 8000)
                                          ↓
                              Modele churnguard@Production
```

## Quickstart — lancer toute la stack en 1 commande

```bash
git clone https://github.com/sofian-ras/ECF5_churnguard.git
cd ECF5_churnguard/repo_depart
docker compose up --build
```

- API disponible sur : http://localhost:8000
- MLflow UI disponible sur : http://localhost:5000
- Sante de l'API : http://localhost:8000/health

> Le modele `churnguard@Production` est deja entraine et versionne dans MLflow (dossier `mlruns/`).
> Aucune action supplementaire n'est necessaire pour obtenir des predictions.

---

## Endpoints API

### Health check
```bash
curl http://localhost:8000/health
# {"status":"ok","model":"churnguard","version":"2"}
```

### Prediction unitaire
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.20
  }'
# {"churn":false,"probability":0.18}
```

### Prediction batch (liste de clients, max 100)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":358.20},
    {"gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.70,"TotalCharges":70.70}
  ]'
```

---

## Re-entrainer les modeles (optionnel)

Si vous souhaitez re-lancer l'entrainement (3 modeles : LogisticRegression, RandomForest, GradientBoosting) :

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

Le meilleur modele est automatiquement enregistre et promu `Production` dans MLflow.

---

## Tests

```bash
cd repo_depart
pip install -e .
pip install pytest pytest-cov mlflow fastapi uvicorn pydantic
pytest tests/ --cov=churnguard --cov-fail-under=70 -v
```

6 tests, coverage >= 70%.

---

## Image Docker publique

```bash
docker pull ghcr.io/sofian-ras/churnguard:v1.0.0
```

---

## MLflow UI

![Experiments](data/ui_mlflow_experiments_run.png)
![Production](data/ui_mlflow_version1_alias_prod.png)

---

## Bonus

### Healthcheck Docker

Le conteneur API expose un healthcheck Docker actif.
`docker ps` affiche `(healthy)` une fois l'API demarree.

### Utilisateur non-root

Un utilisateur `appuser` est cree dans le Dockerfile.
La ligne `USER appuser` est presente et activable en production
apres configuration des permissions du volume `mlruns`.

### Monitoring de drift (Evidently)

Compare la distribution des donnees d'entrainement vs un echantillon de production simule :

```bash
pip install evidently
cd repo_depart
python monitoring/drift.py
# => genere drift_report.html
```

### Deploiement Kubernetes (k3s / kind / k3d)

Manifests dans `k8s/` :

```bash
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```

API accessible sur le port `30800` du noeud.

### Notification Slack sur echec CI

Job `notify` dans `.github/workflows/ci.yml` declenche si un job echoue.
Activation : ajouter le secret `SLACK_WEBHOOK_URL` dans Settings → Secrets → Actions du repo GitHub.

### Release notes automatiques (Conventional Commits)

Le workflow `release.yml` genere automatiquement les notes de release
lors d'un push de tag `v*.*.*`. Categories configurees dans `.github/release.yml`.

---

## Licence

MIT.
