import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from churnguard.data import load_data, preprocess
from churnguard.evaluate import compute_metrics
from churnguard.train import train_model

df = load_data("data/telco_churn.csv")
X, y = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Les 3 modèles à comparer
models = {
    "LogisticRegression": {"model_name": "lr", "params": {}},
    "RandomForest": {"model_name": "rf", "params": {"n_estimators": 200, "max_depth": 10}},
    "GradientBoosting": {"model_name": "gb", "params": {"n_estimators": 200, "max_depth": 5}},
}

mlflow.set_experiment("churnguard")

results = []
run_ids = {}

for run_name, model_info in models.items():
    print(f"Entraînement de {run_name}...")

    with mlflow.start_run(run_name=run_name) as run:
        # Entraînement
        model = train_model(X_train, y_train, model_info["model_name"], model_info["params"])

        # Métriques
        metrics = compute_metrics(model, X_test, y_test)

        # Log des paramètres et métriques dans MLflow
        mlflow.log_params(model_info["params"])
        mlflow.log_metrics(metrics)

        # Sauvegarde du modèle
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

        run_ids[run_name] = run.info.run_id
        results.append({"model": run_name, **metrics})
        print(f"{run_name} -> accuracy: {metrics['accuracy']:.3f} | roc_auc: {metrics['roc_auc']:.3f}")

# on affiche juste le meilleur modèle
best = max(results, key=lambda x: x["roc_auc"])
print(f"\nMeilleur modèle : {best['model']}")
print(f"roc_auc : {best['roc_auc']:.3f}")
print(f"accuracy : {best['accuracy']:.3f}")


# Enregistrer le meilleur modèle dans le registry
from mlflow.tracking import MlflowClient

best_run_id = run_ids[best["model"]]
registered = mlflow.register_model(f"runs:/{best_run_id}/model", "churnguard")

client = MlflowClient()
client.set_registered_model_alias("churnguard", "Production", registered.version)
print("Modèle promu en Production !")

# Vérifier que le modèle est chargeable depuis Production
model_prod = mlflow.pyfunc.load_model("models:/churnguard@Production")
print("Modèle chargé depuis Production :", type(model_prod))