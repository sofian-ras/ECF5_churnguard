import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from churnguard.data import load_data, preprocess
from churnguard.evaluate import compute_metrics
from churnguard.train import train_model


def train_and_register(data_path: str = "data/telco_churn.csv") -> str:
    """Entraîne 3 modèles, enregistre le meilleur et le promeut en Production."""
    df = load_data(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": {"model_name": "lr", "params": {}},
        "RandomForest": {
            "model_name": "rf",
            "params": {"n_estimators": 200, "max_depth": 10},
        },
        "GradientBoosting": {
            "model_name": "gb",
            "params": {"n_estimators": 200, "max_depth": 5},
        },
    }

    mlflow.set_experiment("churnguard")

    results: list[dict[str, float | str]] = []
    run_ids: dict[str, str] = {}

    for run_name, model_info in models.items():
        print(f"Entraînement de {run_name}...")

        with mlflow.start_run(run_name=run_name) as run:
            model = train_model(
                X_train,
                y_train,
                model_info["model_name"],
                model_info["params"],
            )
            metrics = compute_metrics(model, X_test, y_test)

            mlflow.log_params(model_info["params"])
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_train.iloc[:5],
            )

            run_ids[run_name] = run.info.run_id
            results.append({"model": run_name, **metrics})
            print(
                f"{run_name} -> accuracy: {metrics['accuracy']:.3f} | "
                f"roc_auc: {metrics['roc_auc']:.3f}"
            )

    best = max(results, key=lambda x: float(x["roc_auc"]))
    best_name = str(best["model"])
    print(f"\nMeilleur modèle : {best_name}")

    best_run_id = run_ids[best_name]
    registered = mlflow.register_model(f"runs:/{best_run_id}/model", "churnguard")

    client = MlflowClient()
    client.transition_model_version_stage(
        name="churnguard",
        version=registered.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        name="churnguard",
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )

    model_prod = mlflow.pyfunc.load_model("models:/churnguard/Production")
    print("Modèle chargé depuis Production :", type(model_prod))
    return str(registered.version)


if __name__ == "__main__":
    version = train_and_register()
    print(f"Modèle promu en Production (version {version})")