"""Bootstrap MLflow model registry for first start.

If `churnguard` has no model in Production, this script:
1) downloads data if missing,
2) trains + logs 3 models,
3) promotes the best one to Production.
"""

import time

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from download_data import DEST as DATA_PATH
from download_data import download
from train_mlflow import train_and_register


def wait_for_mlflow(timeout_seconds: int = 90) -> None:
    """Wait until MLflow tracking server is reachable."""
    client = MlflowClient()
    start = time.time()

    while time.time() - start < timeout_seconds:
        try:
            client.search_experiments(max_results=1)
            return
        except Exception:
            time.sleep(2)

    raise RuntimeError("MLflow server non disponible après attente")


def production_model_exists() -> bool:
    """Return True only if a Production model exists and is loadable."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions("churnguard", stages=["Production"])
        if len(versions) == 0:
            return False

        mlflow.pyfunc.load_model("models:/churnguard/Production")
        return True
    except Exception:
        return False


def bootstrap() -> None:
    """Ensure a Production model exists for API startup."""
    wait_for_mlflow()

    if production_model_exists():
        print("[bootstrap] modèle Production déjà présent")
        return

    print("[bootstrap] aucun modèle Production, initialisation...")

    if not DATA_PATH.exists():
        download()

    version = train_and_register(str(DATA_PATH))
    print(f"[bootstrap] modèle prêt (version {version})")


if __name__ == "__main__":
    bootstrap()
