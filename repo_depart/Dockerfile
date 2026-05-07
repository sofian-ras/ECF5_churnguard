# ── builder : installe les dépendances API dans un venv isolé ──────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip --no-cache-dir && \
    /opt/venv/bin/pip install --no-cache-dir \
      pandas scikit-learn "mlflow-skinny==2.22.0" fastapi uvicorn pydantic && \
    SP=/opt/venv/lib/python3.11/site-packages && \
    # Nettoyage : caches, répertoires de test, outils de build
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf $SP/pip $SP/setuptools && \
    # databricks : stub minimal (mlflow l'importe au démarrage)
    rm -rf $SP/databricks && \
    mkdir -p $SP/databricks/sdk/errors && \
    printf "" > $SP/databricks/__init__.py && \
    printf "" > $SP/databricks/sdk/__init__.py && \
    printf "" > $SP/databricks/sdk/errors/__init__.py && \
    printf "class NotFound(Exception): pass\nclass ResourceDoesNotExist(Exception): pass\n" \
      > $SP/databricks/sdk/errors/platform.py && \
    # Suppression des modules scipy non requis pour l'inférence
    rm -rf $SP/scipy/optimize $SP/scipy/stats $SP/scipy/signal $SP/scipy/fft \
            $SP/scipy/io $SP/scipy/ndimage $SP/scipy/odr $SP/scipy/cluster \
            $SP/scipy/integrate $SP/scipy/interpolate $SP/scipy/spatial $SP/scipy/misc && \
    # Stubs vides pour les modules vraiment inutilisés
    for m in signal fft io ndimage odr cluster misc; do \
      mkdir -p $SP/scipy/$m && printf "# stub\n" > $SP/scipy/$m/__init__.py; \
    done && \
    # Stubs avec les symboles nécessaires au chargement du modèle sklearn
    mkdir -p $SP/scipy/stats && \
    printf 'import numpy as _n\ndef rankdata(a,**k): return _n.argsort(_n.argsort(_n.asarray(a,float)))+1.0\ndef gmean(*a,**k): raise NotImplementedError\ndef spearmanr(*a,**k): raise NotImplementedError\nclass chi2:\n  ppf=staticmethod(lambda *a,**k: None)\n  sf=staticmethod(lambda *a,**k: None)\n' \
      > $SP/scipy/stats/__init__.py && \
    mkdir -p $SP/scipy/interpolate && \
    printf 'class BSpline: pass\n' > $SP/scipy/interpolate/__init__.py && \
    mkdir -p $SP/scipy/integrate && \
    printf 'def trapezoid(*a,**k): raise NotImplementedError\n' > $SP/scipy/integrate/__init__.py && \
    mkdir -p $SP/scipy/optimize && \
    printf 'def minimize(*a,**k): raise NotImplementedError\ndef minimize_scalar(*a,**k): raise NotImplementedError\ndef linear_sum_assignment(*a,**k): raise NotImplementedError\ndef linprog(*a,**k): raise NotImplementedError\n' \
      > $SP/scipy/optimize/__init__.py && \
    printf 'def line_search_wolfe1(*a,**k): raise NotImplementedError\ndef line_search_wolfe2(*a,**k): raise NotImplementedError\n' \
      > $SP/scipy/optimize/_linesearch.py && \
    mkdir -p $SP/scipy/spatial $SP/scipy/spatial/distance && \
    printf 'def cdist(*a,**k): raise NotImplementedError\ndef pdist(*a,**k): raise NotImplementedError\ndef squareform(*a,**k): raise NotImplementedError\n' \
      > $SP/scipy/spatial/__init__.py && \
    printf 'def cdist(*a,**k): raise NotImplementedError\ndef pdist(*a,**k): raise NotImplementedError\n' \
      > $SP/scipy/spatial/distance/__init__.py && \
    # Strip des symboles de debug pour alléger les .so compilés
    find /opt/venv -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true


# ── api : image de production (< 500 Mo) ────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

COPY --from=builder /opt/venv /opt/venv

COPY churnguard/ ./churnguard/
COPY api/ ./api/

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ── venv-bootstrap : dépendances complètes pour l'entraînement ───────────────
FROM python:3.11-slim AS venv-bootstrap

WORKDIR /build

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip --no-cache-dir && \
    /opt/venv/bin/pip install --no-cache-dir \
      pandas scikit-learn mlflow==2.22.0 fastapi uvicorn pydantic && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# ── bootstrap : initialise le registry MLflow au premier démarrage ───────────
FROM python:3.11-slim AS bootstrap

WORKDIR /app

COPY --from=venv-bootstrap /opt/venv /opt/venv

COPY churnguard/ ./churnguard/
COPY scripts/ ./scripts/

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "scripts/bootstrap_model.py"]
