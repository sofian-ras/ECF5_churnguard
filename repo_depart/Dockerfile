# stage 1 ------------------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip --no-cache-dir && \
    /opt/venv/bin/pip install --no-cache-dir \
      pandas scikit-learn "mlflow[skinny]" fastapi uvicorn pydantic && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "*.pyo" -delete && \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} \; && \
    find /opt/venv -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true


# stage 2 ------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Utilisateur non-root
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

COPY --from=builder /opt/venv /opt/venv

COPY churnguard/ ./churnguard/
COPY api/ ./api/
COPY scripts/ ./scripts/

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]