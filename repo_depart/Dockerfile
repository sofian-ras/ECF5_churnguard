# stage 1 ------------------ 
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install pandas scikit-learn mlflow fastapi uvicorn pydantic


# stage 2 ------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY churnguard/ ./churnguard/
COPY api/ ./api/
COPY scripts/ ./scripts/

# Variable d'environnement pour utiliser le venv
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]