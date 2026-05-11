@echo off
REM ============================================================
REM Démarrage du serveur MLflow tracking pour MLOps_RAG_Project
REM Backend : SQLite (experiments/mlflow.db) — Artifacts : ./mlartifacts
REM UI accessible sur http://127.0.0.1:5000
REM ============================================================

cd /d "%~dp0"

python -m mlflow server ^
    --backend-store-uri sqlite:///experiments/mlflow.db ^
    --default-artifact-root ./mlartifacts ^
    --host 127.0.0.1 ^
    --port 5000
