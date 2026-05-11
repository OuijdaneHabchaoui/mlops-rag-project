"""Smoke test MLflow — verifie que le serveur tracking accepte runs/params/metrics/artifacts."""

import os

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Experience_Fictive_Test")

with mlflow.start_run(run_name="Test_Initial"):
    print("Run demarre.")

    mlflow.log_params(
        {
            "chunk_size": 1000,
            "modele_llm": "mistral-test",
        }
    )

    mlflow.log_metrics(
        {
            "accuracy_score": 0.95,
            "latency_seconds": 1.2,
        }
    )

    artifact_path = "test_artifact.txt"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("Ceci est un fichier test pour valider l'artifact store de MLflow.")
    mlflow.log_artifact(artifact_path)
    os.remove(artifact_path)

print("MLflow OK : run enregistre.")
