"""Configuration centralisée — charge .env et expose les variables typées."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def assert_langfuse_configured() -> None:
    """Lever une erreur claire si les clés Langfuse manquent."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        raise RuntimeError(
            "Langfuse non configuré. Copier .env.example vers .env "
            "et y coller les clés depuis cloud.langfuse.com (Settings → API Keys)."
        )


def assert_mistral_configured() -> None:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquant dans .env")
