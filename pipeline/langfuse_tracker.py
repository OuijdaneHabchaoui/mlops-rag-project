"""
Langfuse observability — wrapper production-grade pour tracer les évaluations RAG.

Architecture :
    - Le RAGClient existant ne change PAS (séparation des responsabilités).
    - Cette classe LangfuseTracker enveloppe les appels au RAG et envoie les traces
      à Langfuse Cloud avec le run_id MLflow comme metadata (cross-link).
    - Si Langfuse n'est pas configuré (pas de clés), le tracker devient un no-op
      silencieux — l'éval continue normalement.

Usage :
    from pipeline.langfuse_tracker import LangfuseTracker

    tracker = LangfuseTracker(mlflow_run_id="abc123", session_name="baseline-dataset1")
    with tracker.trace_query("Qu'est-ce que le RCAR ?", category="rcar") as span:
        response = rag_client.query("Qu'est-ce que le RCAR ?")
        tracker.attach_response(span, response)
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Import safe : si langfuse n'est pas installé, on tombe en no-op
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]
    LANGFUSE_AVAILABLE = False


class LangfuseTracker:
    """Wrapper Langfuse pour observabilité des évaluations RAG.

    Devient automatiquement un **no-op silencieux** si :
        - langfuse n'est pas installé
        - les clés LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY ne sont pas définies
        - l'auth Langfuse échoue (réseau coupé, etc.)

    Le tracker est SAFE à utiliser dans toutes les situations.
    """

    def __init__(
        self,
        mlflow_run_id: Optional[str] = None,
        session_name: Optional[str] = None,
        environment: str = "evaluation",
    ) -> None:
        self.mlflow_run_id = mlflow_run_id
        self.session_name = session_name
        self.environment = environment
        self.enabled = False
        self.client: Optional[Any] = None

        if not LANGFUSE_AVAILABLE:
            logger.info("Langfuse non installe — tracking desactive (no-op)")
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.info("LANGFUSE_PUBLIC_KEY/SECRET_KEY non definis — tracking desactive (no-op)")
            return

        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            if not self.client.auth_check():
                logger.warning("Auth Langfuse echouee — tracking desactive (no-op)")
                self.client = None
                return
            self.enabled = True
            logger.info("Langfuse tracker actif — host=%s session=%s", host, session_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Echec init Langfuse : %s — tracking desactive (no-op)", exc)
            self.client = None

    @contextmanager
    def trace_query(
        self,
        question: str,
        category: Optional[str] = None,
        expected_answer: Optional[str] = None,
    ) -> Iterator[Optional[Any]]:
        """Ouvre une trace pour UNE question RAG. Auto-close à la sortie du with.

        Yields l'objet trace (ou None si Langfuse desactive).
        """
        if not self.enabled or self.client is None:
            yield None
            return

        try:
            trace = self.client.start_span(
                name="rag_query",
                input={"question": question},
                metadata={
                    "mlflow_run_id": self.mlflow_run_id,
                    "session": self.session_name,
                    "environment": self.environment,
                    "category": category,
                    "expected_answer": expected_answer,
                },
            )
            yield trace
            trace.end()
        except Exception as exc:
            logger.warning("Erreur Langfuse trace_query : %s", exc)
            yield None

    def attach_response(
        self,
        span: Optional[Any],
        answer: str,
        contexts: list[str],
        latency_seconds: float,
        error: Optional[str] = None,
    ) -> None:
        """Attache la reponse RAG a la trace en cours."""
        if span is None or not self.enabled:
            return

        try:
            span.update(
                output={
                    "answer": answer,
                    "contexts_count": len(contexts),
                    "latency_seconds": latency_seconds,
                    "error": error,
                },
                metadata={
                    "first_context_preview": contexts[0][:200] if contexts else None,
                },
            )
        except Exception as exc:
            logger.warning("Erreur attach_response : %s", exc)

    def log_metrics(self, run_name: str, metrics: dict[str, float]) -> None:
        """Logue les metriques Ragas finales comme un score Langfuse."""
        if not self.enabled or self.client is None:
            return

        try:
            for metric_name, value in metrics.items():
                self.client.create_score(
                    name=metric_name,
                    value=float(value),
                    comment=f"Run: {run_name}, MLflow: {self.mlflow_run_id}",
                )
        except Exception as exc:
            logger.warning("Erreur log_metrics : %s", exc)

    def flush(self) -> None:
        """Force l'envoi des traces buffered vers Langfuse Cloud."""
        if self.client is not None:
            try:
                self.client.flush()
            except Exception as exc:
                logger.warning("Erreur flush Langfuse : %s", exc)
