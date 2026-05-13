"""Tests unitaires pour pipeline.langfuse_tracker.

Stratégie de mocking :
    - On patch `pipeline.langfuse_tracker.Langfuse` pour remplacer le SDK réel
      par un MagicMock (pas d'appels réseau).
    - On utilise `monkeypatch` pour définir/effacer les vars d'env LANGFUSE_*
      pour tester les branches de configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.langfuse_tracker import LangfuseTracker


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def langfuse_env(monkeypatch):
    """Configure des clés Langfuse valides + retourne un mock Langfuse opérationnel."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-456")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")


@pytest.fixture
def langfuse_no_env(monkeypatch):
    """Efface les vars d'env Langfuse pour tester le mode no-op."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


# ==========================================================================
# TESTS __init__
# ==========================================================================
class TestLangfuseTrackerInit:
    """Tests du constructeur — toutes les branches de configuration."""

    def test_disabled_when_langfuse_not_installed(self, monkeypatch):
        """Si LANGFUSE_AVAILABLE=False → no-op silencieux."""
        monkeypatch.setattr("pipeline.langfuse_tracker.LANGFUSE_AVAILABLE", False)
        tracker = LangfuseTracker(mlflow_run_id="run-123")
        assert tracker.enabled is False
        assert tracker.client is None

    def test_disabled_when_no_keys(self, langfuse_no_env):
        """Pas de clés API → no-op silencieux."""
        tracker = LangfuseTracker(mlflow_run_id="run-123")
        assert tracker.enabled is False
        assert tracker.client is None

    def test_disabled_when_only_public_key(self, monkeypatch):
        """Seulement PUBLIC_KEY sans SECRET_KEY → no-op."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-only")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        tracker = LangfuseTracker()
        assert tracker.enabled is False

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_disabled_when_auth_check_fails(self, mock_langfuse_cls, langfuse_env):
        """Auth Langfuse échoue → no-op silencieux."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = False
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker(session_name="test-session")

        assert tracker.enabled is False
        assert tracker.client is None
        mock_client.auth_check.assert_called_once()

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_enabled_when_all_ok(self, mock_langfuse_cls, langfuse_env):
        """Cas nominal : clés présentes + auth OK → tracker actif."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker(
            mlflow_run_id="mlflow-abc",
            session_name="baseline-v0",
            environment="evaluation",
        )

        assert tracker.enabled is True
        assert tracker.client is mock_client
        assert tracker.mlflow_run_id == "mlflow-abc"
        assert tracker.session_name == "baseline-v0"
        assert tracker.environment == "evaluation"

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_uses_default_host_when_not_set(self, mock_langfuse_cls, monkeypatch):
        """Pas de LANGFUSE_HOST → utilise le default cloud.langfuse.com."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        mock_langfuse_cls.return_value.auth_check.return_value = True

        LangfuseTracker()

        call_kwargs = mock_langfuse_cls.call_args.kwargs
        assert call_kwargs["host"] == "https://cloud.langfuse.com"


# ==========================================================================
# TESTS trace_query (context manager)
# ==========================================================================
class TestTraceQuery:
    """Tests du context manager trace_query."""

    def test_yields_none_when_disabled(self, langfuse_no_env):
        """Tracker désactivé → yields None, pas d'erreur."""
        tracker = LangfuseTracker()
        with tracker.trace_query("question") as span:
            assert span is None

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_yields_span_when_enabled(self, mock_langfuse_cls, langfuse_env):
        """Tracker actif → ouvre une span, la ferme à la sortie."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker()
        with tracker.trace_query("Qu'est-ce que le RCAR ?", category="def") as span:
            assert span is mock_span

        mock_client.start_span.assert_called_once()
        mock_span.end.assert_called_once()

        # Vérifie que les metadata sont bien passées
        call_kwargs = mock_client.start_span.call_args.kwargs
        assert call_kwargs["name"] == "rag_query"
        assert call_kwargs["input"] == {"question": "Qu'est-ce que le RCAR ?"}
        assert call_kwargs["metadata"]["category"] == "def"

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_yields_none_when_start_span_raises(self, mock_langfuse_cls, langfuse_env):
        """Exception dans start_span → yields None, pas de crash."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_client.start_span.side_effect = RuntimeError("Network down")
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker()
        with tracker.trace_query("question") as span:
            assert span is None  # erreur capturée silencieusement


# ==========================================================================
# TESTS attach_response
# ==========================================================================
class TestAttachResponse:
    """Tests de attach_response."""

    def test_noop_when_span_is_none(self, langfuse_no_env):
        """span=None → ne fait rien, pas d'erreur."""
        tracker = LangfuseTracker()
        # Ne doit pas lever d'exception
        tracker.attach_response(span=None, answer="x", contexts=[], latency_seconds=0.1)

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_updates_span_with_response(self, mock_langfuse_cls, langfuse_env):
        """Tracker actif + span valide → appelle span.update() avec les bons args."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker()
        mock_span = MagicMock()

        tracker.attach_response(
            span=mock_span,
            answer="Le RCAR est un régime...",
            contexts=["chunk 1", "chunk 2"],
            latency_seconds=1.23,
        )

        mock_span.update.assert_called_once()
        call_kwargs = mock_span.update.call_args.kwargs
        assert call_kwargs["output"]["answer"] == "Le RCAR est un régime..."
        assert call_kwargs["output"]["contexts_count"] == 2
        assert call_kwargs["output"]["latency_seconds"] == 1.23
        assert call_kwargs["metadata"]["first_context_preview"] == "chunk 1"

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_handles_empty_contexts(self, mock_langfuse_cls, langfuse_env):
        """Contexts vides → first_context_preview=None."""
        mock_langfuse_cls.return_value.auth_check.return_value = True
        tracker = LangfuseTracker()
        mock_span = MagicMock()

        tracker.attach_response(span=mock_span, answer="x", contexts=[], latency_seconds=0.5)

        call_kwargs = mock_span.update.call_args.kwargs
        assert call_kwargs["metadata"]["first_context_preview"] is None

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_swallows_update_exception(self, mock_langfuse_cls, langfuse_env):
        """Exception dans span.update → ne crash pas l'éval."""
        mock_langfuse_cls.return_value.auth_check.return_value = True
        tracker = LangfuseTracker()
        mock_span = MagicMock()
        mock_span.update.side_effect = RuntimeError("API error")

        # Ne doit PAS lever d'exception
        tracker.attach_response(span=mock_span, answer="x", contexts=[], latency_seconds=0.1)


# ==========================================================================
# TESTS log_metrics
# ==========================================================================
class TestLogMetrics:
    """Tests de log_metrics."""

    def test_noop_when_disabled(self, langfuse_no_env):
        """Tracker désactivé → ne fait rien."""
        tracker = LangfuseTracker()
        tracker.log_metrics("run-1", {"faithfulness": 0.85})  # pas d'erreur

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_creates_score_per_metric(self, mock_langfuse_cls, langfuse_env):
        """Tracker actif → 1 appel create_score par métrique."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker(mlflow_run_id="mlflow-xyz")
        metrics = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.92,
            "context_precision": 0.78,
        }
        tracker.log_metrics("baseline-v0-dataset1", metrics)

        assert mock_client.create_score.call_count == 3
        # Vérifie l'un des appels
        first_call = mock_client.create_score.call_args_list[0]
        assert first_call.kwargs["name"] in metrics
        assert "mlflow-xyz" in first_call.kwargs["comment"]

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_swallows_create_score_exception(self, mock_langfuse_cls, langfuse_env):
        """Exception dans create_score → ne crash pas."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_client.create_score.side_effect = RuntimeError("API down")
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker()
        # Ne doit PAS lever d'exception
        tracker.log_metrics("run-x", {"faith": 0.5})


# ==========================================================================
# TESTS flush
# ==========================================================================
class TestFlush:
    """Tests de flush."""

    def test_noop_when_no_client(self, langfuse_no_env):
        """Pas de client → ne fait rien."""
        tracker = LangfuseTracker()
        tracker.flush()  # pas d'erreur

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_calls_client_flush_when_enabled(self, mock_langfuse_cls, langfuse_env):
        """Client présent → appelle client.flush()."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker()
        tracker.flush()

        mock_client.flush.assert_called_once()

    @patch("pipeline.langfuse_tracker.Langfuse")
    def test_swallows_flush_exception(self, mock_langfuse_cls, langfuse_env):
        """Exception dans flush → ne crash pas."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_client.flush.side_effect = RuntimeError("Connection reset")
        mock_langfuse_cls.return_value = mock_client

        tracker = LangfuseTracker()
        # Ne doit PAS lever d'exception
        tracker.flush()
