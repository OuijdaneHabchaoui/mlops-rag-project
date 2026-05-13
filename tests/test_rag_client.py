"""Tests unitaires pour pipeline.rag_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.rag_client import RAGClient, RAGResponse


class TestRAGResponse:
    """Tests du dataclass RAGResponse."""

    def test_is_error_when_empty_answer(self):
        resp = RAGResponse(answer="")
        assert resp.is_error is True

    def test_is_error_when_error_set(self):
        resp = RAGResponse(answer="some answer", error="timeout")
        assert resp.is_error is True

    def test_not_error_when_valid_response(self):
        resp = RAGResponse(answer="Le RCAR est...", contexts=["chunk 1"])
        assert resp.is_error is False

    def test_default_lists_are_empty(self):
        resp = RAGResponse(answer="x")
        assert resp.contexts == []
        assert resp.sources_metadata == []
        assert resp.raw_events == []

    def test_default_lists_are_independent(self):
        """Vérifie qu'on n'a pas le bug du mutable default."""
        r1 = RAGResponse(answer="a")
        r2 = RAGResponse(answer="b")
        r1.contexts.append("x")
        assert r2.contexts == []


class TestRAGClientInit:
    """Tests du constructeur RAGClient."""

    def test_default_base_url(self):
        client = RAGClient()
        assert client.base_url == "http://localhost:5010"

    def test_strips_trailing_slash(self):
        client = RAGClient(base_url="http://example.com/")
        assert client.base_url == "http://example.com"

    def test_custom_timeout(self):
        client = RAGClient(timeout=60.0)
        assert client.timeout == 60.0

    def test_valid_conversation_types(self):
        assert {"rcar", "cnra"} == RAGClient.VALID_CONV_TYPES


class TestCreateConversation:
    """Tests de create_conversation()."""

    def test_invalid_conversation_type_raises(self):
        client = RAGClient()
        with pytest.raises(ValueError, match="conversation_type"):
            client.create_conversation(conversation_type="invalid")

    @patch("pipeline.rag_client.requests.Session")
    def test_create_conversation_returns_id(self, mock_session_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "conversation_id": "conv-abc-123",
            "conversation_type": "rcar",
            "message": "ok",
        }
        mock_response.raise_for_status = MagicMock()
        mock_session = mock_session_cls.return_value
        mock_session.post.return_value = mock_response

        client = RAGClient()
        cid = client.create_conversation(conversation_type="rcar", title="test")

        assert cid == "conv-abc-123"
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args.kwargs
        assert call_kwargs["json"]["conversation_type"] == "rcar"
        assert call_kwargs["json"]["title"] == "test"


class TestSSEParsing:
    """Tests du parsing des événements SSE (cœur fragile du client)."""

    @patch("pipeline.rag_client.requests.Session")
    def test_parses_sources_tokens_and_done(self, mock_session_cls, sample_sse_stream):
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = sample_sse_stream
        mock_response.raise_for_status = MagicMock()
        # __enter__/__exit__ pour le context manager `with self.session.post(...) as r`
        mock_session = mock_session_cls.return_value
        mock_session.post.return_value.__enter__.return_value = mock_response

        client = RAGClient()
        resp = client._stream_query("Test question", "conv-123")

        # On a accumulé tous les tokens en une réponse cohérente
        assert "RCAR" in resp.answer
        assert "régime" in resp.answer
        # On a récupéré les 2 sources
        assert len(resp.contexts) == 2
        assert "chunk 1 RCAR" in resp.contexts[0]
        # Les IDs de fin sont parsés
        assert resp.conversation_id == "conv-123"
        assert resp.exchange_id == "ex-456"
        assert resp.error is None
        # On capture tous les events bruts pour debug
        assert len(resp.raw_events) >= 3

    @patch("pipeline.rag_client.requests.Session")
    def test_ignores_empty_lines(self, mock_session_cls):
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            "",
            "data:",  # ligne vide après le préfixe
            'data: {"type":"token","content":"X"}',
            'data: {"type":"done","conversation_id":"c","exchange_id":"e","error":null}',
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session = mock_session_cls.return_value
        mock_session.post.return_value.__enter__.return_value = mock_response

        client = RAGClient()
        resp = client._stream_query("Q", "c")
        assert resp.answer == "X"

    @patch("pipeline.rag_client.requests.Session")
    def test_ignores_non_json_data_lines(self, mock_session_cls):
        """Lignes data: avec JSON invalide sont ignorées sans crash."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            "data: not a valid json {{{",
            'data: {"type":"token","content":"hello"}',
            'data: {"type":"done","conversation_id":"c","exchange_id":"e","error":null}',
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session = mock_session_cls.return_value
        mock_session.post.return_value.__enter__.return_value = mock_response

        client = RAGClient()
        resp = client._stream_query("Q", "c")
        assert resp.answer == "hello"
        assert resp.error is None

    @patch("pipeline.rag_client.requests.Session")
    def test_captures_error_event(self, mock_session_cls):
        """Event 'done' avec error → la RAGResponse contient l'erreur."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            'data: {"type":"token","content":"part"}',
            'data: {"type":"done","conversation_id":"c","exchange_id":"e","error":"LLM timeout"}',
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session = mock_session_cls.return_value
        mock_session.post.return_value.__enter__.return_value = mock_response

        client = RAGClient()
        resp = client._stream_query("Q", "c")
        assert resp.error == "LLM timeout"

    @patch("pipeline.rag_client.requests.Session")
    def test_empty_answer_becomes_error(self, mock_session_cls):
        """Aucun token reçu → error renseigné."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            'data: {"type":"done","conversation_id":"c","exchange_id":"e","error":null}',
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session = mock_session_cls.return_value
        mock_session.post.return_value.__enter__.return_value = mock_response

        client = RAGClient()
        resp = client._stream_query("Q", "c")
        assert resp.answer == ""
        assert resp.error  # quelque chose est renseigné


# ==========================================================================
# TESTS health()
# ==========================================================================
class TestHealth:
    """Tests de health()."""

    @patch("pipeline.rag_client.requests.Session")
    def test_health_returns_json(self, mock_session_cls):
        """health() retourne le JSON de l'API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "uptime_seconds": 123}
        mock_response.raise_for_status = MagicMock()
        mock_session = mock_session_cls.return_value
        mock_session.get.return_value = mock_response

        client = RAGClient()
        result = client.health()

        assert result == {"status": "ok", "uptime_seconds": 123}
        mock_session.get.assert_called_once()
        # Vérifie l'URL appelée
        call_args = mock_session.get.call_args
        assert "/health" in call_args.args[0]


# ==========================================================================
# TESTS query() avec retry
# ==========================================================================
class TestQueryRetry:
    """Tests de la logique de retry de query()."""

    @patch("pipeline.rag_client.time.sleep")  # accélère le test (pas de vraie attente)
    @patch("pipeline.rag_client.RAGClient._stream_query")
    @patch("pipeline.rag_client.RAGClient.create_conversation")
    def test_query_creates_conv_when_missing(self, mock_create_conv, mock_stream, mock_sleep):
        """conversation_id=None → create_conversation est appelé."""
        mock_create_conv.return_value = "auto-created-conv-id"
        mock_stream.return_value = RAGResponse(answer="ok", contexts=["ctx"])

        client = RAGClient()
        resp = client.query("test question", conversation_id=None)

        mock_create_conv.assert_called_once_with(conversation_type="rcar")
        assert resp.answer == "ok"

    @patch("pipeline.rag_client.time.sleep")
    @patch("pipeline.rag_client.RAGClient._stream_query")
    def test_query_first_attempt_success(self, mock_stream, mock_sleep):
        """Premier essai OK → pas de retry."""
        mock_stream.return_value = RAGResponse(answer="ok", contexts=["c"])

        client = RAGClient()
        resp = client.query("q", conversation_id="conv-1")

        assert resp.answer == "ok"
        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pipeline.rag_client.time.sleep")
    @patch("pipeline.rag_client.RAGClient._stream_query")
    def test_query_retries_on_failure(self, mock_stream, mock_sleep):
        """Échec puis succès → 2 tentatives, 1 sleep."""
        import requests

        ok_resp = RAGResponse(answer="ok", contexts=["c"])
        mock_stream.side_effect = [requests.ConnectionError("net down"), ok_resp]

        client = RAGClient(max_retries=3)
        resp = client.query("q", conversation_id="conv-1")

        assert resp.answer == "ok"
        assert mock_stream.call_count == 2
        mock_sleep.assert_called_once()

    @patch("pipeline.rag_client.time.sleep")
    @patch("pipeline.rag_client.RAGClient._stream_query")
    def test_query_raises_after_max_retries(self, mock_stream, mock_sleep):
        """Toutes les tentatives échouent → RuntimeError."""
        import requests

        mock_stream.side_effect = requests.ConnectionError("net down")

        client = RAGClient(max_retries=3)
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.query("q", conversation_id="conv-1")

        assert mock_stream.call_count == 3


# ==========================================================================
# TESTS close() + context manager
# ==========================================================================
class TestContextManager:
    """Tests du context manager (with RAGClient() as c)."""

    def test_close_closes_session(self):
        """close() appelle session.close()."""
        client = RAGClient()
        client.session = MagicMock()
        client.close()
        client.session.close.assert_called_once()

    def test_context_manager_closes_on_exit(self):
        """`with` ferme automatiquement la session."""
        with RAGClient() as client:
            client.session = MagicMock()
            session_mock = client.session
        session_mock.close.assert_called_once()

    def test_enter_returns_self(self):
        """__enter__ retourne l'instance pour le `with X as y`."""
        client = RAGClient()
        with client as c:
            assert c is client
