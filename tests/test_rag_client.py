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
        assert RAGClient.VALID_CONV_TYPES == {"rcar", "cnra"}


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
