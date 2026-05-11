"""
Client HTTP vers l'API RAG `shipping` (FastAPI sur http://localhost:5010).

Contrat d'API découvert lors de la Phase 0 (cf ROADMAP.md D14, D15) :
  1. POST /api/v1/conversation/new   {title, conversation_type ∈ rcar|cnra}
                                     → {conversation_id, conversation_type, message}
  2. POST /api/v1/query              {query, conversation_id}
                                     → flux SSE :
                                         data: {"type":"sources", sources:[{content,...}], ...}
                                         data: {"type":"token", content:"...", ...}    (répété)
                                         data: {"type":"done", conversation_id, exchange_id, error, ...}
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Réponse structurée d'une requête RAG (format Ragas-friendly)."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    sources_metadata: list[dict] = field(default_factory=list)
    latency_seconds: float = 0.0
    conversation_id: Optional[str] = None
    exchange_id: Optional[str] = None
    error: Optional[str] = None
    raw_events: list[dict] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return bool(self.error) or not self.answer


class RAGClient:
    """Client léger pour l'API RAG `shipping`. Une instance = une session HTTP réutilisée."""

    DEFAULT_BASE_URL = "http://localhost:5010"
    VALID_CONV_TYPES = {"rcar", "cnra"}

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = requests.Session()

    def health(self) -> dict:
        """Statut composants (api, database, redis, llm)."""
        r = self.session.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def create_conversation(
        self,
        conversation_type: str = "rcar",
        title: Optional[str] = None,
    ) -> str:
        if conversation_type not in self.VALID_CONV_TYPES:
            raise ValueError(
                f"conversation_type doit être dans {self.VALID_CONV_TYPES}, reçu: {conversation_type}"
            )

        payload = {
            "title": title or f"mlops-eval-{uuid.uuid4().hex[:8]}",
            "conversation_type": conversation_type,
        }
        r = self.session.post(
            f"{self.base_url}/api/v1/conversation/new",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        cid = data["conversation_id"]
        logger.debug("Conversation créée: %s (type=%s)", cid, conversation_type)
        return cid

    def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        conversation_type: str = "rcar",
    ) -> RAGResponse:
        """Pose une question au RAG, parse le SSE, retourne une RAGResponse."""
        if conversation_id is None:
            conversation_id = self.create_conversation(conversation_type=conversation_type)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                resp = self._stream_query(question, conversation_id)
                resp.latency_seconds = time.perf_counter() - start
                return resp
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Query attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)

        raise RuntimeError(
            f"RAG query failed after {self.max_retries} attempts: {last_exc}"
        )

    def _stream_query(self, question: str, conversation_id: str) -> RAGResponse:
        payload = {"query": question, "conversation_id": conversation_id}
        tokens: list[str] = []
        contexts: list[str] = []
        sources_metadata: list[dict] = []
        exchange_id: Optional[str] = None
        final_conv_id: Optional[str] = None
        error: Optional[str] = None
        raw_events: list[dict] = []

        with self.session.post(
            f"{self.base_url}/api/v1/query",
            json=payload,
            stream=True,
            timeout=self.timeout,
            headers={"Accept": "text/event-stream"},
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("SSE non-JSON ignoré: %s", data_str[:80])
                    continue

                raw_events.append(event)
                ev_type = event.get("type")

                if ev_type == "sources":
                    for src in event.get("sources") or []:
                        if isinstance(src, dict) and "content" in src:
                            contexts.append(src["content"])
                            sources_metadata.append(src)
                elif ev_type == "token":
                    content = event.get("content")
                    if content:
                        tokens.append(content)
                elif ev_type == "done":
                    final_conv_id = event.get("conversation_id") or final_conv_id
                    exchange_id = event.get("exchange_id") or exchange_id
                    if event.get("error"):
                        error = event.get("error")

                if exchange_id is None and event.get("exchange_id"):
                    exchange_id = event.get("exchange_id")

        answer = "".join(tokens).strip()
        if not answer or answer.startswith("Une erreur s'est produite"):
            error = error or answer or "Empty answer from RAG"
            answer = ""

        return RAGResponse(
            answer=answer,
            contexts=contexts,
            sources_metadata=sources_metadata,
            conversation_id=final_conv_id or conversation_id,
            exchange_id=exchange_id,
            error=error,
            raw_events=raw_events,
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "RAGClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _cli() -> None:
    """Test rapide en ligne de commande : `python -m pipeline.rag_client "Qu'est-ce que le RCAR ?"`"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test ad-hoc du RAGClient")
    parser.add_argument("question")
    parser.add_argument("--type", choices=["rcar", "cnra"], default="rcar")
    parser.add_argument("--base-url", default=RAGClient.DEFAULT_BASE_URL)
    args = parser.parse_args()

    with RAGClient(base_url=args.base_url) as client:
        h = client.health()
        print(f"Health: {h.get('status')} (uptime {h.get('uptime_seconds', 0):.0f}s)")

        resp = client.query(args.question, conversation_type=args.type)
        print(f"\n=== Réponse ({resp.latency_seconds:.2f}s) ===")
        print(resp.answer if resp.answer else f"[ERREUR: {resp.error}]")
        print(f"\n=== {len(resp.contexts)} sources récupérées ===")
        for i, ctx in enumerate(resp.contexts[:3], 1):
            preview = ctx[:200].replace("\n", " ")
            print(f"  [{i}] ({len(ctx)} chars) {preview}...")
        print(f"\nconversation_id: {resp.conversation_id}")
        print(f"exchange_id: {resp.exchange_id}")


if __name__ == "__main__":
    _cli()
