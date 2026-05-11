"""Fixtures pytest partagées entre tous les tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_sse_stream() -> list[str]:
    """Stream SSE typique retourné par le RAG shipping.

    Format réel observé : data:{...sources...}, data:{...token...} x N, data:{...done...}
    """
    return [
        'data: {"type":"sources","sources":[{"content":"chunk 1 RCAR","metadata":{"source":"doc1"}},{"content":"chunk 2","metadata":{"source":"doc2"}}],"chunk_count":2}',
        'data: {"type":"token","content":"Le ","metadata":null}',
        'data: {"type":"token","content":"RCAR","metadata":null}',
        'data: {"type":"token","content":" est","metadata":null}',
        'data: {"type":"token","content":" un régime","metadata":null}',
        'data: {"type":"done","conversation_id":"conv-123","exchange_id":"ex-456","error":null}',
    ]


@pytest.fixture
def sample_golden_item() -> dict:
    """Item type du dataset golden Ragas-ready."""
    return {
        "question": "Qu'est-ce que le RCAR ?",
        "expected_answer": "Le RCAR est le Régime Collectif d'Allocation de Retraite.",
        "category": "definition",
        "organization": "RCAR",
        "source_doc": "FAQ_RCAR.pdf",
    }


@pytest.fixture
def project_root() -> Path:
    """Racine du projet MLOps_RAG_Project."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Dossier data/ contenant les golden sets."""
    return project_root / "data"
