"""Tests de validation des datasets golden — format JSONL Ragas-ready."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REQUIRED_FIELDS = {"question", "expected_answer"}
OPTIONAL_FIELDS = {"category", "organization", "source_doc"}


@pytest.fixture
def all_datasets(data_dir: Path) -> list[Path]:
    """Liste tous les fichiers JSONL dans data/."""
    return sorted(data_dir.glob("*.jsonl"))


def test_data_directory_exists(data_dir: Path):
    assert data_dir.exists(), f"Le dossier data/ doit exister : {data_dir}"


def test_at_least_three_golden_datasets(all_datasets: list[Path]):
    """Le projet doit avoir au minimum les 3 datasets baselines."""
    names = {p.name for p in all_datasets}
    expected = {
        "reference_test_set_30.jsonl",
        "natural_language_test_set_20.jsonl",
        "natural_language_rag_style_20.jsonl",
    }
    missing = expected - names
    assert not missing, f"Datasets golden manquants : {missing}"


@pytest.mark.parametrize(
    "dataset_name",
    [
        "reference_test_set_30.jsonl",
        "natural_language_test_set_20.jsonl",
        "natural_language_rag_style_20.jsonl",
    ],
)
def test_dataset_is_valid_jsonl(data_dir: Path, dataset_name: str):
    """Chaque ligne doit être un JSON valide."""
    path = data_dir / dataset_name
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"{dataset_name}:L{line_num} — JSON invalide : {e}")


@pytest.mark.parametrize(
    "dataset_name",
    [
        "reference_test_set_30.jsonl",
        "natural_language_test_set_20.jsonl",
        "natural_language_rag_style_20.jsonl",
    ],
)
def test_dataset_has_required_fields(data_dir: Path, dataset_name: str):
    """Chaque item doit contenir les champs obligatoires Ragas."""
    path = data_dir / dataset_name
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            missing = REQUIRED_FIELDS - set(obj.keys())
            assert not missing, f"{dataset_name}:L{line_num} — champs requis manquants : {missing}"


def test_reference_dataset_has_30_questions(data_dir: Path):
    path = data_dir / "reference_test_set_30.jsonl"
    with open(path, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    assert n == 30, f"reference_test_set_30 doit avoir 30 questions, trouvé : {n}"


@pytest.mark.parametrize(
    "dataset_name",
    [
        "natural_language_test_set_20.jsonl",
        "natural_language_rag_style_20.jsonl",
    ],
)
def test_natural_datasets_have_20_questions(data_dir: Path, dataset_name: str):
    path = data_dir / dataset_name
    with open(path, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    assert n == 20, f"{dataset_name} doit avoir 20 questions, trouvé : {n}"


def test_no_empty_answers(data_dir: Path, all_datasets: list[Path]):
    """Aucun expected_answer ne doit être vide."""
    for path in all_datasets:
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                answer = obj.get("expected_answer", "").strip()
                assert answer, f"{path.name}:L{line_num} — expected_answer vide"
