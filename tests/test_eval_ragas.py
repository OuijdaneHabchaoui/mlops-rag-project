"""Tests unitaires pour evaluation.eval_ragas.

Stratégie :
    - Fonctions pures (load_golden, latency_stats, conversation_type_for) : tests
      directs sans mocks lourds (pandas est installé).
    - Imports lourds (mlflow, ragas, langchain_openai) : injectés comme MagicMock
      dans sys.modules AVANT l'import de eval_ragas, pour éviter d'avoir besoin
      de les installer en local. Le CI utilisera les vraies libs via requirements.txt.
    - Fonctions qui appellent l'API RAG : on patch RAGClient.
    - main() : non testé en unitaire (orchestrateur, validé par tests d'intégration).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------
# Injection des libs lourdes AVANT l'import de eval_ragas.
# Cela permet de tester sans installer mlflow/ragas/langchain-openai en local.
# --------------------------------------------------------------------------
for _mod_name in (
    "mlflow",
    "ragas",
    "ragas.metrics",
    "ragas.metrics.collections",
    "ragas.embeddings",
    "ragas.llms",
    "ragas.run_config",
    "langchain_openai",
):
    sys.modules.setdefault(_mod_name, MagicMock())

# Maintenant on peut importer eval_ragas sans crash
from evaluation.eval_ragas import (  # noqa: E402
    build_ragas_dataset,
    collect_rag_responses,
    conversation_type_for,
    latency_stats,
    load_golden,
    run_ragas,
    setup_openai_judge,
)


# ==========================================================================
# TESTS load_golden
# ==========================================================================
class TestLoadGolden:
    """Tests du chargement JSONL."""

    def test_loads_all_lines(self, tmp_path: Path):
        """Charge toutes les lignes JSONL valides."""
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"question": "Q1", "expected_answer": "A1"}\n'
            '{"question": "Q2", "expected_answer": "A2"}\n'
            '{"question": "Q3", "expected_answer": "A3"}\n',
            encoding="utf-8",
        )
        rows = load_golden(f)
        assert len(rows) == 3
        assert rows[0]["question"] == "Q1"
        assert rows[2]["expected_answer"] == "A3"

    def test_ignores_empty_lines(self, tmp_path: Path):
        """Lignes vides au milieu du fichier ignorées."""
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"question": "Q1", "expected_answer": "A1"}\n'
            "\n"
            '{"question": "Q2", "expected_answer": "A2"}\n'
            "\n",
            encoding="utf-8",
        )
        rows = load_golden(f)
        assert len(rows) == 2

    def test_respects_limit(self, tmp_path: Path):
        """Le paramètre limit tronque à N lignes."""
        f = tmp_path / "test.jsonl"
        lines = [json.dumps({"question": f"Q{i}", "expected_answer": f"A{i}"}) for i in range(10)]
        f.write_text("\n".join(lines), encoding="utf-8")

        rows = load_golden(f, limit=3)
        assert len(rows) == 3
        assert rows[0]["question"] == "Q0"
        assert rows[2]["question"] == "Q2"

    def test_no_limit_returns_all(self, tmp_path: Path):
        """limit=None → toutes les lignes."""
        f = tmp_path / "test.jsonl"
        lines = [json.dumps({"question": f"Q{i}", "expected_answer": f"A{i}"}) for i in range(5)]
        f.write_text("\n".join(lines), encoding="utf-8")

        rows = load_golden(f, limit=None)
        assert len(rows) == 5

    def test_raises_when_file_missing(self, tmp_path: Path):
        """Fichier inexistant → FileNotFoundError."""
        f = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_golden(f)


# ==========================================================================
# TESTS conversation_type_for
# ==========================================================================
class TestConversationTypeFor:
    """Tests du dispatcher rcar/cnra basé sur organization."""

    def test_rcar_in_organization(self):
        assert conversation_type_for({"organization": "RCAR"}) == "rcar"

    def test_rcar_lowercase(self):
        assert conversation_type_for({"organization": "rcar"}) == "rcar"

    def test_rcar_in_longer_string(self):
        assert conversation_type_for({"organization": "RCAR Maroc"}) == "rcar"

    def test_cnra_returns_cnra(self):
        assert conversation_type_for({"organization": "CNRA"}) == "cnra"

    def test_unknown_org_defaults_to_cnra(self):
        """Organization absent ou inconnu → cnra par défaut."""
        assert conversation_type_for({"organization": ""}) == "cnra"
        assert conversation_type_for({}) == "cnra"
        assert conversation_type_for({"organization": "Autre"}) == "cnra"


# ==========================================================================
# TESTS latency_stats (utilise pandas)
# ==========================================================================
class TestLatencyStats:
    """Tests des statistiques de latence."""

    def test_empty_results(self):
        """Pas de latences → seulement le compte à 0."""
        assert latency_stats([]) == {"latency_count": 0}

    def test_all_zero_latencies_ignored(self):
        """Latences à 0 ignorées (erreurs)."""
        results = [
            {"latency_seconds": 0.0},
            {"latency_seconds": 0.0},
        ]
        assert latency_stats(results) == {"latency_count": 0}

    def test_computes_stats(self):
        """Stats calculées sur latences > 0."""
        results = [
            {"latency_seconds": 1.0},
            {"latency_seconds": 2.0},
            {"latency_seconds": 3.0},
            {"latency_seconds": 4.0},
            {"latency_seconds": 5.0},
        ]
        stats = latency_stats(results)
        assert stats["latency_count"] == 5
        assert stats["latency_mean"] == 3.0
        assert stats["latency_median"] == 3.0
        assert stats["latency_max"] == 5.0
        assert stats["latency_p95"] >= 4.0  # p95 ≈ 4.8

    def test_mixes_zero_and_nonzero(self):
        """Les latences nulles sont exclues du calcul."""
        results = [
            {"latency_seconds": 0.0},
            {"latency_seconds": 2.0},
            {"latency_seconds": 4.0},
            {"latency_seconds": 0.0},
        ]
        stats = latency_stats(results)
        assert stats["latency_count"] == 2
        assert stats["latency_mean"] == 3.0


# ==========================================================================
# TESTS build_ragas_dataset
# ==========================================================================
class TestBuildRagasDataset:
    """Tests de la construction du dataset Ragas (filtrage erreurs).

    Note : ragas est mocké via sys.modules → EvaluationDataset et SingleTurnSample
    sont des MagicMock. On vérifie donc la LOGIQUE de filtrage, pas la structure
    interne du dataset Ragas.
    """

    def test_filters_errored_results(self):
        """Résultats avec 'error' sont skippés."""
        results = [
            {
                "question": "Q1",
                "answer": "A1",
                "contexts": ["ctx1"],
                "ground_truth": "GT1",
                "error": None,
            },
            {
                "question": "Q2",
                "answer": "",
                "contexts": [],
                "ground_truth": "GT2",
                "error": "timeout",
            },
        ]
        _, skipped = build_ragas_dataset(results)
        assert skipped == 1

    def test_filters_empty_answers(self):
        """Réponses vides (sans error explicite) sont skippées."""
        results = [
            {
                "question": "Q1",
                "answer": "A1",
                "contexts": ["ctx1"],
                "ground_truth": "GT1",
                "error": None,
            },
            {
                "question": "Q2",
                "answer": "",
                "contexts": ["ctx2"],
                "ground_truth": "GT2",
                "error": None,
            },
        ]
        _, skipped = build_ragas_dataset(results)
        assert skipped == 1

    def test_filters_empty_contexts(self):
        """Réponses sans contexts sont skippées (Ragas a besoin de contexts).

        Note : on garde 1 résultat valide pour éviter le RuntimeError
        levé quand TOUS les résultats sont invalides.
        """
        results = [
            {
                "question": "Q1",
                "answer": "A1",
                "contexts": ["ctx valide"],
                "ground_truth": "GT1",
                "error": None,
            },
            {
                "question": "Q2",
                "answer": "A2",
                "contexts": [],  # vide → skippé
                "ground_truth": "GT2",
                "error": None,
            },
        ]
        _, skipped = build_ragas_dataset(results)
        assert skipped == 1

    def test_raises_when_no_valid_samples(self):
        """Tous les résultats invalides → RuntimeError."""
        results = [
            {
                "question": "Q1",
                "answer": "",
                "contexts": [],
                "ground_truth": "GT1",
                "error": "timeout",
            },
        ]
        with pytest.raises(RuntimeError, match="Aucune réponse RAG exploitable"):
            build_ragas_dataset(results)

    def test_valid_results_pass_filter(self):
        """3 résultats valides → 0 skipped."""
        results = [
            {
                "question": f"Q{i}",
                "answer": f"A{i}",
                "contexts": [f"ctx{i}"],
                "ground_truth": f"GT{i}",
                "error": None,
            }
            for i in range(3)
        ]
        _, skipped = build_ragas_dataset(results)
        assert skipped == 0


# ==========================================================================
# TESTS setup_openai_judge
# ==========================================================================
class TestSetupOpenAIJudge:
    """Tests de la config du judge OpenAI."""

    @patch("evaluation.eval_ragas.config")
    def test_raises_when_no_api_key(self, mock_config):
        """OPENAI_API_KEY manquant → RuntimeError."""
        mock_config.OPENAI_API_KEY = None
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY manquant"):
            setup_openai_judge()

    @patch("evaluation.eval_ragas.config")
    def test_returns_wrappers_when_key_set(self, mock_config):
        """Clé API présente → retourne 2 objets (les wrappers).

        Note : langchain_openai et ragas.* sont mockés via sys.modules,
        donc les "wrappers" sont en fait des MagicMock. On vérifie juste
        que la fonction retourne 2 valeurs non-None.
        """
        mock_config.OPENAI_API_KEY = "sk-test-fake-key"
        result = setup_openai_judge()
        assert result is not None
        assert len(result) == 2
        llm_w, emb_w = result
        assert llm_w is not None
        assert emb_w is not None


# ==========================================================================
# TESTS collect_rag_responses
# ==========================================================================
class TestCollectRagResponses:
    """Tests de la collecte des réponses RAG (mock RAGClient)."""

    @patch("evaluation.eval_ragas.RAGClient")
    def test_collects_successful_responses(self, mock_client_cls):
        """2 questions → 2 résultats avec answer + contexts."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.answer = "Réponse mockée"
        mock_response.contexts = ["ctx 1", "ctx 2"]
        mock_response.latency_seconds = 1.5
        mock_response.conversation_id = "conv-1"
        mock_response.exchange_id = "ex-1"
        mock_response.error = None
        mock_client.query.return_value = mock_response

        questions = [
            {"question": "Q1", "expected_answer": "GT1", "organization": "RCAR"},
            {"question": "Q2", "expected_answer": "GT2", "organization": "CNRA"},
        ]
        results = collect_rag_responses(questions, "http://test", delay=0.0)

        assert len(results) == 2
        assert results[0]["answer"] == "Réponse mockée"
        assert results[0]["contexts"] == ["ctx 1", "ctx 2"]
        assert results[0]["latency_seconds"] == 1.5
        assert results[0]["error"] is None

    @patch("evaluation.eval_ragas.RAGClient")
    def test_continues_on_exception(self, mock_client_cls):
        """Exception sur 1 question → résultat error mais continue."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        ok_response = MagicMock()
        ok_response.answer = "OK"
        ok_response.contexts = ["ctx"]
        ok_response.latency_seconds = 1.0
        ok_response.conversation_id = "c"
        ok_response.exchange_id = "e"
        ok_response.error = None

        mock_client.query.side_effect = [RuntimeError("Timeout!"), ok_response]

        questions = [
            {"question": "Q1", "expected_answer": "GT1", "organization": "RCAR"},
            {"question": "Q2", "expected_answer": "GT2", "organization": "RCAR"},
        ]
        results = collect_rag_responses(questions, "http://test", delay=0.0)

        assert len(results) == 2
        assert results[0]["answer"] == ""
        assert "Timeout!" in results[0]["error"]
        assert results[1]["answer"] == "OK"

    @patch("evaluation.eval_ragas.RAGClient")
    def test_passes_correct_conversation_type(self, mock_client_cls):
        """conversation_type passé selon organization."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_resp = MagicMock(answer="x", contexts=[], latency_seconds=0.1, error=None)
        mock_client.query.return_value = mock_resp

        questions = [
            {"question": "Q1", "expected_answer": "GT", "organization": "CNRA"},
        ]
        collect_rag_responses(questions, "http://test", delay=0.0)

        call_kwargs = mock_client.query.call_args.kwargs
        assert call_kwargs["conversation_type"] == "cnra"


# ==========================================================================
# TESTS run_ragas
# ==========================================================================
class TestRunRagas:
    """Tests de l'orchestration Ragas (ragas mocké via sys.modules)."""

    @patch("evaluation.eval_ragas.setup_openai_judge")
    def test_calls_evaluate(self, mock_setup):
        """run_ragas appelle ragas.evaluate avec le dataset et les 6 métriques."""
        mock_setup.return_value = (MagicMock(), MagicMock())

        # ragas.evaluate est déjà un MagicMock via sys.modules
        import ragas

        ragas.evaluate.reset_mock()
        ragas.evaluate.return_value = MagicMock()

        dataset = MagicMock()
        run_ragas(dataset, max_workers=1, timeout=60, max_wait=5)

        ragas.evaluate.assert_called_once()
        # dataset est passé en argument POSITIONNEL, le reste en kwargs
        call_args = ragas.evaluate.call_args
        assert call_args.args[0] is dataset
        assert "metrics" in call_args.kwargs
        assert len(call_args.kwargs["metrics"]) == 6
        assert call_args.kwargs["raise_exceptions"] is False
