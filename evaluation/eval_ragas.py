"""
Évaluation qualité du RAG via Ragas (6 métriques).

Pipeline :
  1. Charge le golden set (data/reference_test_set_30.jsonl).
  2. Pour chaque question : appelle le RAG via pipeline.rag_client.
  3. Construit un EvaluationDataset Ragas.
  4. Évalue avec Mistral comme LLM-judge + mistral-embed pour les métriques sémantiques.
  5. Logue paramètres + métriques + artefacts dans MLflow.
  6. Sauvegarde rapports JSON/CSV horodatés dans experiments/runs/.

Usage :
  python -m evaluation.eval_ragas                # full 30 questions
  python -m evaluation.eval_ragas --limit 3      # smoke test rapide
  python -m evaluation.eval_ragas --skip-ragas   # collecte seule, sans Ragas
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force UTF-8 sur stdout/stderr (Windows cp1252 fait crasher MLflow sur les emojis).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  charge .env via config.py
from pipeline.rag_client import RAGClient  # noqa: E402

DATA_FILE_DEFAULT = PROJECT_ROOT / "data" / "reference_test_set_30.jsonl"
RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"

JUDGE_LLM_MODEL = "gpt-4o-mini"
JUDGE_EMBED_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


def load_golden(data_file: Path, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def conversation_type_for(row: dict) -> str:
    org = (row.get("organization") or "").lower()
    return "rcar" if "rcar" in org else "cnra"


def collect_rag_responses(questions: list[dict], base_url: str, delay: float = 5.0, rag_timeout: float = 180.0) -> list[dict]:
    """Appelle le RAG pour chaque question. Continue malgré les erreurs.
    delay: secondes d'attente entre chaque question (évite la surcharge serveur).
    rag_timeout: timeout HTTP par requête (s).
    """
    results: list[dict] = []
    with RAGClient(base_url=base_url, timeout=rag_timeout) as client:
        for i, row in enumerate(questions, 1):
            q = row["question"]
            gt = row["expected_answer"]
            ctype = conversation_type_for(row)
            logger.info("[%d/%d] (%s) %s", i, len(questions), ctype, q[:70])
            try:
                resp = client.query(q, conversation_type=ctype)
                status = "OK" if resp.answer else f"VIDE({resp.error})"
                logger.info("  -> %s | %.1fs | %d ctx", status, resp.latency_seconds, len(resp.contexts or []))
                results.append({
                    "question": q,
                    "answer": resp.answer,
                    "contexts": resp.contexts or [],
                    "ground_truth": gt,
                    "latency_seconds": resp.latency_seconds,
                    "conversation_id": resp.conversation_id,
                    "exchange_id": resp.exchange_id,
                    "category": row.get("category", ""),
                    "organization": row.get("organization", ""),
                    "source_doc": row.get("source_doc", ""),
                    "error": resp.error,
                })
            except Exception as exc:
                logger.error("Q%d ERREUR: %s", i, exc)
                results.append({
                    "question": q,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": gt,
                    "latency_seconds": 0.0,
                    "conversation_id": None,
                    "exchange_id": None,
                    "category": row.get("category", ""),
                    "organization": row.get("organization", ""),
                    "source_doc": row.get("source_doc", ""),
                    "error": str(exc),
                })
            if i < len(questions) and delay > 0:
                time.sleep(delay)
    return results


def build_ragas_dataset(results: list[dict]):
    """Filtre les erreurs et construit un EvaluationDataset Ragas (0.4 API)."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    skipped = 0
    for r in results:
        if r.get("error") or not r["answer"] or not r["contexts"]:
            skipped += 1
            continue
        samples.append(SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        ))
    if not samples:
        raise RuntimeError("Aucune réponse RAG exploitable pour Ragas (toutes en erreur ou vides).")
    logger.info("Ragas dataset: %d samples valides, %d skipped", len(samples), skipped)
    return EvaluationDataset(samples=samples), skipped


def setup_openai_judge():
    """Wrappers Ragas autour de OpenAI gpt-4o-mini + text-embedding-3-small."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant — vérifier MLOps_RAG_Project/.env")

    llm = ChatOpenAI(
        model=JUDGE_LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        model=JUDGE_EMBED_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def run_ragas(dataset, max_workers: int = 2, timeout: int = 180, max_wait: int = 15):
    """
    Lance les 6 métriques Ragas avec le judge Mistral.

    `max_workers` est volontairement bas (2) pour rester sous le rate-limit Mistral
    free tier (1 req/s) et obtenir TOUTES les métriques sans timeout.
    `max_wait` à 15s (au lieu de 60) pour ne pas bloquer sur les 429 free tier.
    """
    from ragas import evaluate
    try:
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity,
        )
    except ImportError:
        from ragas.metrics import (  # fallback pre-0.2
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity,
        )
    from ragas.run_config import RunConfig

    llm_wrapper, emb_wrapper = setup_openai_judge()
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity,
    ]
    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=15,
        max_wait=max_wait,
    )
    return evaluate(
        dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=emb_wrapper,
        run_config=run_config,
        raise_exceptions=False,
    )


def latency_stats(results: list[dict]) -> dict:
    lats = [r["latency_seconds"] for r in results if r["latency_seconds"] > 0]
    if not lats:
        return {"latency_count": 0}
    s = pd.Series(lats)
    return {
        "latency_count": len(lats),
        "latency_mean": float(s.mean()),
        "latency_median": float(s.median()),
        "latency_p95": float(s.quantile(0.95)),
        "latency_max": float(s.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation Ragas du RAG")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite de questions (smoke test)")
    parser.add_argument("--base-url", default="http://localhost:5010",
                        help="URL du RAG shipping")
    parser.add_argument("--experiment", default="ragas-baseline",
                        help="Nom de l'expérience MLflow")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Collecter seulement, sans évaluation Ragas")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Workers parallèles Ragas (bas = moins de 429 Mistral)")
    parser.add_argument("--ragas-timeout", type=int, default=240,
                        help="Timeout (s) par job Ragas")
    parser.add_argument("--max-wait", type=int, default=15,
                        help="Attente max entre retries Ragas (s) — 15 pour free tier Mistral")
    parser.add_argument("--from-responses", type=str, default=None,
                        help="Charger les réponses RAG depuis un fichier JSON (sauter la collecte)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Délai (s) entre chaque question RAG pour éviter la surcharge serveur (défaut: 5)")
    parser.add_argument("--rag-timeout", type=float, default=180.0,
                        help="Timeout HTTP (s) par requête RAG (défaut: 180)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Chemin vers le dataset JSONL (défaut: data/reference_test_set_30.jsonl)")
    parser.add_argument("--rag-normalization", type=str, default=None,
                        help="Valeur USE_QUERY_NORMALIZATION du container RAG (true/false)")
    parser.add_argument("--rag-hyde", type=str, default=None,
                        help="Valeur USE_HYDE du container RAG (true/false)")
    parser.add_argument("--rag-multi-query", type=str, default=None,
                        help="Valeur USE_MULTI_QUERY_EXPANSION du container RAG (true/false)")
    args = parser.parse_args()

    DATA_FILE = Path(args.data_file) if args.data_file else DATA_FILE_DEFAULT

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)

    norm_flag = os.getenv("USE_QUERY_NORMALIZATION", "unknown")
    run_label = f"norm_{norm_flag}_{timestamp}"
    with mlflow.start_run(run_name=run_label) as run:

        # 1. Collecte RAG (ou chargement depuis fichier)
        if args.from_responses:
            logger.info("=== ÉTAPE 1/3 : Chargement réponses depuis %s ===", args.from_responses)
            with open(args.from_responses, "r", encoding="utf-8") as f:
                results = json.load(f)
            if args.limit:
                results = results[:args.limit]
            rag_seconds = 0.0
            logger.info("Chargé %d réponses (collecte sautée)", len(results))
        else:
            questions = load_golden(DATA_FILE, limit=args.limit)
            logger.info("Loaded %d questions from %s", len(questions), DATA_FILE.name)
            logger.info("=== ÉTAPE 1/3 : Collecte des réponses RAG ===")
            t0 = time.perf_counter()
            results = collect_rag_responses(questions, args.base_url, delay=args.delay, rag_timeout=args.rag_timeout)
            rag_seconds = time.perf_counter() - t0
            logger.info("Collecte terminée en %.1fs — %d erreurs",
                        rag_seconds, sum(1 for r in results if r.get("error")))

        errors = sum(1 for r in results if r.get("error"))
        mlflow.log_params({
            "dataset": DATA_FILE.name,
            "dataset_size": len(results),
            "rag_base_url": args.base_url,
            "judge_llm": JUDGE_LLM_MODEL,
            "judge_embeddings": JUDGE_EMBED_MODEL,
            "skip_ragas": args.skip_ragas,
            "from_responses": args.from_responses or "",
            "use_query_normalization": args.rag_normalization or os.getenv("USE_QUERY_NORMALIZATION", "unknown"),
            "use_hyde": args.rag_hyde or os.getenv("USE_HYDE", "unknown"),
            "use_multi_query_expansion": args.rag_multi_query or os.getenv("USE_MULTI_QUERY_EXPANSION", "unknown"),
            "multi_part_enabled": os.getenv("MULTI_PART_ENABLED", "unknown"),
            "pertinence_med": os.getenv("PERTINENCE_MED", "unknown"),
            "pertinence_low": os.getenv("PERTINENCE_LOW", "unknown"),
            "rag_delay_between_questions": args.delay,
            "rag_timeout": args.rag_timeout,
        })

        # Stats latence
        lat_stats = latency_stats(results)
        for k, v in lat_stats.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("rag_collection_seconds", rag_seconds)
        mlflow.log_metric("rag_errors", errors)
        mlflow.log_metric("rag_success", len(results) - errors)

        responses_file = run_dir / "rag_responses.json"
        responses_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(responses_file))

        # 2. Évaluation Ragas
        if args.skip_ragas:
            logger.info("Ragas évaluation sautée (--skip-ragas).")
            return

        logger.info("=== ÉTAPE 2/3 : Évaluation Ragas (OpenAI judge) ===")
        try:
            dataset, skipped = build_ragas_dataset(results)
            mlflow.log_metric("ragas_skipped", skipped)

            t0 = time.perf_counter()
            scores = run_ragas(dataset, max_workers=args.max_workers, timeout=args.ragas_timeout, max_wait=args.max_wait)
            ragas_seconds = time.perf_counter() - t0
            mlflow.log_metric("ragas_seconds", ragas_seconds)
            logger.info("Ragas terminé en %.1fs", ragas_seconds)

            scores_df = scores.to_pandas()
            scores_csv = run_dir / "ragas_scores.csv"
            scores_df.to_csv(scores_csv, index=False, encoding="utf-8")
            mlflow.log_artifact(str(scores_csv))

            metric_cols = [
                c for c in scores_df.columns
                if c not in {"user_input", "retrieved_contexts", "response", "reference"}
                and pd.to_numeric(scores_df[c], errors="coerce").notna().any()
            ]
            summary = {}
            for col in metric_cols:
                vals = pd.to_numeric(scores_df[col], errors="coerce").dropna()
                if not vals.empty:
                    mean_val = float(vals.mean())
                    summary[col] = mean_val
                    mlflow.log_metric(col, mean_val)
                    logger.info("  %s : %.3f", col, mean_val)

            (run_dir / "ragas_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            mlflow.log_artifact(str(run_dir / "ragas_summary.json"))

            logger.info("=== ÉTAPE 3/3 : Run MLflow loggé ===")
            logger.info("Run ID : %s", run.info.run_id)
            logger.info("Artifacts: %s", run_dir)
        except Exception as exc:
            logger.exception("Échec Ragas : %s", exc)
            mlflow.log_param("ragas_error", str(exc)[:200])
            raise


if __name__ == "__main__":
    main()
