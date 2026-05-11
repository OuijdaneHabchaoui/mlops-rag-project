"""
Compare baseline runs from MLflow and generate comparison table + graphs.

Lit les 3 experiments dedies (un par dataset) :
    - baseline-reference_test_set_30
    - baseline-natural_language_test_set_20
    - baseline-natural_language_rag_style_20

Usage:
    python -m scripts.compare_baselines

Outputs (in reports/baselines-comparison/):
    - comparison_table.csv      : tableau brut des params + metrics
    - comparison_table.md       : table markdown formatée pour rapport
    - metrics_bar_chart.png     : bar chart des 6 metriques Ragas par dataset
    - metrics_radar_chart.png   : radar chart comparatif
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# Les 3 baselines v0 (etat initial cassi) + les baselines v5 (etat optimisi)
BASELINE_EXPERIMENTS = [
    "baseline-v0-reference_test_set_30",
    "baseline-v0-natural_language_test_set_20",
    "baseline-v0-natural_language_rag_style_20",
]

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_similarity",
]

CONFIG_PARAMS = [
    "dataset",
    "dataset_size",
    "use_query_normalization",
    "use_hyde",
    "use_multi_query_expansion",
    "multi_part_enabled",
    "pertinence_med",
    "pertinence_low",
    "judge_llm",
]


def fetch_runs(experiment_names: list[str]) -> pd.DataFrame:
    """Recupere le run le plus recent de chaque experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    rows = []
    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"[WARN] Experiment '{exp_name}' introuvable, skip")
            continue
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            print(f"[WARN] Aucun run dans '{exp_name}', skip")
            continue
        run = runs.iloc[0]
        row = {
            "experiment": exp_name,
            "run_id": run["run_id"],
            "run_name": run.get("tags.mlflow.runName", ""),
            "start_time": run["start_time"],
        }
        for p in CONFIG_PARAMS:
            row[p] = run.get(f"params.{p}", "")
        for m in RAGAS_METRICS:
            val = run.get(f"metrics.{m}", None)
            row[m] = float(val) if val is not None and not pd.isna(val) else None
        row["rag_collection_seconds"] = run.get("metrics.rag_collection_seconds", None)
        row["rag_errors"] = run.get("metrics.rag_errors", None)
        row["ragas_skipped"] = run.get("metrics.ragas_skipped", None)
        rows.append(row)

    if not rows:
        raise SystemExit("Aucun baseline trouve dans les experiments configures")

    return pd.DataFrame(rows)


def save_csv_and_markdown(df: pd.DataFrame, out_dir: Path) -> None:
    csv_path = out_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] CSV ecrit : {csv_path}")

    short_cols = ["run_name", "dataset"] + RAGAS_METRICS + ["ragas_skipped"]
    short = df[[c for c in short_cols if c in df.columns]].copy()
    for m in RAGAS_METRICS:
        if m in short.columns:
            short[m] = short[m].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    md_path = out_dir / "comparison_table.md"
    md_path.write_text(short.to_markdown(index=False), encoding="utf-8")
    print(f"[OK] Markdown ecrit : {md_path}")


def plot_bar_chart(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df[["dataset"] + RAGAS_METRICS].copy()
    plot_df = plot_df.set_index("dataset")

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Comparaison des metriques Ragas par dataset (baselines)", fontsize=14)
    ax.set_ylabel("Score (0-1)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = out_dir / "metrics_bar_chart.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Bar chart ecrit : {out_path}")


def plot_radar_chart(df: pd.DataFrame, out_dir: Path) -> None:
    angles = np.linspace(0, 2 * np.pi, len(RAGAS_METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for _, row in df.iterrows():
        values = [row[m] if pd.notna(row[m]) else 0 for m in RAGAS_METRICS]
        values += values[:1]
        label = row["dataset"] if row["dataset"] else row["run_name"]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RAGAS_METRICS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Radar comparatif - metriques Ragas par baseline", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    out_path = out_dir / "metrics_radar_chart.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Radar chart ecrit : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baselines MLflow")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=BASELINE_EXPERIMENTS,
        help="Liste des experiments MLflow a comparer (defaut: les 3 baselines)",
    )
    parser.add_argument(
        "--out-name",
        default="baselines-comparison",
        help="Nom du dossier de sortie sous reports/",
    )
    args = parser.parse_args()

    out_dir = REPORTS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {out_dir}")
    print(f"Experiments cibles : {args.experiments}")

    df = fetch_runs(args.experiments)
    print(f"\nBaselines trouvees : {len(df)}")
    print(df[["experiment", "dataset"] + RAGAS_METRICS].to_string(index=False))

    save_csv_and_markdown(df, out_dir)
    plot_bar_chart(df, out_dir)
    plot_radar_chart(df, out_dir)

    print(f"\n=== Comparaison terminee. Tous les outputs dans {out_dir} ===")


if __name__ == "__main__":
    main()
