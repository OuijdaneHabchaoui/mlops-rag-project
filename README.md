# MLOps RAG Project — Chatbot RCAR/CNRA

[![CI](https://github.com/OuijdaneHabchaoui/mlops-rag-project/actions/workflows/ci.yml/badge.svg)](https://github.com/OuijdaneHabchaoui/mlops-rag-project/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red.svg)]()

**Cadre MLOps end-to-end** pour le chatbot RAG des régimes de retraite marocains **RCAR** et **CNRA** (CDG — Caisse de Dépôt et de Gestion).

> **PFE — Ouijdane Habchaoui** · Encadrant faculté : Pr. Boulouard · Stage : CDG · Soutenance : 10 juin 2026

---

## Vue d'ensemble

Ce projet construit l'**infrastructure MLOps complète** autour du chatbot RAG existant (`shipping/`). Le RAG lui-même n'est pas modifié — il est **évalué scientifiquement**, **conteneurisé**, **déployé automatiquement** et **monitoré** en production.

```
┌─────────────────────────────────────────────────────────────┐
│  RAG shipping (FastAPI)  ←──── HTTP ────  MLOps_RAG_Project │
│  - PostgreSQL + pgvector                  - Evaluation Ragas │
│  - LangGraph + Mistral                    - MLflow tracking  │
│  - Reranker Cohere                        - Langfuse tracing │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   GitHub Actions CI/CD             │
        │   - Lint + tests + security        │
        │   - Auto-eval Ragas sur chaque PR  │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   Production (Render / CDG)        │
        │   + Grafana Cloud + Langfuse       │
        └────────────────────────────────────┘
```

---

## Stack MLOps

| Catégorie | Outils | Rôle |
|---|---|---|
| **Évaluation** | Ragas, OpenAI GPT-4o-mini | 6 métriques qualité RAG (judge LLM neutre) |
| **Tracking** | MLflow | Versionnement des expériences (params, métriques, artifacts) |
| **Observabilité** | Langfuse Cloud | Traces LLM en production (requis CDG) |
| **Qualité code** | Ruff, Black, Pre-commit | Linting + formatage automatique |
| **Tests** | Pytest, pytest-cov | Tests unitaires + couverture |
| **Sécurité** | Bandit, pip-audit, Gitleaks | Scans vulnérabilités + détection secrets |
| **CI/CD** | GitHub Actions | Pipeline automatique sur chaque push/PR |
| **Conteneurisation** | Docker (multi-stage), docker-compose | Image production-grade non-root |

---

## Quick start

### Prérequis

- Python 3.11+
- Docker + docker-compose (optionnel pour exécution conteneurisée)
- API actives : Mistral, Cohere, OpenAI, Langfuse Cloud

### 1. Installation

```bash
git clone https://github.com/OuijdaneHabchaoui/mlops-rag-project.git
cd mlops-rag-project

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements-dev.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Editer .env et remplir les cles API
```

### 3. Lancer MLflow tracking

```bash
# Option A : script natif (Windows)
./start_mlflow.bat

# Option B : Docker (multi-OS)
docker compose up -d mlflow
```

UI MLflow : http://127.0.0.1:5000

### 4. Lancer une évaluation Ragas

```bash
python -m evaluation.eval_ragas \
    --experiment baseline-reference_test_set_30 \
    --data-file data/reference_test_set_30.jsonl \
    --delay 8 \
    --rag-timeout 180
```

### 5. Générer le tableau comparatif des baselines

```bash
python -m scripts.compare_baselines
# Outputs dans reports/baselines-comparison/
```

---

## Structure du projet

```
MLOps_RAG_Project/
├── .github/workflows/ci.yml      # Pipeline CI (lint + tests + security + datasets)
├── docker/
│   └── Dockerfile.eval           # Image production multi-stage non-root
├── data/                         # Golden datasets JSONL
│   ├── reference_test_set_30.jsonl
│   ├── natural_language_test_set_20.jsonl
│   └── natural_language_rag_style_20.jsonl
├── evaluation/                   # Évaluation Ragas (6 métriques)
│   └── eval_ragas.py
├── pipeline/                     # Client RAG + observabilité
│   ├── rag_client.py             # Client HTTP vers shipping (SSE parsing)
│   └── langfuse_tracker.py       # Wrapper Langfuse Cloud
├── experiments/                  # MLflow tracking (SQLite + artifacts)
├── scripts/
│   └── compare_baselines.py      # Tableau + graphes comparatifs
├── tests/                        # Tests unitaires pytest
├── docs/                         # Documentation (rapport, ADR)
├── docker-compose.yml            # Orchestration MLflow + eval
├── pyproject.toml                # Config ruff/black/pytest/coverage
├── .pre-commit-config.yaml       # Hooks Git automatiques
└── requirements*.txt             # Dépendances prod + dev
```

---

## Métriques évaluées

### Qualité réponse (Ragas)

| Métrique | Mesure | Cible |
|---|---|---|
| `faithfulness` | Anti-hallucination (réponse ancrée dans les chunks) | > 0.85 |
| `answer_relevancy` | Pertinence par rapport à la question | > 0.80 |
| `context_precision` | Pertinence des chunks récupérés | > 0.75 |
| `context_recall` | Couverture des chunks pertinents | > 0.80 |
| `answer_correctness` | Conformité au ground truth | > 0.75 |
| `answer_similarity` | Similarité sémantique au ground truth | > 0.70 |

### Performance

- Latence p50, p95, p99
- Coût par requête (OpenAI + Mistral + Cohere)
- Throughput (req/s)

---

## Workflow CI/CD

À chaque `push` ou `pull request`, GitHub Actions exécute :

1. **Code quality** — `ruff check` + `black --check`
2. **Tests unitaires** — `pytest` sur Python 3.11 + 3.12 (matrix)
3. **Security scan** — `bandit` + `pip-audit` + `gitleaks`
4. **Validation datasets** — schema JSONL des golden sets
5. **CI Success gate** — bloque le merge si l'un échoue

---

## Développement

### Pre-commit hooks (qualité auto avant chaque commit)

```bash
pre-commit install
# Désormais, chaque `git commit` lance automatiquement :
#   - trailing-whitespace, end-of-file-fixer
#   - ruff (lint + auto-fix)
#   - black (formatage)
#   - bandit (sécurité Python)
#   - gitleaks (détection secrets)
```

### Lancer les tests localement

```bash
pytest tests/ -v --cov=evaluation --cov=pipeline
```

### Build l'image Docker

```bash
docker build -f docker/Dockerfile.eval -t mlops-rag-eval:latest .
```

---

## Documentation

- **`docs/rapport_experimentations.md`** — Rapport complet des 16 expérimentations menées
- **`docs/ADR/`** — Architecture Decision Records (à venir)
- **`ROADMAP.md`** — Plan d'exécution 4 sprints
- **`backups/`** — Snapshots de l'état système avec scripts de restauration

---

## Statut du projet

🟢 **Sprint 1+2 — Évaluation scientifique** : 16 expérimentations, 3 baselines stables
🟡 **Sprint 3 — Qualité code + Containerization + CI** : en cours
⬜ **Sprint 4 — CD + Monitoring + Production** : à venir

---

## Licence

Propriétaire — Projet de fin d'études développé pour la CDG (Caisse de Dépôt et de Gestion, Maroc).

---

## Contact

**Ouijdane Habchaoui** — `ouijdanehabchaoui@gmail.com`
GitHub : [@OuijdaneHabchaoui](https://github.com/OuijdaneHabchaoui)
