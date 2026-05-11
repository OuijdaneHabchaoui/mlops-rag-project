# ROADMAP MLOps RAG — PFE Ouijdane Habchaoui

**Projet :** Cadre MLOps end-to-end pour chatbot RAG RCAR/CNRA (CDG)
**Encadrant faculté :** Pr. Boulouard
**Stage :** CDG
**Périmètre :** MLOps complet (Option A) — évaluation + Docker + CI/CD + monitoring + déploiement
**Dernière mise à jour :** 2026-05-05

---

## 1. Vue d'ensemble

Ce projet construit un **cadre MLOps end-to-end professionnel** autour du chatbot RAG existant (`shipping/`). Le RAG lui-même n'est pas modifié — on l'évalue scientifiquement, on le containerize, on le déploie automatiquement, on le monitore en production.

### Objectifs (couvrant tout le cycle MLOps)
1. **Mesurer** la qualité du RAG (Ragas multi-dimensions, latence, cohérence, robustesse)
2. **Versionner** code + expériences + résultats (Git + MLflow)
3. **Tester automatiquement** à chaque modification (CI GitHub Actions)
4. **Containeriser** l'app pour reproductibilité (Docker)
5. **Déployer automatiquement** en production (CD vers cloud ou serveur CDG)
6. **Monitorer** en temps réel (Prometheus + Grafana + Langfuse + alerting)
7. **Démontrer** au jury la maîtrise complète du cycle MLOps moderne

---

## 2. Architecture COMPLÈTE end-to-end

```
┌─────────────────────────────────────────────────────────────────────┐
│                      VOTRE PC (développement)                       │
│                                                                     │
│  ┌──────────────────┐       ┌────────────────────────────────────┐  │
│  │ shipping/        │ HTTP  │ MLOps_RAG_Project/                 │  │
│  │ uvicorn :8000    │←─────→│ • pipeline/rag_client.py           │  │
│  └──────────────────┘       │ • evaluation/ (5 dimensions)       │  │
│                             │ • experiments/run_full_eval.py     │  │
│  ┌──────────────────┐       │ • tests/                           │  │
│  │ MLflow local     │←──────┤ • docker-compose.yml               │  │
│  │ :5000 (SQLite)   │       └────────────────┬───────────────────┘  │
│  └──────────────────┘                        │                      │
│                                              │ git push             │
└──────────────────────────────────────────────┼──────────────────────┘
                                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       GitHub (cloud)                                │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │ Repo MLOps       │                                               │
│  └────────┬─────────┘                                               │
│           ↓                                                         │
│  ┌──────────────────────────────────────────────────────┐           │
│  │ GitHub Actions (CI/CD)                               │           │
│  │  Stage 1 — CI : lint + tests + évaluations Ragas     │           │
│  │  Stage 2 — Build : image Docker                      │           │
│  │  Stage 3 — Push : Docker Hub                         │           │
│  │  Stage 4 — CD : déploiement automatique              │           │
│  └──────────┬───────────────────────────────────────────┘           │
└─────────────┼───────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│        PRODUCTION (Render free / ou serveur CDG — à confirmer)      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐           │
│  │ shipping/ Docker container — RAG en ligne            │           │
│  │ https://rag-pfe.onrender.com/api/v1/query            │           │
│  └──────────────┬───────────────────────────────────────┘           │
└─────────────────┼───────────────────────────────────────────────────┘
                  │ chaque requête réelle
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       OBSERVABILITÉ (cloud)                         │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │ Langfuse Cloud  │  │ Grafana Cloud    │  │ Sentry (option)  │    │
│  │ traces RAG      │  │ + Prometheus     │  │ erreurs Python   │    │
│  │ qualité convs   │  │ CPU/RAM/latence  │  │                  │    │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘    │
│                                                                     │
│            Alertes Slack/Email en cas d'incident                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Principe :** chaque flèche est automatisée. Aucune action manuelle après le `git push` initial.

---

## 3. Contraintes & adaptations

| Contrainte | Adaptation actée |
|---|---|
| CPU only (pas de GPU) | Reranker Cohere API + Mistral API + Ragas LLM-judge via API. Aucun modèle ML local lourd. |
| Disque C: à 100% (~2.4 GB libres) | Pas de Docker Desktop (3 GB) → **Docker via WSL2** (léger). Pas de Grafana local → **Grafana Cloud** gratuit. Pas de Langfuse self-hosted → **Langfuse Cloud**. |
| Temps limité | Travail en **4 sprints courts** + priorisation. Sprint 1+2 = livrables soutenance minimaux. Sprint 3+4 = MLOps avancé. |
| Pas de carte bancaire pro | Tous les services choisis sont **gratuits sans CB** (Render, Langfuse Cloud, Grafana Cloud, Docker Hub, GitHub). |
| MLflow local | Tracking SQLite pour expériences manuelles. CI exporte aussi en JSON pour persistance GitHub artifacts. |

---

## 4. Stack technique COMPLÈTE

| Catégorie | Outil | Rôle | Hébergement |
|---|---|---|---|
| **Code & versioning** | Git | Versionner code | Local |
|  | GitHub | Cloud du code + collaboration | Cloud (gratuit) |
| **Expérimentation** | MLflow | Tracking expériences (params, métriques, artefacts, model registry) | Local SQLite |
|  | Ragas | Évaluation qualité RAG (6 métriques) | Lib Python locale |
|  | Mistral API | LLM RAG + LLM-judge Ragas | Cloud Mistral |
|  | Cohere API | Reranker | Cloud Cohere |
| **Qualité code** | pytest | Tests unitaires | Local + CI |
|  | black | Formatage Python | Local + CI |
|  | ruff | Linter rapide | Local + CI |
|  | pre-commit | Hooks Git automatiques | Local |
| **CI/CD** | GitHub Actions | Pipelines automatiques | Cloud (gratuit 2000 min/mois) |
|  | Docker | Containerization | Local (WSL2) + CI |
|  | Docker Hub | Registry images | Cloud (gratuit) |
|  | docker-compose | Orchestration multi-services | Local + déploiement |
| **Production** | Render / serveur CDG | Hébergement RAG | Cloud (Render free tier ou CDG) |
| **Observabilité** | Langfuse Cloud | Traces RAG (qualité convs, coûts) | Cloud (50k traces/mois gratuit) |
|  | Prometheus | Métriques système (déjà dans shipping) | Local + Cloud |
|  | Grafana Cloud | Dashboards visuels | Cloud (gratuit 10k metrics) |
|  | Sentry (option) | Erreurs Python | Cloud (5k events/mois gratuit) |
| **Data** | reference_test_set_30.jsonl | Golden set | Repo Git |

---

## 5. Métriques évaluées (vision exhaustive)

### A. Qualité de la réponse (Ragas)
- [ ] Faithfulness — anti-hallucination (cible > 0.85)
- [ ] Answer Relevancy (cible > 0.80)
- [ ] Context Precision (cible > 0.75)
- [ ] Context Recall (cible > 0.80)
- [ ] Answer Correctness (cible > 0.75)
- [ ] Answer Similarity (cible > 0.70)

### B. Performance
- [ ] Latency p50, p95, p99
- [ ] Tokens/sec
- [ ] Coût par requête (USD)
- [ ] Coût total par run d'évaluation

### C. Cohérence conversation (multi-tour)
- [ ] Reference Resolution (pronoms, ellipses)
- [ ] Context Carryover (info des tours précédents)
- [ ] Conversational Coherence

### D. Robustesse
- [ ] Questions hors domaine → refus correct
- [ ] Questions ambiguës → clarification
- [ ] Multilingue (FR / AR / darija)
- [ ] Tolérance aux fautes de frappe

### E. Retrieval pur
- [ ] Hit@K (K=1, 3, 5, 10)
- [ ] MRR (Mean Reciprocal Rank)
- [ ] NDCG@10

### F. Système (production)
- [ ] Taux d'erreur HTTP
- [ ] Disponibilité (uptime API)
- [ ] Throughput (req/s soutenu)
- [ ] CPU/RAM utilisation
- [ ] Alertes incidents

---

## 6. Plan d'exécution — 4 sprints pragmatiques

### 🔵 SPRINT 1 — Fondations & évaluation qualité (~8h, livrable critique)
> **Objectif :** avoir un premier rapport Ragas baseline complet, démontrable au jury.

| Phase | Tâche | Durée | Livrable |
|---|---|---|---|
| 0 | Setup repo + .env + dépendances + verif shipping démarre | 30 min | Environnement OK |
| 1 | `pipeline/rag_client.py` — client HTTP vers shipping | 1 h | Une question → réponse |
| 2 | `evaluation/eval_ragas.py` — qualité (6 métriques) | 2 h | Rapport CSV + JSON |
| 3 | `evaluation/eval_latency.py` — perf (p50/p95/p99) | 1 h | Distribution latence |
| 7a | `experiments/run_full_eval.py` v1 — orchestrateur | 1.5 h | Run unique consolidé |
| - | Intégration MLflow (run baseline) | 1 h | Visible sur :5000 |
| - | Intégration Langfuse (traces baseline) | 1 h | Visible cloud |

**Fin Sprint 1** : vous avez **un rapport Ragas baseline** + traces Langfuse + run MLflow. Vous pouvez déjà montrer quelque chose.

---

### 🟢 SPRINT 2 — Évaluations avancées + expérimentations comparatives (~8h)
> **Objectif :** prouver scientifiquement quelle config RAG est la meilleure.

| Phase | Tâche | Durée | Livrable |
|---|---|---|---|
| 4 | `data/conversation_scenarios.jsonl` (10 dialogues) | 1 h | Dataset multi-tour |
| 4 | `evaluation/eval_conversation.py` (cohérence) | 2 h | Métriques convs |
| 5 | `data/robustness_set.jsonl` + `eval_robustness.py` | 1.5 h | Score robustesse |
| 6 | `evaluation/eval_retrieval.py` (Hit@K, MRR, NDCG) | 1 h | Métriques retrieval |
| 7b | `run_full_eval.py` v2 — orchestrateur complet | 30 min | 5 dimensions consolidées |
| 8 | Expérimentations comparatives (5 expériences clés) | 2 h | ~10 runs MLflow comparés |

**Expériences Sprint 2 :**
1. Baseline (config actuelle)
2. Variation chunk_size (256, 512, 1024)
3. Reranker on/off (Cohere vs sans)
4. top_k variation (3, 5, 10)
5. Variation prompt (2 versions)

**Fin Sprint 2** : le **cœur scientifique du PFE est terminé**. Tableau MLflow présentable.

---

### 🟡 SPRINT 3 — Qualité code + Containerization + CI (~8h)
> **Objectif :** rendre le projet **professionnel** avec tests automatiques.

| Phase | Tâche | Durée | Livrable |
|---|---|---|---|
| 9 | Tests unitaires `tests/` (pytest) sur `pipeline/` et `evaluation/` | 1.5 h | Suite tests verts |
| 10 | Setup `pre-commit` + `black` + `ruff` + `.editorconfig` | 30 min | Code formaté auto |
| 11a | `Dockerfile` pour `shipping/` (basé sur celui existant si présent) | 1.5 h | Image Docker buildable |
| 11b | `Dockerfile` pour `MLOps_RAG_Project/` (eval) | 30 min | Image eval |
| 12 | `docker-compose.yml` (shipping + MLflow + eval) | 1 h | `docker-compose up` lance tout |
| 13a | `.github/workflows/ci.yml` (lint + tests + eval Ragas) | 2 h | CI verte à chaque push |
| 13b | Badge CI dans README | 15 min | Visuel pro |
| - | Documentation `docker-compose` dans README | 45 min | Reproductible |

**Fin Sprint 3** : **CI verte, code professionnel, projet conteneurisé.**

---

### 🟣 SPRINT 4 — CD + Monitoring + Production-ready (~8h)
> **Objectif :** déployer + monitorer + finaliser pour soutenance.

| Phase | Tâche | Durée | Livrable |
|---|---|---|---|
| 14a | Décision finale CD (attendre encadrant CDG) | — | Cible déploiement |
| 14b | `.github/workflows/deploy.yml` (vers Render OU CDG) | 2 h | Déploiement auto |
| 14c | Test end-to-end : push → eval OK → deploy → URL accessible | 1 h | URL publique RAG |
| 15 | Vérifier Prometheus dans `shipping/` exporte bien les métriques | 30 min | `/metrics` accessible |
| 16 | Setup Grafana Cloud + connexion Prometheus | 1.5 h | Dashboards live |
| 16 | Dashboards : latence, CPU, RAM, erreurs, throughput | 1 h | Vues présentables |
| 17 | Alertes Grafana (latence > 5s, erreurs > 5%) | 30 min | Notifications email |
| 18 | MLflow Model Registry — versionner les configs RAG | 1 h | Configs taguées |
| 19 | Validation dataset (Pydantic schema sur jsonl) | 30 min | Erreur claire si mauvais format |
| 20 | README final + diagramme architecture + screenshots | 2 h | Doc soutenance |
| - | Capture vidéo démo end-to-end pour soutenance | 1 h | Vidéo ~3 min |

**Fin Sprint 4** : **MLOps complet déployé + monitoré + documenté.** Prêt pour soutenance.

---

## 7. Structure de fichiers cible (finale)

```
MLOps_RAG_Project/
├── .github/
│   └── workflows/
│       ├── ci.yml                          # Sprint 3
│       └── deploy.yml                      # Sprint 4
├── data/
│   ├── reference_test_set_30.jsonl         # Sprint 1 (copie)
│   ├── conversation_scenarios.jsonl        # Sprint 2
│   └── robustness_set.jsonl                # Sprint 2
├── pipeline/
│   ├── __init__.py
│   └── rag_client.py                       # Sprint 1
├── evaluation/
│   ├── __init__.py
│   ├── eval_ragas.py                       # Sprint 1
│   ├── eval_latency.py                     # Sprint 1
│   ├── eval_conversation.py                # Sprint 2
│   ├── eval_robustness.py                  # Sprint 2
│   └── eval_retrieval.py                   # Sprint 2
├── experiments/
│   ├── __init__.py
│   ├── run_full_eval.py                    # Sprint 1+2
│   ├── exp_chunking.py                     # Sprint 2
│   ├── exp_reranker.py                     # Sprint 2
│   ├── exp_topk.py                         # Sprint 2
│   ├── exp_prompts.py                      # Sprint 2
│   └── runs/                               # résultats horodatés
├── tests/
│   ├── __init__.py
│   ├── test_rag_client.py                  # Sprint 3
│   └── test_evaluation.py                  # Sprint 3
├── reports/                                # HTML générés
├── notebooks/                              # exploration
├── scripts/
│   └── setup_check.py
├── utils/
│   ├── __init__.py
│   └── logger.py
├── docker/
│   ├── Dockerfile.eval                     # Sprint 3
│   └── Dockerfile.shipping                 # Sprint 3 (ou reprise)
├── grafana/
│   └── dashboards/                         # Sprint 4 (JSON exports)
├── docker-compose.yml                      # Sprint 3
├── pyproject.toml                          # Sprint 3 (black/ruff config)
├── .pre-commit-config.yaml                 # Sprint 3
├── config.py
├── requirements.txt
├── requirements-dev.txt                    # Sprint 3 (pytest, black, ruff)
├── .env.example
├── .env                                    # gitignored
├── .gitignore
├── .dockerignore                           # Sprint 3
├── start_mlflow.bat
├── ROADMAP.md                              # ce fichier
└── README.md                               # Sprint 4 (finalisé)
```

---

## 8. État d'avancement

**Légende :** ⬜ pas commencé · 🟨 en cours · ✅ terminé · ⛔ bloqué

### Sprint 1 — Fondations & évaluation qualité
| Phase | Statut | Date début | Date fin | Notes |
|---|---|---|---|---|
| 0 — Setup | ✅ | 2026-05-05 | 2026-05-05 | shipping ✅ tourne via Docker (port **5010**), MLflow ✅, .env complété, RAG répond avec sources + tokens streamés. 401 initial dû à un blip DNS transitoire (résolu après restart). |
| 1 — Client HTTP | ✅ | 2026-05-05 | 2026-05-05 | `pipeline/rag_client.py` opérationnel. Test RCAR : réponse correcte 6.71s, 4 sources. Gère SSE + retries + timeout. |
| 2 — Eval Ragas | 🟨 | 2026-05-05 | — | `evaluation/eval_ragas.py` créé avec 6 métriques + RunConfig (max_workers=1, max_retries=10) pour rate-limit Mistral. Smoke test 3 Q en cours. |
| 3 — Eval latence | ⬜ | — | — | — |
| 7a — Orchestrateur v1 | ⬜ | — | — | — |
| MLflow baseline | ⬜ | — | — | — |
| Langfuse baseline | ⬜ | — | — | — |

### Sprint 2 — Évaluations avancées + expérimentations
| Phase | Statut | Date début | Date fin | Notes |
|---|---|---|---|---|
| 4 — Conversation | ⬜ | — | — | — |
| 5 — Robustesse | ⬜ | — | — | — |
| 6 — Retrieval | ⬜ | — | — | — |
| 7b — Orchestrateur v2 | ⬜ | — | — | — |
| 8 — Expérimentations (5) | ⬜ | — | — | — |

### Sprint 3 — Qualité code + Containerization + CI
| Phase | Statut | Date début | Date fin | Notes |
|---|---|---|---|---|
| 9 — Tests unitaires | ⬜ | — | — | — |
| 10 — pre-commit + black + ruff | ⬜ | — | — | — |
| 11 — Dockerfile | ⬜ | — | — | — |
| 12 — docker-compose | ⬜ | — | — | — |
| 13 — CI GitHub Actions | ⬜ | — | — | — |

### Sprint 4 — CD + Monitoring + Production
| Phase | Statut | Date début | Date fin | Notes |
|---|---|---|---|---|
| 14 — CD (Render ou CDG) | ⬜ | — | — | ⏳ Cible à confirmer encadrant CDG |
| 15 — Prometheus | ⬜ | — | — | Déjà dans shipping, à vérifier |
| 16 — Grafana Cloud | ⬜ | — | — | — |
| 17 — Alerting | ⬜ | — | — | — |
| 18 — MLflow Model Registry | ⬜ | — | — | — |
| 19 — Data validation | ⬜ | — | — | — |
| 20 — Documentation finale | ⬜ | — | — | — |

---

## 9. Décisions architecturales actées

| # | Décision | Justification |
|---|---|---|
| D1 | Couplage HTTP `shipping/` ↔ `MLOps_RAG_Project/` | Découplage propre, simule un vrai client, compatible CI/CD. |
| D2 | MLflow local SQLite | Déjà installé, simple, suffisant pour exp manuelles. CI exporte en JSON. |
| D3 | Langfuse Cloud (pas self-hosted) | Disque C: à 100%, gratuit 50k traces/mois. |
| D4 | Reranker Cohere API | CPU only → reranker local trop lent. Cohere : 200-500 ms. |
| D5 | Golden set existant `reference_test_set_30.jsonl` | Déjà validé, format Ragas-ready. |
| D6 | CI : artifacts JSON+HTML (pas push MLflow distant) | MLflow local non accessible depuis GitHub Actions cloud. |
| D7 | Mistral API comme LLM-judge Ragas | Cohérent avec LLM du RAG, pas de modèle local. |
| D8 | Docker via WSL2 (pas Docker Desktop) | Disque C: limité, WSL2 ~500 MB vs 3 GB. |
| D9 | Grafana Cloud (pas local) | Disque limité, gratuit 10k metrics. |
| D10 | Render free tier comme cloud par défaut pour CD | Pas de carte bancaire, setup 10 min. **À confirmer ou remplacer après réponse encadrant CDG.** |
| D11 | Sprints de ~8h chacun | Contrainte temps : permet livrables intermédiaires démontrables. |
| D12 | Pre-commit hooks (black + ruff) | Standard pro, qualité automatique sans effort. |
| D13 | shipping API tourne sur port **5010** (pas 8000) | Découvert lors Phase 0. Le docker-compose mappe 5010 → app interne. Toutes les requêtes HTTP du MLOps client doivent cibler `http://localhost:5010`. |
| D14 | API contract shipping : conversation-first | `POST /api/v1/conversation/new` (avec `conversation_type ∈ {rcar, cnra}`) → retourne `conversation_id` → puis `POST /api/v1/query` avec `{query, conversation_id}`. Pas de mode "stateless query". |
| D15 | Réponse `/api/v1/query` est en **Server-Sent Events (SSE)** | Le format de retour est `data: {"type":"sources",...}` puis `data: {"type":"token","content":"..."}` répété, puis `data: {"type":"done",...}`. Le client HTTP MLOps doit accumuler les tokens et extraire les sources séparément. |

---

## 10. Risques & mitigations

| Risque | Impact | Mitigation |
|---|---|---|
| API Mistral rate-limit (free tier ≈ 1 req/s) | Métriques Ragas manquantes (TimeoutError) | **Confirmé en pratique :** `RunConfig(max_workers=1, max_retries=10, max_wait=60)` dans `eval_ragas.py`. Coût : ~3-4× plus lent mais TOUTES les métriques calculées. |
| Cohere quota épuisé | Pas de reranker | Fallback `RERANKER_PROVIDER=local` |
| Langfuse quota dépassé | Pas de traces | Sample 10% en CI |
| MLflow SQLite corrompu | Perte historique | Backup `mlflow.db` régulier |
| shipping ne démarre pas | Tout bloqué | Procédure documentée README |
| Docker build échoue en CI | CI rouge | Tester localement avant push |
| Render dort après 15 min | Première requête lente (~30 s) | Acceptable pour démo, mentionner au jury |
| Encadrant CDG tarde à répondre cible déploiement | Sprint 4 retardé | Démarrer avec Render par défaut, switch facile vers CDG après |
| Manque de temps pour Sprint 4 complet | MLOps incomplet | Prioriser : 14 + 16 + 20. Reporter 17/18/19 si nécessaire. |

---

## 11. Critères de succès du PFE

### Minimum viable (Sprints 1+2)
- ✅ Rapport Ragas baseline (6 métriques) sur 30 questions
- ✅ Au moins 5 expérimentations comparatives dans MLflow
- ✅ Évaluations latence + conversation + robustesse + retrieval
- ✅ Traces Langfuse Cloud visibles

### Niveau professionnel (Sprints 1+2+3)
- ✅ Tests unitaires (pytest) verts
- ✅ Code formaté automatiquement (black + ruff + pre-commit)
- ✅ Image Docker buildable + docker-compose fonctionnel
- ✅ CI GitHub Actions verte à chaque push
- ✅ Badge CI dans README

### Excellence MLOps (Sprints 1+2+3+4)
- ✅ Déploiement automatique fonctionnel (URL publique)
- ✅ Dashboards Grafana avec données live
- ✅ Alerting configuré (latence + erreurs)
- ✅ MLflow Model Registry avec ≥ 3 versions taguées
- ✅ README complet + diagramme architecture + vidéo démo

**Cible Ouijdane : Excellence MLOps (les 4 sprints).**

---

## 12. Prochaine action immédiate

**Phase 0 — Setup (Sprint 1, 30 min).**

Avant de démarrer, **3 vérifications** :

1. **shipping démarre ?**
   ```
   cd "shipping (2)/shipping"
   uvicorn app.main:app --port 8000
   ```
   → Voir `Application startup complete` ?

2. **Clés API dans `MLOps_RAG_Project/.env`** :
   - `MISTRAL_API_KEY` ✅ ?
   - `COHERE_API_KEY` ✅ ?
   - `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` ✅ ?

3. **MLflow** : `start_mlflow.bat` lance bien http://127.0.0.1:5000 ?

→ Une fois confirmé, on commit `ROADMAP.md` et on démarre **Phase 1** (`pipeline/rag_client.py`).
