# ADR-0001: MLflow tracking en SQLite local (phase dev)

**Statut** : Acceptée — temporaire (sera remplacée par PostgreSQL en production)
**Date** : 2026-05-05
**Décideurs** : Ouijdane Habchaoui

## Contexte

Le projet nécessite un système de tracking d'expériences ML pour :
- Logger les paramètres de chaque évaluation (config RAG, dataset, etc.)
- Logger les métriques Ragas (faithfulness, answer_relevancy, etc.)
- Versionner les artifacts (rag_responses.json, ragas_scores.csv)
- Permettre la comparaison visuelle entre runs

Plusieurs backends MLflow sont disponibles : SQLite local, MySQL, PostgreSQL, file-based.

### Contraintes

- **Disque local limité** sur le PC de dev (~2.4 GB libres seulement)
- **CPU only** — pas de GPU, donc pas besoin de partage cluster
- **Phase de développement** — usage mono-utilisateur, pas d'accès concurrent

## Décision

**SQLite local** (`experiments/mlflow.db`) avec artifacts dans `mlartifacts/`.

Backend store URI : `sqlite:///experiments/mlflow.db`
Artifact root : `./mlartifacts`

## Alternatives considérées

| Option | Pourquoi écartée |
|---|---|
| **PostgreSQL local** | Nécessite installation + 100+ MB de RAM permanent, overkill pour mono-user |
| **File-based** (mlruns/) | Pas de requêtes SQL, perfs dégradées au-delà de 100 runs |
| **MLflow Cloud** (Databricks) | Payant, pas d'usage CDG en cours |
| **PostgreSQL existant du shipping** | Mélange les responsabilités (RAG data + MLOps tracking) |

## Conséquences

### Positives
- ✅ Setup en 0 minute — pas d'installation supplémentaire
- ✅ Fichier unique facile à backup
- ✅ Performance suffisante pour < 1000 runs (notre cas)
- ✅ Compatible avec les artifacts locaux (pas besoin de S3)

### Négatives
- ⚠️ Pas de partage multi-utilisateurs (mais pas requis en phase dev)
- ⚠️ Pas adapté à un déploiement production CDG

### Risques
- 🟡 Corruption SQLite possible si interruption brutale → **Mitigation** : backups réguliers de `mlflow.db`
- 🟡 Migration future PostgreSQL requise → **Mitigation** : MLflow fournit `mlflow db upgrade` pour migrer

## Migration future (Sprint 4)

Quand le projet passera en mode production CDG, migrer vers PostgreSQL :
```bash
mlflow db upgrade postgresql://user:pwd@host/mlflow_db
```
Voir [ADR-XXXX] (à créer) pour la décision de migration.
