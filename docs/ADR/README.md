# Architecture Decision Records (ADR)

Ce dossier contient les **décisions architecturales** majeures du projet MLOps_RAG_Project.

## Pourquoi des ADR ?

Un ADR documente :
- **La décision** prise
- **Le contexte** (pourquoi cette décision était nécessaire)
- **Les alternatives** considérées
- **Les conséquences** (positives et négatives)

Cela permet à un nouvel arrivant (ou à soi-même 6 mois plus tard) de comprendre **pourquoi** le code est structuré comme il l'est — sans avoir à reverse-engineer le `git log`.

## Format

Chaque ADR suit la convention [Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) :

```markdown
# ADR-NNNN: Titre de la décision

**Statut** : Acceptée | Proposée | Dépréciée | Remplacée par ADR-XXXX
**Date** : YYYY-MM-DD
**Décideurs** : Ouijdane Habchaoui, [encadrant si applicable]

## Contexte
[Le problème et les contraintes]

## Décision
[Le choix fait]

## Alternatives considérées
[Les autres options écartées et pourquoi]

## Conséquences
### Positives
### Négatives
### Risques
```

## Index des ADR

| # | Titre | Statut | Date |
|---|---|---|---|
| [ADR-0001](0001-mlflow-sqlite-local.md) | MLflow tracking SQLite local | Acceptée | 2026-05-05 |
| [ADR-0002](0002-langfuse-cloud-vs-self-hosted.md) | Langfuse Cloud plutôt que self-hosted | Acceptée | 2026-05-05 |
| [ADR-0003](0003-decouplage-shipping-mlops-http.md) | Découplage HTTP shipping ↔ MLOps_RAG_Project | Acceptée | 2026-05-05 |
| [ADR-0004](0004-judge-llm-gpt4o-mini-vs-mistral.md) | GPT-4o-mini comme judge LLM Ragas | Acceptée | 2026-05-05 |
| [ADR-0005](0005-3-datasets-evaluation-strategy.md) | 3 datasets d'évaluation complémentaires | Acceptée | 2026-05-06 |
| [ADR-0006](0006-docker-multistage-non-root.md) | Dockerfile multi-stage + utilisateur non-root | Acceptée | 2026-05-11 |
| [ADR-0007](0007-ablation-study-methodology.md) | Méthodologie d'ablation pour valider les optimisations | Acceptée | 2026-05-11 |
