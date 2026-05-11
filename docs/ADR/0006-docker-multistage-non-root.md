# ADR-0006: Dockerfile multi-stage + utilisateur non-root

**Statut** : Acceptée
**Date** : 2026-05-11
**Décideurs** : Ouijdane Habchaoui

## Contexte

Le container d'évaluation Ragas doit être :
- **Reproductible** entre dev local, CI GitHub Actions, et production
- **Léger** pour démarrer vite en CI (économise minutes GitHub Actions)
- **Sécurisé** pour un usage CDG (institution financière, standards stricts)

Un Dockerfile naïf single-stage avec utilisateur root présente plusieurs problèmes :
- Image finale lourde (~900 MB avec compilateurs gcc, build-essential)
- Privilèges root = surface d'attaque maximale en cas de compromission
- Pas de séparation build / runtime → tooling de dev embarqué en production

## Décision

**Multi-stage build avec utilisateur non-root (`mlops:1001`)**.

Architecture :
1. **Stage 1 (builder)** — `python:3.11-slim` + compilateurs → installe deps dans un venv isolé
2. **Stage 2 (runtime)** — `python:3.11-slim` → copie uniquement le venv + le code

Sécurité runtime :
- Utilisateur dédié `mlops:1001` (pas root, pas UID 0)
- `WORKDIR /app` avec ownership explicite `chown=mlops:mlops`
- Healthcheck intégré avec curl vers MLflow

## Alternatives considérées

| Option | Pourquoi écartée |
|---|---|
| **Single-stage + root** | Image 900 MB, surface d'attaque inacceptable pour CDG |
| **Distroless (gcr.io/distroless/python3)** | Pas de shell pour debug en cas de crash, trop strict en phase dev |
| **Alpine Linux** | musl libc incompatible avec certaines wheels Python binaires (numpy, pandas) |
| **scratch** | Trop minimaliste, impossible à debug |

## Conséquences

### Positives
- ✅ Image finale **~250 MB** (vs 900 MB single-stage) → 3.6× plus rapide en CI
- ✅ **Sécurité accrue** : impossibilité d'escalade de privilèges (CIS Docker Benchmark 4.1)
- ✅ **Layer caching optimisé** : `requirements.txt` copié en premier → rebuild rapide sur changement de code
- ✅ Healthcheck Docker → orchestrateur peut redémarrer auto le container si KO
- ✅ Métadonnées OCI (`org.opencontainers.image.*`) → traçabilité version/source

### Négatives
- ⚠️ Build plus long de ~30s (2 étapes) — acceptable
- ⚠️ Plus complexe à comprendre pour un débutant Docker — mitigé par les commentaires inline

### Risques
- 🟢 Aucun majeur. Bonne pratique industry-standard.

## Validation

Tester localement :
```bash
docker build -f docker/Dockerfile.eval -t mlops-rag-eval:latest .
docker run --rm mlops-rag-eval:latest python -c "import ragas; print(ragas.__version__)"
docker scan mlops-rag-eval:latest  # scan vulnérabilités CVE
```

## Références

- [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Docker Top 10](https://owasp.org/www-project-docker-top-10/)
