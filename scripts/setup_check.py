"""Verifie que toute la stack MLOps est operationnelle.

Usage : python scripts/setup_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    MLFLOW_TRACKING_URI,
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "[OK]  " if ok else "[FAIL]"
    print(f"  {mark}  {label}{(' — ' + detail) if detail else ''}")
    return ok


def check_python() -> bool:
    return check(
        "Python >= 3.10", sys.version_info >= (3, 10), f"actuel : {sys.version.split()[0]}"
    )


def check_imports() -> bool:
    results = []
    for pkg in ("mlflow", "langfuse", "dotenv"):
        try:
            __import__(pkg)
            results.append(check(f"import {pkg}", True))
        except ImportError:
            results.append(check(f"import {pkg}", False, "Package non trouvé"))
    return all(results)


def check_mlflow_server() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=3)
        return check(f"serveur MLflow joignable ({MLFLOW_TRACKING_URI})", True)
    except Exception:
        return check(
            f"serveur MLflow joignable ({MLFLOW_TRACKING_URI})",
            False,
            "lance start_mlflow.bat dans un autre terminal",
        )


def check_langfuse_keys() -> bool:
    has_keys = bool(LANGFUSE_PUBLIC_KEY) and bool(LANGFUSE_SECRET_KEY)
    return check(
        f"cles Langfuse presentes (host : {LANGFUSE_HOST})",
        has_keys,
        "copie .env.example -> .env et colle les cles" if not has_keys else "",
    )


def check_langfuse_auth() -> bool:
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return check("Langfuse auth", False, "skip — cles absentes")
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST
        )
        ok = client.auth_check() if hasattr(client, "auth_check") else True
        return check("Langfuse auth_check", bool(ok))
    except Exception as e:
        return check("Langfuse auth_check", False, str(e))


def main() -> int:
    print("== MLOps RAG — setup check ==\n")
    print("[Python & deps]")
    deps_ok = check_python() and check_imports()
    print("\n[MLflow]")
    mlflow_ok = check_mlflow_server()
    print("\n[Langfuse]")
    langfuse_ok = check_langfuse_keys() and check_langfuse_auth()

    print("\n" + "=" * 40)
    if deps_ok and mlflow_ok and langfuse_ok:
        print("Tout est pret. Tu peux lancer test_mlflow.py et test_langfuse.py.")
        return 0
    print("Setup incomplet — corrige les [FAIL] ci-dessus.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
