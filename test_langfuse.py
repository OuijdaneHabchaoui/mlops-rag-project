"""Smoke test Langfuse Cloud — envoie une trace fictive pour valider le setup."""

from langfuse import Langfuse, observe

from config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    assert_langfuse_configured,
)

assert_langfuse_configured()

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

if not langfuse.auth_check():
    raise SystemExit(f"Echec auth Langfuse sur {LANGFUSE_HOST}. Verifie les cles dans .env.")


@observe(name="retrieval")
def fake_retrieval(query: str) -> list[str]:
    return ["chunk_1_RCAR_carte", "chunk_2_RCAR_demarche"]


@observe(name="llm_call", as_type="generation")
def fake_llm(query: str, chunks: list[str]) -> str:
    langfuse.update_current_generation(
        model="ministral-14b-latest",
        input=[{"role": "user", "content": query}],
        usage_details={"input": 150, "output": 80},
        metadata={"temperature": 0.0},
    )
    return "Pour renouveler votre carte RCAR, ..."


@observe(name="smoke_test_rag")
def fake_rag_pipeline(query: str) -> str:
    chunks = fake_retrieval(query)
    return fake_llm(query, chunks)


fake_rag_pipeline("Comment renouveler ma carte RCAR ?")

langfuse.flush()
print(f"Trace envoyee a Langfuse Cloud : {LANGFUSE_HOST}")
print("Va sur https://cloud.langfuse.com -> ton projet -> Tracing -> Traces.")
