"""
Probe script: envoie une question au pipeline et capture tous les evenements SSE bruts.
Sert a prouver les hypotheses sur la cause des echecs.
"""
import json
import requests

BASE_URL = "http://localhost:5010"


def new_conversation(conv_type="rcar"):
    r = requests.post(f"{BASE_URL}/api/v1/conversation/new",
                      json={"title": "probe-test", "conversation_type": conv_type},
                      timeout=10)
    return r.json()["conversation_id"]


def probe(question: str, label: str, conv_type: str = "rcar"):
    conv_id = new_conversation(conv_type)

    print(f"\n{'='*70}")
    print(f"LABEL   : {label}")
    print(f"QUESTION: {question}")
    print(f"{'='*70}")

    events = []
    tokens = []

    with requests.post(
        f"{BASE_URL}/api/v1/query",
        json={"query": question, "conversation_id": conv_id},
        stream=True,
        timeout=180,
        headers={"Accept": "text/event-stream"},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                event = json.loads(data_str)
            except Exception:
                continue

            ev_type = event.get("type", "unknown")
            events.append(event)

            if ev_type == "token":
                tokens.append(event.get("content", ""))
            else:
                # Affiche tous les evenements non-token
                print(f"  [{ev_type}] {json.dumps(event, ensure_ascii=False)[:500]}")

    answer = "".join(tokens).strip()
    sources_events = [e for e in events if e.get("type") == "sources"]
    final_state = next((e for e in events if e.get("type") == "final_state"), None)
    types_recus = [e.get("type") for e in events if e.get("type") != "token"]

    print(f"\n--- RESUME ---")
    print(f"Types evenements (hors tokens) : {types_recus}")
    print(f"Evenement 'sources' emis       : {'OUI' if sources_events else 'NON <- PROUVE: pas de sources'}")
    if sources_events:
        srcs = sources_events[0].get("sources", [])
        print(f"Nombre de chunks recus         : {len(srcs)}")

    if final_state:
        state = final_state.get("state", {})
        print(f"Intent detecte                 : {state.get('intent')}  (tier={state.get('tier')})")
        sub_q = state.get("sub_queries")
        if sub_q:
            print(f"Sous-questions generees        :")
            for sq in sub_q:
                print(f"  -> [{sq.get('intent')}] {sq.get('text', sq.get('question', ''))}")
        else:
            print(f"Sous-questions generees        : aucune")
        print(f"Chunks dans le state           : {len(state.get('retrieved_chunks') or [])}")
        print(f"Sources dans le state          : {len(state.get('sources') or [])}")

    print(f"\nREPONSE FINALE:")
    safe = (answer[:600] if answer else "(vide)").encode('ascii', errors='replace').decode('ascii')
    print(safe)
    print()


if __name__ == "__main__":
    # --- HYPOTHESE 1 ---
    # Ces questions sont classifiees multi_part -> 0 contextes
    probe(
        "Je travaille comme agent temporaire dans une commune depuis 5 ans, est-ce que j ai droit a une retraite ?",
        "GROUPE A (echoue) - doit prouver: intent=multi_part, sources=NON"
    )

    # --- HYPOTHESE 2 ---
    # Ces questions ont un bon intent + 5 contextes mais LLM refuse
    probe(
        "Je viens d etre recrute comme contractuel dans un etablissement public, comment m inscrire pour avoir une retraite ?",
        "GROUPE B (echoue) - doit prouver: intent correct, sources=OUI, mais Je ne retrouve pas"
    )

    # --- CONTROLE ---
    # Cette question fonctionne
    probe(
        "Je suis avocat inscrit au barreau depuis 10 ans, est-ce qu il existe une caisse de retraite speciale pour les avocats au Maroc ?",
        "GROUPE C (fonctionne) - doit prouver: intent correct, sources=OUI, bonne reponse",
        conv_type="cnra"
    )
