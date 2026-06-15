import re


def extract_last_ai_text(messages: list) -> str:
    """Récupère le dernier message assistant avec du texte (évite fin de run ambiguë)."""
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        c = m.content
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            c = "".join(parts)
        if isinstance(c, str) and c.strip():
            return c
    for m in reversed(messages):
        c = getattr(m, "content", None)
        if isinstance(c, list):
            continue
        if isinstance(c, str) and c.strip():
            return c
    return ""


def strip_final_answer_prefix(text: str) -> str:
    """Retire le préfixe 'FINAL ANSWER:' pour ne garder que la valeur brute à soumettre."""
    marker = "FINAL ANSWER:"
    stripped = text.strip()
    if stripped.upper().startswith(marker):
        return stripped[len(marker):].strip()
    return stripped


def normalize_gaia_answer(raw: str) -> str:
    """normalise la sortie du LLM en un format uniforme FINAL ANSWER: <valeur>"""
    if not raw:
        return "FINAL ANSWER: "
    text = raw.strip()
    marker = "FINAL ANSWER:"
    idx = text.lower().rfind(marker.lower())
    if idx != -1:
        tail = text[idx + len(marker) :].strip()
        first = tail.split("\n", 1)[0].strip() if tail else ""
        first = re.sub(r"[*_`]+", "", first).strip()
        return f"{marker} {first}".strip() if first else f"{marker} "
    first_line = text.split("\n", 1)[0].strip()
    first_line = re.sub(r"[*_`]+", "", first_line).strip()
    return f"{marker} {first_line}"