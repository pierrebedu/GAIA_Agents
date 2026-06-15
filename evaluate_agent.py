import argparse
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agent import build_graph
from regexs import extract_last_ai_text, normalize_gaia_answer, strip_final_answer_prefix

BASE = os.getenv("SCORING_API_URL", "https://agents-course-unit4-scoring.hf.space").rstrip("/")


def file_name(item: dict) -> str:
    return (item.get("file_name") or "").strip()


def has_file(item: dict) -> bool:
    return bool(file_name(item))


TOOL_USE_FAILED = os.getenv("TOOL_USE_FAILED", "tool_use_failed")
MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))


_GAIA_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaia_files")


def answer(graph, item: dict, cfg: dict) -> str:
    msg = f"question: {item['question']}"
    if fn := file_name(item):
        full_path = os.path.join(_GAIA_FILES_DIR, fn)
        msg += f"\nfile_path: {full_path}"

    out = graph.invoke({"messages": [HumanMessage(content=msg)]}, config=cfg)
    raw = normalize_gaia_answer(extract_last_ai_text(out["messages"]))
    return strip_final_answer_prefix(raw)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep", type=float, default=float(os.getenv("GROQ_EVAL_SLEEP_SECONDS", "2")))
    args = p.parse_args()

    questions = requests.get(f"{BASE}/questions", timeout=30).json()
    if args.limit:
        questions = questions[: args.limit]

    graph = build_graph()
    base_cfg = {"recursion_limit": int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "80"))}
    answers = []
    run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M")
    lf = CallbackHandler() if os.getenv("TRACE_WITH_LANGFUSE") else None

    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['task_id'][:8]}…")

        result = "error when calling the agent"

        if has_file(item) and ".py" not in file_name(item) and ".xlsx" not in file_name(item) and ".mp3" not in file_name(item):
            answers.append({"task_id": item["task_id"], "submitted_answer": "has file, not processed yet"})
            print(f"la question {i} a un fichier ({file_name(item)!r}), donc non traitée.")
            continue
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                cfg = base_cfg if not lf else {
                    **base_cfg,
                    "callbacks": [lf],
                    "run_name": f"{run_id} question {i:02d}",
                    "metadata": {"langfuse_session_id": run_id},
                }
                result = answer(graph, item, cfg)
                break
            except Exception as e:
                if TOOL_USE_FAILED in str(e) and attempt < MAX_RETRIES:
                    print(f"    tentative {attempt + 1}/{MAX_RETRIES} : erreur tool calling, retry…")
                else:
                    print(f"    tentative {attempt + 1}/{MAX_RETRIES} : erreur non-tool-calling : {e}")
                time.sleep(args.sleep)
        answers.append({"task_id": item["task_id"], "submitted_answer": result})
        
        if args.sleep and i < len(questions):
            time.sleep(args.sleep)


    for i, ans in enumerate(answers):
        print(f"[{i+1}/{len(answers)}], {ans['submitted_answer']} \n")
    
    try:
        resp = requests.post(
        f"{BASE}/submit",
        json={
            "username": os.environ["HF_USERNAME"],
            "agent_code": os.environ["AGENT_CODE_URL"],
            "answers": answers,
        },
        timeout=120,
        )
        r = resp.json()
    except requests.JSONDecodeError:
        print(f"Submit failed: HTTP {resp.status_code}\n{resp.text[:500]}")
        return
    if resp.status_code != 200 or "score" not in r:
        print(f"Submit error (HTTP {resp.status_code}): {r}")
        return


    print( f"SCORE: {r['score']}% ({r['correct_count']}/{r['total_attempted']})")
    if msg := r.get("message"):
        print(msg)

    if lf:
        get_client().flush()


if __name__ == "__main__":
    main()
