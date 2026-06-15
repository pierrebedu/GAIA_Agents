# GAIA Agent

AI agent designed to automate multi-step research and reasoning tasks across diverse data sources. It extracts and processes information from documents, spreadsheets, audio, video, and the web to answer complex questions — the same kind of workflows that show up in report generation, invoice data extraction, and competitive intelligence.

Built and evaluated on the [GAIA benchmark](https://arxiv.org/pdf/2311.12983) (static questions), this project explores how a tool-augmented agent can reduce manual workload and improve productivity on tasks that normally require humans to search, read, compute, and synthesize.

## What it does

The agent receives a natural-language question (sometimes with an attached file) and autonomously decides which tools to call, in what order, until it can produce a precise final answer.

**Supported inputs & sources**

| Source | Capability |
|--------|------------|
| Web | DuckDuckGo search for recent pages, news, and reference material |
| Wikipedia | Factual lookups (biographies, sports, history, etc.) |
| arXiv | Academic preprints when explicitly needed |
| Excel (`.xlsx`) | Full sheet extraction as structured tables |
| YouTube | Transcript retrieval and analysis |
| Audio (`.mp3`) | Speech-to-text via Whisper |
| Python code | Execution of attached or embedded scripts |
| Plain reasoning | Direct inference on data provided in the prompt |

**Reliability**

- Automatic retries when tool-calling fails (configurable)
- Strict answer formatting aligned with GAIA scoring rules
- Batch evaluation pipeline with automated submission and scoring

## Observability

Runs are traced with **[Langfuse](https://langfuse.com/)** for performance evaluation on the GAIA dataset and step-by-step debugging. Each evaluation run is tagged with a session ID so you can compare attempts, inspect tool calls, and iterate on prompts and tool selection.

Set `TRACE_WITH_LANGFUSE=true` in your environment to enable tracing during evaluation.

## Stack

- **Orchestration:** LangGraph ReAct agent (`create_agent`)
- **LLM providers:** Groq (default), Google Gemini, Hugging Face
- **Tools:** LangChain tool interface
- **Evaluation:** Hugging Face scoring API + GAIA test questions
- **Observability:** Langfuse (cloud; self-hosting also possible)

## Project structure

```
agent.py              # Agent graph and LLM provider selection
tools.py              # Web, Wikipedia, arXiv, Excel, YouTube, Whisper, Python exec
evaluate_agent.py     # Batch evaluation, retries, Langfuse tracing, HF submission
system_prompt.txt     # Agent instructions and answer format rules
explore_gaia.ipynb    # Exploratory analysis of the GAIA dataset
regexs.py             # Answer extraction and normalization helpers
```

---

## Note perso (détails techniques)

> Section perso — setup, commandes, et détails d’implémentation.

### Environnement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur factorius : `source .venv/bin/activate`

Variables d’environnement utiles (via `.env`) :

- `LLM_PROVIDER` — `groq` (défaut), `google`, ou `huggingface`
- `GROQ_MODEL`, `GROQ_SEED` — modèle et seed Groq
- `HF_TOKEN` — requis pour la transcription MP3 (Whisper)
- `HF_USERNAME`, `AGENT_CODE_URL` — soumission des réponses au serveur de scoring HF
- `TRACE_WITH_LANGFUSE` — activer Langfuse (`true`)
- `AGENT_MAX_RETRIES` — retries en cas d’échec tool-calling (défaut : 2)
- `GROQ_EVAL_SLEEP_SECONDS` — pause entre questions (rate limiting)

### Lancer une évaluation

```bash
python evaluate_agent.py              # toutes les questions
python evaluate_agent.py --limit 20   # sous-ensemble
```

Le script récupère les questions via l’API HF, invoque l’agent, normalise la réponse (`FINAL ANSWER: …`), soumet les résultats et affiche le score.

Fichiers attachés : placés dans `gaia_files/`. Actuellement traités : `.py`, `.xlsx`, `.mp3`. Les autres types sont ignorés (placeholder `"has file, not processed yet"`).

### Agent (`agent.py`)

- Graph LangGraph via `create_agent(llm, tools, system_prompt)`
- Prompt système chargé depuis `system_prompt.txt`
- Outils : `multiply`, `wiki_search`, `web_search`, `arvix_search`, `execute_python_code`, `YouTubeVideoAnalysisTool`, `read_excel_format`, `transcribe_mp3`

### Traces Langfuse

Callback LangChain branché dans `evaluate_agent.py` quand `TRACE_WITH_LANGFUSE` est défini. Chaque question a un `run_name` du type `run_YYYY-MM-DD_HH-MM question NN` et un `langfuse_session_id` commun par run.

### TODO perso

- Documenter ce qui tourne sous le capot dans l’instance ReAct (boucle think → act → observe)
- Étendre le support des types de fichiers GAIA non encore traités
