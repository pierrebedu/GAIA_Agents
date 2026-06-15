from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.tools import tool

from youtube_transcript_api import YouTubeTranscriptApi

import os

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return up to 4 articles.
    
    Args:
        query: The search query."""
    try:
        import wikipedia
        wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
        wikipedia.set_rate_limiting(True)
        search_docs = WikipediaLoader(query=query, load_max_docs=4).load()
    except Exception as e:
        return f"Wikipedia search failed: {e}"
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs or "(no Wikipedia results)"

@tool
def web_search(query: str) -> str:
    """Search the public web via DuckDuckGo (no API key). Returns titles, URLs and short snippets.

    Args:
        query: The search query."""
    try:
        from ddgs import DDGS
    except ImportError as e:
        return f"Web search unavailable (install ddgs): {e}"
    max_results = int(os.getenv("DDG_MAX_RESULTS", "8"))
    q = (query or "").strip()
    if not q:
        return "(empty query)"
    timeout = int(os.getenv("DDG_TIMEOUT", "25"))
    try:
        with DDGS(timeout=timeout) as ddgs:
            hits = list(ddgs.text(q, max_results=max_results))
    except Exception as e:
        return f"DuckDuckGo search failed: {e}"
    if not hits:
        return "(no web results)"
    parts: list[str] = []
    for r in hits:
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or "")[:1500]
        parts.append(f'<Document source="{url}" page=""/>\n{title}\n{body}\n</Document>')
    return "\n\n---\n\n".join(parts)

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    except Exception as e:
        return f"Arxiv search failed: {e}"
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", doc.metadata.get("entry_id", ""))}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs or "(no Arxiv results)"


@tool
def execute_python_code(source: str) -> str:
    """Run Python source in an isolated subprocess (same interpreter). Returns stdout; includes stderr if non-zero exit.

    Use when the question embeds or attaches Python code and you need the actual printed/numeric output.
    Args:
        source: Python source code to execute as a single string."""
    import subprocess
    import sys
    import os
    proc = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=int(os.getenv("PYTHON_TOOL_TIMEOUT", "45")),
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        combined = f"exit={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}".strip()
        return combined[:8000]
    text = out if out else "(empty stdout)"
    if err:
        text = f"{text}\nSTDERR:\n{err}"
    return text[:8000]

@tool
def read_excel_format(file_path: str) -> str:
    """Read an Excel (.xlsx) file and return all its sheets as Markdown tables.

    Use this tool whenever the question references a spreadsheet or .xlsx file.
    Prefer this over execute_python_code when you just need to read and reason about
    tabular data — no need to write any code.

    Args:
        file_path: Absolute path to the .xlsx file as provided in the 'file_path' field of the question.
    """
    try:
        import pandas as pd
    except ImportError:
        return "pandas is not installed. Run: pip install pandas openpyxl"

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        return f"Failed to open Excel file: {e}"

    filename = os.path.basename(file_path)
    parts: list[str] = [f"**File:** `{filename}`\n"]

    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name)
        except Exception as e:
            parts.append(f"### Sheet: {sheet_name}\n(error reading sheet: {e})\n")
            continue

        parts.append(f"### Sheet: `{sheet_name}` — {df.shape[0]} rows × {df.shape[1]} columns\n")
        parts.append(df.to_markdown(index=False))
        parts.append("")

    return "\n".join(parts)


@tool
def YouTubeVideoAnalysisTool(video_id: str) -> str:
    """
    Fetches the transcript of a YouTube video by its ID and performs.
    Args:
        video_id: The ID of the YouTube video.
        
    Returns:
        video transcript in text format.
    """

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
        full_transcript = " ".join([snippet.text for snippet in fetched])
    except Exception as e:
        return f"An error occurred while fetching the YouTube transcript: {e}"
    
    return "the transcript of the youtube video is the following: "+ full_transcript

@tool
def transcribe_mp3(file_path: str) -> str:
    """Transcribe an MP3 audio file to text using Whisper (Hugging Face Inference API).

    Use this tool when the question references an .mp3 audio file.

    Args:
        file_path: Absolute path to the .mp3 file.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    token = os.getenv("HF_TOKEN")
    if not token:
        return "HF_TOKEN is not set in the environment."

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=token)
        with open(file_path, "rb") as f:
            output = client.automatic_speech_recognition(
                f.read(),
                model="openai/whisper-large-v3",
            )
        return output.text or "(empty transcription)"
    except Exception as e:
        return f"Transcription failed: {e}"