"""LangGraph Agent"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage

from tools import multiply, wiki_search, web_search, arvix_search, execute_python_code, YouTubeVideoAnalysisTool, read_excel_format, transcribe_mp3


load_dotenv()

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# load the system prompt from the file
with open(os.path.join(_AGENT_DIR, "system_prompt.txt"), "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

tools = [
    multiply,
    wiki_search,
    web_search,
    arvix_search,
    execute_python_code,
    YouTubeVideoAnalysisTool,
    read_excel_format,
    transcribe_mp3,
]


# Build graph function
def build_graph(provider: str | None = None):

    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    elif provider == "groq":
        model = os.getenv("GROQ_MODEL")
        seed = int(os.getenv("GROQ_SEED", "42"))
        llm = ChatGroq(model=model, temperature=0, model_kwargs={"seed": seed})
        
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint. crédits tres limités...
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    return create_agent(llm, tools, system_prompt=sys_msg)