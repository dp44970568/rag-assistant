"""
Configuration settings for RAG AI Assistant.
Supports environment variables and dynamic runtime updates.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_documents"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Load existing .env if present
load_dotenv(BASE_DIR / ".env")


class AppConfig(BaseModel):
    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # LLM Settings
    # Options: "offline", "gemini", "openai", "groq", "ollama"
    llm_provider: str = os.getenv("LLM_PROVIDER", "offline")
    model_name: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")

    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    similarity_threshold: float = 0.15
    embedding_mode: str = "local"  # "local" (TF-IDF/BM25) or "dense" (OpenAI/Gemini embeddings)

    # System Prompt
    system_prompt: str = (
        "You are an intelligent, precise RAG (Retrieval-Augmented Generation) AI Assistant. "
        "Answer the user's question clearly and accurately using ONLY the retrieved context chunks provided below. "
        "If the context does not contain enough information to answer the question, state honestly that the information "
        "is not available in the uploaded documents rather than making up facts. "
        "Always cite your sources using chunk references like [Source: DocumentName (Chunk #)] when appropriate."
    )


# Active global configuration instance
config = AppConfig()
