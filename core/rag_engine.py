"""
RAG Engine orchestrator module.
Coordinates document parsing, chunking, indexing, context retrieval, and LLM response generation.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator

from core.document_loader import DocumentLoader
from core.chunker import TextChunker
from core.vector_store import VectorStore
from core.llm_service import LLMService
from core.memory import SessionMemory
from config import config, DATA_DIR

logger = logging.getLogger(__name__)


class RAGEngine:
    """Core Retrieval-Augmented Generation Engine."""

    def __init__(self):
        index_file = DATA_DIR / "vector_store.json"
        self.vector_store = VectorStore(storage_path=index_file)
        self.chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        self.memory = SessionMemory()
        self.llm_service = LLMService(
            provider=config.llm_provider,
            model_name=config.model_name,
            gemini_api_key=config.gemini_api_key,
            openai_api_key=config.openai_api_key,
            groq_api_key=config.groq_api_key,
            ollama_base_url=config.ollama_base_url
        )

    def update_settings(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """Dynamically updates runtime engine parameters."""
        if provider is not None:
            config.llm_provider = provider
            self.llm_service.provider = provider
        if model_name is not None:
            config.model_name = model_name
            self.llm_service.model_name = model_name
        if gemini_api_key is not None:
            config.gemini_api_key = gemini_api_key
            self.llm_service.gemini_api_key = gemini_api_key
        if openai_api_key is not None:
            config.openai_api_key = openai_api_key
            self.llm_service.openai_api_key = openai_api_key
        if groq_api_key is not None:
            config.groq_api_key = groq_api_key
            self.llm_service.groq_api_key = groq_api_key
        if chunk_size is not None and chunk_overlap is not None:
            config.chunk_size = chunk_size
            config.chunk_overlap = chunk_overlap
            self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if top_k is not None:
            config.top_k = top_k
        if similarity_threshold is not None:
            config.similarity_threshold = similarity_threshold
        if system_prompt is not None:
            config.system_prompt = system_prompt

    def ingest_document(self, filepath: str | Path, doc_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses, chunks, and indexes a file.
        """
        path = Path(filepath)
        name = doc_name or path.name

        doc_data = DocumentLoader.load_file(path)
        doc_data["indexed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        chunks = self.chunker.split_text(doc_data["content"], doc_name=name)
        indexed_count = self.vector_store.add_document_chunks(name, doc_data, chunks)

        return {
            "document": name,
            "char_count": doc_data["char_count"],
            "pages": doc_data["pages"],
            "chunks_created": indexed_count,
            "status": "success"
        }

    def delete_document(self, doc_name: str) -> bool:
        """Removes a document from the vector store."""
        return self.vector_store.delete_document(doc_name)

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves matching chunks from vector store."""
        k = top_k if top_k is not None else config.top_k
        thresh = threshold if threshold is not None else config.similarity_threshold
        return self.vector_store.search(query, top_k=k, threshold=thresh, doc_filter=doc_filter)

    def construct_rag_prompt(self, user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Assembles the user query and retrieved context chunks into an augmented prompt."""
        if not context_chunks:
            return user_query

        context_str_parts = []
        for i, chunk in enumerate(context_chunks):
            doc = chunk.get("doc_name", "Unknown")
            idx = chunk.get("chunk_index", i)
            text = chunk.get("text", "")
            context_str_parts.append(f"[Document: {doc} | Chunk #{idx}]\n{text}")

        context_block = "\n\n---\n\n".join(context_str_parts)

        return (
            f"Context Information from uploaded knowledge base:\n"
            f"================================================\n"
            f"{context_block}\n"
            f"================================================\n\n"
            f"User Question: {user_query}\n\n"
            f"Instructions:\n"
            f"Please answer the user question based on the provided context information. "
            f"Include citations like [Doc: {context_chunks[0]['doc_name']}] when referencing facts."
        )

    async def stream_query(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        doc_filter: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes a RAG query and yields structured SSE packets:
        - First packet: metadata with retrieved sources/citations
        - Subsequent packets: token deltas
        - Final packet: completion signal
        """
        session_id = self.memory.get_or_create_session(session_id)
        
        # 1. Retrieve context chunks
        chunks = self.retrieve_context(query, top_k=top_k, threshold=threshold, doc_filter=doc_filter)
        
        # Yield metadata event with source citations
        yield {
            "type": "sources",
            "sources": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_name": c.get("doc_name"),
                    "chunk_index": c.get("chunk_index"),
                    "score": c.get("score"),
                    "preview": c.get("text")[:180] + ("..." if len(c.get("text", "")) > 180 else ""),
                    "full_text": c.get("text"),
                    "char_count": c.get("char_count")
                }
                for c in chunks
            ]
        }

        # 2. Build prompt
        augmented_prompt = self.construct_rag_prompt(query, chunks)
        history = self.memory.get_history(session_id)

        # 3. Stream LLM tokens
        full_response = []
        async for token in self.llm_service.stream_response(
            prompt=augmented_prompt,
            system_prompt=config.system_prompt,
            history=history,
            retrieved_chunks=chunks
        ):
            full_response.append(token)
            yield {
                "type": "token",
                "delta": token
            }

        final_text = "".join(full_response)

        # 4. Save to session memory
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", final_text, sources=chunks)

        yield {
            "type": "done",
            "session_id": session_id,
            "char_count": len(final_text)
        }
