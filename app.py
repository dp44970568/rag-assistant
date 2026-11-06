"""
Main FastAPI Application for RAG AI Assistant.
Provides high-performance REST APIs and Server-Sent Events (SSE) streaming without Streamlit.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config, BASE_DIR, UPLOAD_DIR, SAMPLE_DIR
from core.rag_engine import RAGEngine
from core.document_loader import DocumentLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_app")

app = FastAPI(
    title="RAG AI Assistant",
    description="High-performance Retrieval-Augmented Generation AI Assistant (FastAPI + Modern Web UI)",
    version="2.0.0"
)

# CORS setup for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
rag_engine = RAGEngine()

# Static directory setup
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    doc_filter: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    system_prompt: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """If vector store is completely empty, automatically ingest sample documents."""
    stats = rag_engine.vector_store.get_stats()
    if stats["total_chunks"] == 0:
        logger.info("Knowledge base is empty. Pre-loading sample documents...")
        for sample_file in SAMPLE_DIR.glob("*.txt"):
            try:
                rag_engine.ingest_document(sample_file)
                logger.info(f"Loaded sample: {sample_file.name}")
            except Exception as e:
                logger.warning(f"Could not load sample {sample_file.name}: {e}")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main single-page application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>RAG AI Assistant</h1><p>UI loading error. Check static/index.html</p>")


@app.get("/api/status")
async def get_status():
    """Returns application status, stats, and active config."""
    stats = rag_engine.vector_store.get_stats()
    return {
        "status": "online",
        "provider": config.llm_provider,
        "model_name": config.model_name,
        "has_gemini_key": bool(config.gemini_api_key.strip()),
        "has_openai_key": bool(config.openai_api_key.strip()),
        "has_groq_key": bool(config.groq_api_key.strip()),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "similarity_threshold": config.similarity_threshold,
        "stats": stats
    }


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads and indexes one or more documents."""
    uploaded_results = []
    errors = []

    for file in files:
        if not DocumentLoader.is_supported(file.filename):
            errors.append(f"File '{file.filename}' is not a supported document type.")
            continue

        dest_path = UPLOAD_DIR / file.filename
        try:
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            result = rag_engine.ingest_document(dest_path, doc_name=file.filename)
            uploaded_results.append(result)
        except Exception as e:
            logger.error(f"Failed processing upload {file.filename}: {e}")
            errors.append(f"Error processing '{file.filename}': {str(e)}")

    return {
        "uploaded": uploaded_results,
        "errors": errors,
        "current_stats": rag_engine.vector_store.get_stats()
    }


@app.post("/api/load-sample")
async def load_sample_knowledge():
    """Manually triggers loading of sample knowledge files."""
    loaded = []
    for sample_file in SAMPLE_DIR.glob("*.txt"):
        try:
            res = rag_engine.ingest_document(sample_file)
            loaded.append(res)
        except Exception as e:
            logger.error(f"Failed to load sample {sample_file.name}: {e}")

    return {
        "message": f"Successfully loaded {len(loaded)} sample documents.",
        "documents": loaded,
        "stats": rag_engine.vector_store.get_stats()
    }


@app.get("/api/documents")
async def list_documents():
    """Lists all currently indexed documents."""
    return rag_engine.vector_store.get_stats()


@app.delete("/api/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Deletes a document and its chunks from index."""
    success = rag_engine.delete_document(doc_name)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Try deleting file from uploads if present
    upload_file = UPLOAD_DIR / doc_name
    if upload_file.exists():
        try:
            upload_file.unlink()
        except Exception:
            pass

    return {
        "message": f"Document '{doc_name}' deleted successfully",
        "stats": rag_engine.vector_store.get_stats()
    }


@app.delete("/api/documents")
async def clear_all_documents():
    """Clears all indexed documents from vector store."""
    rag_engine.vector_store.clear()
    return {"message": "All documents cleared", "stats": rag_engine.vector_store.get_stats()}


@app.get("/api/chunks/{doc_name}")
async def get_document_chunks(doc_name: str):
    """Returns chunk details for a given document for inspection."""
    chunks = [c for c in rag_engine.vector_store.chunks if c.get("doc_name") == doc_name]
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for document")
    return {
        "document": doc_name,
        "chunk_count": len(chunks),
        "chunks": chunks
    }


@app.post("/api/search-test")
async def test_search(req: QueryRequest):
    """Previews chunk retrieval without generating an LLM response."""
    results = rag_engine.retrieve_context(
        req.query,
        top_k=req.top_k,
        threshold=req.threshold,
        doc_filter=req.doc_filter
    )
    return {
        "query": req.query,
        "retrieved_count": len(results),
        "results": results
    }


@app.post("/api/chat")
async def non_streaming_chat(req: QueryRequest):
    """Non-streaming conversational RAG response."""
    session_id = rag_engine.memory.get_or_create_session(req.session_id)
    chunks = rag_engine.retrieve_context(
        req.query,
        top_k=req.top_k,
        threshold=req.threshold,
        doc_filter=req.doc_filter
    )
    prompt = rag_engine.construct_rag_prompt(req.query, chunks)
    history = rag_engine.memory.get_history(session_id)

    response_text = await rag_engine.llm_service.generate_response(
        prompt=prompt,
        system_prompt=config.system_prompt,
        history=history
    )

    rag_engine.memory.add_message(session_id, "user", req.query)
    rag_engine.memory.add_message(session_id, "assistant", response_text, sources=chunks)

    return {
        "session_id": session_id,
        "response": response_text,
        "sources": chunks
    }


@app.get("/api/stream")
async def stream_chat(
    query: str = Query(...),
    session_id: Optional[str] = Query(None),
    top_k: Optional[int] = Query(None),
    threshold: Optional[float] = Query(None),
    doc_filter: Optional[str] = Query(None)
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits events: 'sources', 'token', 'done'.
    """
    async def event_generator():
        try:
            async for packet in rag_engine.stream_query(
                query=query,
                session_id=session_id or "default",
                top_k=top_k,
                threshold=threshold,
                doc_filter=doc_filter
            ):
                payload = json.dumps(packet)
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            err_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Fetches chat history for the session."""
    return {"session_id": session_id, "history": rag_engine.memory.get_history(session_id)}


@app.delete("/api/session/{session_id}")
async def clear_session_history(session_id: str):
    """Clears history for the session."""
    rag_engine.memory.clear_session(session_id)
    return {"message": "Session history cleared"}


@app.get("/api/session/{session_id}/export")
async def export_session_history(session_id: str):
    """Exports session conversation as Markdown."""
    markdown_content = rag_engine.memory.export_markdown(session_id)
    return {
        "session_id": session_id,
        "markdown": markdown_content
    }


@app.post("/api/config")
async def update_configuration(req: ConfigUpdateRequest):
    """Updates runtime configuration settings."""
    rag_engine.update_settings(
        provider=req.llm_provider,
        model_name=req.model_name,
        gemini_api_key=req.gemini_api_key,
        openai_api_key=req.openai_api_key,
        groq_api_key=req.groq_api_key,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        system_prompt=req.system_prompt
    )
    return {
        "message": "Configuration updated successfully",
        "current_config": {
            "llm_provider": config.llm_provider,
            "model_name": config.model_name,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
            "similarity_threshold": config.similarity_threshold,
            "has_gemini_key": bool(config.gemini_api_key.strip()),
            "has_openai_key": bool(config.openai_api_key.strip()),
            "has_groq_key": bool(config.groq_api_key.strip()),
        }
    }
