"""
Vector Store module.
Supports high-speed local indexing (TF-IDF + Cosine Similarity) and dense neural embeddings.
Maintains persistent index and metadata across server restarts.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory and disk-persistent vector index."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.chunks: List[Dict[str, Any]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        
        if self.storage_path and self.storage_path.exists():
            self.load_index()

    def add_document_chunks(self, doc_name: str, doc_meta: Dict[str, Any], chunks: List[Dict[str, Any]]) -> int:
        """
        Adds new document chunks to the store and refreshes the search index.
        """
        # Remove any previous version of this document
        self.delete_document(doc_name, persist=False)

        # Store document metadata
        self.documents[doc_name] = {
            "name": doc_name,
            "filename": doc_meta.get("filename", doc_name),
            "char_count": doc_meta.get("char_count", 0),
            "pages": doc_meta.get("pages"),
            "chunk_count": len(chunks),
            "extension": doc_meta.get("extension", ""),
            "indexed_at": doc_meta.get("indexed_at", "")
        }

        # Append chunks
        self.chunks.extend(chunks)

        # Re-index
        self._rebuild_index()

        if self.storage_path:
            self.save_index()

        return len(chunks)

    def delete_document(self, doc_name: str, persist: bool = True) -> bool:
        """Deletes a document and its chunks from the store."""
        initial_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("doc_name") != doc_name]
        deleted_from_docs = self.documents.pop(doc_name, None) is not None

        if len(self.chunks) != initial_count or deleted_from_docs:
            self._rebuild_index()
            if persist and self.storage_path:
                self.save_index()
            return True
        return False

    def clear(self):
        """Clears all indexed documents and chunks."""
        self.chunks = []
        self.documents = {}
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        if self.storage_path and self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete storage path: {e}")

    def _rebuild_index(self):
        """Rebuilds the TF-IDF representation for fast local cosine retrieval."""
        if not self.chunks:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = [c["text"] for c in self.chunks]
            # ngram_range (1, 2) provides both keyword and phrase matching
            self._tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=25000,
                sublinear_tf=True
            )
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None

    def search(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.05,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks for the given query.
        Returns chunks sorted by descending relevance_score.
        """
        if not self.chunks or not query.strip():
            return []

        # If TF-IDF matrix is ready, perform cosine similarity
        if self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                query_vec = self._tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

                scored_chunks = []
                for idx, score in enumerate(similarities):
                    chunk = self.chunks[idx]
                    
                    if doc_filter and chunk.get("doc_name") != doc_filter:
                        continue

                    # Keyword boost: bonus if query keywords directly appear in chunk
                    q_words = set(query.lower().split())
                    chunk_lower = chunk["text"].lower()
                    kw_matches = sum(1 for w in q_words if len(w) > 3 and w in chunk_lower)
                    boost = min(0.15, kw_matches * 0.04)

                    final_score = float(score) + boost

                    if final_score >= threshold:
                        chunk_copy = dict(chunk)
                        chunk_copy["score"] = round(final_score, 4)
                        scored_chunks.append(chunk_copy)

                scored_chunks.sort(key=lambda x: x["score"], reverse=True)
                return scored_chunks[:top_k]
            except Exception as e:
                logger.error(f"TF-IDF search error: {e}")

        # Fallback keyword-overlap search if vectorizer fails
        return self._keyword_search(query, top_k, threshold, doc_filter)

    def _keyword_search(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.05,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Simple BM25-like keyword overlap scoring fallback."""
        q_tokens = [t.lower() for t in query.split() if len(t) > 2]
        if not q_tokens:
            return self.chunks[:top_k]

        scored = []
        for chunk in self.chunks:
            if doc_filter and chunk.get("doc_name") != doc_filter:
                continue

            c_text = chunk["text"].lower()
            matches = sum(1 for t in q_tokens if t in c_text)
            if matches > 0:
                score = matches / (len(q_tokens) + math.log1p(len(chunk["text"].split())))
                if score >= threshold:
                    item = dict(chunk)
                    item["score"] = round(float(score), 4)
                    scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current index."""
        total_chars = sum(c.get("char_count", 0) for c in self.chunks)
        total_tokens = sum(c.get("token_estimate", 0) for c in self.chunks)
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "total_characters": total_chars,
            "estimated_tokens": total_tokens,
            "documents": list(self.documents.values())
        }

    def save_index(self):
        """Serializes chunks and documents to JSON."""
        if not self.storage_path:
            return
        try:
            data = {
                "documents": self.documents,
                "chunks": self.chunks
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save index to {self.storage_path}: {e}")

    def load_index(self):
        """Deserializes chunks and documents from JSON."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", {})
                self.chunks = data.get("chunks", [])
            self._rebuild_index()
            logger.info(f"Loaded {len(self.chunks)} chunks from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load index from {self.storage_path}: {e}")
