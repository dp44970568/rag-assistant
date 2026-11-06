"""
Text chunking module.
Provides recursive character chunking with overlap and rich chunk metadata.
"""

import re
from typing import List, Dict, Any


class TextChunker:
    """Intelligent text splitting into overlapping semantic chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)

    def split_text(self, text: str, doc_name: str = "document") -> List[Dict[str, Any]]:
        """
        Recursively splits text into chunks within `chunk_size` limits,
        respecting natural delimiters (double newline, single newline, sentence punctuation, spaces).
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.chunk_size, self.chunk_overlap)
        
        results = []
        current_pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            start_char = text.find(chunk_text[:30], current_pos) if len(chunk_text) >= 30 else text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)
            current_pos = max(current_pos, start_char + max(1, len(chunk_text) - self.chunk_overlap))

            results.append({
                "chunk_id": f"{doc_name}_chk_{i}",
                "doc_name": doc_name,
                "chunk_index": i,
                "text": chunk_text,
                "start_char": start_char,
                "end_char": end_char,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "token_estimate": max(1, len(chunk_text) // 4)
            })

        return results

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
        return self._split_with_separators(text, separators, chunk_size, chunk_overlap)

    def _split_with_separators(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator and good_splits:
                # keep separator with string
                good_splits.append(separator + s)
            else:
                good_splits.append(s)

        current_chunk = ""
        for s in good_splits:
            if len(current_chunk) + len(s) <= chunk_size:
                current_chunk += s
            else:
                if current_chunk.strip():
                    if len(current_chunk) > chunk_size and new_separators:
                        sub_chunks = self._split_with_separators(current_chunk, new_separators, chunk_size, chunk_overlap)
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(current_chunk)

                # Overlap calculation
                overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 and len(current_chunk) > chunk_overlap else ""
                current_chunk = (overlap_text + s) if len(overlap_text + s) <= chunk_size else s

        if current_chunk.strip():
            if len(current_chunk) > chunk_size and new_separators:
                sub_chunks = self._split_with_separators(current_chunk, new_separators, chunk_size, chunk_overlap)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(current_chunk)

        return [c for c in final_chunks if c.strip()]
