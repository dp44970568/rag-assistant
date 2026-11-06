"""
Document Loader module.
Extracts clean plain text and metadata from PDF, DOCX, TXT, MD, CSV, JSON, and code files.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class DocumentLoader:
    """Multi-format document extractor."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css", ".log"
    }

    @classmethod
    def is_supported(cls, filepath: str | Path) -> bool:
        ext = Path(filepath).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def load_file(cls, filepath: str | Path) -> Dict[str, Any]:
        """
        Loads and parses a single file into text and metadata.
        Returns a dictionary:
        {
            "filename": str,
            "filepath": str,
            "extension": str,
            "content": str,
            "char_count": int,
            "pages": Optional[int],
            "metadata": dict
        }
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = path.suffix.lower()
        filename = path.name
        content = ""
        pages = None
        extra_meta = {}

        if ext == ".pdf":
            content, pages = cls._load_pdf(path)
        elif ext == ".docx":
            content = cls._load_docx(path)
        elif ext in {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css", ".log"}:
            content = cls._load_text(path)
        else:
            # Fallback text attempt
            content = cls._load_text(path)

        content = content.strip()
        return {
            "filename": filename,
            "filepath": str(path.resolve()),
            "extension": ext,
            "content": content,
            "char_count": len(content),
            "pages": pages,
            "metadata": extra_meta,
        }

    @classmethod
    def _load_pdf(cls, path: Path) -> tuple[str, int]:
        text_parts = []
        page_count = 0
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- [Page {idx + 1}] ---\n" + page_text.strip())
        except Exception as e:
            # Fallback if pypdf encounters an issue
            text_parts.append(f"Error parsing PDF with pypdf: {str(e)}")

        return "\n\n".join(text_parts), page_count

    @classmethod
    def _load_docx(cls, path: Path) -> str:
        try:
            import docx
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            return "\n\n".join(paragraphs)
        except Exception as e:
            return f"Error parsing DOCX: {str(e)}"

    @classmethod
    def _load_text(cls, path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
