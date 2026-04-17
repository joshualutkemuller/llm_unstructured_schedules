"""
Document skills: load, inspect, and classify collateral schedule documents.

Skills exposed:
  - load_document       Load a file and return page text + metadata
  - classify_schedule   Identify IM / VM / REPO + governing law
  - chunk_text          Split raw text into LLM-friendly chunks
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from skills.registry import SkillResult, _SkillEntry


class DocumentSkills:

    def entries(self) -> List[_SkillEntry]:
        return [
            _SkillEntry("load_document",    self.load_document,    LOAD_SCHEMA),
            _SkillEntry("classify_schedule", self.classify_schedule, CLASSIFY_SCHEMA),
            _SkillEntry("chunk_text",        self.chunk_text,        CHUNK_SCHEMA),
        ]

    # ── load_document ─────────────────────────────────────────────────────────

    def load_document(self, file_path: str) -> SkillResult:
        """Load a PDF/DOCX/XLSX/TXT and return structured page content."""
        from ingestion.document_loader import DocumentLoader
        try:
            loader = DocumentLoader(ocr_fallback=True)
            doc = loader.load(Path(file_path))
            return SkillResult(
                success=True,
                data={
                    "filename": doc.filename,
                    "format": doc.raw_format,
                    "page_count": doc.page_count,
                    "pages": [
                        {"page": p.page_number, "text": p.text, "is_ocr": p.is_ocr}
                        for p in doc.pages
                    ],
                    "full_text": doc.full_text,
                },
                metadata={"page_count": doc.page_count},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

    # ── classify_schedule ─────────────────────────────────────────────────────

    def classify_schedule(self, text: str) -> SkillResult:
        """Classify document text as IM, VM, or REPO and detect governing law."""
        from ingestion.document_classifier import DocumentClassifier
        try:
            clf = DocumentClassifier()
            result = clf.classify(text)
            return SkillResult(
                success=True,
                data={
                    "schedule_type": result.schedule_type.value,
                    "governing_law": result.governing_law.value,
                    "type_confidence": round(result.type_confidence, 4),
                    "law_confidence": round(result.law_confidence, 4),
                    "method": result.method,
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

    # ── chunk_text ────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, max_tokens: int = 3000) -> SkillResult:
        """Split text into overlapping chunks suitable for an LLM context window."""
        from ingestion.document_loader import DocumentPage, LoadedDocument, chunk_document
        doc = LoadedDocument(
            filename="<raw>",
            pages=[DocumentPage(page_number=1, text=text)],
            raw_format="txt",
        )
        chunks = chunk_document(doc, max_tokens=max_tokens)
        return SkillResult(
            success=True,
            data={"chunks": chunks, "count": len(chunks)},
        )


# ── Anthropic tool schemas ────────────────────────────────────────────────────

LOAD_SCHEMA = {
    "name": "load_document",
    "description": (
        "Load a collateral schedule document from disk (PDF, DOCX, XLSX, or TXT). "
        "Returns structured page-level text and document metadata. "
        "Use this before classifying or extracting fields from a file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the document file.",
            }
        },
        "required": ["file_path"],
    },
}

CLASSIFY_SCHEMA = {
    "name": "classify_schedule",
    "description": (
        "Classify raw document text as one of: IM (Initial Margin CSA), "
        "VM (Variation Margin CSA), or REPO (GMRA Repurchase Agreement). "
        "Also detects governing law (NEW_YORK, ENGLISH, JAPANESE). "
        "Returns confidence scores and the method used (heuristic or llm)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Raw text of the collateral schedule document.",
            }
        },
        "required": ["text"],
    },
}

CHUNK_SCHEMA = {
    "name": "chunk_text",
    "description": (
        "Split a long document text into overlapping chunks that fit within "
        "an LLM context window. Use this before calling extract_fields on "
        "documents longer than ~3000 tokens."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Document text to split."},
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens per chunk (default 3000).",
                "default": 3000,
            },
        },
        "required": ["text"],
    },
}
