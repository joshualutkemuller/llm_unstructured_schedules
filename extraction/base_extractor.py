"""
Abstract base for all schedule-type extractors.

Handles:
  - Chunking long documents
  - Calling the LLM (Anthropic Claude by default)
  - Merging multi-chunk results (highest-confidence field wins)
  - Parsing + validating the JSON response
  - Flagging low-confidence fields for human review
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from config.settings import Settings
from ingestion.document_loader import LoadedDocument, chunk_document

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    raw_json: Dict[str, Any]
    validated_model: Optional[BaseModel]
    low_confidence_fields: List[str]
    validation_errors: List[str] = field(default_factory=list)
    chunk_count: int = 1


class BaseExtractor(ABC):
    """
    Subclass this for each schedule type (IM, VM, REPO).

    Subclasses must implement:
      - build_messages(chunk: str) -> list[dict]
      - schema_class -> Type[BaseModel]
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._client = self._build_client()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_messages(self, chunk: str) -> list[dict]:
        """Return the message list for the LLM call."""

    @property
    @abstractmethod
    def schema_class(self) -> Type[BaseModel]:
        """The Pydantic schema to validate against."""

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, doc: LoadedDocument) -> ExtractionResult:
        """
        Run extraction on a full document.
        Multi-chunk documents are processed individually and merged.
        """
        chunks = chunk_document(doc, max_tokens=self.settings.chunk_max_tokens)
        logger.info("Extracting from %d chunks", len(chunks))

        chunk_results: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk %d/%d", i + 1, len(chunks))
            raw = self._call_llm(chunk)
            chunk_results.append(raw)

        merged = self._merge_chunks(chunk_results)
        return self._validate_and_build(merged, chunk_count=len(chunks))

    def extract_from_text(self, text: str) -> ExtractionResult:
        """Convenience method when you already have a text string."""
        from ingestion.document_loader import DocumentPage, LoadedDocument
        doc = LoadedDocument(
            filename="<text>",
            pages=[DocumentPage(page_number=1, text=text)],
            raw_format="txt",
        )
        return self.extract(doc)

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, chunk: str) -> Dict[str, Any]:
        messages = self.build_messages(chunk)

        try:
            response = self._client.messages.create(
                model=self.settings.extraction_model,
                max_tokens=self.settings.extraction_max_tokens,
                messages=messages,
            )
            content = response.content[0].text
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

        return self._parse_json(content)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("JSON parse error: %s\nRaw: %.200s", e, text)
            raise

    # ── Chunk merging ─────────────────────────────────────────────────────────

    def _merge_chunks(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge extracted fields across chunks.
        For each field, keep the value with the highest confidence score.
        """
        if len(results) == 1:
            return results[0]

        merged: Dict[str, Any] = {}
        all_keys = set(k for r in results for k in r)

        for key in all_keys:
            candidates = [r[key] for r in results if key in r and r[key] is not None]
            if not candidates:
                continue

            # Each candidate is a dict with "value" and "confidence"
            if isinstance(candidates[0], dict) and "confidence" in candidates[0]:
                best = max(candidates, key=lambda c: c.get("confidence", 0))
                merged[key] = best
            else:
                merged[key] = candidates[0]

        return merged

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_and_build(
        self, raw: Dict[str, Any], chunk_count: int = 1
    ) -> ExtractionResult:
        errors: List[str] = []
        validated = None
        low_conf: List[str] = []

        try:
            validated = self.schema_class(**self._raw_to_model_kwargs(raw))
            low_conf = validated.low_confidence_fields(
                threshold=self.settings.review_confidence_threshold
            )
            # Mark fields for review
            for fname in low_conf:
                field_val = getattr(validated, fname)
                if hasattr(field_val, "needs_review"):
                    object.__setattr__(field_val, "needs_review", True)
        except Exception as e:
            errors.append(str(e))
            logger.warning("Schema validation failed: %s", e)

        return ExtractionResult(
            raw_json=raw,
            validated_model=validated,
            low_confidence_fields=low_conf,
            validation_errors=errors,
            chunk_count=chunk_count,
        )

    def _raw_to_model_kwargs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw LLM output dict to keyword arguments for the Pydantic model.
        Each field in the raw output is already in ExtractedField format.
        """
        from schemas.base import ExtractedField
        kwargs: Dict[str, Any] = {}

        for key, val in raw.items():
            if isinstance(val, dict) and "value" in val:
                kwargs[key] = ExtractedField(
                    value=val.get("value"),
                    confidence=float(val.get("confidence", 0.0)),
                    source_text=val.get("source_text"),
                )
            else:
                kwargs[key] = val

        return kwargs

    # ── Client factory ────────────────────────────────────────────────────────

    def _build_client(self):
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
