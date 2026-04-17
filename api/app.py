"""
FastAPI service for collateral schedule standardization.

Endpoints:
  POST /extract          Upload a document file, get standardized JSON back
  POST /extract/text     Submit raw text + schedule_type, get standardized JSON
  GET  /health           Health check

Run:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import Settings
from pipeline.standardizer import CollateralStandardizer
from schemas.base import ScheduleType

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Collateral Schedule Standardizer",
    description="Extract and standardize IM, VM, and REPO collateral schedule fields using LLM.",
    version="1.0.0",
)

_settings = Settings()
_standardizer = CollateralStandardizer(settings=_settings)


# ── Request / Response models ─────────────────────────────────────────────────

class TextExtractionRequest(BaseModel):
    text: str
    schedule_type: Literal["IM", "VM", "REPO"]


class ExtractionResponse(BaseModel):
    success: bool
    schedule_type: Optional[str]
    validated_fields: Optional[dict]
    low_confidence_fields: list[str]
    validation_errors: list[str]
    chunk_count: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": _settings.extraction_model}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(
    file: UploadFile = File(..., description="PDF, DOCX, XLSX, or TXT collateral schedule"),
):
    """Upload a document and receive standardized schedule fields."""
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    allowed = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = _standardizer.process(tmp_path)
    except Exception as e:
        logger.error("Extraction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return _build_response(result)


@app.post("/extract/text", response_model=ExtractionResponse)
def extract_text(request: TextExtractionRequest):
    """Submit raw text (already extracted from a document) for field extraction."""
    try:
        stype = ScheduleType(request.schedule_type)
        result = _standardizer.process_text(request.text, stype)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Extraction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(result)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_response(result) -> ExtractionResponse:
    validated_fields = None
    schedule_type = None

    if result.validated_model is not None:
        data = result.validated_model.model_dump()
        validated_fields = data
        schedule_type = data.get("schedule_type")

    return ExtractionResponse(
        success=result.validated_model is not None,
        schedule_type=schedule_type,
        validated_fields=validated_fields,
        low_confidence_fields=result.low_confidence_fields,
        validation_errors=result.validation_errors,
        chunk_count=result.chunk_count,
    )
