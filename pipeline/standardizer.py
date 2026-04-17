"""
End-to-end collateral schedule standardizer.

Orchestrates:
  1. Document loading
  2. Type classification (IM / VM / REPO)
  3. LLM field extraction
  4. Pydantic validation
  5. JSON output with low-confidence flagging

Usage (CLI):
    python -m pipeline.standardizer --file path/to/csa.pdf --output output.json

Usage (library):
    from pipeline.standardizer import CollateralStandardizer
    result = CollateralStandardizer().process("path/to/csa.pdf")
    print(result.validated_model.model_dump_json(indent=2))
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import Settings
from extraction.base_extractor import ExtractionResult
from ingestion.document_classifier import ClassificationResult, DocumentClassifier
from ingestion.document_loader import DocumentLoader, LoadedDocument
from schemas.base import ScheduleType

logger = logging.getLogger(__name__)


class CollateralStandardizer:
    """
    Single entry point for the full standardization pipeline.

    Args:
        settings: Optional Settings object (reads from env by default).
        llm_client: Optional pre-built Anthropic client (reused across calls).
    """

    def __init__(self, settings: Optional[Settings] = None, llm_client=None):
        self.settings = settings or Settings()
        self._loader = DocumentLoader(ocr_fallback=True)
        self._classifier = DocumentClassifier(llm_client=llm_client)
        self._llm_client = llm_client

    def process(self, file_path: str | Path) -> ExtractionResult:
        """
        Full pipeline: load → classify → extract → validate.
        Returns an ExtractionResult with the validated model and any flags.
        """
        path = Path(file_path)
        logger.info("Processing: %s", path.name)

        # 1. Load document
        doc: LoadedDocument = self._loader.load(path)
        logger.info("Loaded %d pages (%s)", doc.page_count, doc.raw_format)

        # 2. Classify
        classification: ClassificationResult = self._classifier.classify(doc.full_text)
        logger.info(
            "Classified as %s (conf=%.2f, method=%s)",
            classification.schedule_type,
            classification.type_confidence,
            classification.method,
        )

        # 3. Select extractor
        extractor = self._build_extractor(classification.schedule_type)

        # 4. Extract
        result: ExtractionResult = extractor.extract(doc)

        # 5. Stamp metadata onto the validated model
        if result.validated_model is not None:
            ts = datetime.now(tz=timezone.utc).isoformat()
            object.__setattr__(result.validated_model, "source_filename", path.name)
            object.__setattr__(result.validated_model, "extraction_model", self.settings.extraction_model)
            object.__setattr__(result.validated_model, "extraction_timestamp", ts)

        # 6. Log summary
        self._log_summary(result)
        return result

    def process_text(self, text: str, schedule_type: ScheduleType) -> ExtractionResult:
        """Process raw text when the type is already known."""
        extractor = self._build_extractor(schedule_type)
        return extractor.extract_from_text(text)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_extractor(self, schedule_type: ScheduleType):
        from extraction.im_extractor import IMExtractor
        from extraction.vm_extractor import VMExtractor
        from extraction.repo_extractor import REPOExtractor

        map_ = {
            ScheduleType.IM: IMExtractor,
            ScheduleType.VM: VMExtractor,
            ScheduleType.REPO: REPOExtractor,
        }
        cls = map_.get(schedule_type)
        if cls is None:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        return cls(settings=self.settings)

    def _log_summary(self, result: ExtractionResult) -> None:
        if result.validated_model is None:
            logger.warning(
                "Validation failed: %s", "; ".join(result.validation_errors)
            )
        else:
            n_low = len(result.low_confidence_fields)
            logger.info(
                "Extraction complete. %d chunks, %d low-confidence fields: %s",
                result.chunk_count,
                n_low,
                result.low_confidence_fields or "none",
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Standardize a collateral schedule document")
    parser.add_argument("--file", required=True, help="Path to schedule document")
    parser.add_argument("--output", default=None, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    standardizer = CollateralStandardizer()
    result = standardizer.process(args.file)

    if result.validated_model is not None:
        output_json = result.validated_model.model_dump_json(indent=2)
    else:
        output_json = json.dumps(
            {
                "error": "validation_failed",
                "errors": result.validation_errors,
                "raw": result.raw_json,
            },
            indent=2,
        )

    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
