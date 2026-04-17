"""
Pipeline skills: high-level orchestration over document loading + extraction.

Skills exposed:
  - standardize_document    Full end-to-end pipeline for one file
  - batch_standardize       Process a directory of documents
  - export_schedule         Serialize a standardized schedule to JSON or CSV
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from skills.registry import SkillResult, _SkillEntry


class PipelineSkills:

    def __init__(self, settings=None):
        from config.settings import Settings
        self.settings = settings or Settings()

    def entries(self) -> List[_SkillEntry]:
        return [
            _SkillEntry("standardize_document", self.standardize_document, STANDARDIZE_SCHEMA),
            _SkillEntry("batch_standardize",    self.batch_standardize,    BATCH_SCHEMA),
            _SkillEntry("export_schedule",      self.export_schedule,      EXPORT_SCHEMA),
        ]

    # ── standardize_document ──────────────────────────────────────────────────

    def standardize_document(
        self,
        file_path: str,
        schedule_type: Optional[str] = None,
    ) -> SkillResult:
        """
        Full pipeline: load → classify → extract → validate.
        Optionally override the schedule_type if already known.
        """
        from pipeline.standardizer import CollateralStandardizer
        from schemas.base import ScheduleType

        try:
            standardizer = CollateralStandardizer(settings=self.settings)

            if schedule_type:
                stype = ScheduleType(schedule_type.upper())
                path = Path(file_path)
                text = path.read_text(encoding="utf-8", errors="replace") if path.suffix == ".txt" else None
                if text:
                    result = standardizer.process_text(text, stype)
                else:
                    result = standardizer.process(file_path)
            else:
                result = standardizer.process(file_path)

            model_data = result.validated_model.model_dump() if result.validated_model else None

            return SkillResult(
                success=result.validated_model is not None,
                data={
                    "standardized": model_data,
                    "low_confidence_fields": result.low_confidence_fields,
                    "validation_errors": result.validation_errors,
                    "chunk_count": result.chunk_count,
                    "needs_review": len(result.low_confidence_fields) > 0,
                },
                metadata={"file": file_path},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

    # ── batch_standardize ─────────────────────────────────────────────────────

    def batch_standardize(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> SkillResult:
        """
        Process all supported documents in a directory.
        Saves per-file JSON results to output_dir if provided.
        Returns a summary with per-file status.
        """
        extensions = extensions or [".pdf", ".docx", ".xlsx", ".txt"]
        dir_path = Path(directory)

        if not dir_path.is_dir():
            return SkillResult(success=False, error=f"Not a directory: {directory}")

        files = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]
        if not files:
            return SkillResult(success=False, error=f"No supported files found in {directory}")

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        summary = []
        for file in sorted(files):
            result = self.standardize_document(str(file))
            entry = {
                "file": file.name,
                "success": result.success,
                "low_confidence_count": len(result.data.get("low_confidence_fields", [])) if result.data else 0,
                "error": result.error,
            }
            if result.success and output_dir:
                out_path = Path(output_dir) / f"{file.stem}.json"
                out_path.write_text(json.dumps(result.data, default=str, indent=2))
                entry["output_file"] = str(out_path)
            summary.append(entry)

        succeeded = sum(1 for s in summary if s["success"])
        return SkillResult(
            success=True,
            data={
                "total": len(files),
                "succeeded": succeeded,
                "failed": len(files) - succeeded,
                "results": summary,
            },
        )

    # ── export_schedule ───────────────────────────────────────────────────────

    def export_schedule(
        self,
        schedule: dict,
        format: str = "json",
        output_path: Optional[str] = None,
    ) -> SkillResult:
        """
        Serialize a standardized schedule dict to JSON or CSV.
        If output_path is provided, writes to disk; otherwise returns the content.
        """
        format = format.lower()
        if format not in ("json", "csv"):
            return SkillResult(success=False, error="format must be 'json' or 'csv'")

        try:
            if format == "json":
                content = json.dumps(schedule, default=str, indent=2)
            else:
                import csv, io
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(["field", "value", "confidence", "needs_review"])
                for field_name, entry in schedule.items():
                    if isinstance(entry, dict) and "value" in entry:
                        writer.writerow([
                            field_name,
                            entry.get("value", ""),
                            entry.get("confidence", ""),
                            entry.get("needs_review", False),
                        ])
                    else:
                        writer.writerow([field_name, entry, "", ""])
                content = buf.getvalue()

            if output_path:
                Path(output_path).write_text(content)
                return SkillResult(success=True, data={"written_to": output_path, "format": format})
            return SkillResult(success=True, data={"content": content, "format": format})

        except Exception as e:
            return SkillResult(success=False, error=str(e))


# ── Anthropic tool schemas ────────────────────────────────────────────────────

STANDARDIZE_SCHEMA = {
    "name": "standardize_document",
    "description": (
        "Run the full collateral schedule standardization pipeline on a single document: "
        "load → classify → extract fields → validate against schema. "
        "Returns the standardized field dict, any low-confidence fields that need human "
        "review, and validation errors. This is the primary skill for processing one file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the collateral schedule document (PDF/DOCX/XLSX/TXT).",
            },
            "schedule_type": {
                "type": "string",
                "enum": ["IM", "VM", "REPO"],
                "description": "Optional. Provide if the type is already known to skip classification.",
            },
        },
        "required": ["file_path"],
    },
}

BATCH_SCHEMA = {
    "name": "batch_standardize",
    "description": (
        "Process all collateral schedule documents in a directory. "
        "Returns a per-file summary of success/failure and low-confidence field counts. "
        "Optionally writes each result as a JSON file to an output directory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory containing schedule documents to process.",
            },
            "extensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File extensions to include (default: ['.pdf', '.docx', '.xlsx', '.txt']).",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional directory to write per-file JSON results.",
            },
        },
        "required": ["directory"],
    },
}

EXPORT_SCHEMA = {
    "name": "export_schedule",
    "description": (
        "Serialize a standardized schedule dict to JSON or CSV format. "
        "Optionally writes to a file path. Use this to hand off results to a "
        "downstream system, database loader, or ops review tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "object",
                "description": "Standardized schedule dict (from standardize_document).",
            },
            "format": {
                "type": "string",
                "enum": ["json", "csv"],
                "description": "Output format (default: json).",
            },
            "output_path": {
                "type": "string",
                "description": "Optional file path to write the output to.",
            },
        },
        "required": ["schedule"],
    },
}
