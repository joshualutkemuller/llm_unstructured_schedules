"""
Extraction skills: run LLM field extraction and validate against schemas.

Skills exposed:
  - extract_fields          Extract all fields from a text chunk for a given type
  - validate_schedule       Validate a raw extracted dict against the Pydantic schema
  - get_low_confidence      Return fields below a confidence threshold
  - compare_schedules       Diff two standardized schedules field-by-field
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from skills.registry import SkillResult, _SkillEntry


class ExtractionSkills:

    def __init__(self, settings=None):
        from config.settings import Settings
        self.settings = settings or Settings()

    def entries(self) -> List[_SkillEntry]:
        return [
            _SkillEntry("extract_fields",       self.extract_fields,       EXTRACT_SCHEMA),
            _SkillEntry("validate_schedule",     self.validate_schedule,    VALIDATE_SCHEMA),
            _SkillEntry("get_low_confidence",    self.get_low_confidence,   LOW_CONF_SCHEMA),
            _SkillEntry("compare_schedules",     self.compare_schedules,    COMPARE_SCHEMA),
        ]

    # ── extract_fields ────────────────────────────────────────────────────────

    def extract_fields(self, text: str, schedule_type: str) -> SkillResult:
        """
        Run LLM extraction on a text chunk and return raw extracted fields.
        schedule_type must be one of: IM, VM, REPO.
        """
        schedule_type = schedule_type.upper()
        extractors = {
            "IM":   "extraction.im_extractor.IMExtractor",
            "VM":   "extraction.vm_extractor.VMExtractor",
            "REPO": "extraction.repo_extractor.REPOExtractor",
        }
        if schedule_type not in extractors:
            return SkillResult(success=False, error=f"Unknown schedule_type '{schedule_type}'. Use IM, VM, or REPO.")

        try:
            import importlib
            module_path, cls_name = extractors[schedule_type].rsplit(".", 1)
            extractor = getattr(importlib.import_module(module_path), cls_name)(settings=self.settings)
            result = extractor.extract_from_text(text)

            return SkillResult(
                success=True,
                data={
                    "raw_fields": result.raw_json,
                    "validation_errors": result.validation_errors,
                    "low_confidence_fields": result.low_confidence_fields,
                    "chunk_count": result.chunk_count,
                    "schema_valid": result.validated_model is not None,
                },
                metadata={"schedule_type": schedule_type},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

    # ── validate_schedule ─────────────────────────────────────────────────────

    def validate_schedule(self, fields: Dict[str, Any], schedule_type: str) -> SkillResult:
        """
        Validate a dict of extracted fields against the Pydantic schema.
        Returns validation errors and the serialized model if valid.
        """
        schedule_type = schedule_type.upper()
        schema_map = {
            "IM":   "schemas.im_schedule.IMSchedule",
            "VM":   "schemas.vm_schedule.VMSchedule",
            "REPO": "schemas.repo_schedule.REPOSchedule",
        }
        if schedule_type not in schema_map:
            return SkillResult(success=False, error=f"Unknown schedule_type '{schedule_type}'.")

        try:
            import importlib
            from schemas.base import ExtractedField

            module_path, cls_name = schema_map[schedule_type].rsplit(".", 1)
            schema_cls = getattr(importlib.import_module(module_path), cls_name)

            # Wrap raw values in ExtractedField if not already
            kwargs = {}
            for k, v in fields.items():
                if isinstance(v, dict) and "value" in v:
                    kwargs[k] = ExtractedField(**v)
                else:
                    kwargs[k] = v

            model = schema_cls(**kwargs)
            return SkillResult(
                success=True,
                data={
                    "valid": True,
                    "model": model.model_dump(),
                    "low_confidence_fields": model.low_confidence_fields(),
                },
            )
        except Exception as e:
            return SkillResult(
                success=True,  # skill ran — validation itself failed
                data={"valid": False, "errors": str(e)},
            )

    # ── get_low_confidence ────────────────────────────────────────────────────

    def get_low_confidence(
        self, fields: Dict[str, Any], threshold: float = 0.7
    ) -> SkillResult:
        """
        Given a raw extracted fields dict, return every field whose confidence
        is below the threshold — these need human review.
        """
        flagged = {}
        for name, entry in fields.items():
            if isinstance(entry, dict):
                conf = float(entry.get("confidence", 0.0))
                if conf < threshold:
                    flagged[name] = {
                        "value": entry.get("value"),
                        "confidence": conf,
                        "source_text": entry.get("source_text", ""),
                    }
        return SkillResult(
            success=True,
            data={
                "flagged_fields": flagged,
                "count": len(flagged),
                "threshold": threshold,
            },
        )

    # ── compare_schedules ─────────────────────────────────────────────────────

    def compare_schedules(
        self,
        schedule_a: Dict[str, Any],
        schedule_b: Dict[str, Any],
        label_a: str = "Schedule A",
        label_b: str = "Schedule B",
    ) -> SkillResult:
        """
        Field-by-field diff of two standardized schedule dicts.
        Useful for comparing counterparty terms or spotting renegotiated fields.
        """
        all_keys = set(schedule_a) | set(schedule_b)
        diffs = {}
        matches = []

        for key in sorted(all_keys):
            val_a = schedule_a.get(key)
            val_b = schedule_b.get(key)

            # Unwrap ExtractedField dicts
            if isinstance(val_a, dict) and "value" in val_a:
                val_a = val_a["value"]
            if isinstance(val_b, dict) and "value" in val_b:
                val_b = val_b["value"]

            if str(val_a) != str(val_b):
                diffs[key] = {label_a: val_a, label_b: val_b}
            else:
                matches.append(key)

        return SkillResult(
            success=True,
            data={
                "differences": diffs,
                "matching_fields": matches,
                "diff_count": len(diffs),
                "match_count": len(matches),
            },
        )


# ── Anthropic tool schemas ────────────────────────────────────────────────────

EXTRACT_SCHEMA = {
    "name": "extract_fields",
    "description": (
        "Use the LLM to extract structured fields from a collateral schedule text chunk. "
        "Returns raw field values with per-field confidence scores (0–1). "
        "Fields with confidence < 0.7 are flagged for human review. "
        "Call classify_schedule first if you don't know the schedule_type."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Raw schedule text (one chunk, max ~3000 tokens).",
            },
            "schedule_type": {
                "type": "string",
                "enum": ["IM", "VM", "REPO"],
                "description": "Type of collateral schedule.",
            },
        },
        "required": ["text", "schedule_type"],
    },
}

VALIDATE_SCHEMA = {
    "name": "validate_schedule",
    "description": (
        "Validate a dictionary of extracted schedule fields against the Pydantic schema "
        "for the given schedule type. Returns whether the output is schema-valid, "
        "any validation errors, and a list of low-confidence fields."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "description": "Extracted fields dict (from extract_fields output).",
            },
            "schedule_type": {
                "type": "string",
                "enum": ["IM", "VM", "REPO"],
            },
        },
        "required": ["fields", "schedule_type"],
    },
}

LOW_CONF_SCHEMA = {
    "name": "get_low_confidence",
    "description": (
        "Inspect a raw extracted fields dict and return every field whose confidence "
        "score is below the threshold. Use this to decide which fields need human review "
        "before writing results to a downstream system."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "description": "Raw extracted fields dict (from extract_fields).",
            },
            "threshold": {
                "type": "number",
                "description": "Confidence threshold below which a field is flagged (default 0.7).",
                "default": 0.7,
            },
        },
        "required": ["fields"],
    },
}

COMPARE_SCHEMA = {
    "name": "compare_schedules",
    "description": (
        "Compare two standardized schedule dicts field-by-field. "
        "Returns a diff showing which fields differ between the two counterparty schedules "
        "and which match. Useful for renegotiation analysis or change detection."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "schedule_a": {"type": "object", "description": "First schedule fields dict."},
            "schedule_b": {"type": "object", "description": "Second schedule fields dict."},
            "label_a":    {"type": "string", "description": "Label for schedule A (e.g. counterparty name)."},
            "label_b":    {"type": "string", "description": "Label for schedule B."},
        },
        "required": ["schedule_a", "schedule_b"],
    },
}
