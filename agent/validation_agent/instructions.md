# ValidationAgent — Instructions

## Purpose
The ValidationAgent is the **quality gate** of the pipeline. It takes raw
extracted field dicts from the ExtractionAgent and:

1. Validates them against typed Pydantic schemas
2. Enforces domain-specific quality rules (regulatory constraints, data integrity)
3. Flags fields requiring human review
4. Compares schedules across counterparties or versions
5. Exports validated results for downstream systems

---

## Skills

| Skill | Description |
|-------|-------------|
| `validate_schedule` | Validates extracted fields against the IM / VM / REPO Pydantic schema. Catches type errors, missing required fields, out-of-range values. |
| `get_low_confidence` | Returns all fields below a confidence threshold. Default 0.7; raise to 0.85 for Tier-1 counterparties. |
| `compare_schedules` | Field-by-field diff of two schedule dicts — useful for detecting amendments and renegotiations. |
| `export_schedule` | Serializes validated schedule to JSON or CSV for downstream use. |

---

## Quality Gates

These rules are enforced **in addition** to Pydantic schema validation:

| Rule | Schedule Type | Severity |
|------|--------------|----------|
| `threshold_party_a > 0` or `threshold_party_b > 0` | VM | **BLOCK** — regulatory breach under EMIR/Dodd-Frank |
| `initial_margin_ratio < 1.0` | REPO | **BLOCK** — economically impossible |
| `rehypothecation_permitted = True` | IM | **WARN** — restricted under UMR Phase 1–6 |
| `counterparty_lei` null or not 20 chars | All | **WARN** — required for trade reporting |
| `effective_date` null | All | **WARN** — required for legal validity |
| Any field `confidence < 0.7` | All | **REVIEW** — human must confirm |

---

## Validation Flow

```
receive extracted_fields + schedule_type
    │
    ▼
validate_schedule(fields, schedule_type)
    │
    ├─ FAILED? → report all errors, BLOCK export
    │
    ▼
get_low_confidence(fields, threshold)
    │
    ▼
apply quality gate rules
    │
    ├─ any BLOCK triggers? → stop, report
    ├─ any WARN/REVIEW?    → flag, continue
    │
    ▼
export_schedule(validated_fields, format) ← only if no BLOCKs
```

---

## Comparison Mode

When comparing two schedules (e.g., current vs. amended):

```bash
python agent/validation_agent/run.py \
    --compare \
    --schedule-a extracted_2023.json \
    --schedule-b extracted_2024.json \
    --label-a "2023 CSA" \
    --label-b "2024 Amendment"
```

The agent reports:
- Fields that changed (with before/after values)
- Fields that are new in the amendment
- Fields that were removed
- Total diff count

---

## Export Formats

| Format | Use case |
|--------|----------|
| `json` | API ingestion, audit trail, downstream LLM inputs |
| `csv`  | Ops team spreadsheet review, collateral management system upload |

---

## Example Tasks

```bash
# Validate and export an IM extraction result
python agent/validation_agent/run.py \
    --fields extracted_im.json \
    --type IM \
    --export json \
    --output validated_im.json

# Compare two VM schedules
python agent/validation_agent/run.py \
    --compare \
    --schedule-a vm_2023.json \
    --schedule-b vm_2024.json

# Strict review for Tier-1 counterparty
python agent/validation_agent/run.py \
    --fields extracted_im.json \
    --type IM \
    --threshold 0.85
```
