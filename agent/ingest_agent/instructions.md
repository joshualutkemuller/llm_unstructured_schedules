# IngestAgent — Instructions

## Purpose
The IngestAgent is the **first stage** of the collateral schedule pipeline.
It is responsible for turning raw document files (PDF, DOCX, XLSX, TXT) into
clean, classified, chunked text that downstream agents can process.

It does **not** extract schedule fields — that belongs to the ExtractionAgent.

---

## Skills

| Skill | Description |
|-------|-------------|
| `load_document` | Loads a file from disk. Supports PDF (with OCR fallback for scans), DOCX, XLSX, and TXT. Returns page-level text and metadata. |
| `classify_schedule` | Identifies the schedule type (IM / VM / REPO) and governing law using keyword heuristics, with optional LLM fallback for ambiguous documents. |
| `chunk_text` | Splits document text into overlapping chunks of ≤3000 tokens for downstream LLM processing. |

---

## Decision Logic

```
receive file_path
    │
    ▼
load_document(file_path)
    │
    ├─ error? → report failure, stop
    │
    ▼
classify_schedule(full_text)
    │
    ├─ type_confidence < 0.6? → flag as NEEDS REVIEW, stop
    │
    ▼
page_count > 1 OR word_count > 2500?
    ├─ YES → chunk_text(full_text)
    └─ NO  → single chunk, no split needed
    │
    ▼
report: type, law, confidence, chunks, OCR pages
```

---

## Output Contract

The IngestAgent always ends with a structured status report:

```
Document:           sample_im.txt (txt, 1 page)
Schedule type:      IM (confidence: 0.87)
Governing law:      NEW_YORK (confidence: 0.95)
Classification:     heuristic
OCR pages:          none
Chunks produced:    1
Status:             READY FOR EXTRACTION
```

If classification confidence is below 0.6:
```
Status: NEEDS REVIEW — low classification confidence (0.42).
        Document may not be a standard collateral schedule.
        Recommend manual review before extraction.
```

---

## Escalation Rules

- **confidence < 0.6**: Halt and flag for human review.
- **load_document error**: Report the error and the file format. Do not retry with different parameters.
- **All pages OCR**: Warn that OCR quality may affect extraction accuracy.
- **XLSX files**: Note that tabular schedules may need manual mapping — structured tables are harder for chunked extraction.

---

## Example Tasks

```bash
# Single file
python agent/ingest_agent/run.py --file path/to/csa.pdf

# With verbose step output
python agent/ingest_agent/run.py --file path/to/csa.pdf --verbose

# Provide raw text directly (skips load_document)
python agent/ingest_agent/run.py --text "CREDIT SUPPORT ANNEX..." 
```
