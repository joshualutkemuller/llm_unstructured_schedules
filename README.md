# LLM Collateral Schedule Standardizer

An end-to-end system for ingesting unstructured collateral schedule documents
(IM, VM, REPO) across counterparties and standardizing them into typed,
validated templates — with a fine-tuned LLM at the core.

---

## Problem

Collateral schedules arrive as PDFs, Word documents, and Excel files with no
consistent structure. Legal language varies by counterparty, governing law
(NY / English / Japanese), and agreement vintage (ISDA 1994, 2016, GMRA 2000,
GMRA 2011). Ops teams spend hours manually keying fields into downstream
systems.

## Solution Architecture

```
Raw Document (PDF / DOCX / XLSX / scan)
        │
        ▼
┌──────────────────────┐
│  Ingestion Pipeline  │  ← multi-format loader, OCR, page chunking
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Document Classifier  │  ← zero-shot: IM / VM / REPO + governing law
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   LLM Extractor      │  ← schedule-type-specific prompts → JSON
│  (fine-tuned model)  │     + confidence scoring per field
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Schema Validator    │  ← Pydantic models, cross-field rules
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Standardized Output │  ← JSON / DB / downstream system
└──────────────────────┘
```

---

## Schedule Types

| Type | Agreement | Key Fields |
|------|-----------|------------|
| **IM** | ISDA CSA (1994/2016) | Threshold, MTA, IA, eligible collateral, haircuts, custodian |
| **VM** | ISDA CSA (2016 VM) | Threshold (=0), MTA, base currency, eligible currencies, interest |
| **REPO** | GMRA (2000/2011) | Margin ratio, eligible securities, repricing, substitution |

---

## Project Layout

```
llm_unstructured_schedules/
├── schemas/              # Pydantic data models (source of truth)
│   ├── base.py
│   ├── im_schedule.py
│   ├── vm_schedule.py
│   └── repo_schedule.py
├── ingestion/            # Document loading, OCR, classification
│   ├── document_loader.py
│   ├── ocr_processor.py
│   └── document_classifier.py
├── extraction/           # LLM prompt templates + field extractors
│   ├── base_extractor.py
│   ├── im_extractor.py
│   ├── vm_extractor.py
│   ├── repo_extractor.py
│   └── prompts/
│       ├── im_prompts.py
│       ├── vm_prompts.py
│       └── repo_prompts.py
├── training/             # Fine-tuning pipeline
│   ├── data_generator.py    # Synthetic schedule generation
│   ├── dataset_builder.py   # HuggingFace Dataset construction
│   ├── fine_tune.py         # LoRA/QLoRA training script
│   └── evaluate.py          # F1, field-level accuracy metrics
├── pipeline/             # Orchestration
│   └── standardizer.py
├── api/                  # FastAPI service
│   └── app.py
├── config/
│   └── settings.py
└── tests/
    ├── test_schemas.py
    ├── test_extraction.py
    └── fixtures/
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run extraction on a single document
python -m pipeline.standardizer --file path/to/csa.pdf --output output.json

# Start the API
uvicorn api.app:app --reload

# Generate synthetic training data
python -m training.data_generator --count 1000 --output data/synthetic/

# Fine-tune
python -m training.fine_tune \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training/ \
  --output models/collateral-v1/
```

---

## LLM Training Strategy

### 1. Data Collection
- **Synthetic generation**: Use GPT-4 / Claude to generate realistic schedule
  text with known ground-truth field values (see `training/data_generator.py`)
- **Human-annotated real docs**: Redacted real schedules labelled by ops/legal
- **Augmentation**: Paraphrase, reorder sections, introduce legal-language
  variants

### 2. Task Formulation
Instruction-following format (Alpaca/ChatML style):

```
SYSTEM: You are a collateral operations specialist. Extract structured fields
        from the following {schedule_type} schedule.

USER:   [raw schedule text chunk]

ASSISTANT: {"counterparty_id": "...", "threshold_party_a": ..., ...}
```

### 3. Fine-tuning
- Base model: Llama-3.1-8B-Instruct or Mistral-7B-Instruct
- Method: QLoRA (4-bit quantization + LoRA adapters) — fits on a single A100
- Library: `transformers` + `peft` + `trl` (SFTTrainer)
- Epochs: 3–5, learning rate 2e-4

### 4. Evaluation
- **Field-level F1**: per-field exact match and partial match
- **Schema validity rate**: % outputs that pass Pydantic validation
- **Hallucination rate**: fields present in output but absent from source text

---

## Confidence & Human-in-the-Loop

Each extracted field carries a `confidence` score (0–1). Fields below the
`LOW_CONFIDENCE_THRESHOLD` (default 0.7) are flagged for human review. The
review UI writes corrections back as training examples, closing the loop.
