# TrainingAgent — Instructions

## Purpose
The TrainingAgent manages the **model improvement loop**: generating synthetic
training data, assembling fine-tuning datasets, and evaluating extraction quality.

This agent does not run inference on live documents — it feeds the pipeline
that makes the ExtractionAgent better over time.

---

## Skills

| Skill | Description |
|-------|-------------|
| `generate_synthetic_samples` | Generate realistic (document_text, ground_truth) pairs for IM, VM, or REPO using randomised legal-language templates. Returns a preview and optionally writes a JSONL file. |
| `evaluate_extraction` | Score a prediction against ground truth. Reports field-level exact match, coverage, and hallucination rates. |
| `build_training_dataset` | Convert JSONL files into a HuggingFace Dataset in ChatML format with train/val split — ready for `training/fine_tune.py`. |

---

## The Training Data Flywheel

```
generate_synthetic_samples
        │
        ▼
build_training_dataset   ←──────────────────────────────┐
        │                                                │
        ▼                                                │
  fine_tune.py (QLoRA)                                  │
        │                                                │
        ▼                                                │
  deploy ExtractionAgent                                 │
        │                                                │
        ▼                                             (corrections
  ops team reviews low-confidence fields          become new samples)
        │                                                │
        └────────────────────────────────────────────────┘
```

---

## Recommended Data Volumes

| Stage | Per Type | Total | Notes |
|-------|----------|-------|-------|
| Proof of concept | 100 | 300 | Validates the approach |
| Initial fine-tune | 500 | 1,500 | Usable production quality |
| Production model | 2,000 | 6,000 | Strong generalisation |
| With real data | 500 real + 1500 synth | 6,000 | Best results |

---

## Evaluation Metrics Explained

| Metric | Definition | Target |
|--------|-----------|--------|
| `exact_match_rate` | % fields where predicted value exactly matches ground truth | > 85% |
| `coverage_rate` | % non-null GT fields that were extracted (not left null) | > 90% |
| `schema_valid_rate` | % outputs that pass Pydantic validation | > 95% |
| `hallucination_rate` | % extracted values not found verbatim in source text | < 5% |

---

## Synthetic Data Quality Checks

The TrainingAgent automatically flags generated samples where:
- `threshold_party_a > 0` or `threshold_party_b > 0` on VM schedules
- `initial_margin_ratio < 1.0` on REPO schedules
- `counterparty_lei` length ≠ 20 characters
- `haircut_pct > 50` on any collateral row (implausible)

---

## Example Tasks

```bash
# Generate a balanced dataset (100 per type)
python agent/training_agent/run.py --task generate --count 100

# Generate only IM samples
python agent/training_agent/run.py --task generate --type IM --count 500 \
    --output data/synthetic/im_synthetic.jsonl

# Build HuggingFace dataset from JSONL files
python agent/training_agent/run.py --task build \
    --data-dir data/synthetic/ \
    --output data/training/

# Evaluate extraction quality
python agent/training_agent/run.py --task evaluate \
    --predictions predictions.json \
    --ground-truth ground_truth.json \
    --source-text tests/fixtures/sample_im.txt
```
