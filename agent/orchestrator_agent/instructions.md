# OrchestratorAgent — Instructions

## Purpose
The OrchestratorAgent is the **master coordinator** for the collateral schedule
standardization platform. It is the primary interface for ops users and
automated workflows, and has access to all skills across all pipeline stages.

For most production use cases, you only need the OrchestratorAgent — it
delegates to the right skills automatically based on your task.

---

## When to Use Each Agent

| Use this agent | When you want to... |
|---------------|---------------------|
| **OrchestratorAgent** | End-to-end processing, batch jobs, mixed workflows, anything that spans multiple pipeline stages |
| **IngestAgent** | Debug document loading or classification in isolation |
| **ExtractionAgent** | Re-run extraction with a different model or threshold on pre-classified text |
| **ValidationAgent** | Validate a JSON file of extracted fields, compare schedules, or export results |
| **TrainingAgent** | Generate training data, evaluate model quality, build fine-tuning datasets |

---

## Skills Available

The OrchestratorAgent has access to **all 12 skills**:

**Document**: `load_document`, `classify_schedule`, `chunk_text`

**Extraction**: `extract_fields`, `validate_schedule`, `get_low_confidence`, `compare_schedules`

**Pipeline**: `standardize_document`, `batch_standardize`, `export_schedule`

**Training**: `generate_synthetic_samples`, `evaluate_extraction`, `build_training_dataset`

---

## Decision Trees

### Single Document
```
standardize_document(file_path)
    ├─ success → get_low_confidence → export_schedule (if requested)
    └─ failure → load_document + classify_schedule + extract_fields (decomposed)
```

### Batch
```
batch_standardize(directory)
    └─ summarise: succeeded, failed, total review flags
```

### Amendment Detection
```
standardize_document(old_file) → schedule_a
standardize_document(new_file) → schedule_b
compare_schedules(schedule_a, schedule_b)
    └─ report: N fields changed, list diffs
```

### Training Run
```
generate_synthetic_samples (IM + VM + REPO)
build_training_dataset
→ run training/fine_tune.py externally
→ evaluate_extraction on held-out set
```

---

## Output Standards

### Single file report
```
┌─────────────────────────────────────────────
│ File:         sample_im.txt
│ Type:         IM (ISDA 2016, New York Law)
│ Counterparty: Meridian Securities Ltd (LEI: 5493001RKX5PVOA2GM83)
│ Governing Law: NEW_YORK
│ Status:       NEEDS REVIEW
│ Review flags: 3 fields (custodian_account_party_a, simm_version, effective_date)
│ Export:       Ready (pending review)
└─────────────────────────────────────────────
```

### Batch report
```
Processed 8 files: 7 succeeded, 1 failed, 12 fields flagged for review.

File                    Type  Status        Review Flags
----------------------  ----  ------------  ------------
counterparty_a_vm.pdf   VM    VALIDATED     0
counterparty_b_im.pdf   IM    NEEDS REVIEW  4
counterparty_c_repo.txt REPO  VALIDATED     1
...
```

---

## Example Tasks

```bash
# Single file, full pipeline
python agent/orchestrator_agent/run.py --file path/to/csa.pdf

# Batch process a directory
python agent/orchestrator_agent/run.py --batch --dir path/to/schedules/ --output results/

# Compare two versions of the same CSA
python agent/orchestrator_agent/run.py \
    --compare --file-a old_csa.pdf --file-b new_csa.pdf

# Natural language task
python agent/orchestrator_agent/run.py \
    --prompt "Process all PDFs in ./incoming/, flag any IM schedules where threshold > 0, and export validated results to ./processed/"

# Generate 200 training samples per type and build dataset
python agent/orchestrator_agent/run.py \
    --prompt "Generate 200 synthetic samples for each of IM, VM, REPO and build a training dataset in data/training/"
```
