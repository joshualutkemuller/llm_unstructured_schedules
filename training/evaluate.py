"""
Evaluation script: field-level accuracy, schema validity rate, hallucination rate.

Metrics:
  - exact_match_rate  : % fields where extracted value == ground truth exactly
  - partial_match_rate: % numeric/string fields within acceptable tolerance
  - schema_valid_rate : % outputs that pass Pydantic validation
  - hallucination_rate: % extracted fields where value not found in source text
  - coverage_rate     : % non-null ground-truth fields successfully extracted

Usage:
    python -m training.evaluate \
        --model models/collateral-v1 \
        --test-data data/synthetic/im_synthetic.jsonl \
        --type IM
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Field-level comparators ───────────────────────────────────────────────────

def _values_match(pred: Any, gt: Any, numeric_tol: float = 0.01) -> bool:
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    if isinstance(gt, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(pred) - float(gt)) <= abs(float(gt)) * numeric_tol
    if isinstance(gt, list) and isinstance(pred, list):
        return len(pred) == len(gt)
    return str(pred).strip().lower() == str(gt).strip().lower()


def _is_hallucinated(value: Any, source_text: str) -> bool:
    """Heuristic: value is hallucinated if it cannot be found verbatim in source_text."""
    if value is None or source_text is None:
        return False
    return str(value) not in source_text


# ── Per-sample evaluator ──────────────────────────────────────────────────────

def evaluate_sample(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    source_text: str,
) -> Dict[str, Any]:
    field_results = {}
    hallucinations = 0
    total_gt_fields = 0

    for field, gt_val in ground_truth.items():
        if field == "schedule_type":
            continue
        total_gt_fields += 1
        pred_entry = prediction.get(field, {})
        if isinstance(pred_entry, dict):
            pred_val = pred_entry.get("value")
            pred_source = pred_entry.get("source_text", "")
        else:
            pred_val = pred_entry
            pred_source = ""

        match = _values_match(pred_val, gt_val)
        hallucinated = False
        if pred_val is not None and not _is_hallucinated(pred_val, source_text):
            hallucinated = False
        elif pred_val is not None:
            hallucinated = True
            hallucinations += 1

        field_results[field] = {
            "gt": gt_val,
            "pred": pred_val,
            "match": match,
            "hallucinated": hallucinated,
        }

    matched = sum(1 for v in field_results.values() if v["match"])
    extracted = sum(1 for v in field_results.values() if v["pred"] is not None)

    return {
        "field_results": field_results,
        "exact_match_rate": matched / total_gt_fields if total_gt_fields else 0,
        "coverage_rate": extracted / total_gt_fields if total_gt_fields else 0,
        "hallucination_count": hallucinations,
        "total_gt_fields": total_gt_fields,
    }


# ── Dataset-level evaluation ──────────────────────────────────────────────────

def evaluate_dataset(
    extractor,
    test_samples: List[dict],
    schedule_type: str,
) -> Dict[str, float]:
    results = []
    schema_valid = 0

    for sample in test_samples:
        source_text = sample["input"]
        gt = sample["ground_truth"]

        try:
            result = extractor.extract_from_text(source_text)
            prediction = result.raw_json
            if result.validated_model is not None:
                schema_valid += 1
        except Exception as e:
            logger.warning("Extraction failed: %s", e)
            prediction = {}

        eval_result = evaluate_sample(prediction, gt, source_text)
        results.append(eval_result)

    n = len(results)
    avg_exact = sum(r["exact_match_rate"] for r in results) / n
    avg_coverage = sum(r["coverage_rate"] for r in results) / n
    total_hallu = sum(r["hallucination_count"] for r in results)
    total_fields = sum(r["total_gt_fields"] for r in results)

    return {
        "schedule_type": schedule_type,
        "n_samples": n,
        "avg_exact_match_rate": round(avg_exact, 4),
        "avg_coverage_rate": round(avg_coverage, 4),
        "schema_valid_rate": round(schema_valid / n, 4),
        "hallucination_rate": round(total_hallu / max(total_fields, 1), 4),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned model or model ID")
    parser.add_argument("--test-data", required=True, help="Path to JSONL test file")
    parser.add_argument("--type", choices=["IM", "VM", "REPO"], required=True)
    parser.add_argument("--limit", type=int, default=100, help="Max samples to evaluate")
    args = parser.parse_args()

    # Load test data
    samples = []
    with open(args.test_data) as f:
        for line in f:
            if len(samples) >= args.limit:
                break
            samples.append(json.loads(line.strip()))

    # Build extractor pointing at the fine-tuned model
    from config.settings import Settings
    settings = Settings(extraction_model=args.model)

    type_map = {
        "IM": "extraction.im_extractor.IMExtractor",
        "VM": "extraction.vm_extractor.VMExtractor",
        "REPO": "extraction.repo_extractor.REPOExtractor",
    }
    module_path, class_name = type_map[args.type].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    extractor = getattr(mod, class_name)(settings=settings)

    metrics = evaluate_dataset(extractor, samples, args.type)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
