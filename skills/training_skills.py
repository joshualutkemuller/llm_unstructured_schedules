"""
Training skills: generate synthetic data, build datasets, and evaluate models.

Skills exposed:
  - generate_synthetic_samples    Produce (text, ground_truth) pairs for a schedule type
  - evaluate_extraction           Score an extraction against ground truth
  - build_training_dataset        Convert JSONL files into a HuggingFace Dataset
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from skills.registry import SkillResult, _SkillEntry


class TrainingSkills:

    def entries(self) -> List[_SkillEntry]:
        return [
            _SkillEntry("generate_synthetic_samples", self.generate_synthetic_samples, GENERATE_SCHEMA),
            _SkillEntry("evaluate_extraction",        self.evaluate_extraction,        EVALUATE_SCHEMA),
            _SkillEntry("build_training_dataset",     self.build_training_dataset,     BUILD_DS_SCHEMA),
        ]

    # ── generate_synthetic_samples ────────────────────────────────────────────

    def generate_synthetic_samples(
        self, schedule_type: str, count: int = 10, output_path: Optional[str] = None
    ) -> SkillResult:
        """
        Generate synthetic (document_text, ground_truth) pairs for a schedule type.
        Returns the samples and optionally writes a JSONL file.
        """
        import json
        from training.data_generator import generate_im_sample, generate_vm_sample, generate_repo_sample

        generators = {"IM": generate_im_sample, "VM": generate_vm_sample, "REPO": generate_repo_sample}
        schedule_type = schedule_type.upper()
        if schedule_type not in generators:
            return SkillResult(success=False, error=f"Unknown type '{schedule_type}'. Use IM, VM, or REPO.")

        gen = generators[schedule_type]
        samples = []
        for _ in range(count):
            text, gt = gen()
            samples.append({"input": text, "ground_truth": gt})

        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

        return SkillResult(
            success=True,
            data={
                "count": len(samples),
                "schedule_type": schedule_type,
                "samples": samples[:3],   # return first 3 as preview
                "output_path": output_path,
            },
        )

    # ── evaluate_extraction ───────────────────────────────────────────────────

    def evaluate_extraction(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
        source_text: str = "",
    ) -> SkillResult:
        """
        Score an extraction result against known ground truth.
        Returns field-level match rates, coverage, and hallucination count.
        """
        from training.evaluate import evaluate_sample
        try:
            result = evaluate_sample(prediction, ground_truth, source_text)
            return SkillResult(success=True, data=result)
        except Exception as e:
            return SkillResult(success=False, error=str(e))

    # ── build_training_dataset ────────────────────────────────────────────────

    def build_training_dataset(
        self, data_dir: str, output_dir: str, val_split: float = 0.1
    ) -> SkillResult:
        """
        Convert JSONL files in data_dir into a HuggingFace Dataset saved to output_dir.
        Splits into train / validation sets.
        """
        from pathlib import Path
        from training.dataset_builder import build_examples

        try:
            from datasets import Dataset
        except ImportError:
            return SkillResult(success=False, error="pip install datasets")

        try:
            examples = build_examples(Path(data_dir))
            if not examples:
                return SkillResult(success=False, error=f"No examples found in {data_dir}")

            ds = Dataset.from_list(examples)
            split = ds.train_test_split(test_size=val_split, seed=42)
            split.save_to_disk(output_dir)

            return SkillResult(
                success=True,
                data={
                    "total_examples": len(examples),
                    "train_count": len(split["train"]),
                    "val_count": len(split["test"]),
                    "output_dir": output_dir,
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))


# ── Anthropic tool schemas ────────────────────────────────────────────────────

GENERATE_SCHEMA = {
    "name": "generate_synthetic_samples",
    "description": (
        "Generate synthetic collateral schedule documents with known ground-truth field "
        "values for model training. Returns (document_text, ground_truth) pairs. "
        "Optionally writes a JSONL file for use in the training pipeline."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "schedule_type": {
                "type": "string",
                "enum": ["IM", "VM", "REPO"],
                "description": "Type of schedule to generate.",
            },
            "count": {
                "type": "integer",
                "description": "Number of samples to generate (default 10).",
                "default": 10,
            },
            "output_path": {
                "type": "string",
                "description": "Optional JSONL file path to write samples to.",
            },
        },
        "required": ["schedule_type"],
    },
}

EVALUATE_SCHEMA = {
    "name": "evaluate_extraction",
    "description": (
        "Score an LLM extraction result against known ground truth. "
        "Returns field-level exact match rate, coverage rate (non-null GT fields extracted), "
        "and hallucination count (values not found in source text). "
        "Use this during model development to measure extraction quality."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "object",
                "description": "Extracted fields dict (from extract_fields).",
            },
            "ground_truth": {
                "type": "object",
                "description": "Known correct field values.",
            },
            "source_text": {
                "type": "string",
                "description": "Original document text (used for hallucination detection).",
            },
        },
        "required": ["prediction", "ground_truth"],
    },
}

BUILD_DS_SCHEMA = {
    "name": "build_training_dataset",
    "description": (
        "Convert JSONL synthetic data files into a HuggingFace Dataset with train/val split, "
        "formatted in ChatML style for SFTTrainer fine-tuning. "
        "Run this after generate_synthetic_samples and before fine-tuning."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "data_dir": {
                "type": "string",
                "description": "Directory containing *_synthetic.jsonl files.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save the HuggingFace Dataset.",
            },
            "val_split": {
                "type": "number",
                "description": "Fraction of data to use for validation (default 0.1).",
                "default": 0.1,
            },
        },
        "required": ["data_dir", "output_dir"],
    },
}
