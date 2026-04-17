"""
Build a HuggingFace Dataset from JSONL files for fine-tuning.

Input format (each line):
  {"input": "<raw schedule text>", "ground_truth": {<field dict>}}

Output: instruction-following examples in ChatML format, saved to disk as a
HuggingFace Dataset so they can be loaded directly by SFTTrainer.

Usage:
    python -m training.dataset_builder \
        --data-dir data/synthetic \
        --output data/training \
        --val-split 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from extraction.prompts.im_prompts import SYSTEM_PROMPT as IM_SYSTEM
from extraction.prompts.vm_prompts import SYSTEM_PROMPT as VM_SYSTEM
from extraction.prompts.repo_prompts import SYSTEM_PROMPT as REPO_SYSTEM

_SYSTEM_MAP = {
    "im": IM_SYSTEM,
    "vm": VM_SYSTEM,
    "repo": REPO_SYSTEM,
}


def _make_chatml(system: str, user_text: str, assistant_json: str) -> str:
    """Format a single training example in ChatML style."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_json}<|im_end|>"
    )


def build_examples(data_dir: Path) -> List[dict]:
    examples = []

    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        # Infer schedule type from filename prefix (im_, vm_, repo_)
        stem = jsonl_path.stem.split("_")[0].lower()
        system_prompt = _SYSTEM_MAP.get(stem)
        if system_prompt is None:
            print(f"[WARN] Unknown schedule type in filename {jsonl_path.name}, skipping")
            continue

        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                raw_text = sample["input"]
                gt = sample["ground_truth"]

                # Wrap each ground_truth field into ExtractedField format
                # so the model learns to output confidence scores too
                assistant_dict = {
                    k: {"value": v, "confidence": 1.0, "source_text": ""}
                    for k, v in gt.items()
                    if k != "schedule_type"
                }

                user_msg = (
                    f"Extract {stem.upper()} schedule fields from the following "
                    f"document fragment.\n\n"
                    f"--- BEGIN DOCUMENT FRAGMENT ---\n{raw_text}\n--- END ---\n\n"
                    "Return ONLY the JSON object."
                )

                text = _make_chatml(
                    system=system_prompt,
                    user_text=user_msg,
                    assistant_json=json.dumps(assistant_dict, indent=2),
                )
                examples.append({"text": text, "schedule_type": stem.upper()})

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--output", default="data/training")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets")

    examples = build_examples(Path(args.data_dir))
    print(f"Built {len(examples)} training examples")

    ds = Dataset.from_list(examples)
    split = ds.train_test_split(test_size=args.val_split, seed=42)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(out))
    print(f"Saved train/val split → {out}")
    print(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")


if __name__ == "__main__":
    main()
