"""
TrainingAgent CLI — generate training data, build datasets, evaluate models.

Usage:
    python agent/training_agent/run.py --task generate --count 100
    python agent/training_agent/run.py --task generate --type IM --count 500 --output data/synthetic/im_synthetic.jsonl
    python agent/training_agent/run.py --task build --data-dir data/synthetic/ --output data/training/
    python agent/training_agent/run.py --task evaluate --predictions pred.json --ground-truth gt.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="TrainingAgent: synthetic data generation, dataset building, evaluation."
    )
    parser.add_argument("--task",         required=True, choices=["generate", "build", "evaluate"],
                        help="Task to perform")
    parser.add_argument("--type",         choices=["IM", "VM", "REPO", "ALL"], default="ALL",
                        help="Schedule type for generation (default: ALL)")
    parser.add_argument("--count",        type=int, default=100,
                        help="Samples to generate per type (default: 100)")
    parser.add_argument("--output",       help="Output path (JSONL for generate, dir for build)")
    parser.add_argument("--data-dir",     dest="data_dir", default="data/synthetic/",
                        help="JSONL source directory for build task")
    parser.add_argument("--predictions",  help="Path to predictions JSON (evaluate task)")
    parser.add_argument("--ground-truth", dest="ground_truth",
                        help="Path to ground truth JSON (evaluate task)")
    parser.add_argument("--source-text",  dest="source_text",
                        help="Path to source text file (for hallucination detection)")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--model",        default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    from agent.training_agent.agent import TrainingAgent
    from config.settings import Settings

    agent = TrainingAgent(settings=Settings(anthropic_api_key=api_key), model=args.model)
    on_step = print if args.verbose else None

    if args.task == "generate":
        types = ["IM", "VM", "REPO"] if args.type == "ALL" else [args.type]
        type_str = ", ".join(types)
        output_note = f" Write each to '{args.output}'." if args.output else ""
        task = (
            f"Generate {args.count} synthetic training samples for each of these "
            f"schedule types: {type_str}.{output_note} "
            "Show a preview of the first sample for each type and confirm "
            "all ground truth values are plausible."
        )

    elif args.task == "build":
        output = args.output or "data/training/"
        task = (
            f"Build a HuggingFace training dataset from the JSONL files in "
            f"'{args.data_dir}'. Save to '{output}' with a 10% validation split. "
            "Report train/val counts and which schedule types are represented."
        )

    elif args.task == "evaluate":
        if not args.predictions or not args.ground_truth:
            parser.error("--evaluate requires --predictions and --ground-truth")
        preds = json.loads(Path(args.predictions).read_text())
        gt = json.loads(Path(args.ground_truth).read_text())
        src = Path(args.source_text).read_text() if args.source_text else ""
        task = (
            f"Evaluate this extraction against ground truth. "
            f"Predictions: {json.dumps(preds)[:2000]}. "
            f"Ground truth: {json.dumps(gt)[:2000]}. "
            f"Source text (for hallucination detection): {src[:500]}. "
            "Report exact_match_rate, coverage_rate, schema_valid_rate, and hallucination_rate. "
            "Explain what each score means and recommend next steps."
        )

    print(f"\n=== TrainingAgent ({args.task}) ===")
    result = agent.run(task, on_step=on_step)
    print(f"\n{result}\n")


if __name__ == "__main__":
    main()
