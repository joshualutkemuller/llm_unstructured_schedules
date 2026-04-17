"""
ValidationAgent CLI — validate, QA-review, compare, and export schedule data.

Usage:
    python agent/validation_agent/run.py --fields extracted.json --type IM
    python agent/validation_agent/run.py --fields extracted.json --type VM --export json --output out.json
    python agent/validation_agent/run.py --compare --schedule-a a.json --schedule-b b.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(
        description="ValidationAgent: validate, review, compare, and export schedule data."
    )
    parser.add_argument("--fields",      help="Path to extracted fields JSON file")
    parser.add_argument("--type",        choices=["IM", "VM", "REPO"],
                        help="Schedule type (required unless --compare)")
    parser.add_argument("--threshold",   type=float, default=0.7,
                        help="Confidence threshold for review flags (default: 0.7)")
    parser.add_argument("--export",      choices=["json", "csv"],
                        help="Export format if validation passes")
    parser.add_argument("--output",      help="Output file path for export")
    parser.add_argument("--compare",     action="store_true",
                        help="Run in comparison mode")
    parser.add_argument("--schedule-a",  dest="schedule_a",
                        help="Path to first schedule JSON (comparison mode)")
    parser.add_argument("--schedule-b",  dest="schedule_b",
                        help="Path to second schedule JSON (comparison mode)")
    parser.add_argument("--label-a",     default="Schedule A")
    parser.add_argument("--label-b",     default="Schedule B")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--model",       default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    from agent.validation_agent.agent import ValidationAgent
    from config.settings import Settings

    agent = ValidationAgent(settings=Settings(anthropic_api_key=api_key), model=args.model)
    on_step = print if args.verbose else None

    if args.compare:
        if not args.schedule_a or not args.schedule_b:
            parser.error("--compare requires --schedule-a and --schedule-b")
        sched_a = _load_json(args.schedule_a)
        sched_b = _load_json(args.schedule_b)
        task = (
            f"Compare the two collateral schedules. "
            f"Schedule A ('{args.label_a}'): {json.dumps(sched_a)[:2000]}. "
            f"Schedule B ('{args.label_b}'): {json.dumps(sched_b)[:2000]}. "
            "Report every field that differs, every field that matches, and the total diff count."
        )
    else:
        if not args.fields or not args.type:
            parser.error("--fields and --type are required unless using --compare")
        fields = _load_json(args.fields)
        export_instruction = (
            f" If validation passes, export the result as {args.export}"
            + (f" to '{args.output}'" if args.output else "")
            + "."
            if args.export else ""
        )
        task = (
            f"Validate these {args.type} schedule fields against the schema. "
            f"Apply quality gates. Use confidence threshold {args.threshold}. "
            f"Fields: {json.dumps(fields)[:3000]}.{export_instruction}"
        )

    print(f"\n=== ValidationAgent ===")
    result = agent.run(task, on_step=on_step)
    print(f"\n{result}\n")


if __name__ == "__main__":
    main()
