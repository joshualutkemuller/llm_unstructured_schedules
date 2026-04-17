"""
ExtractionAgent CLI — extract structured fields from a collateral schedule.

Usage:
    python agent/extraction_agent/run.py --file tests/fixtures/sample_im.txt --type IM
    python agent/extraction_agent/run.py --file path/to/csa.pdf --type VM --threshold 0.85
    python agent/extraction_agent/run.py --file path/to/repo.txt --type REPO --verbose
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="ExtractionAgent: extract structured fields from a collateral schedule."
    )
    parser.add_argument("--file",      required=True, help="Path to schedule document or text file")
    parser.add_argument("--type",      required=True, choices=["IM", "VM", "REPO"],
                        help="Schedule type (IM / VM / REPO)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold for flagging review fields (default: 0.7)")
    parser.add_argument("--verbose",   action="store_true", help="Print each skill call")
    parser.add_argument("--model",     default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    from agent.extraction_agent.agent import ExtractionAgent
    from config.settings import Settings

    agent = ExtractionAgent(settings=Settings(anthropic_api_key=api_key), model=args.model)
    on_step = print if args.verbose else None

    task = (
        f"Extract all {args.type} schedule fields from the document at '{args.file}'. "
        f"Use a confidence threshold of {args.threshold} when reporting low-confidence fields. "
        "Summarise: total fields, non-null fields, and which fields need human review."
    )

    print(f"\n=== ExtractionAgent ({args.type}) ===")
    result = agent.run(task, on_step=on_step)
    print(f"\n{result}\n")


if __name__ == "__main__":
    main()
