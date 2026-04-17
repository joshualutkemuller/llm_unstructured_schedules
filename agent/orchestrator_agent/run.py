"""
OrchestratorAgent CLI — end-to-end collateral schedule processing.

Usage:
    python agent/orchestrator_agent/run.py --file path/to/csa.pdf
    python agent/orchestrator_agent/run.py --batch --dir path/to/docs/ --output results/
    python agent/orchestrator_agent/run.py --compare --file-a old.pdf --file-b new.pdf
    python agent/orchestrator_agent/run.py --prompt "Process all files in ./docs and export to JSON"
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="OrchestratorAgent: full-pipeline collateral schedule standardization."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file",   help="Single document to standardize")
    mode.add_argument("--batch",  action="store_true", help="Batch-process a directory")
    mode.add_argument("--compare",action="store_true", help="Compare two schedule documents")
    mode.add_argument("--prompt", help="Free-form natural language task")

    parser.add_argument("--dir",     help="Directory for batch mode")
    parser.add_argument("--output",  help="Output directory or file for results")
    parser.add_argument("--file-a",  dest="file_a", help="First file for comparison")
    parser.add_argument("--file-b",  dest="file_b", help="Second file for comparison")
    parser.add_argument("--export",  choices=["json", "csv"], default=None,
                        help="Export format for validated results")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model",   default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    from agent.orchestrator_agent.agent import OrchestratorAgent
    from config.settings import Settings

    agent = OrchestratorAgent(settings=Settings(anthropic_api_key=api_key), model=args.model)
    on_step = print if args.verbose else None

    if args.file:
        export_note = f" Export the validated result as {args.export}" + \
                      (f" to '{args.output}'." if args.output else ".") \
                      if args.export else ""
        task = (
            f"Standardize the collateral schedule at '{args.file}'. "
            "Run the full pipeline: load, classify, extract, validate, and review confidence. "
            f"Produce a structured report.{export_note}"
        )

    elif args.batch:
        if not args.dir:
            parser.error("--batch requires --dir")
        output_note = f" Save per-file JSON results to '{args.output}'." if args.output else ""
        task = (
            f"Batch-process all collateral schedule documents in '{args.dir}'.{output_note} "
            "Produce a summary table: file name, schedule type, status, and review flag count."
        )

    elif args.compare:
        if not args.file_a or not args.file_b:
            parser.error("--compare requires --file-a and --file-b")
        task = (
            f"Standardize both '{args.file_a}' and '{args.file_b}', then compare them "
            "field-by-field. Report every field that differs with before/after values, "
            "and every field that matches. Highlight any regulatory or economic changes."
        )

    else:
        task = args.prompt

    print(f"\n=== OrchestratorAgent ===")
    result = agent.run(task, on_step=on_step)
    print(f"\n{result}\n")


if __name__ == "__main__":
    main()
