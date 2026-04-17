"""
IngestAgent CLI — load, classify, and chunk a collateral schedule document.

Usage:
    python agent/ingest_agent/run.py --file path/to/csa.pdf
    python agent/ingest_agent/run.py --file path/to/csa.pdf --verbose
    python agent/ingest_agent/run.py --text "CREDIT SUPPORT ANNEX dated..."
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="IngestAgent: load, classify, and chunk a collateral schedule document."
    )
    parser.add_argument("--file",    help="Path to schedule document (PDF/DOCX/XLSX/TXT)")
    parser.add_argument("--text",    help="Raw document text (alternative to --file)")
    parser.add_argument("--verbose", action="store_true", help="Print each skill call as it runs")
    parser.add_argument("--model",   default=None, help="Override Anthropic model")
    args = parser.parse_args()

    if not args.file and not args.text:
        parser.error("Provide --file or --text")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    from agent.ingest_agent.agent import IngestAgent
    from config.settings import Settings

    agent = IngestAgent(settings=Settings(anthropic_api_key=api_key), model=args.model)
    on_step = print if args.verbose else None

    if args.file:
        task = (
            f"Load and classify the collateral schedule document at '{args.file}'. "
            "If the document is long, chunk it. Report the document type, governing law, "
            "confidence scores, OCR pages if any, chunks produced, and readiness status."
        )
    else:
        task = (
            f"Classify the following collateral schedule text and chunk it if needed.\n\n"
            f"{args.text}"
        )

    print(f"\n=== IngestAgent ===")
    result = agent.run(task, on_step=on_step)
    print(f"\n{result}\n")


if __name__ == "__main__":
    main()
