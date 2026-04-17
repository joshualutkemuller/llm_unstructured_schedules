"""
Example: deploy the CollateralAgent for various agentic tasks.

Run any example:
    python examples/run_agent.py --task standardize --file tests/fixtures/sample_im.txt
    python examples/run_agent.py --task batch       --dir  tests/fixtures/
    python examples/run_agent.py --task compare     --file tests/fixtures/sample_im.txt
    python examples/run_agent.py --task generate    --type IM --count 5
    python examples/run_agent.py --task custom      --prompt "What fields does the IM fixture have with confidence below 0.8?"
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def _step_printer(msg: str) -> None:
    print(f"  {msg}")


def task_standardize(agent, file: str) -> None:
    print(f"\n── Standardizing: {file} ──")
    response = agent.run(
        f"Standardize the collateral schedule at '{file}'. "
        "Extract all fields, validate the schema, then list any fields that need human review.",
        on_step=_step_printer,
    )
    print("\n── Agent response ──")
    print(response)


def task_batch(agent, directory: str) -> None:
    print(f"\n── Batch processing: {directory} ──")
    response = agent.run(
        f"Process all collateral schedule documents in the directory '{directory}'. "
        "Summarise how many succeeded, how many failed, and the total number of "
        "fields flagged for human review across all documents.",
        on_step=_step_printer,
    )
    print("\n── Agent response ──")
    print(response)


def task_compare(agent, file: str) -> None:
    print(f"\n── Compare: {file} against itself (demo diff) ──")
    response = agent.run(
        f"Load and extract fields from '{file}'. "
        "Then compare those fields against themselves as a sanity check — "
        "there should be zero differences. Report what you find.",
        on_step=_step_printer,
    )
    print("\n── Agent response ──")
    print(response)


def task_generate(agent, schedule_type: str, count: int) -> None:
    print(f"\n── Generating {count} synthetic {schedule_type} samples ──")
    response = agent.run(
        f"Generate {count} synthetic {schedule_type} collateral schedule samples "
        "and show me a preview of the first one (document text and ground truth fields).",
        on_step=_step_printer,
    )
    print("\n── Agent response ──")
    print(response)


def task_custom(agent, prompt: str) -> None:
    print(f"\n── Custom task ──")
    response = agent.run(prompt, on_step=_step_printer)
    print("\n── Agent response ──")
    print(response)


def main():
    parser = argparse.ArgumentParser(description="Run the CollateralAgent")
    parser.add_argument("--task", required=True,
                        choices=["standardize", "batch", "compare", "generate", "custom"])
    parser.add_argument("--file",   default="tests/fixtures/sample_im.txt")
    parser.add_argument("--dir",    default="tests/fixtures/")
    parser.add_argument("--type",   default="IM", choices=["IM", "VM", "REPO"])
    parser.add_argument("--count",  type=int, default=5)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model",  default=None,
                        help="Override the Anthropic model (e.g. claude-opus-4-7)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    from agent.loop import CollateralAgent
    from config.settings import Settings

    settings = Settings(anthropic_api_key=api_key)
    agent = CollateralAgent(settings=settings, model=args.model)

    dispatch = {
        "standardize": lambda: task_standardize(agent, args.file),
        "batch":       lambda: task_batch(agent, args.dir),
        "compare":     lambda: task_compare(agent, args.file),
        "generate":    lambda: task_generate(agent, args.type, args.count),
        "custom":      lambda: task_custom(agent, args.prompt),
    }
    dispatch[args.task]()


if __name__ == "__main__":
    main()
