"""
Agentic loop for collateral schedule processing.

The agent receives a natural-language task (e.g. "standardize this IM CSA and
flag anything needing review"), autonomously decides which skills to call, and
iterates until it produces a final answer.

Built on the Anthropic tool-use API:
  - skills map 1-to-1 with Anthropic tool schemas
  - the loop handles tool_use → skill dispatch → tool_result → next step
  - max_iterations prevents runaway loops

Usage:
    agent = CollateralAgent()
    response = agent.run("Standardize tests/fixtures/sample_im.txt")
    print(response)

    # Stream progress to a callback:
    agent.run("Process all files in ./docs/", on_step=print)
"""

from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

from config.settings import Settings
from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """\
You are a collateral operations specialist agent with access to a suite of
tools for processing collateral schedule documents.

Your skills cover the full workflow:
  1. Loading documents (PDF, DOCX, XLSX, TXT)
  2. Classifying the schedule type (IM / VM / REPO) and governing law
  3. Extracting structured fields with per-field confidence scores
  4. Validating extracted data against typed schemas
  5. Flagging low-confidence fields for human review
  6. Comparing schedules across counterparties
  7. Exporting standardized results
  8. Generating synthetic training data and evaluating model quality

Guidelines:
- Always classify the document type before extracting fields if the type is unknown.
- After extracting, always call get_low_confidence to surface fields needing review.
- When processing a batch, summarize the results — count successes, failures,
  and total fields flagged for review.
- Be concise: report what was done, what succeeded, and what needs human attention.
- If a skill returns an error, diagnose and try an alternative approach once before
  reporting failure to the user.
"""


class CollateralAgent:
    """
    Agentic loop: natural-language task → skill calls → final answer.

    Args:
        settings:       Application settings (API key, model, etc.)
        registry:       Pre-built SkillRegistry. Defaults to SkillRegistry.default().
        max_iterations: Hard cap on tool-use rounds to prevent infinite loops.
        model:          Anthropic model to use for the agent.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        registry: Optional[SkillRegistry] = None,
        max_iterations: int = 15,
        model: Optional[str] = None,
    ):
        self.settings = settings or Settings()
        self.registry = registry or SkillRegistry.default(settings=self.settings)
        self.max_iterations = max_iterations
        self.model = model or self.settings.extraction_model
        self._client = self._build_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Run the agent on a task description.

        Args:
            task:    Natural-language instruction (e.g. "Standardize ./csa.pdf")
            on_step: Optional callback called with a status string at each step.

        Returns:
            The agent's final text response.
        """
        messages: List[dict] = [{"role": "user", "content": task}]
        tools = self.registry.tool_schemas()

        for iteration in range(self.max_iterations):
            logger.debug("Agent iteration %d", iteration + 1)

            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=AGENT_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract final text
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if response.stop_reason != "tool_use":
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

            # Dispatch all tool calls in this turn
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                skill_name = block.name
                skill_input = block.input

                if on_step:
                    on_step(f"[{iteration+1}] Calling skill: {skill_name}({json.dumps(skill_input, default=str)[:120]})")

                logger.info("Calling skill: %s", skill_name)
                result = self.registry.call_from_tool_use({"name": block.name, "input": block.input})

                if on_step and not result.success:
                    on_step(f"  ⚠ Skill error: {result.error}")

                tool_results.append(result.to_tool_result(block.id))

            messages.append({"role": "user", "content": tool_results})

        return "Agent reached maximum iterations without completing the task."

    # ── Client factory ────────────────────────────────────────────────────────

    def _build_client(self):
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
