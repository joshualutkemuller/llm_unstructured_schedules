"""
BaseAgent: shared Anthropic tool-use loop inherited by all five agents.

Each subclass provides:
  - SYSTEM_PROMPT  : the agent's identity, scope, and decision rules
  - skill_names    : the subset of registered skills this agent may use
  - max_iterations : hard cap on tool-use rounds (override per agent)

The loop:
  1. Build a filtered tool schema list from skill_names
  2. Call Claude with tools + system prompt
  3. If stop_reason == tool_use: dispatch each tool_use block through
     the SkillRegistry, append tool_result, repeat
  4. If stop_reason == end_turn: return final text
"""

from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

from config.settings import Settings
from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Inherit this class to create a focused agent with a fixed skill subset.

    Subclass pattern:
        class MyAgent(BaseAgent):
            SYSTEM_PROMPT = "You are a ..."
            skill_names   = ["load_document", "classify_schedule"]
            max_iterations = 10
    """

    SYSTEM_PROMPT: str = "You are a helpful assistant."
    skill_names: List[str] = []   # empty = all skills available
    max_iterations: int = 15

    def __init__(
        self,
        settings: Optional[Settings] = None,
        registry: Optional[SkillRegistry] = None,
        model: Optional[str] = None,
    ):
        self.settings = settings or Settings()
        self.registry = registry or SkillRegistry.default(settings=self.settings)
        self.model = model or self.settings.extraction_model
        self._client = self._build_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute a natural-language task, returning the agent's final answer.

        Args:
            task:    Natural-language instruction.
            on_step: Optional callback(str) fired at each tool-use step —
                     useful for streaming progress to a UI or log.
        """
        tools = self._scoped_tools()
        messages: List[dict] = [{"role": "user", "content": task}]

        for iteration in range(self.max_iterations):
            logger.debug("[%s] iteration %d", self.__class__.__name__, iteration + 1)

            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return self._extract_text(response.content)

            if response.stop_reason != "tool_use":
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if on_step:
                    preview = json.dumps(block.input, default=str)[:100]
                    on_step(f"[step {iteration+1}] {block.name}({preview})")

                result = self.registry.call_from_tool_use(
                    {"name": block.name, "input": block.input}
                )

                if on_step and not result.success:
                    on_step(f"  ERROR: {result.error}")

                tool_results.append(result.to_tool_result(block.id))

            messages.append({"role": "user", "content": tool_results})

        return "Agent reached max_iterations without finishing."

    # ── Internals ─────────────────────────────────────────────────────────────

    def _scoped_tools(self) -> List[dict]:
        all_schemas = {s["name"]: s for s in self.registry.tool_schemas()}
        if not self.skill_names:
            return list(all_schemas.values())
        return [all_schemas[n] for n in self.skill_names if n in all_schemas]

    def _extract_text(self, content) -> str:
        for block in content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _build_client(self):
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
