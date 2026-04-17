"""
Skill registry: maps tool names → Python callables and holds Anthropic
tool schemas so an agent loop can pass them directly to the API.

Usage:
    registry = SkillRegistry.default()
    result = registry.call("classify_schedule", text="...")
    tools  = registry.tool_schemas()   # pass to client.messages.create(tools=...)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SkillResult:
    """Uniform return envelope for every skill call."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tool_result(self, tool_use_id: str) -> dict:
        """Format for Anthropic tool_result content block."""
        import json
        if self.success:
            content = json.dumps(self.data, default=str)
        else:
            content = json.dumps({"error": self.error})
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }


@dataclass
class _SkillEntry:
    name: str
    fn: Callable
    schema: dict  # Anthropic tool schema


class SkillRegistry:
    """
    Central registry of all callable skills and their tool schemas.

    Register skills individually or use SkillRegistry.default() which
    registers every skill defined in this library.
    """

    def __init__(self):
        self._skills: Dict[str, _SkillEntry] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, fn: Callable, schema: dict) -> None:
        self._skills[name] = _SkillEntry(name=name, fn=fn, schema=schema)

    def register_group(self, group) -> None:
        """Register all skills from a skill group object (DocumentSkills etc.)."""
        for entry in group.entries():
            self._skills[entry.name] = entry

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def call(self, name: str, **kwargs) -> SkillResult:
        if name not in self._skills:
            return SkillResult(success=False, error=f"Unknown skill: {name}")
        try:
            result = self._skills[name].fn(**kwargs)
            if isinstance(result, SkillResult):
                return result
            return SkillResult(success=True, data=result)
        except Exception as e:
            return SkillResult(success=False, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def call_from_tool_use(self, tool_use_block: dict) -> SkillResult:
        """Convenience: unpack an Anthropic tool_use content block and dispatch."""
        return self.call(tool_use_block["name"], **tool_use_block.get("input", {}))

    # ── Schema export ─────────────────────────────────────────────────────────

    def tool_schemas(self) -> List[dict]:
        """Return all registered Anthropic tool schemas (pass to tools= param)."""
        return [e.schema for e in self._skills.values()]

    def names(self) -> List[str]:
        return list(self._skills.keys())

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def default(cls, settings=None) -> "SkillRegistry":
        """Build a registry with all skill groups pre-registered."""
        from config.settings import Settings
        from skills.document_skills import DocumentSkills
        from skills.extraction_skills import ExtractionSkills
        from skills.pipeline_skills import PipelineSkills
        from skills.training_skills import TrainingSkills

        settings = settings or Settings()
        registry = cls()
        registry.register_group(DocumentSkills())
        registry.register_group(ExtractionSkills(settings=settings))
        registry.register_group(PipelineSkills(settings=settings))
        registry.register_group(TrainingSkills())
        return registry
