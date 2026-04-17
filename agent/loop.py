# Backwards-compatible re-export.
# New code should use agent.orchestrator_agent.OrchestratorAgent or BaseAgent directly.
from agent.base import BaseAgent
from agent.orchestrator_agent.agent import OrchestratorAgent as CollateralAgent

__all__ = ["BaseAgent", "CollateralAgent"]
