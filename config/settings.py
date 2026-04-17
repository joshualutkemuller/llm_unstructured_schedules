"""
Central configuration.  All values can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Anthropic / LLM ───────────────────────────────────────────────────────
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    # Model for zero-shot / few-shot extraction before fine-tuning
    extraction_model: str = field(
        default_factory=lambda: os.environ.get(
            "EXTRACTION_MODEL", "claude-sonnet-4-6"
        )
    )
    extraction_max_tokens: int = 4096

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_max_tokens: int = 3000

    # ── Review threshold ──────────────────────────────────────────────────────
    # Fields below this confidence are flagged for human review
    review_confidence_threshold: float = 0.70

    # ── Training ──────────────────────────────────────────────────────────────
    base_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "BASE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"
        )
    )
    training_output_dir: str = "models/collateral-v1"
    synthetic_data_dir: str = "data/synthetic"
    training_data_dir: str = "data/training"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
