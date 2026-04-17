from __future__ import annotations
from typing import Type
from pydantic import BaseModel
from extraction.base_extractor import BaseExtractor
from extraction.prompts.repo_prompts import build_repo_extraction_messages
from schemas.repo_schedule import REPOSchedule


class REPOExtractor(BaseExtractor):
    """Extracts and validates REPO / GMRA schedule fields."""

    def build_messages(self, chunk: str) -> list[dict]:
        return build_repo_extraction_messages(chunk)

    @property
    def schema_class(self) -> Type[BaseModel]:
        return REPOSchedule
