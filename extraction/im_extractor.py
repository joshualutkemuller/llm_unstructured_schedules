from __future__ import annotations
from typing import Type
from pydantic import BaseModel
from extraction.base_extractor import BaseExtractor
from extraction.prompts.im_prompts import build_im_extraction_messages
from schemas.im_schedule import IMSchedule


class IMExtractor(BaseExtractor):
    """Extracts and validates Initial Margin CSA fields."""

    def build_messages(self, chunk: str) -> list[dict]:
        return build_im_extraction_messages(chunk)

    @property
    def schema_class(self) -> Type[BaseModel]:
        return IMSchedule
