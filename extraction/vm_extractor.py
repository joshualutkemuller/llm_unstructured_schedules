from __future__ import annotations
from typing import Type
from pydantic import BaseModel
from extraction.base_extractor import BaseExtractor
from extraction.prompts.vm_prompts import build_vm_extraction_messages
from schemas.vm_schedule import VMSchedule


class VMExtractor(BaseExtractor):
    """Extracts and validates Variation Margin CSA fields."""

    def build_messages(self, chunk: str) -> list[dict]:
        return build_vm_extraction_messages(chunk)

    @property
    def schema_class(self) -> Type[BaseModel]:
        return VMSchedule
