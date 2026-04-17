"""
Core types and the shared base for all collateral schedule schemas.

Every extracted field is wrapped in ExtractedField so downstream consumers
always know how confident the model was and where in the source text the value
came from.  The schedule-level models inherit from CollateralScheduleBase.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


class ScheduleType(str, Enum):
    IM = "IM"       # Initial Margin (ISDA CSA)
    VM = "VM"       # Variation Margin (ISDA CSA 2016)
    REPO = "REPO"   # Repurchase Agreement (GMRA)


class GoverningLaw(str, Enum):
    NEW_YORK = "NEW_YORK"
    ENGLISH = "ENGLISH"
    JAPANESE = "JAPANESE"
    OTHER = "OTHER"


class AgreementType(str, Enum):
    ISDA_1992 = "ISDA_1992"
    ISDA_1994_NY = "ISDA_1994_NY"
    ISDA_1995_ENGLISH = "ISDA_1995_ENGLISH"
    ISDA_2016_VM = "ISDA_2016_VM"
    ISDA_2016_IM = "ISDA_2016_IM"
    GMRA_2000 = "GMRA_2000"
    GMRA_2011 = "GMRA_2011"
    OTHER = "OTHER"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    OTHER = "OTHER"


class ExtractedField(BaseModel, Generic[T]):
    """Wraps a single extracted value with provenance and confidence metadata."""

    value: Optional[T] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source_text: Optional[str] = Field(
        default=None,
        description="Verbatim excerpt from source document that supports this value.",
    )
    page_ref: Optional[int] = Field(
        default=None, description="Page number in source document."
    )
    needs_review: bool = Field(
        default=False,
        description="Set True when confidence falls below the review threshold.",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class EligibleCollateral(BaseModel):
    """A single eligible collateral row (used in IM and REPO schedules)."""

    asset_class: str
    issuer: Optional[str] = None
    min_rating: Optional[str] = None
    max_maturity_years: Optional[float] = None
    currency: Optional[str] = None
    haircut_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    concentration_limit_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class CollateralScheduleBase(BaseModel):
    """Fields common to all schedule types."""

    # --- Identity ---
    schedule_type: ScheduleType
    agreement_type: ExtractedField[AgreementType]
    governing_law: ExtractedField[GoverningLaw]
    effective_date: ExtractedField[date]

    # --- Counterparty ---
    counterparty_name: ExtractedField[str]
    counterparty_lei: ExtractedField[str]          # Legal Entity Identifier
    local_entity_name: ExtractedField[str]
    local_entity_lei: ExtractedField[str]

    # --- Transfer mechanics ---
    base_currency: ExtractedField[Currency]
    notification_time: ExtractedField[str]          # e.g. "10:00 AM New York"
    settlement_day: ExtractedField[int]             # T+N
    rounding_nearest: ExtractedField[float]         # nearest unit of base currency

    # --- Metadata (not extracted from doc) ---
    source_filename: Optional[str] = None
    extraction_model: Optional[str] = None
    extraction_timestamp: Optional[str] = None
    schema_version: str = "1.0.0"

    class Config:
        use_enum_values = True

    def low_confidence_fields(self, threshold: float = 0.7) -> list[str]:
        """Return names of ExtractedField attributes below the confidence threshold."""
        low = []
        for name, field_info in self.model_fields.items():
            val = getattr(self, name)
            if isinstance(val, ExtractedField) and val.confidence < threshold:
                low.append(name)
        return low
