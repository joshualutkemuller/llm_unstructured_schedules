"""
REPO (Repurchase Agreement) schedule schema.

Governed by GMRA 2000 or GMRA 2011 (Global Master Repurchase Agreement).
The key parameters are the initial margin / margin ratio, eligible purchased
securities, repricing mechanics, and substitution rights.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.base import (
    CollateralScheduleBase,
    EligibleCollateral,
    ExtractedField,
    ScheduleType,
)


class RepricingMethod(str):
    REPRICING = "REPRICING"          # Adjust repo price to reflect market movement
    MARGIN_MAINTENANCE = "MARGIN_MAINTENANCE"  # Cash top-up / return


class REPOSchedule(CollateralScheduleBase):
    """Standardized representation of a GMRA Annex / REPO schedule."""

    schedule_type: ScheduleType = ScheduleType.REPO

    # ── Margin Mechanics ──────────────────────────────────────────────────────
    initial_margin_ratio: ExtractedField[float] = Field(
        description="Margin ratio as a decimal, e.g. 1.02 = 102% of purchase price."
    )
    margin_maintenance_threshold: ExtractedField[float] = Field(
        description="Exposure level that triggers a margin call."
    )
    net_margin_applies: ExtractedField[bool] = Field(
        description="Whether margin exposures across all repos are netted before calls."
    )
    margin_call_method: ExtractedField[str] = Field(
        description="REPRICING | MARGIN_MAINTENANCE – how margin shortfalls are remedied."
    )

    # ── Eligible Purchased Securities ─────────────────────────────────────────
    eligible_securities: ExtractedField[List[EligibleCollateral]] = Field(
        description="Securities eligible to be sold under repo (with haircuts)."
    )
    concentration_limits: ExtractedField[str] = Field(
        description="Issuer / sector concentration caps on eligible securities."
    )

    # ── Repricing & Settlement ────────────────────────────────────────────────
    repricing_date: ExtractedField[str] = Field(
        description="Frequency of repricing (e.g. 'daily', 'weekly', specific day)."
    )
    repricing_notice_hours: ExtractedField[int] = Field(
        description="Notice period in business hours for a repricing demand."
    )
    settlement_lag: ExtractedField[int] = Field(
        description="T+N settlement for margin transfers."
    )

    # ── Substitution Rights ───────────────────────────────────────────────────
    substitution_permitted: ExtractedField[bool] = Field(
        description="Whether the seller may substitute equivalent securities during the term."
    )
    substitution_notice_days: ExtractedField[int] = Field(
        description="Business days notice required before substituting securities."
    )

    # ── Income & Manufactured Payments ───────────────────────────────────────
    income_payment_method: ExtractedField[str] = Field(
        description="How coupon/dividend income on purchased securities is handled."
    )
    manufactured_payment_timing: ExtractedField[str] = Field(
        description="Timing rule for manufactured dividend / coupon payments."
    )

    # ── Default & Close-out ───────────────────────────────────────────────────
    default_interest_rate: ExtractedField[str] = Field(
        description="Rate charged on overdue amounts (e.g. 'overnight + 200bps')."
    )
    mini_close_out_applies: ExtractedField[bool] = Field(
        description="Whether mini close-out provisions apply on a per-transaction basis."
    )
    set_off_rights: ExtractedField[bool] = Field(
        description="Whether the non-defaulting party may set off obligations on close-out."
    )

    # ── Custody ───────────────────────────────────────────────────────────────
    tri_party_agent: ExtractedField[str] = Field(
        description="Tri-party agent name if repo is settled tri-party (e.g. BNY, Clearstream)."
    )
    delivery_by_value: ExtractedField[bool] = Field(
        description="Whether securities are delivered by value (DBV) rather than specific bonds."
    )

    # ── Rates ─────────────────────────────────────────────────────────────────
    pricing_rate_basis: ExtractedField[str] = Field(
        description="Benchmark for repo rate (e.g. 'SOFR', 'SONIA', fixed)."
    )

    special_provisions: ExtractedField[str] = Field(
        default=ExtractedField(value=None, confidence=0.0),
        description="Non-standard or bespoke provisions.",
    )
