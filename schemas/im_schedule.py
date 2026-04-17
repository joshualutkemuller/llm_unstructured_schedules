"""
Initial Margin (IM) schedule schema.

Covers ISDA 2016 Credit Support Annex (Security Interest – New York Law),
ISDA 2016 CSA (Title Transfer – English Law), and the 1994/1995 variants.

Key regulatory context: UMR (Uncleared Margin Rules) mandates IM posting for
Phase 1–6 entities; custodian / tri-party arrangements are standard.
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


class CustodyArrangement(str):
    THIRD_PARTY = "THIRD_PARTY"
    BILATERAL = "BILATERAL"
    TRI_PARTY = "TRI_PARTY"


class IMSchedule(CollateralScheduleBase):
    """Standardized representation of an Initial Margin CSA."""

    schedule_type: ScheduleType = ScheduleType.IM

    # ── Threshold & MTA ───────────────────────────────────────────────────────
    threshold_party_a: ExtractedField[float] = Field(
        description="Threshold amount for Party A (posting party). Often 0 under UMR."
    )
    threshold_party_b: ExtractedField[float] = Field(
        description="Threshold amount for Party B (receiving party). Often 0 under UMR."
    )
    minimum_transfer_amount_party_a: ExtractedField[float] = Field(
        description="MTA for Party A – smallest transfer obligation triggered."
    )
    minimum_transfer_amount_party_b: ExtractedField[float] = Field(
        description="MTA for Party B."
    )

    # ── Independent Amount ────────────────────────────────────────────────────
    independent_amount_party_a: ExtractedField[float] = Field(
        description="Independent Amount (IA) for Party A – upfront IM add-on."
    )
    independent_amount_party_b: ExtractedField[float] = Field(
        description="Independent Amount (IA) for Party B."
    )

    # ── Eligible Collateral ───────────────────────────────────────────────────
    eligible_collateral: ExtractedField[List[EligibleCollateral]] = Field(
        description="Schedule of eligible collateral assets with haircuts."
    )
    concentration_limits_apply: ExtractedField[bool] = Field(
        description="Whether concentration limits are specified in the schedule."
    )

    # ── Custody ───────────────────────────────────────────────────────────────
    custody_arrangement: ExtractedField[str] = Field(
        description="THIRD_PARTY | BILATERAL | TRI_PARTY"
    )
    custodian_name: ExtractedField[str] = Field(
        description="Name of the custody bank or tri-party agent (e.g. BNY, Euroclear)."
    )
    custodian_account_party_a: ExtractedField[str]
    custodian_account_party_b: ExtractedField[str]
    rehypothecation_permitted: ExtractedField[bool] = Field(
        description="Whether the secured party may re-use posted collateral."
    )

    # ── Interest & Valuation ──────────────────────────────────────────────────
    interest_rate_cash_usd: ExtractedField[str] = Field(
        description="Interest rate applicable to USD cash collateral (e.g. SOFR, Fed Funds)."
    )
    interest_rate_cash_eur: ExtractedField[str]
    interest_rate_cash_gbp: ExtractedField[str]
    valuation_agent: ExtractedField[str] = Field(
        description="Party responsible for calculating IM exposure (often the dealer)."
    )
    valuation_date: ExtractedField[str] = Field(
        description="Frequency / timing of valuation (e.g. 'each Local Business Day')."
    )

    # ── Dispute Resolution ────────────────────────────────────────────────────
    dispute_resolution_time: ExtractedField[int] = Field(
        description="Number of business days to resolve a disputed call."
    )

    # ── ISDA SIMM ─────────────────────────────────────────────────────────────
    im_calculation_method: ExtractedField[str] = Field(
        description="SIMM | SCHEDULE | OTHER – method used to calculate IM requirement."
    )
    simm_version: ExtractedField[str] = Field(
        description="ISDA SIMM version agreed (e.g. 'SIMM v2.6')."
    )

    # ── Additional ────────────────────────────────────────────────────────────
    additional_termination_events: ExtractedField[str] = Field(
        default=ExtractedField(value=None, confidence=0.0),
        description="Any bespoke ATEs that affect collateral obligations.",
    )
    special_provisions: ExtractedField[str] = Field(
        default=ExtractedField(value=None, confidence=0.0),
        description="Free-text catch-all for non-standard provisions.",
    )
