"""
Variation Margin (VM) schedule schema.

Variation margin is the daily mark-to-market settlement of unrealised P&L.
Post-2016, VM CSAs are typically standalone ISDA 2016 Credit Support Annexes.
Threshold is almost universally zero; the key variables are eligible currencies,
interest rates, and settlement mechanics.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.base import (
    CollateralScheduleBase,
    ExtractedField,
    ScheduleType,
)


class EligibleCurrency(str):
    pass


class VMInterestRate(object):
    """Interest rate terms for a single cash currency."""

    currency: str
    rate_index: str    # e.g. "SOFR", "€STR", "SONIA"
    spread_bps: float  # spread over the index in basis points


class VMSchedule(CollateralScheduleBase):
    """Standardized representation of a Variation Margin CSA."""

    schedule_type: ScheduleType = ScheduleType.VM

    # ── Threshold & MTA ───────────────────────────────────────────────────────
    threshold_party_a: ExtractedField[float] = Field(
        description="Threshold for Party A. Regulatory VM requires this to be 0."
    )
    threshold_party_b: ExtractedField[float] = Field(
        description="Threshold for Party B. Regulatory VM requires this to be 0."
    )
    minimum_transfer_amount_party_a: ExtractedField[float]
    minimum_transfer_amount_party_b: ExtractedField[float]
    rounding_amount: ExtractedField[float] = Field(
        description="MTA + threshold rounded to nearest N units of base currency."
    )

    # ── Eligible Collateral / Currencies ─────────────────────────────────────
    eligible_currencies: ExtractedField[List[str]] = Field(
        description="ISO 4217 currency codes accepted as VM collateral."
    )
    securities_eligible: ExtractedField[bool] = Field(
        description="Whether non-cash securities are permitted as VM collateral."
    )

    # ── Interest on Cash Collateral ───────────────────────────────────────────
    interest_rate_usd: ExtractedField[str] = Field(
        description="Rate on USD cash VM (e.g. 'SOFR – 5bps', 'Fed Funds flat')."
    )
    interest_rate_eur: ExtractedField[str]
    interest_rate_gbp: ExtractedField[str]
    interest_rate_jpy: ExtractedField[str]
    interest_payment_netting: ExtractedField[bool] = Field(
        description="Whether interest amounts are netted against margin calls."
    )

    # ── Valuation ─────────────────────────────────────────────────────────────
    valuation_agent: ExtractedField[str]
    valuation_time: ExtractedField[str] = Field(
        description="Time and city for close-of-business valuation (e.g. '4pm New York')."
    )
    close_out_netting_applies: ExtractedField[bool]

    # ── Settlement ────────────────────────────────────────────────────────────
    delivery_amount_floor: ExtractedField[float] = Field(
        description="Minimum delivery amount before a call is issued."
    )
    regular_settlement_day: ExtractedField[int] = Field(
        description="Standard settlement lag T+N for margin transfers."
    )

    # ── Netting Set ───────────────────────────────────────────────────────────
    netting_set_identifier: ExtractedField[str] = Field(
        description="Identifier linking this CSA to its ISDA Master Agreement netting set."
    )
    covered_transactions: ExtractedField[str] = Field(
        description="Transaction types covered (e.g. 'All Transactions', 'FX only')."
    )

    # ── Credit Support Obligations ────────────────────────────────────────────
    credit_support_obligations_party_a: ExtractedField[str]
    credit_support_obligations_party_b: ExtractedField[str]

    special_provisions: ExtractedField[str] = Field(
        default=ExtractedField(value=None, confidence=0.0),
        description="Non-standard or bespoke provisions.",
    )
