"""
Unit tests for the Pydantic schema layer.
No LLM calls required — pure validation logic.
"""

import pytest
from schemas.base import ExtractedField, GoverningLaw, ScheduleType
from schemas.im_schedule import IMSchedule
from schemas.vm_schedule import VMSchedule
from schemas.repo_schedule import REPOSchedule


def _ef(value, confidence=0.9, source="test"):
    return ExtractedField(value=value, confidence=confidence, source_text=source)


class TestExtractedField:
    def test_confidence_clamped_to_range(self):
        with pytest.raises(Exception):
            ExtractedField(value="x", confidence=1.5)

    def test_confidence_rounded(self):
        ef = ExtractedField(value=1, confidence=0.123456789)
        assert len(str(ef.confidence).split(".")[-1]) <= 4

    def test_needs_review_default_false(self):
        ef = ExtractedField(value="test", confidence=0.8)
        assert ef.needs_review is False


class TestIMSchedule:
    def _base_kwargs(self):
        """Minimum valid IM kwargs."""
        from schemas.base import AgreementType, Currency
        from datetime import date
        return dict(
            agreement_type=_ef(AgreementType.ISDA_2016_IM),
            governing_law=_ef(GoverningLaw.NEW_YORK),
            effective_date=_ef(date(2024, 1, 15)),
            counterparty_name=_ef("Test Bank"),
            counterparty_lei=_ef("5493001RKX5PVOA2GM83"),
            local_entity_name=_ef("Test Fund"),
            local_entity_lei=_ef("5493001RKX5PVOA2GM84"),
            base_currency=_ef(Currency.USD),
            notification_time=_ef("10:00 AM NY"),
            settlement_day=_ef(2),
            rounding_nearest=_ef(1000.0),
            threshold_party_a=_ef(0.0),
            threshold_party_b=_ef(0.0),
            minimum_transfer_amount_party_a=_ef(500_000.0),
            minimum_transfer_amount_party_b=_ef(500_000.0),
            independent_amount_party_a=_ef(0.0),
            independent_amount_party_b=_ef(0.0),
            eligible_collateral=_ef([]),
            concentration_limits_apply=_ef(False),
            custody_arrangement=_ef("THIRD_PARTY"),
            custodian_name=_ef("BNY Mellon"),
            custodian_account_party_a=_ef("ACC-A"),
            custodian_account_party_b=_ef("ACC-B"),
            rehypothecation_permitted=_ef(False),
            interest_rate_cash_usd=_ef("SOFR flat"),
            interest_rate_cash_eur=_ef(None, confidence=0.0),
            interest_rate_cash_gbp=_ef(None, confidence=0.0),
            valuation_agent=_ef("Party B"),
            valuation_date=_ef("each Local Business Day"),
            dispute_resolution_time=_ef(3),
            im_calculation_method=_ef("SIMM"),
            simm_version=_ef("SIMM v2.6"),
        )

    def test_valid_im_schedule_creates_successfully(self):
        schedule = IMSchedule(**self._base_kwargs())
        assert schedule.schedule_type == ScheduleType.IM

    def test_low_confidence_fields_detected(self):
        kwargs = self._base_kwargs()
        kwargs["custodian_name"] = ExtractedField(value="BNY Mellon", confidence=0.4)
        schedule = IMSchedule(**kwargs)
        assert "custodian_name" in schedule.low_confidence_fields(threshold=0.7)

    def test_all_high_confidence_no_low_fields(self):
        schedule = IMSchedule(**self._base_kwargs())
        assert schedule.low_confidence_fields(threshold=0.7) == []


class TestVMSchedule:
    def test_schedule_type_is_vm(self):
        from schemas.base import AgreementType, Currency
        from datetime import date
        vm = VMSchedule(
            agreement_type=_ef(AgreementType.ISDA_2016_VM),
            governing_law=_ef(GoverningLaw.ENGLISH),
            effective_date=_ef(date(2024, 1, 1)),
            counterparty_name=_ef("Bank X"),
            counterparty_lei=_ef("213800MBWEIJDM5CU638"),
            local_entity_name=_ef("Fund Y"),
            local_entity_lei=_ef("213800MBWEIJDM5CU639"),
            base_currency=_ef(Currency.USD),
            notification_time=_ef("10:00 AM"),
            settlement_day=_ef(1),
            rounding_nearest=_ef(1000.0),
            threshold_party_a=_ef(0.0),
            threshold_party_b=_ef(0.0),
            minimum_transfer_amount_party_a=_ef(1_000_000.0),
            minimum_transfer_amount_party_b=_ef(1_000_000.0),
            rounding_amount=_ef(1000.0),
            eligible_currencies=_ef(["USD", "EUR"]),
            securities_eligible=_ef(False),
            interest_rate_usd=_ef("SOFR flat"),
            interest_rate_eur=_ef("€STR flat"),
            interest_rate_gbp=_ef(None, confidence=0.0),
            interest_rate_jpy=_ef(None, confidence=0.0),
            interest_payment_netting=_ef(True),
            valuation_agent=_ef("Party B"),
            valuation_time=_ef("4pm London"),
            close_out_netting_applies=_ef(True),
            delivery_amount_floor=_ef(0.0),
            regular_settlement_day=_ef(1),
            netting_set_identifier=_ef("NS-001"),
            covered_transactions=_ef("All Transactions"),
            credit_support_obligations_party_a=_ef("Transfer Eligible Credit Support"),
            credit_support_obligations_party_b=_ef("Transfer Eligible Credit Support"),
        )
        assert vm.schedule_type == ScheduleType.VM


class TestREPOSchedule:
    def test_schedule_type_is_repo(self):
        from schemas.base import AgreementType, Currency
        from datetime import date
        repo = REPOSchedule(
            agreement_type=_ef(AgreementType.GMRA_2011),
            governing_law=_ef(GoverningLaw.ENGLISH),
            effective_date=_ef(date(2024, 1, 1)),
            counterparty_name=_ef("Pinnacle Fixed Income Ltd"),
            counterparty_lei=_ef("549300NF0HQLTPIQH841"),
            local_entity_name=_ef("StoneRidge Capital"),
            local_entity_lei=_ef("549300NF0HQLTPIQH842"),
            base_currency=_ef(Currency.GBP),
            notification_time=_ef("9:00 AM London"),
            settlement_day=_ef(1),
            rounding_nearest=_ef(1000.0),
            initial_margin_ratio=_ef(1.02),
            margin_maintenance_threshold=_ef(1.01),
            net_margin_applies=_ef(True),
            margin_call_method=_ef("REPRICING"),
            eligible_securities=_ef([]),
            concentration_limits=_ef("60% per issuer"),
            repricing_date=_ef("daily"),
            repricing_notice_hours=_ef(2),
            settlement_lag=_ef(1),
            substitution_permitted=_ef(True),
            substitution_notice_days=_ef(3),
            income_payment_method=_ef("manufactured payment next business day"),
            manufactured_payment_timing=_ef("same business day as issuer payment"),
            default_interest_rate=_ef("SONIA + 200bps"),
            mini_close_out_applies=_ef(False),
            set_off_rights=_ef(True),
            tri_party_agent=_ef("Euroclear Bank SA/NV"),
            delivery_by_value=_ef(False),
            pricing_rate_basis=_ef("SONIA"),
        )
        assert repo.schedule_type == ScheduleType.REPO
