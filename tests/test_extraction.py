"""
Tests for the extraction layer.
Uses the fixture text files and mocks the LLM client so no API key is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_anthropic_response(payload: dict):
    """Build a mock Anthropic SDK response object."""
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    return msg


class TestDocumentClassifier:
    def test_im_keywords_classify_as_im(self):
        from ingestion.document_classifier import DocumentClassifier
        text = (FIXTURES / "sample_im.txt").read_text()
        clf = DocumentClassifier()
        result = clf.classify(text)
        assert result.schedule_type.value == "IM"
        assert result.type_confidence > 0.5

    def test_vm_keywords_classify_as_vm(self):
        from ingestion.document_classifier import DocumentClassifier
        text = (FIXTURES / "sample_vm.txt").read_text()
        clf = DocumentClassifier()
        result = clf.classify(text)
        assert result.schedule_type.value == "VM"

    def test_repo_keywords_classify_as_repo(self):
        from ingestion.document_classifier import DocumentClassifier
        text = (FIXTURES / "sample_repo.txt").read_text()
        clf = DocumentClassifier()
        result = clf.classify(text)
        assert result.schedule_type.value == "REPO"

    def test_governing_law_new_york(self):
        from ingestion.document_classifier import DocumentClassifier
        text = (FIXTURES / "sample_im.txt").read_text()
        clf = DocumentClassifier()
        result = clf.classify(text)
        assert result.governing_law.value == "NEW_YORK"

    def test_governing_law_english(self):
        from ingestion.document_classifier import DocumentClassifier
        text = (FIXTURES / "sample_vm.txt").read_text()
        clf = DocumentClassifier()
        result = clf.classify(text)
        assert result.governing_law.value == "ENGLISH"


class TestDocumentLoader:
    def test_load_txt_file(self):
        from ingestion.document_loader import DocumentLoader
        loader = DocumentLoader(ocr_fallback=False)
        doc = loader.load(FIXTURES / "sample_im.txt")
        assert doc.page_count == 1
        assert "CREDIT SUPPORT ANNEX" in doc.full_text
        assert doc.raw_format == "txt"

    def test_chunk_document_splits_large_text(self):
        from ingestion.document_loader import DocumentLoader, chunk_document
        loader = DocumentLoader(ocr_fallback=False)
        doc = loader.load(FIXTURES / "sample_im.txt")
        chunks = chunk_document(doc, max_tokens=100)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)


class TestSyntheticDataGenerator:
    def test_im_generator_produces_valid_sample(self):
        from training.data_generator import generate_im_sample
        text, gt = generate_im_sample()
        assert isinstance(text, str) and len(text) > 100
        assert "threshold_party_a" in gt
        assert "counterparty_name" in gt
        assert gt["schedule_type"] == "IM"

    def test_vm_generator_produces_zero_threshold(self):
        from training.data_generator import generate_vm_sample
        _, gt = generate_vm_sample()
        assert gt["threshold_party_a"] == 0
        assert gt["threshold_party_b"] == 0

    def test_repo_generator_margin_ratio_above_one(self):
        from training.data_generator import generate_repo_sample
        _, gt = generate_repo_sample()
        assert gt["initial_margin_ratio"] > 1.0


class TestIMExtractorWithMock:
    """Test the IM extractor end-to-end with a mocked LLM response."""

    def _mock_im_payload(self):
        return {
            "threshold_party_a": {"value": 0, "confidence": 0.98, "source_text": "USD 0"},
            "threshold_party_b": {"value": 0, "confidence": 0.98, "source_text": "USD 0"},
            "minimum_transfer_amount_party_a": {"value": 500000, "confidence": 0.97, "source_text": "USD 500,000"},
            "minimum_transfer_amount_party_b": {"value": 500000, "confidence": 0.97, "source_text": "USD 500,000"},
            "independent_amount_party_a": {"value": 2000000, "confidence": 0.95, "source_text": "USD 2,000,000"},
            "independent_amount_party_b": {"value": 0, "confidence": 0.95, "source_text": "USD 0"},
            "eligible_collateral": {"value": [], "confidence": 0.80, "source_text": "US Treasury Securities"},
            "concentration_limits_apply": {"value": True, "confidence": 0.85, "source_text": "Concentration Limits"},
            "custody_arrangement": {"value": "TRI_PARTY", "confidence": 0.92, "source_text": "tri-party"},
            "custodian_name": {"value": "BNY Mellon", "confidence": 0.98, "source_text": "BNY Mellon"},
            "custodian_account_party_a": {"value": None, "confidence": 0.0, "source_text": ""},
            "custodian_account_party_b": {"value": None, "confidence": 0.0, "source_text": ""},
            "rehypothecation_permitted": {"value": False, "confidence": 0.97, "source_text": "not permitted"},
            "interest_rate_cash_usd": {"value": "SOFR flat", "confidence": 0.96, "source_text": "SOFR flat"},
            "interest_rate_cash_eur": {"value": "EUR STR flat", "confidence": 0.90, "source_text": "€STR flat"},
            "interest_rate_cash_gbp": {"value": None, "confidence": 0.0, "source_text": ""},
            "valuation_agent": {"value": "Party B", "confidence": 0.95, "source_text": "Party B"},
            "valuation_date": {"value": "each Local Business Day", "confidence": 0.90, "source_text": ""},
            "dispute_resolution_time": {"value": 3, "confidence": 0.93, "source_text": "3 Business Days"},
            "im_calculation_method": {"value": "SIMM", "confidence": 0.98, "source_text": "SIMM v2.6"},
            "simm_version": {"value": "SIMM v2.6", "confidence": 0.97, "source_text": "SIMM v2.6"},
            "notification_time": {"value": "10:00 AM New York time", "confidence": 0.96, "source_text": ""},
            "settlement_day": {"value": 2, "confidence": 0.95, "source_text": "T+2"},
            "base_currency": {"value": "USD", "confidence": 0.90, "source_text": ""},
            "governing_law": {"value": "NEW_YORK", "confidence": 0.99, "source_text": "New York"},
            "agreement_type": {"value": "ISDA_2016_IM", "confidence": 0.95, "source_text": "2002 ISDA"},
            "counterparty_name": {"value": "Meridian Securities Ltd", "confidence": 0.99, "source_text": "Meridian Securities Ltd"},
            "counterparty_lei": {"value": "5493001RKX5PVOA2GM83", "confidence": 0.99, "source_text": "5493001RKX5PVOA2GM83"},
            "effective_date": {"value": "2023-10-01", "confidence": 0.97, "source_text": "October 1, 2023"},
            "special_provisions": {"value": None, "confidence": 0.0, "source_text": ""},
        }

    @patch("anthropic.Anthropic")
    def test_im_extractor_returns_extraction_result(self, MockAnthropic):
        from config.settings import Settings
        from extraction.im_extractor import IMExtractor

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            self._mock_im_payload()
        )
        MockAnthropic.return_value = mock_client

        settings = Settings(anthropic_api_key="test-key")
        extractor = IMExtractor(settings=settings)
        extractor._client = mock_client

        text = (FIXTURES / "sample_im.txt").read_text()
        result = extractor.extract_from_text(text)

        assert result.raw_json is not None
        assert result.chunk_count >= 1
