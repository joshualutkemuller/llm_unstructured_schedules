"""
Zero-shot document classifier.

Determines the schedule type (IM / VM / REPO) and governing law from the
first ~2000 words of the document before the full extraction pass.

Strategy: keyword heuristics first (fast, interpretable), then LLM fallback
for ambiguous documents.  This avoids wasting LLM calls on obvious cases.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

from schemas.base import GoverningLaw, ScheduleType

logger = logging.getLogger(__name__)

# ── Keyword heuristics ────────────────────────────────────────────────────────

_IM_SIGNALS = [
    r"\binitial margin\b",
    r"\bindependent amount\b",
    r"\bcustodian\b",
    r"\btri.?party\b",
    r"\bsimm\b",
    r"\buncleared margin\b",
    r"\buma\b",           # Uncleared Margin Agreement
    r"2016 credit support annex.*security interest",
]

_VM_SIGNALS = [
    r"\bvariation margin\b",
    r"\bmark.to.market\b",
    r"\bdaily margin\b",
    r"2016 credit support annex.*title transfer",
    r"\bregulatory vm\b",
    r"\bcredit support amount\b",
]

_REPO_SIGNALS = [
    r"\bglobal master repurchase\b",
    r"\bgmra\b",
    r"\brepurchase price\b",
    r"\bpurchased securities\b",
    r"\bmargin ratio\b",
    r"\brepricing\b",
    r"\bsell.*buy back\b",
]

_NY_SIGNALS = [
    r"new york law",
    r"laws of the state of new york",
    r"governed by.*new york",
]

_ENGLISH_SIGNALS = [
    r"english law",
    r"laws of england",
    r"governed by.*english",
    r"england and wales",
]

_JAPANESE_SIGNALS = [
    r"japanese law",
    r"laws of japan",
    r"governed by.*japan",
]


def _count_matches(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


class ClassificationResult:
    def __init__(
        self,
        schedule_type: ScheduleType,
        governing_law: GoverningLaw,
        type_confidence: float,
        law_confidence: float,
        method: str,
    ):
        self.schedule_type = schedule_type
        self.governing_law = governing_law
        self.type_confidence = type_confidence
        self.law_confidence = law_confidence
        self.method = method  # "heuristic" | "llm"

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(type={self.schedule_type}, "
            f"law={self.governing_law}, "
            f"type_conf={self.type_confidence:.2f}, method={self.method})"
        )


class DocumentClassifier:
    """
    Two-stage classifier: keyword heuristics → LLM fallback.

    Args:
        llm_client: Optional Anthropic / OpenAI client for ambiguous docs.
        llm_threshold: Signal-count ratio below which we call the LLM.
    """

    def __init__(self, llm_client=None, llm_threshold: float = 0.4):
        self.llm_client = llm_client
        self.llm_threshold = llm_threshold

    def classify(self, text: str) -> ClassificationResult:
        # Use first 2000 words for speed
        snippet = " ".join(text.split()[:2000])

        schedule_type, type_conf = self._classify_type_heuristic(snippet)
        governing_law, law_conf = self._classify_law_heuristic(snippet)

        if type_conf < self.llm_threshold and self.llm_client is not None:
            logger.info(
                "Low heuristic confidence (%.2f) — falling back to LLM classifier",
                type_conf,
            )
            schedule_type, type_conf = self._classify_type_llm(snippet)
            method = "llm"
        else:
            method = "heuristic"

        return ClassificationResult(
            schedule_type=schedule_type,
            governing_law=governing_law,
            type_confidence=type_conf,
            law_confidence=law_conf,
            method=method,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_type_heuristic(self, text: str) -> Tuple[ScheduleType, float]:
        im_count = _count_matches(text, _IM_SIGNALS)
        vm_count = _count_matches(text, _VM_SIGNALS)
        repo_count = _count_matches(text, _REPO_SIGNALS)
        total = im_count + vm_count + repo_count

        if total == 0:
            return ScheduleType.IM, 0.0  # default, low confidence

        scores = {
            ScheduleType.IM: im_count / total,
            ScheduleType.VM: vm_count / total,
            ScheduleType.REPO: repo_count / total,
        }
        best = max(scores, key=scores.__getitem__)
        return best, scores[best]

    def _classify_law_heuristic(self, text: str) -> Tuple[GoverningLaw, float]:
        ny = _count_matches(text, _NY_SIGNALS)
        en = _count_matches(text, _ENGLISH_SIGNALS)
        jp = _count_matches(text, _JAPANESE_SIGNALS)
        total = ny + en + jp

        if total == 0:
            return GoverningLaw.OTHER, 0.0

        if ny >= en and ny >= jp:
            return GoverningLaw.NEW_YORK, ny / total
        if en >= jp:
            return GoverningLaw.ENGLISH, en / total
        return GoverningLaw.JAPANESE, jp / total

    def _classify_type_llm(self, text: str) -> Tuple[ScheduleType, float]:
        """
        Ask the LLM to classify the document type.
        Stub: wire up your preferred client (anthropic / openai).
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client not configured")

        prompt = (
            "Classify the following financial document as exactly one of: "
            "IM (Initial Margin), VM (Variation Margin), or REPO (Repurchase Agreement).\n\n"
            f"Document excerpt:\n{text[:1500]}\n\n"
            "Respond with JSON: {\"type\": \"IM|VM|REPO\", \"confidence\": 0.0-1.0}"
        )

        # TODO: implement actual LLM call via self.llm_client
        # response = self.llm_client.complete(prompt)
        raise NotImplementedError("Wire up LLM client in _classify_type_llm")
