"""
OCR processor for scanned collateral schedule pages.

Uses Tesseract via pytesseract. For production, replace with a cloud OCR
service (AWS Textract, Google Document AI) for better table extraction —
collateral schedules often present eligible collateral as tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Converts a PIL Image to text using Tesseract.

    For financial documents, '--psm 6' (assume a uniform block of text) works
    better than the default for dense legal paragraphs.  For tables, '--psm 4'
    or AWS Textract is recommended.
    """

    def __init__(self, lang: str = "eng", psm: int = 6):
        self.lang = lang
        self.psm = psm

    def image_to_text(self, image: "Image") -> str:
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pip install pytesseract  # also needs Tesseract binary")

        config = f"--oem 3 --psm {self.psm}"
        text = pytesseract.image_to_string(image, lang=self.lang, config=config)
        logger.debug("OCR extracted %d chars", len(text))
        return text

    def image_to_data(self, image: "Image") -> dict:
        """Return word-level bounding boxes — useful for table parsing."""
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pip install pytesseract")

        return pytesseract.image_to_data(
            image, lang=self.lang, output_type=pytesseract.Output.DICT
        )
