"""
OCR extraction module using PaddleOCR.
"""
import logging
from pathlib import Path
from typing import Optional
import tempfile

from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract text from PDF documents using PaddleOCR."""

    def __init__(self, config: dict, max_pages: int = 10):
        """
        Initialize the OCR extractor.

        Args:
            config: PaddleOCR configuration dict
            max_pages: Maximum number of pages to process per PDF
        """
        self.config = config
        self.max_pages = max_pages
        self._ocr = None

    def _init_ocr(self):
        """Lazy initialization of PaddleOCR engine."""
        if self._ocr is None:
            logger.info("Initializing PaddleOCR engine...")
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(**self.config)
            logger.info("PaddleOCR engine initialized")

    def extract_text(self, pdf_path: str | Path) -> dict:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            dict with:
                - text: extracted text (concatenated from all pages)
                - n_pages_processed: number of pages processed
                - n_pages_total: total number of pages in PDF
                - success: True if extraction successful
                - error: error message if failed
        """
        self._init_ocr()
        pdf_path = Path(pdf_path)
        temp_pdf_path = None

        try:
            # Read PDF and get page count
            reader = PdfReader(str(pdf_path))
            n_pages_total = len(reader.pages)
            n_pages_to_extract = min(n_pages_total, self.max_pages)

            # Create temporary PDF with only the pages we need
            writer = PdfWriter()
            for i in range(n_pages_to_extract):
                writer.add_page(reader.pages[i])

            # Save temporary PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                temp_pdf_path = Path(tmp.name)
                writer.write(tmp)

            # Run OCR
            result = self._ocr.predict(str(temp_pdf_path))

            if result is None:
                return {
                    "text": "",
                    "n_pages_processed": 0,
                    "n_pages_total": n_pages_total,
                    "success": False,
                    "error": "OCR returned None",
                }

            # Extract text from results
            n_pages_processed = len(result)
            pages_text = []

            for page_result in result:
                if page_result is None:
                    continue

                if isinstance(page_result, dict):
                    # PaddleOCR 3.x structure
                    if "rec_texts" in page_result:
                        page_text = " ".join(page_result["rec_texts"])
                        pages_text.append(page_text)
                elif isinstance(page_result, list):
                    # Fallback for older structure
                    page_lines = []
                    for line in page_result:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else line[1]
                            page_lines.append(str(text))
                    pages_text.append(" ".join(page_lines))

            text_full = "\n\n".join(pages_text)

            return {
                "text": text_full,
                "n_pages_processed": n_pages_processed,
                "n_pages_total": n_pages_total,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return {
                "text": "",
                "n_pages_processed": 0,
                "n_pages_total": 0,
                "success": False,
                "error": str(e),
            }
        finally:
            # Clean up temporary file
            if temp_pdf_path and temp_pdf_path.exists():
                try:
                    temp_pdf_path.unlink()
                except Exception:
                    pass
