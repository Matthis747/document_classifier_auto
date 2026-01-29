"""
Document classifier for inference.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib

from .ocr import OCRExtractor
from .normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classify PDF documents using a trained model."""

    def __init__(
        self,
        models_dir: Path,
        ocr_config: dict,
        max_pages: int = 10,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize the document classifier.

        Args:
            models_dir: Directory containing model artifacts
            ocr_config: Configuration for PaddleOCR
            max_pages: Maximum pages to OCR per PDF
            confidence_threshold: Minimum confidence for prediction (else UNKNOWN)
        """
        self.models_dir = Path(models_dir)
        self.ocr_config = ocr_config
        self.max_pages = max_pages
        self.confidence_threshold = confidence_threshold

        self.model_file = self.models_dir / "svm_model.joblib"
        self.tfidf_file = self.models_dir / "tfidf_vectorizer.joblib"
        self.metadata_file = self.models_dir / "model_metadata.json"

        self._model = None
        self._tfidf = None
        self._metadata = None
        self._ocr = None
        self._normalizer = None

    def _load_model(self):
        """Load model artifacts."""
        if self._model is None:
            if not self.model_file.exists():
                raise FileNotFoundError(f"Model not found: {self.model_file}. Train a model first.")

            logger.info("Loading model artifacts...")
            self._model = joblib.load(self.model_file)
            self._tfidf = joblib.load(self.tfidf_file)

            with open(self.metadata_file) as f:
                self._metadata = json.load(f)

            logger.info(f"Model loaded with {len(self._metadata['classes'])} classes")

    def _get_ocr(self) -> OCRExtractor:
        """Get or create OCR extractor."""
        if self._ocr is None:
            self._ocr = OCRExtractor(self.ocr_config, self.max_pages)
        return self._ocr

    def _get_normalizer(self) -> TextNormalizer:
        """Get or create text normalizer."""
        if self._normalizer is None:
            self._normalizer = TextNormalizer()
        return self._normalizer

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def classes(self) -> list:
        """Get list of classes."""
        self._load_model()
        return self._metadata["classes"]

    @property
    def metadata(self) -> dict:
        """Get model metadata."""
        self._load_model()
        return self._metadata

    def predict_text(self, text: str) -> dict:
        """
        Predict class from raw text.

        Args:
            text: Raw text to classify

        Returns:
            dict with predicted_class, confidence, top_3
        """
        self._load_model()

        # Normalize text
        text_clean = self._get_normalizer().normalize(text)

        # Vectorize
        X = self._tfidf.transform([text_clean])

        # Predict
        predicted_class = self._model.predict(X)[0]
        probas = self._model.predict_proba(X)[0]

        # Get top 3
        classes = self._model.classes_
        top_indices = probas.argsort()[-3:][::-1]
        top_3 = [{"class": classes[i], "score": float(probas[i])} for i in top_indices]

        confidence = float(max(probas))

        # Check threshold
        if confidence < self.confidence_threshold:
            predicted_class = "UNKNOWN"

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3": top_3,
        }

    def predict_file(self, pdf_path: str | Path) -> dict:
        """
        Predict class from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            dict with filename, predicted_class, confidence, top_3, ocr_stats
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Extract text
        ocr_result = self._get_ocr().extract_text(pdf_path)

        if not ocr_result["success"]:
            return {
                "filename": pdf_path.name,
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "top_3": [],
                "error": ocr_result["error"],
                "ocr_stats": {
                    "n_pages_processed": 0,
                    "n_pages_total": ocr_result["n_pages_total"],
                },
            }

        # Predict
        prediction = self.predict_text(ocr_result["text"])

        return {
            "filename": pdf_path.name,
            "predicted_class": prediction["predicted_class"],
            "confidence": prediction["confidence"],
            "top_3": prediction["top_3"],
            "ocr_stats": {
                "n_pages_processed": ocr_result["n_pages_processed"],
                "n_pages_total": ocr_result["n_pages_total"],
                "text_length": len(ocr_result["text"]),
            },
        }

    def get_status(self) -> dict:
        """
        Get classifier status.

        Returns:
            dict with status information
        """
        try:
            self._load_model()
            return {
                "status": "ready",
                "model_loaded": True,
                "classes": self._metadata["classes"],
                "n_classes": len(self._metadata["classes"]),
                "test_accuracy": self._metadata.get("test_accuracy"),
                "test_f1_macro": self._metadata.get("test_f1_macro"),
            }
        except FileNotFoundError:
            return {
                "status": "no_model",
                "model_loaded": False,
                "message": "No trained model found. Run training first.",
            }
        except Exception as e:
            return {
                "status": "error",
                "model_loaded": False,
                "error": str(e),
            }
