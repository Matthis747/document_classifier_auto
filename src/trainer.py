"""
Model training module with incremental support.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

from .ocr import OCRExtractor
from .normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage the document classification model."""

    def __init__(
        self,
        dataset_dir: Path,
        data_dir: Path,
        models_dir: Path,
        ocr_config: dict,
        tfidf_config: dict,
        svm_config: dict,
        max_pages: int = 10,
        test_size: float = 0.15,
        val_size: float = 0.15,
        stopwords: Optional[set] = None,
    ):
        """
        Initialize the model trainer.

        Args:
            dataset_dir: Directory containing class subdirectories with PDFs
            data_dir: Directory for processed data files
            models_dir: Directory for model artifacts
            ocr_config: Configuration for PaddleOCR
            tfidf_config: Configuration for TF-IDF vectorizer
            svm_config: Configuration for SVM classifier
            max_pages: Maximum pages to OCR per PDF
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            stopwords: Set of stopwords to exclude
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.ocr_config = ocr_config
        self.tfidf_config = tfidf_config
        self.svm_config = svm_config
        self.max_pages = max_pages
        self.test_size = test_size
        self.val_size = val_size
        self.stopwords = stopwords or set()

        # Paths for data files
        self.raw_pkl = self.data_dir / "dataset_raw.pkl"
        self.clean_pkl = self.data_dir / "dataset_clean.pkl"
        self.model_file = self.models_dir / "svm_model.joblib"
        self.tfidf_file = self.models_dir / "tfidf_vectorizer.joblib"
        self.metadata_file = self.models_dir / "model_metadata.json"

        # Components
        self.ocr = OCRExtractor(ocr_config, max_pages)
        self.normalizer = TextNormalizer()

    def scan_dataset(self) -> pd.DataFrame:
        """
        Scan the dataset directory and list all PDFs.

        Returns:
            DataFrame with doc_id, label, full_path columns
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        classes = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(classes)} classes: {classes}")

        all_files = []
        for cls in classes:
            cls_path = self.dataset_dir / cls
            pdf_files = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
            for pdf in pdf_files:
                all_files.append({
                    "doc_id": pdf.stem,
                    "label": cls,
                    "full_path": str(pdf),
                })

        df = pd.DataFrame(all_files)
        logger.info(f"Found {len(df)} PDFs total")
        return df

    def extract_texts(self, force: bool = False) -> dict:
        """
        Extract text from PDFs using OCR (incremental).

        Args:
            force: If True, re-extract all documents

        Returns:
            dict with stats about extraction
        """
        df_current = self.scan_dataset()

        # Load existing data if available
        df_existing = None
        if self.raw_pkl.exists() and not force:
            df_existing = pd.read_pickle(self.raw_pkl)
            logger.info(f"Loaded existing data: {len(df_existing)} documents")

        # Determine what needs processing
        if df_existing is not None and not force:
            existing_keys = set(df_existing["doc_id"] + "|" + df_existing["label"])
            current_keys = set(df_current["doc_id"] + "|" + df_current["label"])

            new_keys = current_keys - existing_keys
            removed_keys = existing_keys - current_keys

            df_new = df_current[df_current.apply(lambda x: f"{x['doc_id']}|{x['label']}" in new_keys, axis=1)]
            removed_docs = [k.split("|") for k in removed_keys]
        else:
            df_new = df_current
            removed_docs = []

        logger.info(f"New documents to process: {len(df_new)}")
        logger.info(f"Documents to remove: {len(removed_docs)}")

        # Process new documents
        new_results = []
        errors = []

        if len(df_new) > 0:
            for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc="OCR Extraction"):
                result = self.ocr.extract_text(row["full_path"])
                if result["success"]:
                    new_results.append({
                        "doc_id": row["doc_id"],
                        "label": row["label"],
                        "text_raw": result["text"],
                        "n_pages_processed": result["n_pages_processed"],
                        "n_pages_total": result["n_pages_total"],
                        "text_length": len(result["text"]),
                    })
                else:
                    errors.append({
                        "doc_id": row["doc_id"],
                        "label": row["label"],
                        "error": result["error"],
                    })

        # Merge results
        if df_existing is not None and not force:
            df_dataset = df_existing.copy()

            # Remove deleted documents
            for doc_id, label in removed_docs:
                mask = (df_dataset["doc_id"] == doc_id) & (df_dataset["label"] == label)
                df_dataset = df_dataset[~mask]

            # Add new documents
            if new_results:
                df_new_data = pd.DataFrame(new_results)
                df_dataset = pd.concat([df_dataset, df_new_data], ignore_index=True)
        else:
            df_dataset = pd.DataFrame(new_results)

        # Sort and save
        df_dataset = df_dataset.sort_values(["label", "doc_id"]).reset_index(drop=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df_dataset.to_pickle(self.raw_pkl)
        df_dataset.to_csv(self.raw_pkl.with_suffix(".csv"), index=False)

        logger.info(f"Saved {len(df_dataset)} documents to {self.raw_pkl}")

        return {
            "total": len(df_dataset),
            "new_processed": len(new_results),
            "removed": len(removed_docs),
            "errors": len(errors),
        }

    def normalize_texts(self, force: bool = False) -> dict:
        """
        Normalize extracted texts (incremental).

        Args:
            force: If True, re-normalize all documents

        Returns:
            dict with stats about normalization
        """
        if not self.raw_pkl.exists():
            raise FileNotFoundError(f"Raw data not found: {self.raw_pkl}. Run extract_texts first.")

        df_raw = pd.read_pickle(self.raw_pkl)

        # Load existing clean data if available
        df_existing = None
        if self.clean_pkl.exists() and not force:
            df_existing = pd.read_pickle(self.clean_pkl)
            logger.info(f"Loaded existing clean data: {len(df_existing)} documents")

        # Determine what needs processing
        if df_existing is not None and not force:
            existing_keys = set(df_existing["doc_id"] + "|" + df_existing["label"])
            raw_keys = set(df_raw["doc_id"] + "|" + df_raw["label"])

            new_keys = raw_keys - existing_keys
            removed_keys = existing_keys - raw_keys

            df_new = df_raw[df_raw.apply(lambda x: f"{x['doc_id']}|{x['label']}" in new_keys, axis=1)]
            removed_docs = [k.split("|") for k in removed_keys]
        else:
            df_new = df_raw
            removed_docs = []

        logger.info(f"New documents to normalize: {len(df_new)}")
        logger.info(f"Documents to remove: {len(removed_docs)}")

        # Normalize new documents
        new_normalized = []
        for _, row in df_new.iterrows():
            text_clean = self.normalizer.normalize(row["text_raw"])
            new_normalized.append({
                "doc_id": row["doc_id"],
                "label": row["label"],
                "text_clean": text_clean,
                "text_length": len(text_clean),
                "n_pages_processed": row["n_pages_processed"],
                "n_pages_total": row["n_pages_total"],
            })

        # Merge results
        if df_existing is not None and not force:
            df_clean = df_existing.copy()

            # Remove deleted documents
            for doc_id, label in removed_docs:
                mask = (df_clean["doc_id"] == doc_id) & (df_clean["label"] == label)
                df_clean = df_clean[~mask]

            # Add new documents
            if new_normalized:
                df_new_data = pd.DataFrame(new_normalized)
                df_clean = pd.concat([df_clean, df_new_data], ignore_index=True)
        else:
            df_clean = pd.DataFrame(new_normalized)

        # Sort and save
        df_clean = df_clean.sort_values(["label", "doc_id"]).reset_index(drop=True)
        df_clean.to_pickle(self.clean_pkl)
        df_clean.to_csv(self.clean_pkl.with_suffix(".csv"), index=False)

        logger.info(f"Saved {len(df_clean)} normalized documents to {self.clean_pkl}")

        return {
            "total": len(df_clean),
            "new_normalized": len(new_normalized),
            "removed": len(removed_docs),
        }

    def train_model(self) -> dict:
        """
        Train the TF-IDF + SVM model.

        Returns:
            dict with training metrics
        """
        if not self.clean_pkl.exists():
            raise FileNotFoundError(f"Clean data not found: {self.clean_pkl}. Run normalize_texts first.")

        df = pd.read_pickle(self.clean_pkl)
        logger.info(f"Training on {len(df)} documents")

        X = df["text_clean"].values
        y = df["label"].values

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=42,
            stratify=y,
        )

        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp,
        )

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Build stopwords list
        try:
            import nltk
            from nltk.corpus import stopwords
            nltk.download("stopwords", quiet=True)
            all_stopwords = set(stopwords.words("french")) | set(stopwords.words("english")) | self.stopwords
        except Exception:
            all_stopwords = self.stopwords

        # Create and fit TF-IDF vectorizer
        tfidf_params = self.tfidf_config.copy()
        tfidf_params["stop_words"] = list(all_stopwords)
        tfidf = TfidfVectorizer(**tfidf_params)

        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)
        X_test_tfidf = tfidf.transform(X_test)

        logger.info(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")

        # Train SVM
        svm = LinearSVC(**self.svm_config)
        svm.fit(X_train_tfidf, y_train)

        # Calibrate for probability estimates
        svm_calibrated = CalibratedClassifierCV(svm, cv=3, method="sigmoid")
        svm_calibrated.fit(X_train_tfidf, y_train)

        # Evaluate
        y_val_pred = svm_calibrated.predict(X_val_tfidf)
        y_test_pred = svm_calibrated.predict(X_test_tfidf)

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        logger.info(f"Validation: accuracy={val_accuracy:.3f}, f1={val_f1:.3f}")
        logger.info(f"Test: accuracy={test_accuracy:.3f}, f1={test_f1:.3f}")

        # Save models
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(svm_calibrated, self.model_file)
        joblib.dump(tfidf, self.tfidf_file)

        # Save metadata
        metadata = {
            "classes": list(svm_calibrated.classes_),
            "n_features": len(tfidf.vocabulary_),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1,
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1,
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {self.model_file}")

        return {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1,
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1,
            "classes": list(svm_calibrated.classes_),
        }

    def train(self, force: bool = False) -> dict:
        """
        Run the complete training pipeline.

        Args:
            force: If True, re-process everything from scratch

        Returns:
            dict with complete training stats
        """
        logger.info("Starting training pipeline...")

        # Step 1: Extract texts
        extract_stats = self.extract_texts(force=force)
        logger.info(f"Extraction: {extract_stats}")

        # Step 2: Normalize texts
        normalize_stats = self.normalize_texts(force=force)
        logger.info(f"Normalization: {normalize_stats}")

        # Step 3: Train model (always retrain if data changed)
        needs_retrain = (
            force
            or extract_stats["new_processed"] > 0
            or extract_stats["removed"] > 0
            or not self.model_file.exists()
        )

        if needs_retrain:
            train_stats = self.train_model()
            logger.info(f"Training: {train_stats}")
        else:
            logger.info("No changes detected, skipping training")
            with open(self.metadata_file) as f:
                train_stats = json.load(f)

        return {
            "extraction": extract_stats,
            "normalization": normalize_stats,
            "training": train_stats,
            "retrained": needs_retrain,
        }
