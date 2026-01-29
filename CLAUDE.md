# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated aviation document classification system with CLI and REST API. Uses PaddleOCR for text extraction and TF-IDF + SVM for classification.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py train

# Classify a PDF
python main.py predict document.pdf

# Start API server
python main.py api

# Check model status
python main.py status
```

## Architecture

```
src/
├── ocr.py         # OCRExtractor - PaddleOCR wrapper
├── normalizer.py  # TextNormalizer - text preprocessing
├── trainer.py     # ModelTrainer - training pipeline with incremental support
└── classifier.py  # DocumentClassifier - inference
```

**Data Flow:**
1. `OCRExtractor.extract_text()` → raw text from PDF
2. `TextNormalizer.normalize()` → cleaned text
3. `TfidfVectorizer.transform()` → feature vector
4. `CalibratedClassifierCV.predict()` → class + probabilities

## Key Files

- `config.py` - All configuration (paths, OCR settings, model params)
- `main.py` - CLI entry point (train, predict, status, api commands)
- `api.py` - FastAPI application
- `data/dataset/` - Training PDFs organized by class
- `models/` - Trained model artifacts (svm_model.joblib, tfidf_vectorizer.joblib)

## Incremental Training

The trainer tracks processed documents via pickle files:
- `data/dataset_raw.pkl` - OCR results
- `data/dataset_clean.pkl` - Normalized text

On each training run, it compares current PDFs with processed data to:
- Process only new documents
- Remove deleted documents
- Skip unchanged documents

## API Endpoints

- `GET /status` - Model status and metrics
- `GET /classes` - Available document classes
- `POST /predict` - Classify uploaded PDF
- `POST /train` - Trigger background training
- `GET /train/status` - Training progress
