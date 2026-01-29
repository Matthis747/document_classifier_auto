# Document Classifier

Automated aviation document classification using OCR and machine learning.

## Features

- **OCR Extraction**: Extract text from PDF documents using PaddleOCR
- **Incremental Processing**: Only process new/changed documents
- **ML Classification**: TF-IDF + SVM classifier with ~97% accuracy
- **CLI Interface**: Command-line tools for training and prediction
- **REST API**: FastAPI-based API for integration

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PaddlePaddle GPU (CUDA 12.9)
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# Install other dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

## Project Structure

```
document_classifier_auto/
├── src/
│   ├── __init__.py
│   ├── ocr.py           # OCR extraction with PaddleOCR
│   ├── normalizer.py    # Text normalization
│   ├── trainer.py       # Model training pipeline
│   └── classifier.py    # Inference
├── data/
│   └── dataset/         # Place your PDFs here (organized by class)
├── models/              # Trained model artifacts
├── config.py            # Configuration
├── main.py              # CLI entry point
├── api.py               # FastAPI application
├── requirements.txt
└── README.md
```

## Dataset Structure

Organize your PDF documents in class subdirectories:

```
data/dataset/
├── AD_SB/
│   ├── document1.pdf
│   └── document2.pdf
├── AMP/
│   └── document3.pdf
├── ATL/
│   └── ...
└── work_order/
    └── ...
```

## Usage

### Training

```bash
# Train with default dataset path (data/dataset/)
python main.py train

# Train with custom dataset path
python main.py train --dataset /path/to/dataset

# Force full re-processing (no incremental)
python main.py train --force
```

### Prediction

```bash
# Classify a single PDF
python main.py predict document.pdf

# Get JSON output
python main.py predict document.pdf --json
```

### Model Status

```bash
python main.py status
```

### API Server

```bash
# Start API server
python main.py api

# With custom host/port
python main.py api --host 0.0.0.0 --port 8080

# Development mode with auto-reload
python main.py api --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/status` | GET | Model status and metrics |
| `/classes` | GET | List of document classes |
| `/predict` | POST | Classify a PDF (upload file) |
| `/train` | POST | Trigger training (background) |
| `/train/status` | GET | Training status |

### Example API Usage

```bash
# Health check
curl http://localhost:8080/

# Get classes
curl http://localhost:8080/classes

# Classify a document
curl -X POST -F "file=@document.pdf" http://localhost:8080/predict

# Trigger training
curl -X POST http://localhost:8080/train

# Trigger full retraining
curl -X POST "http://localhost:8080/train?force=true"
```

### API Response Example

```json
{
  "filename": "document.pdf",
  "predicted_class": "work_order",
  "confidence": 0.87,
  "top_3": [
    {"class": "work_order", "score": 0.87},
    {"class": "work_package", "score": 0.08},
    {"class": "ATL", "score": 0.03}
  ],
  "ocr_stats": {
    "n_pages_processed": 10,
    "n_pages_total": 25,
    "text_length": 15432
  }
}
```

## Configuration

Edit `config.py` to customize:

- OCR settings (GPU/CPU, language, etc.)
- Model parameters (TF-IDF, SVM)
- API settings (host, port)
- File paths

## Document Classes

| Class | Description |
|-------|-------------|
| AD_SB | Airworthiness Directives / Service Bulletins |
| AMP | Aircraft Maintenance Program |
| ATL | Aircraft Technical Log |
| KARDEX | Life Limited Parts tracking |
| MOD | Modifications / STCs |
| MT | Maintenance Tasks / LDND Status |
| REP | Repair documents |
| release_certificate_component | EASA Form 1 |
| work_order | Transfer tickets / Work orders |
| work_package | Maintenance work packages |

## Incremental Mode

The system automatically detects changes:

- **New PDFs**: Documents added to dataset are OCR'd and processed
- **Removed PDFs**: Documents removed from dataset are cleaned from data files
- **Unchanged**: Previously processed documents are preserved

This saves significant time when updating the training set.
