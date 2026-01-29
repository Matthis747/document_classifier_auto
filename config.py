"""
Configuration for the document classifier.
"""
from pathlib import Path
import os

# =============================================================================
# PATHS
# =============================================================================

# Base directory (where this file is located)
BASE_DIR = Path(__file__).parent

# Dataset directory (contains subdirectories for each class with PDFs)
DATASET_DIR = BASE_DIR / "data" / "dataset"

# Output directory for processed data
DATA_DIR = BASE_DIR / "data"

# Models directory
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# OCR SETTINGS
# =============================================================================

def _detect_device():
    """Auto-detect GPU availability."""
    # Allow override via environment variable
    env_device = os.environ.get("PADDLE_DEVICE")
    if env_device:
        return env_device

    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        pass
    return "cpu"

OCR_CONFIG = {
    "device": _detect_device(),
    "lang": "en",
    "ocr_version": "PP-OCRv5",
    "use_doc_orientation_classify": True,
    "use_textline_orientation": True,
}

# Maximum number of pages to OCR per PDF
MAX_PAGES_PER_PDF = 10

# =============================================================================
# MODEL SETTINGS
# =============================================================================

TFIDF_CONFIG = {
    "ngram_range": (1, 3),
    "min_df": 2,
    "max_df": 0.85,
    "max_features": 10000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}

SVM_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 10000,
    "random_state": 42,
    "dual": "auto",
}

# Train/val/test split ratios
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Confidence threshold for predictions (below this = "UNKNOWN")
CONFIDENCE_THRESHOLD = 0.3

# =============================================================================
# API SETTINGS
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000

# =============================================================================
# STOPWORDS (French + English + Domain-specific)
# =============================================================================

DOMAIN_STOPWORDS = {
    'air', 'chine', 'china', 'aircraft', 'atr', 'antilles', 'baie', 'mahault',
    'guadeloupe', 'france', 'de', 'in', 'oui', 'non', 'yes', 'no',
    'utc', 'release', 'service', 'life', 'limit', 'omyr', 'faa', 'approved',
    'iaw', 'article', 'see', 'related', 'right', 'fh', 'etc', 'terminating',
    'repair', 'open', 'model', 'fuselage', 'number', 'page', 'gztj', 'act',
    'work', 'identified', 'block', 'described', 'specified', 'certifies',
    'unless', 'zzz', 'sure', 'step', 'xx', 'tail', 'airlines', 'pp', 'dhc',
    'emma', 'visite', 'compte', 'department', 'package', 'left', 'engine',
}
