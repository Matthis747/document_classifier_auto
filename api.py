"""
FastAPI application for document classification.
"""
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

import config
from src.classifier import DocumentClassifier
from src.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Classifier API",
    description="API for classifying aviation PDF documents",
    version="1.0.0",
)

# Initialize classifier (lazy loading)
classifier = DocumentClassifier(
    models_dir=config.MODELS_DIR,
    ocr_config=config.OCR_CONFIG,
    max_pages=config.MAX_PAGES_PER_PDF,
    confidence_threshold=config.CONFIDENCE_THRESHOLD,
)


# =============================================================================
# Response Models
# =============================================================================


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    top_3: list
    ocr_stats: Optional[dict] = None
    error: Optional[str] = None


class TrainingResponse(BaseModel):
    status: str
    message: str
    stats: Optional[dict] = None


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: Optional[list] = None
    n_classes: Optional[int] = None
    test_accuracy: Optional[float] = None
    test_f1_macro: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ClassesResponse(BaseModel):
    classes: list
    n_classes: int


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Document Classifier API is running"}


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    """
    Get the current status of the classifier.

    Returns information about the loaded model and its performance metrics.
    """
    return classifier.get_status()


@app.get("/classes", response_model=ClassesResponse, tags=["Status"])
async def get_classes():
    """
    Get the list of document classes the model can predict.
    """
    try:
        classes = classifier.classes
        return {"classes": classes, "n_classes": len(classes)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model found. Run training first.")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a PDF document.

    Upload a PDF file and get the predicted document class with confidence scores.

    - **file**: PDF file to classify

    Returns the predicted class, confidence score, and top 3 predictions.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Classify the document
        result = classifier.predict_file(tmp_path)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


# Training state
training_in_progress = False
last_training_result = None


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train(background_tasks: BackgroundTasks, force: bool = False):
    """
    Train or update the classification model.

    This endpoint triggers the training pipeline which:
    1. Extracts text from PDFs in the dataset directory (incremental)
    2. Normalizes the extracted text (incremental)
    3. Trains/retrains the model if data changed

    - **force**: If true, re-process everything from scratch

    Note: Training runs in the background. Check /train/status for progress.
    """
    global training_in_progress, last_training_result

    if training_in_progress:
        return {
            "status": "in_progress",
            "message": "Training is already in progress",
            "stats": None,
        }

    def run_training(force_flag: bool):
        global training_in_progress, last_training_result
        training_in_progress = True
        try:
            trainer = ModelTrainer(
                dataset_dir=config.DATASET_DIR,
                data_dir=config.DATA_DIR,
                models_dir=config.MODELS_DIR,
                ocr_config=config.OCR_CONFIG,
                tfidf_config=config.TFIDF_CONFIG,
                svm_config=config.SVM_CONFIG,
                max_pages=config.MAX_PAGES_PER_PDF,
                test_size=config.TEST_SIZE,
                val_size=config.VAL_SIZE,
                stopwords=config.DOMAIN_STOPWORDS,
            )
            last_training_result = trainer.train(force=force_flag)
            last_training_result["status"] = "success"
        except Exception as e:
            logger.error(f"Training failed: {e}")
            last_training_result = {"status": "error", "error": str(e)}
        finally:
            training_in_progress = False
            # Reload classifier model
            classifier._model = None
            classifier._tfidf = None
            classifier._metadata = None

    background_tasks.add_task(run_training, force)

    return {
        "status": "started",
        "message": "Training started in background. Check /train/status for progress.",
        "stats": None,
    }


@app.get("/train/status", tags=["Training"])
async def train_status():
    """
    Get the status of the last training run.
    """
    global training_in_progress, last_training_result

    if training_in_progress:
        return {"status": "in_progress", "message": "Training is currently running"}

    if last_training_result is None:
        return {"status": "no_training", "message": "No training has been run yet"}

    return last_training_result


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
