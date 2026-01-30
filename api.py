"""
FastAPI application for document classification.
"""
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

import config
from src.classifier import DocumentClassifier
from src.trainer import ModelTrainer
from monitoring import get_monitoring_service, MonitoringService

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

# Initialize monitoring service
monitoring = get_monitoring_service()


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


class MetricsResponse(BaseModel):
    metrics: list
    total: int
    limit: int
    offset: int
    summary: dict


class AlertsResponse(BaseModel):
    alerts: list
    total: int
    limit: int
    offset: int
    severity_counts: dict
    unresolved_count: int


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    health = monitoring.get_health()
    return {
        "status": "ok",
        "message": "Document Classifier API is running",
        "monitoring": health,
    }


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

    # Start timing
    start_time = time.time()
    tmp_path = None

    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Classify the document
        result = classifier.predict_file(tmp_path)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Log prediction to monitoring
        monitoring.log_prediction(
            filename=file.filename,
            predicted_class=result.get("predicted_class", "ERROR"),
            confidence=result.get("confidence", 0.0),
            processing_time_ms=processing_time_ms,
            top_3=result.get("top_3", []),
            ocr_stats=result.get("ocr_stats"),
        )

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path and tmp_path.exists():
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
        start_time = time.time()

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

            # Log training metrics to monitoring
            training_time_s = time.time() - start_time
            training_stats = last_training_result.get("training", {})

            if last_training_result.get("retrained", False):
                monitoring.log_training(
                    n_samples=training_stats.get("n_train", 0) + training_stats.get("n_val", 0) + training_stats.get("n_test", 0),
                    n_classes=len(training_stats.get("classes", [])),
                    train_accuracy=training_stats.get("val_accuracy", 0),  # Using val as proxy
                    val_accuracy=training_stats.get("val_accuracy", 0),
                    test_accuracy=training_stats.get("test_accuracy", 0),
                    test_f1_macro=training_stats.get("test_f1_macro", 0),
                    training_time_s=training_time_s,
                    new_documents=last_training_result.get("extraction", {}).get("new_processed", 0),
                    removed_documents=last_training_result.get("extraction", {}).get("removed", 0),
                )

            # Version models with DVC after training
            if last_training_result.get("retrained", False):
                try:
                    subprocess.run(["dvc", "add", "models"], check=True)
                    subprocess.run(["dvc", "push"], check=True)
                    logger.info("Models versioned and pushed with DVC")
                    last_training_result["dvc_versioned"] = True
                except Exception as dvc_err:
                    logger.warning(f"DVC versioning failed: {dvc_err}")
                    last_training_result["dvc_versioned"] = False

        except Exception as e:
            logger.error(f"Training failed: {e}")
            last_training_result = {"status": "error", "error": str(e)}

            # Log training failure alert
            monitoring.create_custom_alert(
                alert_type="training_failed",
                severity="critical",
                message=f"Training failed: {str(e)}",
                details={"error": str(e)},
            )
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
# Monitoring Endpoints
# =============================================================================


@app.get("/monitoring/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(
    type: Optional[str] = Query(None, description="Filter by metric type (prediction, training)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
):
    """
    Get logged metrics.

    Returns prediction and training metrics with summary statistics.

    - **type**: Filter by metric type (prediction, training, or omit for all)
    - **limit**: Maximum number of records to return (default: 100, max: 1000)
    - **offset**: Number of records to skip for pagination
    """
    return monitoring.get_metrics(metric_type=type, limit=limit, offset=offset)


@app.get("/monitoring/alerts", response_model=AlertsResponse, tags=["Monitoring"])
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (info, warning, critical)"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
):
    """
    Get logged alerts.

    Returns alerts with counts by severity level.

    - **severity**: Filter by severity (info, warning, critical)
    - **resolved**: Filter by resolved status (true/false)
    - **limit**: Maximum number of records to return (default: 100)
    - **offset**: Number of records to skip for pagination
    """
    return monitoring.get_alerts(severity=severity, resolved=resolved, limit=limit, offset=offset)


@app.get("/monitoring/health", tags=["Monitoring"])
async def get_monitoring_health():
    """
    Get monitoring health status.

    Returns summary of system health based on alerts and metrics.
    """
    return monitoring.get_health()


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
