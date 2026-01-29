"""
Monitoring module for document classifier.
Logs predictions, training metrics, and generates alerts.
"""
import json
import logging
import os
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Slack configuration via environment variable
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_ENABLED = bool(SLACK_WEBHOOK_URL)


class MonitoringService:
    """Service for logging metrics and generating alerts."""

    # Alert thresholds
    CONFIDENCE_THRESHOLD = 0.50  # Alert if confidence < 50%
    ACCURACY_THRESHOLD = 0.90   # Alert if accuracy < 90%

    def __init__(self, logs_dir: Optional[Path] = None):
        """
        Initialize the monitoring service.

        Args:
            logs_dir: Directory for log files. Defaults to monitoring/logs/
        """
        self.logs_dir = logs_dir or Path(__file__).parent / "monitoring" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.logs_dir / "metrics.jsonl"
        self.alerts_file = self.logs_dir / "alerts.jsonl"

        # Thread safety for file writes
        self._metrics_lock = Lock()
        self._alerts_lock = Lock()

        # Initialize files if they don't exist
        self._init_files()

        logger.info(f"Monitoring initialized. Logs dir: {self.logs_dir}")

    def _init_files(self):
        """Create log files if they don't exist."""
        for file_path in [self.metrics_file, self.alerts_file]:
            if not file_path.exists():
                file_path.touch()

    def _append_jsonl(self, file_path: Path, data: dict, lock: Lock):
        """Append a JSON line to a file with thread safety."""
        with lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _read_jsonl(self, file_path: Path, limit: int = 100, offset: int = 0) -> List[dict]:
        """Read JSON lines from a file."""
        if not file_path.exists():
            return []

        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        # Get last N lines (most recent first)
        all_lines = all_lines[::-1]  # Reverse to get most recent first

        for line in all_lines[offset:offset + limit]:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return lines

    # =========================================================================
    # PREDICTION LOGGING
    # =========================================================================

    def log_prediction(
        self,
        filename: str,
        predicted_class: str,
        confidence: float,
        processing_time_ms: float,
        top_3: List[dict],
        ocr_stats: Optional[dict] = None,
    ):
        """
        Log a prediction event.

        Args:
            filename: Name of the classified file
            predicted_class: Predicted document class
            confidence: Confidence score (0-1)
            processing_time_ms: Processing time in milliseconds
            top_3: Top 3 predictions with scores
            ocr_stats: OCR statistics (pages processed, text length)
        """
        metric = {
            "type": "prediction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time_ms, 2),
            "top_3": top_3,
            "ocr_stats": ocr_stats,
        }

        self._append_jsonl(self.metrics_file, metric, self._metrics_lock)
        logger.debug(f"Logged prediction: {filename} -> {predicted_class} ({confidence:.2%})")

        # Check for low confidence alert
        if confidence < self.CONFIDENCE_THRESHOLD:
            self._create_alert(
                alert_type="low_confidence",
                severity="warning",
                message=f"Low confidence prediction: {confidence:.2%}",
                details={
                    "filename": filename,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "threshold": self.CONFIDENCE_THRESHOLD,
                },
            )

    # =========================================================================
    # TRAINING LOGGING
    # =========================================================================

    def log_training(
        self,
        n_samples: int,
        n_classes: int,
        train_accuracy: float,
        val_accuracy: float,
        test_accuracy: float,
        test_f1_macro: float,
        training_time_s: float,
        new_documents: int = 0,
        removed_documents: int = 0,
    ):
        """
        Log a training event.

        Args:
            n_samples: Total number of training samples
            n_classes: Number of classes
            train_accuracy: Training set accuracy
            val_accuracy: Validation set accuracy
            test_accuracy: Test set accuracy
            test_f1_macro: Test set macro F1 score
            training_time_s: Training time in seconds
            new_documents: Number of new documents added
            removed_documents: Number of documents removed
        """
        metric = {
            "type": "training",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_samples": n_samples,
            "n_classes": n_classes,
            "train_accuracy": round(train_accuracy, 4),
            "val_accuracy": round(val_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "test_f1_macro": round(test_f1_macro, 4),
            "training_time_s": round(training_time_s, 2),
            "new_documents": new_documents,
            "removed_documents": removed_documents,
        }

        self._append_jsonl(self.metrics_file, metric, self._metrics_lock)
        logger.info(f"Logged training: {n_samples} samples, accuracy={test_accuracy:.2%}")

        # Check for low accuracy alert
        if test_accuracy < self.ACCURACY_THRESHOLD:
            self._create_alert(
                alert_type="low_accuracy",
                severity="critical",
                message=f"Model accuracy below threshold: {test_accuracy:.2%}",
                details={
                    "test_accuracy": test_accuracy,
                    "test_f1_macro": test_f1_macro,
                    "threshold": self.ACCURACY_THRESHOLD,
                    "n_samples": n_samples,
                },
            )

    # =========================================================================
    # SLACK NOTIFICATIONS
    # =========================================================================

    def _send_slack_notification(
        self,
        message: str,
        severity: str = "info",
        details: Optional[dict] = None,
    ) -> bool:
        """
        Send a notification to Slack.

        Args:
            message: Main message text
            severity: Alert severity (info, warning, critical)
            details: Additional details to include

        Returns:
            True if sent successfully, False otherwise
        """
        if not SLACK_ENABLED:
            logger.debug("Slack notifications disabled (no SLACK_WEBHOOK_URL set)")
            return False

        # Emoji based on severity
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🔴",
        }
        emoji = emoji_map.get(severity, "📢")

        # Build Slack message
        slack_message = {
            "text": f"{emoji} *Document Classifier Alert*\n{message}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Document Classifier Alert",
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:*\n{severity.upper()}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"},
                    ]
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Message:*\n{message}"}
                },
            ]
        }

        # Add details if provided
        if details:
            details_text = "\n".join([f"• *{k}:* {v}" for k, v in details.items()])
            slack_message["blocks"].append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Details:*\n{details_text}"}
            })

        try:
            data = json.dumps(slack_message).encode("utf-8")
            req = urllib.request.Request(
                SLACK_WEBHOOK_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info(f"Slack notification sent: {message[:50]}...")
                    return True
        except urllib.error.URLError as e:
            logger.error(f"Failed to send Slack notification: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Slack notification: {e}")

        return False

    def send_slack_message(self, message: str, severity: str = "info") -> bool:
        """Public method to send a custom Slack message."""
        return self._send_slack_notification(message, severity)

    # =========================================================================
    # ALERTS
    # =========================================================================

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[dict] = None,
    ):
        """
        Create and log an alert.

        Args:
            alert_type: Type of alert (low_confidence, low_accuracy, etc.)
            severity: Alert severity (info, warning, critical)
            message: Human-readable message
            details: Additional details
        """
        alert = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "resolved": False,
        }

        self._append_jsonl(self.alerts_file, alert, self._alerts_lock)
        logger.warning(f"Alert [{severity}] {alert_type}: {message}")

        # Send to Slack for warning and critical alerts
        if severity in ("warning", "critical"):
            self._send_slack_notification(message, severity, details)

    def create_custom_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[dict] = None,
    ):
        """Public method to create custom alerts."""
        self._create_alert(alert_type, severity, message, details)

    # =========================================================================
    # METRICS RETRIEVAL
    # =========================================================================

    def get_metrics(
        self,
        metric_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get logged metrics.

        Args:
            metric_type: Filter by type (prediction, training, or None for all)
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            Dict with metrics and summary statistics
        """
        all_metrics = self._read_jsonl(self.metrics_file, limit=1000, offset=0)

        # Filter by type if specified
        if metric_type:
            filtered = [m for m in all_metrics if m.get("type") == metric_type]
        else:
            filtered = all_metrics

        # Apply pagination
        paginated = filtered[offset:offset + limit]

        # Calculate summary statistics
        summary = self._calculate_summary(all_metrics)

        return {
            "metrics": paginated,
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "summary": summary,
        }

    def _calculate_summary(self, metrics: List[dict]) -> dict:
        """Calculate summary statistics from metrics."""
        predictions = [m for m in metrics if m.get("type") == "prediction"]
        trainings = [m for m in metrics if m.get("type") == "training"]

        summary = {
            "total_predictions": len(predictions),
            "total_trainings": len(trainings),
        }

        # Prediction statistics
        if predictions:
            confidences = [p["confidence"] for p in predictions if "confidence" in p]
            processing_times = [p["processing_time_ms"] for p in predictions if "processing_time_ms" in p]

            if confidences:
                summary["predictions"] = {
                    "avg_confidence": round(float(np.mean(confidences)), 4),
                    "min_confidence": round(float(np.min(confidences)), 4),
                    "max_confidence": round(float(np.max(confidences)), 4),
                    "low_confidence_count": sum(1 for c in confidences if c < self.CONFIDENCE_THRESHOLD),
                }

            if processing_times:
                summary["predictions"]["avg_processing_time_ms"] = round(float(np.mean(processing_times)), 2)
                summary["predictions"]["p95_processing_time_ms"] = round(float(np.percentile(processing_times, 95)), 2)

            # Class distribution
            class_counts = {}
            for p in predictions:
                cls = p.get("predicted_class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
            summary["predictions"]["class_distribution"] = class_counts

        # Training statistics
        if trainings:
            latest_training = trainings[0]  # Most recent
            summary["latest_training"] = {
                "timestamp": latest_training.get("timestamp"),
                "test_accuracy": latest_training.get("test_accuracy"),
                "test_f1_macro": latest_training.get("test_f1_macro"),
                "n_samples": latest_training.get("n_samples"),
            }

        return summary

    def get_alerts(
        self,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get logged alerts.

        Args:
            severity: Filter by severity (info, warning, critical)
            resolved: Filter by resolved status
            limit: Maximum number of records
            offset: Number of records to skip

        Returns:
            Dict with alerts and counts by severity
        """
        all_alerts = self._read_jsonl(self.alerts_file, limit=1000, offset=0)

        # Apply filters
        filtered = all_alerts
        if severity:
            filtered = [a for a in filtered if a.get("severity") == severity]
        if resolved is not None:
            filtered = [a for a in filtered if a.get("resolved") == resolved]

        # Apply pagination
        paginated = filtered[offset:offset + limit]

        # Count by severity
        severity_counts = {
            "info": sum(1 for a in all_alerts if a.get("severity") == "info"),
            "warning": sum(1 for a in all_alerts if a.get("severity") == "warning"),
            "critical": sum(1 for a in all_alerts if a.get("severity") == "critical"),
        }

        return {
            "alerts": paginated,
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "severity_counts": severity_counts,
            "unresolved_count": sum(1 for a in all_alerts if not a.get("resolved", True)),
        }

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def get_health(self) -> dict:
        """Get monitoring health status."""
        alerts = self.get_alerts(limit=10)
        metrics = self.get_metrics(limit=1)

        critical_alerts = alerts["severity_counts"].get("critical", 0)
        warning_alerts = alerts["severity_counts"].get("warning", 0)

        if critical_alerts > 0:
            status = "critical"
        elif warning_alerts > 5:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "total_predictions": metrics["summary"].get("total_predictions", 0),
            "total_trainings": metrics["summary"].get("total_trainings", 0),
            "unresolved_alerts": alerts["unresolved_count"],
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
        }


# Global monitoring instance
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get or create the global monitoring service instance."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service
