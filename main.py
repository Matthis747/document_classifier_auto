"""
CLI entry point for the document classifier.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import config
from src.classifier import DocumentClassifier
from src.trainer import ModelTrainer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_train(args):
    """Train the model."""
    trainer = ModelTrainer(
        dataset_dir=Path(args.dataset) if args.dataset else config.DATASET_DIR,
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

    result = trainer.train(force=args.force)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nExtraction:")
    print(f"  - Total documents: {result['extraction']['total']}")
    print(f"  - New processed: {result['extraction']['new_processed']}")
    print(f"  - Removed: {result['extraction']['removed']}")

    print(f"\nNormalization:")
    print(f"  - Total documents: {result['normalization']['total']}")
    print(f"  - New normalized: {result['normalization']['new_normalized']}")

    if result['retrained']:
        print(f"\nModel Training:")
        print(f"  - Train set: {result['training']['n_train']}")
        print(f"  - Validation set: {result['training']['n_val']}")
        print(f"  - Test set: {result['training']['n_test']}")
        print(f"  - Test Accuracy: {result['training']['test_accuracy']:.3f}")
        print(f"  - Test F1 Macro: {result['training']['test_f1_macro']:.3f}")
        print(f"  - Classes: {result['training']['classes']}")
    else:
        print("\nNo changes detected - model not retrained")

    print("=" * 60)


def cmd_predict(args):
    """Predict document class."""
    classifier = DocumentClassifier(
        models_dir=config.MODELS_DIR,
        ocr_config=config.OCR_CONFIG,
        max_pages=config.MAX_PAGES_PER_PDF,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Classifying: {pdf_path.name}")
    print("-" * 40)

    result = classifier.predict_file(pdf_path)

    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop 3 Predictions:")
    for i, pred in enumerate(result['top_3'], 1):
        print(f"  {i}. {pred['class']}: {pred['score']:.2%}")

    if result.get('ocr_stats'):
        print(f"\nOCR Stats:")
        print(f"  - Pages processed: {result['ocr_stats']['n_pages_processed']}/{result['ocr_stats']['n_pages_total']}")
        print(f"  - Text length: {result['ocr_stats']['text_length']} chars")

    if args.json:
        print(f"\nJSON Output:")
        print(json.dumps(result, indent=2))


def cmd_status(args):
    """Get model status."""
    classifier = DocumentClassifier(
        models_dir=config.MODELS_DIR,
        ocr_config=config.OCR_CONFIG,
        max_pages=config.MAX_PAGES_PER_PDF,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    status = classifier.get_status()

    print("\n" + "=" * 60)
    print("MODEL STATUS")
    print("=" * 60)

    print(f"\nStatus: {status['status']}")
    print(f"Model loaded: {status['model_loaded']}")

    if status['model_loaded']:
        print(f"\nClasses ({status['n_classes']}):")
        for cls in status['classes']:
            print(f"  - {cls}")
        print(f"\nPerformance:")
        print(f"  - Test Accuracy: {status['test_accuracy']:.3f}")
        print(f"  - Test F1 Macro: {status['test_f1_macro']:.3f}")
    else:
        print(f"\nMessage: {status.get('message', status.get('error', 'Unknown'))}")

    print("=" * 60)


def cmd_api(args):
    """Start the API server."""
    import uvicorn

    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Document Classifier - Train and classify aviation documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train or update the model")
    train_parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset directory (default: data/dataset)",
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all documents",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Classify a PDF document")
    predict_parser.add_argument("file", type=str, help="Path to PDF file to classify")
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Get model status")

    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument(
        "--host",
        type=str,
        default=config.API_HOST,
        help=f"Host to bind to (default: {config.API_HOST})",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help=f"Port to bind to (default: {config.API_PORT})",
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "api":
        cmd_api(args)


if __name__ == "__main__":
    main()
