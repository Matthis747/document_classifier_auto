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


def _create_trainer(dataset=None):
    """Create a ModelTrainer with standard config."""
    return ModelTrainer(
        dataset_dir=Path(dataset) if dataset else config.DATASET_DIR,
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


def cmd_extract(args):
    """Extract text from PDFs using OCR."""
    trainer = _create_trainer(getattr(args, 'dataset', None))
    result = trainer.extract_texts(force=args.force)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  - Total documents: {result['total']}")
    print(f"  - New processed: {result['new_processed']}")
    print(f"  - Removed: {result['removed']}")
    print(f"  - Errors: {result['errors']}")
    print("=" * 60)


def cmd_preprocess(args):
    """Normalize extracted texts."""
    trainer = _create_trainer()
    result = trainer.normalize_texts(force=args.force)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  - Total documents: {result['total']}")
    print(f"  - New normalized: {result['new_normalized']}")
    print(f"  - Removed: {result['removed']}")
    print("=" * 60)


def cmd_train_model(args):
    """Train the model only (requires preprocessed data)."""
    trainer = _create_trainer()
    result = trainer.train_model()

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"  - Train set: {result['n_train']}")
    print(f"  - Validation set: {result['n_val']}")
    print(f"  - Test set: {result['n_test']}")
    print(f"  - Test Accuracy: {result['test_accuracy']:.3f}")
    print(f"  - Test F1 Macro: {result['test_f1_macro']:.3f}")
    print(f"  - Classes: {result['classes']}")
    print("=" * 60)


def cmd_evaluate(args):
    """Evaluate the trained model."""
    metadata_file = config.MODELS_DIR / "model_metadata.json"
    if not metadata_file.exists():
        print("Error: No trained model found. Run training first.")
        sys.exit(1)

    with open(metadata_file) as f:
        metadata = json.load(f)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"  - Test Accuracy: {metadata['test_accuracy']:.4f}")
    print(f"  - Test F1 Macro: {metadata['test_f1_macro']:.4f}")
    print(f"  - Classes: {metadata['classes']}")
    print(f"  - Samples: train={metadata['n_train']}, val={metadata['n_val']}, test={metadata['n_test']}")

    if metadata['test_accuracy'] < 0.90:
        print("\n  WARNING: Model accuracy below 90%!")
        sys.exit(1)

    print("\n  Model evaluation PASSED")
    print("=" * 60)


def cmd_train(args):
    """Train the model (full pipeline)."""
    trainer = _create_trainer(getattr(args, 'dataset', None))

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


def cmd_predict_dir(args):
    """Classify all PDFs in a directory."""
    classifier = DocumentClassifier(
        models_dir=config.MODELS_DIR,
        ocr_config=config.OCR_CONFIG,
        max_pages=config.MAX_PAGES_PER_PDF,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        sys.exit(1)

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {dir_path}")
        sys.exit(1)

    print(f"Classifying {len(pdf_files)} PDFs from {dir_path}")
    print("=" * 60)

    results = []
    for pdf in pdf_files:
        result = classifier.predict_file(pdf)
        results.append(result)
        status = "OK" if not result.get("error") else "ERR"
        print(f"  [{status}] {result['filename']} -> {result['predicted_class']} ({result['confidence']:.2%})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total: {len(results)}")
    class_counts = {}
    for r in results:
        cls = r['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, count in sorted(class_counts.items()):
        print(f"  - {cls}: {count}")
    print("=" * 60)

    if args.json:
        print(f"\nJSON Output:")
        print(json.dumps(results, indent=2))


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

    # Extract command (for DVC pipeline)
    extract_parser = subparsers.add_parser("extract", help="Extract text from PDFs using OCR")
    extract_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    extract_parser.add_argument("--force", action="store_true", help="Force re-extraction")

    # Preprocess command (for DVC pipeline)
    preprocess_parser = subparsers.add_parser("preprocess", help="Normalize extracted texts")
    preprocess_parser.add_argument("--force", action="store_true", help="Force re-normalization")

    # Train-model command (for DVC pipeline - train step only)
    train_model_parser = subparsers.add_parser("train-model", help="Train the model (requires preprocessed data)")

    # Evaluate command (for DVC pipeline)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model")

    # Train command (full pipeline)
    train_parser = subparsers.add_parser("train", help="Run full training pipeline (extract + preprocess + train)")
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

    # Predict-dir command
    predict_dir_parser = subparsers.add_parser("predict-dir", help="Classify all PDFs in a directory")
    predict_dir_parser.add_argument("directory", type=str, help="Path to directory with PDFs")
    predict_dir_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
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

    commands = {
        "extract": cmd_extract,
        "preprocess": cmd_preprocess,
        "train-model": cmd_train_model,
        "evaluate": cmd_evaluate,
        "train": cmd_train,
        "predict": cmd_predict,
        "predict-dir": cmd_predict_dir,
        "status": cmd_status,
        "api": cmd_api,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
