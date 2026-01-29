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

# Install PaddlePaddle (choose one)
# GPU (CUDA 12.9):
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
# CPU only:
# python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

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
├── monitoring/
│   └── logs/            # Metrics and alerts (JSONL)
├── config.py            # Configuration
├── main.py              # CLI entry point
├── api.py               # FastAPI application
├── monitoring.py        # Monitoring service
├── dvc.yaml             # DVC pipeline definition
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
| `/monitoring/metrics` | GET | Get prediction/training metrics |
| `/monitoring/alerts` | GET | Get system alerts |
| `/monitoring/health` | GET | Get monitoring health status |

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

---

## Monitoring

The system includes built-in monitoring for predictions and training.

### Metrics Logged

- **Predictions**: filename, class, confidence, processing time, top 3 predictions
- **Training**: accuracy, F1 score, number of samples, training time

### Alerts

Automatic alerts are generated when:
- Prediction confidence < 50% (warning)
- Model accuracy < 90% (critical)
- Training fails (critical)

### Slack Notifications

Pour recevoir les alertes dans Slack :

1. Créer un Webhook Slack sur https://api.slack.com/apps
2. Configurer la variable d'environnement :

```bash
# Windows PowerShell
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"

# Linux/Mac
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

3. Lancer l'API - les alertes `warning` et `critical` seront envoyées à Slack automatiquement.

### Log Files

Logs are stored in JSONL format:
```
monitoring/logs/
├── metrics.jsonl    # Prediction and training metrics
└── alerts.jsonl     # System alerts
```

### Monitoring API Endpoints

```bash
# Get metrics (with optional filters)
curl "http://localhost:8080/monitoring/metrics?type=prediction&limit=50"

# Get alerts
curl "http://localhost:8080/monitoring/alerts?severity=critical"

# Get monitoring health
curl http://localhost:8080/monitoring/health
```

### Metrics Response Example

```json
{
  "metrics": [...],
  "total": 150,
  "summary": {
    "total_predictions": 150,
    "predictions": {
      "avg_confidence": 0.85,
      "min_confidence": 0.42,
      "low_confidence_count": 3,
      "avg_processing_time_ms": 2500,
      "class_distribution": {
        "work_order": 45,
        "ATL": 38,
        ...
      }
    },
    "latest_training": {
      "test_accuracy": 0.97,
      "n_samples": 231
    }
  }
}
```

---

## DVC (Data Version Control)

Le projet est configuré pour utiliser DVC pour la gestion des données et la reproductibilité.

### Installation DVC

```bash
pip install dvc
# Avec support S3/GCS (optionnel)
pip install "dvc[s3]"  # ou dvc[gs], dvc[azure]
```

### Initialisation

```bash
# Initialiser DVC dans le projet
dvc init

# Configurer le remote storage (exemple S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage
```

### Pipeline DVC

Le fichier `dvc.yaml` définit 4 stages :

```
extract_text → preprocess → train → evaluate
```

### Commandes DVC

```bash
# Exécuter le pipeline complet
dvc repro

# Exécuter un stage spécifique
dvc repro train

# Voir le statut du pipeline
dvc status

# Voir le DAG du pipeline
dvc dag

# Pousser les données vers le remote
dvc push

# Récupérer les données depuis le remote
dvc pull
```

### Tracking des données

```bash
# Tracker le dataset
dvc add data/dataset/

# Tracker les modèles (si pas dans dvc.yaml outputs)
dvc add models/svm_model.joblib
```

### Expériences avec DVC

```bash
# Lancer une expérience avec paramètres modifiés
dvc exp run -S config.py:TFIDF_CONFIG.max_features=5000

# Voir les expériences
dvc exp show

# Comparer les expériences
dvc exp diff
```

---

## Amélioration Continue avec Argo CronWorkflow

Pour réentraîner automatiquement le modèle à intervalles réguliers, utilisez un **CronWorkflow Argo**.

### CronWorkflow YAML

Créez le fichier `argo/cron-retrain.yaml` :

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: document-classifier-retrain
  namespace: argo
spec:
  # Tous les dimanches à 2h du matin
  schedule: "0 2 * * 0"
  # Timezone
  timezone: "Europe/Paris"
  # Concurrence: ne pas lancer si le précédent tourne encore
  concurrencyPolicy: "Forbid"
  # Conserver les 3 dernières exécutions
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3

  workflowSpec:
    entrypoint: retrain-pipeline

    # Service account avec les permissions nécessaires
    serviceAccountName: argo-workflow

    # Volume pour les données
    volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: classifier-data-pvc

    templates:
      - name: retrain-pipeline
        dag:
          tasks:
            # 1. Synchroniser les nouvelles données
            - name: sync-data
              template: sync-data-template

            # 2. Entraîner le modèle (incrémental)
            - name: train-model
              template: train-template
              dependencies: [sync-data]

            # 3. Évaluer le modèle
            - name: evaluate
              template: evaluate-template
              dependencies: [train-model]

            # 4. Déployer si accuracy > seuil
            - name: deploy
              template: deploy-template
              dependencies: [evaluate]
              when: "{{tasks.evaluate.outputs.parameters.should-deploy}} == true"

      # Template: Sync data from S3/GCS
      - name: sync-data-template
        container:
          image: amazon/aws-cli:latest
          command: [sh, -c]
          args:
            - |
              aws s3 sync s3://my-bucket/dataset/ /data/dataset/
          volumeMounts:
            - name: data-volume
              mountPath: /data

      # Template: Train model
      - name: train-template
        container:
          image: registry.example.com/document-classifier:latest
          command: [python, main.py, train]
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
          resources:
            requests:
              memory: "8Gi"
              cpu: "2"
              nvidia.com/gpu: "1"
            limits:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: data-volume
              mountPath: /app/data

      # Template: Evaluate and decide deployment
      - name: evaluate-template
        container:
          image: registry.example.com/document-classifier:latest
          command: [sh, -c]
          args:
            - |
              python -c "
              import json
              metadata = json.load(open('models/model_metadata.json'))
              accuracy = metadata['test_accuracy']
              print(f'Test Accuracy: {accuracy}')
              # Output parameter for conditional deployment
              should_deploy = 'true' if accuracy >= 0.90 else 'false'
              with open('/tmp/should-deploy.txt', 'w') as f:
                  f.write(should_deploy)
              "
          volumeMounts:
            - name: data-volume
              mountPath: /app/data
        outputs:
          parameters:
            - name: should-deploy
              valueFrom:
                path: /tmp/should-deploy.txt

      # Template: Deploy new model
      - name: deploy-template
        container:
          image: bitnami/kubectl:latest
          command: [sh, -c]
          args:
            - |
              # Rolling restart du deployment pour charger le nouveau modèle
              kubectl rollout restart deployment/document-classifier-api -n production
              kubectl rollout status deployment/document-classifier-api -n production
```

### Déploiement du CronWorkflow

```bash
# Appliquer le CronWorkflow
kubectl apply -f argo/cron-retrain.yaml

# Vérifier le statut
kubectl get cronworkflows -n argo

# Voir les exécutions
argo cron list -n argo

# Déclencher manuellement une exécution
argo submit --from cronwf/document-classifier-retrain -n argo

# Voir les logs
argo logs -n argo @latest
```

### Alerting sur échec

Ajoutez un template de notification en cas d'échec :

```yaml
# Dans workflowSpec
onExit: exit-handler

templates:
  - name: exit-handler
    steps:
      - - name: notify-failure
          template: slack-notification
          when: "{{workflow.status}} != Succeeded"

  - name: slack-notification
    container:
      image: curlimages/curl:latest
      command: [sh, -c]
      args:
        - |
          curl -X POST $SLACK_WEBHOOK_URL \
            -H 'Content-type: application/json' \
            -d '{"text":"⚠️ Document Classifier retraining failed! Status: {{workflow.status}}"}'
      env:
        - name: SLACK_WEBHOOK_URL
          valueFrom:
            secretKeyRef:
              name: slack-secrets
              key: webhook-url
```

### Bonnes pratiques

1. **Monitoring des CronWorkflows**
   - Configurez Prometheus/Grafana pour surveiller les exécutions
   - Alertez sur les échecs répétés

2. **Gestion des ressources**
   - Utilisez des `ResourceQuota` pour limiter les ressources
   - Configurez l'auto-scaling du cluster si nécessaire

3. **Rollback automatique**
   - Si l'accuracy chute après déploiement, déclenchez un rollback
   - Gardez les N dernières versions du modèle

4. **Tests de régression**
   - Ajoutez un stage de test sur un dataset de validation fixe
   - Comparez les performances avec le modèle en production
