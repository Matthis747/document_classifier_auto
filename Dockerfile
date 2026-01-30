# Dockerfile for Document Classifier
# Uses pre-built base image with all dependencies

FROM 192.168.49.2:5000/doc-classifier-base:latest

WORKDIR /app

# Install DVC
RUN pip install dvc --break-system-packages

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/dataset models monitoring/logs

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Default command - run API server
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8080"]