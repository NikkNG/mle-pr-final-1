#!/bin/bash

# Скрипт запуска MLflow сервера
# Запускает MLflow сервер с локальным файловым хранилищем

echo "Starting MLflow Tracking Server..."

# Set environment variables
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlruns"

# Create necessary directories
mkdir -p mlruns
mkdir -p experiments

# Start MLflow server
mlflow server \
    --backend-store-uri file:./mlruns \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5001 \
    --serve-artifacts

echo "MLflow server started at http://127.0.0.1:5001"
echo "Use Ctrl+C to stop the server"
