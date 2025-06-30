#!/bin/bash

# Start API script for E-commerce Recommendation System
# This script starts the FastAPI server with proper configuration

set -e  # Exit on any error

echo "🚀 Starting E-commerce Recommendation API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn, mlflow, implicit" 2>/dev/null || {
    echo "❌ Required packages not installed. Installing..."
    pip install -r requirements.txt
}

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p mlruns

# Check if MLflow server is running
echo "🔍 Checking MLflow server..."
if ! curl -s http://127.0.0.1:5000/health > /dev/null 2>&1; then
    echo "⚠️  MLflow server not running. Starting MLflow server..."
    mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
    sleep 5
    echo "✅ MLflow server started"
fi

# Default configuration
HOST=${API_HOST:-"127.0.0.1"}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-1}
RELOAD=${API_RELOAD:-true}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "🌐 API Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo "   Reload: $RELOAD"
echo "   Log Level: $LOG_LEVEL"

# Start the API server
echo "🚀 Starting FastAPI server..."

if [ "$RELOAD" = "true" ]; then
    # Development mode with auto-reload
    uvicorn src.api.main:app \
        --host $HOST \
        --port $PORT \
        --reload \
        --log-level $LOG_LEVEL \
        --access-log \
        --reload-dir src/
else
    # Production mode
    uvicorn src.api.main:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --log-level $LOG_LEVEL \
        --access-log
fi 