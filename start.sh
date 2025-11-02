#!/bin/bash
# Render deployment start script

# Get PORT from environment (Render provides this)
PORT=${PORT:-8000}

echo "Starting FastAPI application on port $PORT..."

# Start uvicorn with proper configuration for Render
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
