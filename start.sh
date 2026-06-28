#!/usr/bin/env bash
set -e

# Exit on error
trap 'echo "Error on line $LINENO"; exit 1' ERR

# Get PORT from environment or default to 8000
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

echo "🚀 Starting ReviewRadar AI..."
echo "📍 Server: http://$HOST:$PORT"

# Run the application
exec uvicorn backend.main:app --host $HOST --port $PORT --log-level info

