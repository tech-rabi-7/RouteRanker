#!/usr/bin/env bash
# ==============================================================================
# RouteRanker v2.0 - Local Server Runner
# Starts both FastAPI Backend (port 8000) and React Frontend (port 5173)
# ==============================================================================

set -e

# Change to project root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=========================================================="
echo " Starting RouteRanker v2.0 Local Services"
echo "=========================================================="

# Trap SIGINT (Ctrl+C) to gracefully stop both background processes
cleanup() {
    echo ""
    echo "Stopping RouteRanker services..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Backend
echo "1. Starting FastAPI Backend on http://127.0.0.1:8000 ..."
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &

# Wait briefly for backend to bind port
sleep 2

# 2. Start Frontend
echo "2. Starting React Vite Frontend on http://127.0.0.1:5173 ..."
cd "$ROOT_DIR/frontend"
npm run dev -- --host 0.0.0.0 --port 5173 &

echo "=========================================================="
echo " RouteRanker is now running!"
echo " Frontend UI:  http://localhost:5173"
echo " Backend API:  http://localhost:8000 (Docs: /docs)"
echo " Press [Ctrl + C] anytime to stop both servers."
echo "=========================================================="

# Wait for background processes
wait
