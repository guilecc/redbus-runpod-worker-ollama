#!/bin/bash
# ─── RunPod Ollama Worker — Startup Script ───────────────────────
# 1. Kill stale processes
# 2. Start Ollama server
# 3. Wait for readiness (health-check loop)
# 4. Pull model if needed
# 5. Preload model into VRAM (warm start)
# 6. Launch RunPod handler

set -euo pipefail

cleanup() {
    echo "⏹  Cleaning up..."
    pkill -P $$ 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# ─── 1. Kill stale Ollama processes ──────────────────────────────
pgrep ollama | xargs kill 2>/dev/null || true

# ─── 2. Start Ollama server ─────────────────────────────────────
echo "🚀 Starting Ollama server..."
ollama serve 2>&1 | tee /tmp/ollama.server.log &
OLLAMA_PID=$!

# ─── 3. Wait for readiness (A3: curl health check) ──────────────
MAX_RETRIES=60
RETRY_INTERVAL=2
echo "⏳ Waiting for Ollama to be ready..."

for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready (attempt $i)"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "❌ Ollama failed to start after $((MAX_RETRIES * RETRY_INTERVAL))s"
        cat /tmp/ollama.server.log
        exit 1
    fi
    sleep $RETRY_INTERVAL
done

# ─── 4. Pull model if OLLAMA_MODEL_NAME is set ──────────────────
if [ -z "${OLLAMA_MODEL_NAME:-}" ]; then
    echo "ℹ️  No OLLAMA_MODEL_NAME set. Skipping model pull."
else
    echo "📦 Pulling model: $OLLAMA_MODEL_NAME..."
    ollama pull "$OLLAMA_MODEL_NAME"

    # ─── 5. Preload model into VRAM (A1: warm start) ─────────────
    echo "🔥 Preloading $OLLAMA_MODEL_NAME into VRAM..."
    curl -sf http://localhost:11434/api/generate \
        -d "{\"model\": \"$OLLAMA_MODEL_NAME\", \"prompt\": \"\", \"stream\": false}" \
        > /dev/null 2>&1 || echo "⚠️  Preload request failed (non-fatal)"
    echo "✅ Model $OLLAMA_MODEL_NAME loaded into VRAM"
fi

# ─── 6. Launch RunPod handler ────────────────────────────────────
echo "🎯 Starting RunPod handler..."
python -u handler.py "$@"