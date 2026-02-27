#!/bin/bash
# ─── Preload models into /runpod-volume during Docker build ──────
set -euo pipefail

echo "🚀 Starting Ollama server to preload models: $MODEL_NAMES"
OLLAMA_MODELS="/runpod-volume" ollama serve &
OLLAMA_PID=$!

# Wait for readiness with curl health check
echo "⏳ Waiting for Ollama server..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama ready (attempt $i)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama failed to start"
        kill $OLLAMA_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Split comma-separated model names and pull each
IFS=',' read -r -a MODELS <<< "$MODEL_NAMES"

for MODEL_NAME in "${MODELS[@]}"; do
    echo "📦 Pulling model: $MODEL_NAME"
    if ollama pull "$MODEL_NAME"; then
        echo "✅ Successfully pulled: $MODEL_NAME"
    else
        echo "❌ Failed to pull: $MODEL_NAME"
        kill $OLLAMA_PID 2>/dev/null || true
        exit 1
    fi
done

echo "⏹  Stopping Ollama server..."
kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true

echo "✅ Model preloading complete."
