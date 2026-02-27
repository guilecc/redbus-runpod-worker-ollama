# redbus-runpod-worker-ollama

RunPod Serverless worker for [Ollama](https://ollama.com), purpose-built for the [Redbus Agent](https://github.com/guilecc/redbusagent) ecosystem.

This worker acts as the **Worker Engine** in Redbus's Dual-Local Architecture — offloading heavy LLM inference to RunPod's serverless GPUs while the Live Engine handles real-time chat locally.

## Purpose

Redbus Agent uses a "Two Brains, One Machine" architecture. This worker bridges the gap between RunPod's serverless infrastructure and Ollama's native API, providing:

- **Native Ollama protocol support** — forwards `/api/chat` and `/api/generate` payloads directly, preserving tools, options (`num_ctx`, `temperature`, etc.), and per-request model selection.
- **Tool calling** — full pass-through of Ollama's native `tools` and `tool_calls` for agentic workflows.
- **Enterprise-grade reliability** — structured error responses, configurable timeouts, request tracing, and health checks.
- **GPU-optimized** — model preloading into VRAM, `OLLAMA_KEEP_ALIVE=-1`, full GPU offload, and batch size tuning.
- **OpenAI-compatible fallback** — legacy `/v1/chat/completions` and `/v1/completions` routes still work.

## How it works

The Redbus daemon (`runpod-ollama.ts`) sends Ollama payloads wrapped in RunPod's envelope:

```json
{
  "input": {
    "method": "/api/chat",
    "data": {
      "model": "gemma3:27b",
      "messages": [{"role": "user", "content": "Hello"}],
      "tools": [...],
      "stream": false
    }
  }
}
```

The worker routes to the correct engine based on input format:
- `method` + `data` → **OllamaNativeEngine** (Redbus path)
- `openai_route` → **OllamaOpenAiEngine** (legacy OpenAI-compatible)
- raw `messages`/`prompt` → **OllamaEngine** (legacy fallback)

## Deployment

[![Runpod](https://api.runpod.io/badge/guilecc/redbus-runpod-worker-ollama)](https://console.runpod.io/hub/guilecc/redbus-runpod-worker-ollama)

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL_NAME` | Model to download and preload into VRAM | `llama3.2:1b` |
| `OLLAMA_TIMEOUT` | Request timeout in seconds | `300` |
| `OLLAMA_KEEP_ALIVE` | How long to keep model in VRAM (`-1` = forever) | `-1` |
| `OLLAMA_NUM_GPU` | Number of GPU layers to offload | `999` |
| `OLLAMA_NUM_PARALLEL` | Max parallel requests Ollama handles | `1` |
| `OLLAMA_NUM_BATCH` | Batch size for parallel token processing | `512` |
| `MAX_CONCURRENCY` | RunPod job concurrency | `1` |
| `LOG_LEVEL` | Python logging level | `INFO` |

## Test requests

See the [test_inputs](./test_inputs) directory for example payloads.

## Preload model into the docker image

See the [embed_model](./embed_model/) directory for instructions.

## Licence

This project is a fork of [svenbrnn/runpod-worker-ollama](https://github.com/svenbrnn/runpod-worker-ollama), licensed under the Creative Commons Attribution 4.0 International License.

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **Reference**: Original repository at [https://github.com/svenbrnn/runpod-worker-ollama](https://github.com/svenbrnn/runpod-worker-ollama).

For more details, see the [license](https://creativecommons.org/licenses/by/4.0/).
