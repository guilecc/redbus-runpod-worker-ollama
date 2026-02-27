import json
import logging
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
from utils import JobInput

logger = logging.getLogger("worker.engine")

OLLAMA_BASE = "http://localhost:11434"

client = OpenAI(
    base_url=f"{OLLAMA_BASE}/v1/",
    # required but ignored
    api_key="ollama",
)

# ─── Singleton engines (D4) ──────────────────────────────────────
_native_engine = None
_openai_engine = None
_legacy_engine = None


def get_native_engine():
    global _native_engine
    if _native_engine is None:
        _native_engine = OllamaNativeEngine()
    return _native_engine


def get_openai_engine():
    global _openai_engine
    if _openai_engine is None:
        _openai_engine = OllamaOpenAiEngine()
    return _openai_engine


def get_legacy_engine():
    global _legacy_engine
    if _legacy_engine is None:
        _legacy_engine = OllamaEngine()
    return _legacy_engine


# ─── Native Ollama Engine ────────────────────────────────────────
# Calls Ollama's native HTTP API directly (e.g. /api/chat, /api/generate).
# This preserves full Ollama features: tools, options (num_ctx, etc.), format.

class OllamaNativeEngine:
    """
    Forwards the raw Ollama payload to the local Ollama server.
    Supports /api/chat, /api/generate, /api/tags, and health checks.
    """

    SUPPORTED_METHODS = {"/api/chat", "/api/generate", "/api/tags", "health"}
    # D5: Configurable timeout via env var (default 300s = 5 min)
    TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    def __init__(self):
        logger.info("OllamaNativeEngine initialized")

    async def generate(self, job_input: JobInput, job_id: str = "unknown"):
        method = job_input.method
        data = job_input.data or {}

        # ── D1: Health check — fast path ──
        if method == "health":
            model = os.getenv("OLLAMA_MODEL_NAME", "unknown")
            try:
                resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
                resp.raise_for_status()
                yield {"status": "ok", "model": model, "ollama": "reachable"}
            except Exception:
                yield {"status": "degraded", "model": model, "ollama": "unreachable"}
            return

        if method not in self.SUPPORTED_METHODS:
            yield self._error("UNSUPPORTED_METHOD", f"Unsupported method: {method}. Supported: {', '.join(self.SUPPORTED_METHODS)}")
            return

        # /api/tags is a GET — just proxy it
        if method == "/api/tags":
            try:
                resp = requests.get(f"{OLLAMA_BASE}{method}", timeout=30)
                resp.raise_for_status()
                yield resp.json()
            except Exception as e:
                yield self._error("OLLAMA_TAGS_FAILED", f"Ollama {method} failed: {str(e)}")
            return

        # ── /api/chat or /api/generate ──
        # Use model from payload, fallback to env var
        if "model" not in data or not data["model"]:
            data["model"] = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b")

        # Force stream=false for RunPod /runsync (synchronous)
        data["stream"] = False

        url = f"{OLLAMA_BASE}{method}"
        logger.info("[%s] POST %s — model: %s, messages: %d, tools: %d",
                     job_id, url, data.get("model"),
                     len(data.get("messages", [])),
                     len(data.get("tools", [])))

        try:
            resp = requests.post(
                url,
                json=data,
                timeout=self.TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info("[%s] Response: %d chars, tool_calls: %d",
                         job_id,
                         len(result.get("message", {}).get("content", "")),
                         len(result.get("message", {}).get("tool_calls", [])))
            yield result
        except requests.exceptions.Timeout:
            logger.error("[%s] Timeout after %ds on %s", job_id, self.TIMEOUT_SECONDS, method)
            yield self._error("TIMEOUT", f"Ollama {method} timed out after {self.TIMEOUT_SECONDS}s",
                              {"method": method, "timeout_seconds": self.TIMEOUT_SECONDS})
        except requests.exceptions.ConnectionError:
            logger.error("[%s] Cannot connect to Ollama at localhost:11434", job_id)
            yield self._error("CONNECTION_ERROR", "Cannot connect to Ollama server at localhost:11434. Is it running?")
        except Exception as e:
            logger.error("[%s] %s failed: %s", job_id, method, str(e))
            yield self._error("OLLAMA_ERROR", f"Ollama {method} failed: {str(e)}",
                              {"method": method, "exception": type(e).__name__})

    @staticmethod
    def _error(code: str, message: str, details: dict = None) -> dict:
        """C4: Structured error format for consistent error handling."""
        err = {"error": message, "code": code}
        if details:
            err["details"] = details
        return err


# ─── Legacy OpenAI-compatible Engine ─────────────────────────────

class OllamaEngine:
    def __init__(self):
        load_dotenv()
        logger.info("OllamaEngine initialized")

    async def generate(self, job_input, job_id: str = "unknown"):
        model = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b")

        if isinstance(job_input.llm_input, str):
            openAiJob = JobInput({
                "openai_route": "/v1/completions",
                "openai_input": {
                    "model": model,
                    "prompt": job_input.llm_input,
                    "stream": job_input.stream,
                },
            })
        else:
            openAiJob = JobInput({
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": model,
                    "messages": job_input.llm_input,
                    "stream": job_input.stream,
                },
            })

        openAIEngine = get_openai_engine()
        async for batch in openAIEngine.generate(openAiJob, job_id=job_id):
            yield batch


class OllamaOpenAiEngine(OllamaEngine):
    def __init__(self):
        load_dotenv()
        logger.info("OllamaOpenAiEngine initialized")

    async def generate(self, job_input, job_id: str = "unknown"):
        openai_input = job_input.openai_input

        if job_input.openai_route == "/v1/models":
            async for response in self._handle_model_request():
                yield response
        elif job_input.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(
                openai_input, chat=job_input.openai_route == "/v1/chat/completions"
            ):
                yield response
        else:
            yield {"error": "Invalid route"}

    async def _handle_model_request(self):
        try:
            response = client.models.list()
            yield {"object": "list", "data": [model.to_dict() for model in response.data]}
        except Exception as e:
            yield {"error": str(e)}

    async def _handle_chat_or_completion_request(self, openai_input, chat=False):
        try:
            if chat:
                response = client.chat.completions.create(**openai_input)
            else:
                response = client.completions.create(**openai_input)

            if not openai_input.get("stream", False):
                yield response.to_dict()
                return

            for chunk in response:
                yield "data: " + json.dumps(chunk.to_dict(), separators=(",", ":")) + "\n\n"

            yield "data: [DONE]"
        except Exception as e:
            yield {"error": str(e)}