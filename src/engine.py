import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
from utils import JobInput

OLLAMA_BASE = "http://localhost:11434"

client = OpenAI(
    base_url=f"{OLLAMA_BASE}/v1/",
    # required but ignored
    api_key="ollama",
)


# ─── Native Ollama Engine ────────────────────────────────────────
# Calls Ollama's native HTTP API directly (e.g. /api/chat, /api/generate).
# This preserves full Ollama features: tools, options (num_ctx, etc.), format.

class OllamaNativeEngine:
    """
    Forwards the raw Ollama payload to the local Ollama server.
    Supports /api/chat and /api/generate with full tool_calls passthrough.
    """

    SUPPORTED_METHODS = {"/api/chat", "/api/generate", "/api/tags"}
    TIMEOUT_SECONDS = 300  # 5 min — matches RunPod /runsync max

    def __init__(self):
        print("OllamaNativeEngine initialized")

    async def generate(self, job_input: JobInput):
        method = job_input.method
        data = job_input.data or {}

        if method not in self.SUPPORTED_METHODS:
            yield {"error": f"Unsupported method: {method}. Supported: {', '.join(self.SUPPORTED_METHODS)}"}
            return

        # /api/tags is a GET — just proxy it
        if method == "/api/tags":
            try:
                resp = requests.get(f"{OLLAMA_BASE}{method}", timeout=30)
                resp.raise_for_status()
                yield resp.json()
            except Exception as e:
                yield {"error": f"Ollama {method} failed: {str(e)}"}
            return

        # ── /api/chat or /api/generate ──
        # Use model from payload, fallback to env var
        if "model" not in data or not data["model"]:
            data["model"] = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b")

        # Force stream=false for RunPod /runsync (synchronous)
        data["stream"] = False

        url = f"{OLLAMA_BASE}{method}"
        print(f"  🔗 [native] POST {url} — model: {data.get('model')}, "
              f"messages: {len(data.get('messages', []))}, "
              f"tools: {len(data.get('tools', []))}")

        try:
            resp = requests.post(
                url,
                json=data,
                timeout=self.TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            print(f"  🔗 [native] Response: {len(result.get('message', {}).get('content', ''))} chars, "
                  f"tool_calls: {len(result.get('message', {}).get('tool_calls', []))}")
            yield result
        except requests.exceptions.Timeout:
            yield {"error": f"Ollama {method} timed out after {self.TIMEOUT_SECONDS}s"}
        except requests.exceptions.ConnectionError:
            yield {"error": "Cannot connect to Ollama server at localhost:11434. Is it running?"}
        except Exception as e:
            yield {"error": f"Ollama {method} failed: {str(e)}"}


# ─── Legacy OpenAI-compatible Engine ─────────────────────────────

class OllamaEngine:
    def __init__(self):
        load_dotenv()
        print("OllamaEngine initialized")

    async def generate(self, job_input):
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

        openAIEngine = OllamaOpenAiEngine()
        async for batch in openAIEngine.generate(openAiJob):
            yield batch


class OllamaOpenAiEngine(OllamaEngine):
    def __init__(self):
        load_dotenv()
        print("OllamaOpenAiEngine initialized")

    async def generate(self, job_input):
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