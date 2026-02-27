import runpod
import os
from utils import JobInput
from engine import OllamaEngine, OllamaOpenAiEngine, OllamaNativeEngine

DEFAULT_MAX_CONCURRENCY = 1  # Ollama serializes GPU inference; keep at 1
max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))


async def handler(job: any):
    """
    RunPod serverless handler.

    Routes to the correct engine based on input format:
    - Native Ollama (method + data): OllamaNativeEngine  ← redbusagent path
    - OpenAI-compatible (openai_route): OllamaOpenAiEngine
    - Legacy (messages/prompt): OllamaEngine → wraps to OpenAI
    """
    print(f"Job: {job.get('id', 'unknown')}")

    job_input = JobInput(job["input"])

    if job_input.is_native_ollama:
        # ── redbusagent path: native Ollama /api/chat with tools support ──
        print(f"  → Routing to OllamaNativeEngine (method={job_input.method})")
        engine = OllamaNativeEngine()
    elif job_input.openai_route:
        # ── Legacy OpenAI-compatible path ──
        print(f"  → Routing to OllamaOpenAiEngine (route={job_input.openai_route})")
        engine = OllamaOpenAiEngine()
    else:
        # ── Legacy raw messages/prompt path ──
        print("  → Routing to OllamaEngine (legacy)")
        engine = OllamaEngine()

    async for batch in engine.generate(job_input):
        yield batch


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: max_concurrency,
        "return_aggregate_stream": True,
    }
)