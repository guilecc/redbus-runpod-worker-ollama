import logging
import os
import sys

import runpod
from utils import JobInput
from engine import get_native_engine, get_openai_engine, get_legacy_engine

# ─── D8: Structured logging ─────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("worker.handler")

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
    job_id = job.get("id", "unknown")
    logger.info("Job received: %s", job_id)

    job_input = JobInput(job["input"])

    if job_input.is_native_ollama:
        # ── redbusagent path: native Ollama /api/chat with tools support ──
        logger.info("[%s] → OllamaNativeEngine (method=%s)", job_id, job_input.method)
        engine = get_native_engine()
    elif job_input.openai_route:
        # ── Legacy OpenAI-compatible path ──
        logger.info("[%s] → OllamaOpenAiEngine (route=%s)", job_id, job_input.openai_route)
        engine = get_openai_engine()
    else:
        # ── Legacy raw messages/prompt path ──
        logger.info("[%s] → OllamaEngine (legacy)", job_id)
        engine = get_legacy_engine()

    async for batch in engine.generate(job_input, job_id=job_id):
        yield batch


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: max_concurrency,
        "return_aggregate_stream": True,
    }
)