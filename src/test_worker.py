"""
Unit tests for the RunPod Ollama worker.
Covers: JobInput parsing, OllamaNativeEngine, handler routing, health check.
"""

import asyncio
import json
import logging
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Suppress noisy logs during tests
logging.disable(logging.CRITICAL)

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils import JobInput


# ─── JobInput Parsing ───────────────────────────────────────────


class TestJobInputNativeOllama(unittest.TestCase):
    """Test JobInput with the redbusagent native Ollama format."""

    def test_native_ollama_basic(self):
        job = {
            "method": "/api/chat",
            "data": {
                "model": "gemma3:27b",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        }
        ji = JobInput(job)
        self.assertTrue(ji.is_native_ollama)
        self.assertEqual(ji.method, "/api/chat")
        self.assertEqual(ji.data["model"], "gemma3:27b")
        self.assertEqual(len(ji.data["messages"]), 1)

    def test_native_ollama_with_tools(self):
        job = {
            "method": "/api/chat",
            "data": {
                "model": "gemma3:27b",
                "messages": [{"role": "user", "content": "Search for X"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
                "options": {"num_ctx": 8192},
            },
        }
        ji = JobInput(job)
        self.assertTrue(ji.is_native_ollama)
        self.assertEqual(len(ji.data["tools"]), 1)
        self.assertEqual(ji.data["options"]["num_ctx"], 8192)

    def test_native_ollama_generate(self):
        job = {"method": "/api/generate", "data": {"model": "gemma3:27b", "prompt": "Hi"}}
        ji = JobInput(job)
        self.assertTrue(ji.is_native_ollama)
        self.assertEqual(ji.method, "/api/generate")

    def test_method_without_data_is_not_native(self):
        ji = JobInput({"method": "/api/chat"})
        self.assertFalse(ji.is_native_ollama)

    def test_data_without_method_is_not_native(self):
        ji = JobInput({"data": {"model": "test"}})
        self.assertFalse(ji.is_native_ollama)


class TestJobInputLegacy(unittest.TestCase):
    """Test JobInput with legacy OpenAI / raw formats."""

    def test_openai_route(self):
        job = {
            "openai_route": "/v1/chat/completions",
            "openai_input": {"model": "llama3.2:1b", "messages": []},
        }
        ji = JobInput(job)
        self.assertFalse(ji.is_native_ollama)
        self.assertEqual(ji.openai_route, "/v1/chat/completions")

    def test_raw_messages(self):
        job = {"messages": [{"role": "user", "content": "Hi"}]}
        ji = JobInput(job)
        self.assertFalse(ji.is_native_ollama)
        self.assertEqual(len(ji.llm_input), 1)

    def test_raw_prompt(self):
        job = {"prompt": "Hello"}
        ji = JobInput(job)
        self.assertFalse(ji.is_native_ollama)
        self.assertEqual(ji.llm_input, "Hello")


# ─── OllamaNativeEngine ────────────────────────────────────────


class TestOllamaNativeEngine(unittest.TestCase):
    """Test OllamaNativeEngine with mocked HTTP calls."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(self._collect(coro))

    async def _collect(self, gen):
        results = []
        async for item in gen:
            results.append(item)
        return results

    @patch("engine.requests")
    def test_chat_success(self, mock_requests):
        from engine import OllamaNativeEngine

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "gemma3:27b",
            "message": {"role": "assistant", "content": "Hello!", "tool_calls": []},
            "done": True,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "/api/chat", "data": {"model": "gemma3:27b", "messages": [{"role": "user", "content": "Hi"}]}})
        results = self._run(engine.generate(ji))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["message"]["content"], "Hello!")
        # Verify stream was forced false
        call_kwargs = mock_requests.post.call_args
        self.assertFalse(call_kwargs.kwargs["json"]["stream"])

    @patch("engine.requests")
    def test_chat_with_tool_calls(self, mock_requests):
        from engine import OllamaNativeEngine

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "gemma3:27b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "weather"}}}],
            },
            "done": True,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions


    @patch("engine.requests")
    @patch.dict(os.environ, {"OLLAMA_MODEL_NAME": "fallback-model:7b"})
    def test_model_fallback_to_env(self, mock_requests):
        from engine import OllamaNativeEngine

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"model": "fallback-model:7b", "message": {"role": "assistant", "content": "ok"}, "done": True}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions

        engine = OllamaNativeEngine()
        # No model in data — should fallback to OLLAMA_MODEL_NAME
        ji = JobInput({"method": "/api/chat", "data": {"messages": [{"role": "user", "content": "Hi"}]}})
        results = self._run(engine.generate(ji))

        call_kwargs = mock_requests.post.call_args
        self.assertEqual(call_kwargs.kwargs["json"]["model"], "fallback-model:7b")

    def test_unsupported_method(self):
        from engine import OllamaNativeEngine

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "/api/unsupported", "data": {}})
        results = self._run(engine.generate(ji))

        self.assertIn("error", results[0])
        self.assertIn("Unsupported method", results[0]["error"])
        self.assertEqual(results[0]["code"], "UNSUPPORTED_METHOD")

    @patch("engine.requests")
    def test_connection_error(self, mock_requests):
        import requests as real_requests
        from engine import OllamaNativeEngine

        mock_requests.post.side_effect = real_requests.exceptions.ConnectionError("refused")
        mock_requests.exceptions = real_requests.exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "/api/chat", "data": {"model": "test", "messages": []}})
        results = self._run(engine.generate(ji))

        self.assertIn("error", results[0])
        self.assertIn("Cannot connect", results[0]["error"])
        self.assertEqual(results[0]["code"], "CONNECTION_ERROR")

    @patch("engine.requests")
    def test_timeout_error(self, mock_requests):
        import requests as real_requests
        from engine import OllamaNativeEngine

        mock_requests.post.side_effect = real_requests.exceptions.Timeout("timed out")
        mock_requests.exceptions = real_requests.exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "/api/chat", "data": {"model": "test", "messages": []}})
        results = self._run(engine.generate(ji))

        self.assertIn("error", results[0])
        self.assertIn("timed out", results[0]["error"])
        self.assertEqual(results[0]["code"], "TIMEOUT")
        self.assertIn("details", results[0])
        self.assertEqual(results[0]["details"]["method"], "/api/chat")

    @patch("engine.requests")
    def test_tags_endpoint(self, mock_requests):
        from engine import OllamaNativeEngine

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "gemma3:27b"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "/api/tags", "data": {}})
        results = self._run(engine.generate(ji))

        self.assertEqual(results[0]["models"][0]["name"], "gemma3:27b")
        mock_requests.get.assert_called_once()

    @patch("engine.requests")
    def test_health_check_ok(self, mock_requests):
        """D1: Health check returns ok when Ollama is reachable."""
        from engine import OllamaNativeEngine

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "health", "data": {}})
        results = self._run(engine.generate(ji))

        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["ollama"], "reachable")

    @patch("engine.requests")
    def test_health_check_degraded(self, mock_requests):
        """D1: Health check returns degraded when Ollama unreachable."""
        import requests as real_requests
        from engine import OllamaNativeEngine

        mock_requests.get.side_effect = real_requests.exceptions.ConnectionError("refused")
        mock_requests.exceptions = real_requests.exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "health", "data": {}})
        results = self._run(engine.generate(ji))

        self.assertEqual(results[0]["status"], "degraded")
        self.assertEqual(results[0]["ollama"], "unreachable")

    @patch("engine.requests")
    def test_job_id_passed_through(self, mock_requests):
        """C6: job_id is accepted by generate()."""
        from engine import OllamaNativeEngine

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"model": "test", "message": {"role": "assistant", "content": "hi"}, "done": True}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions

        engine = OllamaNativeEngine()
        ji = JobInput({"method": "/api/chat", "data": {"model": "test", "messages": []}})
        # Should not raise — job_id parameter is accepted
        results = self._run(engine.generate(ji, job_id="test-job-123"))
        self.assertEqual(len(results), 1)


# ─── P2: Structured Errors & Configurable Timeout ─────────────


class TestStructuredErrors(unittest.TestCase):
    """C4: Structured error format with code and optional details."""

    def test_error_helper_basic(self):
        from engine import OllamaNativeEngine
        err = OllamaNativeEngine._error("TEST_CODE", "Something went wrong")
        self.assertEqual(err["error"], "Something went wrong")
        self.assertEqual(err["code"], "TEST_CODE")
        self.assertNotIn("details", err)

    def test_error_helper_with_details(self):
        from engine import OllamaNativeEngine
        err = OllamaNativeEngine._error("TIMEOUT", "Timed out", {"method": "/api/chat", "timeout_seconds": 300})
        self.assertEqual(err["code"], "TIMEOUT")
        self.assertEqual(err["details"]["method"], "/api/chat")
        self.assertEqual(err["details"]["timeout_seconds"], 300)

    @patch.dict(os.environ, {"OLLAMA_TIMEOUT": "60"})
    def test_configurable_timeout(self):
        """D5: OLLAMA_TIMEOUT env var is respected."""
        # Need to reimport to pick up env var
        import importlib
        import engine
        importlib.reload(engine)
        e = engine.OllamaNativeEngine()
        self.assertEqual(e.TIMEOUT_SECONDS, 60)
        # Reset
        importlib.reload(engine)


# ─── Singleton Engines (D4) ────────────────────────────────────


class TestSingletonEngines(unittest.TestCase):
    """D4: Engine instances are reused across calls."""

    def test_native_engine_singleton(self):
        from engine import get_native_engine
        e1 = get_native_engine()
        e2 = get_native_engine()
        self.assertIs(e1, e2)

    def test_openai_engine_singleton(self):
        from engine import get_openai_engine
        e1 = get_openai_engine()
        e2 = get_openai_engine()
        self.assertIs(e1, e2)

    def test_legacy_engine_singleton(self):
        from engine import get_legacy_engine
        e1 = get_legacy_engine()
        e2 = get_legacy_engine()
        self.assertIs(e1, e2)


# ─── Handler Routing ────────────────────────────────────────────


class TestHandlerRouting(unittest.TestCase):
    """Verify handler routes to the correct engine based on input format."""

    @patch("engine.requests")
    def test_routes_native_ollama(self, mock_requests):
        """Native method+data input → OllamaNativeEngine"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"model": "test", "message": {"role": "assistant", "content": "hi"}, "done": True}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp
        mock_requests.exceptions = __import__("requests").exceptions

        job = {
            "id": "test-job-1",
            "input": {
                "method": "/api/chat",
                "data": {"model": "gemma3:27b", "messages": [{"role": "user", "content": "Hello"}]},
            },
        }

        # Import handler function — but we can't use it directly because it calls
        # runpod.serverless.start at import time. Instead we test via the engine directly.
        ji = JobInput(job["input"])
        self.assertTrue(ji.is_native_ollama)

    def test_routes_openai(self):
        """openai_route input → detected as OpenAI path"""
        job_input = JobInput({
            "openai_route": "/v1/chat/completions",
            "openai_input": {"model": "llama3.2:1b", "messages": []},
        })
        self.assertFalse(job_input.is_native_ollama)
        self.assertIsNotNone(job_input.openai_route)

    def test_routes_legacy(self):
        """Raw messages input → detected as legacy path"""
        job_input = JobInput({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertFalse(job_input.is_native_ollama)
        self.assertIsNone(job_input.openai_route)
        self.assertIsNotNone(job_input.llm_input)


if __name__ == "__main__":
    unittest.main()

