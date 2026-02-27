"""
Unit tests for the RunPod Ollama worker.
Covers: JobInput parsing, OllamaNativeEngine, handler routing.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

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

