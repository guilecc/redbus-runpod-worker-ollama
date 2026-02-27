class JobInput:
    """
    Parses RunPod job input into a structured object.

    Supports two formats:

    1. Native Ollama format (redbusagent):
       { "method": "/api/chat", "data": { model, messages, tools?, options?, stream? } }

    2. OpenAI-compatible format (legacy):
       { "openai_route": "/v1/chat/completions", "openai_input": { ... } }
       OR
       { "messages": [...] } / { "prompt": "..." }
    """
    def __init__(self, job: dict):
        # ── Native Ollama path (method + data envelope) ──
        self.method = job.get("method")        # e.g. "/api/chat", "/api/generate"
        self.data = job.get("data")            # raw Ollama payload dict

        # ── Legacy OpenAI-compatible path ──
        self.llm_input = job.get("messages", job.get("prompt"))
        self.stream = job.get("stream", False)
        self.openai_route = job.get("openai_route")
        self.openai_input = job.get("openai_input")

    @property
    def is_native_ollama(self) -> bool:
        """True when the input uses the native Ollama method/data envelope."""
        return self.method is not None and self.data is not None