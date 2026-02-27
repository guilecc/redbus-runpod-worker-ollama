# ─── Stage 1: Pull model weights ─────────────────────────────────
# Must match the Ollama version in the main Dockerfile
ARG OLLAMA_VERSION=0.17.4
FROM ollama/ollama:${OLLAMA_VERSION} AS builder

ARG MODEL_NAMES
ENV MODEL_NAMES=$MODEL_NAMES

RUN apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    bash curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ADD preload_model.sh /preload_model.sh
RUN chmod +x /preload_model.sh && /preload_model.sh

# ─── Stage 2: Copy baked weights into clean image ────────────────
FROM ollama/ollama:${OLLAMA_VERSION}
COPY --from=builder /runpod-volume /runpod-volume
