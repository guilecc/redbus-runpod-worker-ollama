ARG OLLAMA_VERSION=0.17.4

# ─── Base image with Ollama pre-installed ─────────────────────────
FROM ollama/ollama:${OLLAMA_VERSION}

ENV PYTHONUNBUFFERED=1

WORKDIR /

# ─── Install Python 3.11 (minimal — no tk, gdbm, dev headers) ───
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    ca-certificates \
    && add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    bash \
    curl && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    # Cleanup to reduce image size
    apt-get remove --yes --quiet software-properties-common gpg-agent && \
    apt-get autoremove --yes --quiet && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/pip

WORKDIR /work

ADD ./src /work

# ─── Ollama runtime config ───────────────────────────────────────
# Models directory — RunPod mounts a volume here
ENV OLLAMA_MODELS="/runpod-volume"
# A2: Keep model in VRAM forever (worker is killed by RunPod anyway)
ENV OLLAMA_KEEP_ALIVE="-1"
# Force full GPU offload
ENV OLLAMA_NUM_GPU="999"

# ─── Install Python dependencies ─────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt && chmod +x /work/start.sh

ENTRYPOINT ["/bin/sh", "-c", "/work/start.sh"]