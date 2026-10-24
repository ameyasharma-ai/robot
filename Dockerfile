# ==============================================================================
# Robot Core — Production Dockerfile
# Genuine multi-stage build (builder + runtime) for Python 3.12 with Audio support.
# NOTE: CPU-only inference. The python:3.12-slim base has no CUDA toolkit, so
# torch/onnxruntime run on CPU here; the optional GPU block in docker-compose.yml
# is only useful with a CUDA-enabled base image.
# ==============================================================================

# ---------- Stage 1: builder — compile/install python wheels ----------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Build tools for wheels that need compiling (portaudio bindings etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    portaudio19-dev \
    libasound2-dev \
    libespeak-ng-dev \
    espeak-ng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Copy pyproject.toml first to cache heavy pip dependencies (torch, etc.)
COPY pyproject.toml README.md /app/

# Install CPU-only torch first to prevent PyPI pulling multi-GB CUDA wheels,
# then install project dependencies against a dummy package stub to leverage Docker caching.
RUN mkdir -p /app/src/robot && touch /app/src/robot/__init__.py && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "torch>=2.4.0" torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir . && \
    rm -rf /app/src

# Step 2: Copy real source tree and install package without re-downloading dependencies
COPY src/ /app/src/
COPY config/ /app/config/
RUN pip install --no-cache-dir --no-deps .

# ---------- Stage 2: runtime — minimal image, no build toolchain ----------
FROM python:3.12-slim AS runtime

LABEL maintainer="Ameya Sharma"
LABEL description="Local-first humanoid robot brain: full-duplex voice, persistent memory, agentic tools (CPU-only build)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONWARNINGS="ignore::DeprecationWarning" \
    PORT=8888 \
    HOST=0.0.0.0

# Runtime system packages: audio (ALSA/PortAudio), video/OpenCV, Vulkan, OCR, phonemes, curl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libasound2 \
    libportaudio2 \
    libgl1 \
    libglib2.0-0 \
    alsa-utils \
    libvulkan1 \
    vulkan-tools \
    espeak-ng \
    libespeak-ng1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create necessary runtime directories
RUN mkdir -p /app/data /app/models /app/logs /app/config /app/src

# Copy the pre-built install + site-packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and scripts
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/docker_entrypoint.sh /app/scripts/docker_entrypoint.sh

RUN chmod +x /app/scripts/docker_entrypoint.sh

# Expose Web UI / API (8888), LLM server (8080), Whisper STT (8081)
EXPOSE 8888 8080 8081

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8888/health || exit 1

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["python3", "-m", "robot"]
