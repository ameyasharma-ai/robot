# ==============================================================================
# Robot Core — Production Dockerfile
# Multi-stage optimized build for Python 3.12 with GPU & Audio support
# ==============================================================================

FROM python:3.12-slim AS base

LABEL maintainer="Ameya Sharma"
LABEL description="Local-first humanoid robot brain: full-duplex voice, persistent memory, agentic tools"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONWARNINGS="ignore::DeprecationWarning" \
    PORT=8888 \
    HOST=0.0.0.0

# Install required system packages: audio (ALSA/PortAudio), video/OpenCV, Vulkan, curl, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libasound2 \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    alsa-utils \
    libvulkan1 \
    vulkan-tools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create necessary runtime directories
RUN mkdir -p /app/data /app/models /app/logs /app/config /app/src

# Copy dependency specifications first for layer caching
COPY pyproject.toml README.md /app/

# Install python build dependencies and wheels
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

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
