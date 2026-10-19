#!/usr/bin/env bash
# ==============================================================================
# Docker & Docker Compose Installer Script for Ubuntu / Debian
# ==============================================================================
set -e

echo "🐳 Checking Docker installation..."

if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker is already installed: $(docker --version)"
    if command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1; then
        echo "✅ Docker Compose is already installed."
    fi
    exit 0
fi

echo "📦 Docker not found. Installing official Docker Engine & Docker Compose..."
echo "This requires sudo privileges."

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Install Docker via official repository or docker.io
sudo apt-get install -y docker.io docker-compose-v2

# Start and enable docker service
sudo systemctl enable --now docker

# Add current user to docker group to run without sudo
sudo usermod -aG docker "$USER"

echo ""
echo "✅ Docker installation complete!"
echo "Version: $(docker --version)"
echo "Note: To apply docker group membership without logging out, run: newgrp docker"
