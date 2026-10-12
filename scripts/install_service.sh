#!/bin/bash
# Install Robot Core as a systemd system service for automated boot on Linux Mint.
# Usage: sudo ./scripts/install_service.sh
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SRC="$DIR/scripts/robot.service"
SERVICE_DEST="/etc/systemd/system/robot.service"

if [ "$EUID" -ne 0 ]; then
    echo "⚠️ This script requires root privileges to install the systemd service."
    echo "Please run: sudo $0"
    exit 1
fi

echo "🤖 Installing Robot Core systemd service..."

if [ ! -f "$SERVICE_SRC" ]; then
    echo "❌ Service file not found at $SERVICE_SRC"
    exit 1
fi

cp "$SERVICE_SRC" "$SERVICE_DEST"
chmod 644 "$SERVICE_DEST"

echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

echo "⚡ Enabling robot.service for automatic startup at system boot..."
systemctl enable robot.service

echo "✅ Robot Core service successfully installed and enabled!"
echo ""
echo "Management commands:"
echo "  • Start service:   systemctl start robot.service"
echo "  • Check status:    systemctl status robot.service"
echo "  • View live logs:  journalctl -u robot.service -f"
echo "  • Stop service:    systemctl stop robot.service"
echo "  • Restart service: systemctl restart robot.service"
