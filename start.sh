#!/usr/bin/env bash
# defectgen - Arrancar stack (sin reconstruir). Uso: ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
docker compose up -d
docker compose ps
echo "Abre http://localhost/"
