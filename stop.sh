#!/usr/bin/env bash
# defectgen - Parar stack (conserva datos en ./projects, ./crop-state, ./hf-cache*). Uso: ./stop.sh
# Para eliminar contenedores huerfanos: docker compose down
set -euo pipefail
cd "$(dirname "$0")"
docker compose stop
docker compose ps
