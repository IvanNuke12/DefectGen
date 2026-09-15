#!/usr/bin/env bash
# ============================================================
# defectgen - Instalador Linux
# Comprueba Docker, prepara .env y carpetas, y arranca el stack:
#   http://localhost/ -> portal | /hmi/ crop-hmi | /train/ defectfill
# Uso: ./install.sh  (desde la raiz de defectgen)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] Comprobando Docker..."
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] Docker no esta instalado."
  echo "Instala Docker Engine: https://docs.docker.com/engine/install/"
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "[ERROR] No se encontro el plugin 'docker compose' (Compose v2)."
  exit 1
fi
echo "OK."

echo "[2/4] Comprobando GPU (opcional, solo defectfill/entrenamiento)..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "OK: GPU accesible desde Docker."
else
  echo "[AVISO] No se pudo usar la GPU desde Docker."
  echo "El HMI funcionara en CPU, pero el entrenamiento requiere"
  echo "GPU NVIDIA + drivers + NVIDIA Container Toolkit:"
  echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

echo "[3/4] Preparando entorno (.env y carpetas de datos)..."
if [ ! -f ".env" ]; then
  cp ".env.example" ".env"
  echo 'Creado ".env" desde ".env.example" (rellena HF_TOKEN si quieres).'
else
  echo '".env" ya existe, no se toca.'
fi
mkdir -p projects crop-state hf-cache hf-cache-torch
echo "OK."

echo "[4/4] Construyendo y arrancando (primera vez: ~10-15 min por PyTorch)..."
docker compose up -d --build

echo ""
docker compose ps
echo ""
echo "Instalacion completa. Abre http://localhost/ en tu navegador."
echo "Si el portal tarda unos segundos, es el healthcheck inicial (espera y recarga)."
