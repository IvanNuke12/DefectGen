#!/usr/bin/env bash
# ============================================================
# entrypoint.sh — Punto de entrada del contenedor DefectFill
# ============================================================
set -euo pipefail

echo "============================================================"
echo "  DefectFill Web App"
echo "  PID 1: $$"
echo "============================================================"

# --- Verificar GPU ---
echo "[init] Comprobando GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        | while IFS=, read -r gpu driver mem; do
            echo "[gpu]  ${gpu} | Driver ${driver} | VRAM ${mem}"
        done
else
    echo "[warn] nvidia-smi no encontrado — ¿se pasó --gpus all a docker run?"
fi

# --- Verificar Python y PyTorch ---
echo "[init] Python: $(python3 --version)"
if python3 -c "import torch; print(f'[init] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    :
else
    echo "[warn] PyTorch no pudo importarse correctamente"
fi

# --- Crear directorios de trabajo ---
mkdir -p /app/data /app/output /app/generated /app/.cache/huggingface

echo "[init] Arrancando uvicorn en ${HOST:-0.0.0.0}:${PORT:-8000}"
echo "============================================================"

exec uvicorn app:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info
