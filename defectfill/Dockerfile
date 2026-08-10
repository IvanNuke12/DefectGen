# ============================================================
# DefectFill Web App — Dockerfile (GPU / CUDA)
# ============================================================
# Imagen base con CUDA 12.4 + cuDNN 8 para PyTorch con soporte GPU.
# La primera vez que se lanza un entrenamiento/inferencia, el modelo
# base Stable Diffusion 2 Inpainting se descarga desde Hugging Face
# y se cachea en el volumen ~/.cache/huggingface.
#
# Build:   docker build -t defectfill-webapp .
# Run:     docker compose up -d
# ============================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Metadatos
LABEL maintainer="DefectFill"
LABEL description="Consola web de entrenamiento e inferencia para DefectFill (CVPR 2024)"

# Evita interactividad en apt
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------
# 1. Dependencias del sistema
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Asegura python3 apunta a 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ------------------------------------------------------------------
# 2. Entorno virtual e instalación de dependencias
# ------------------------------------------------------------------
WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt

# Instalar PyTorch con soporte CUDA 12.4 primero (evita build desde fuente)
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# 3. Copiar código de la aplicación
# ------------------------------------------------------------------
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY sample_dataset/ /app/sample_dataset/

# ------------------------------------------------------------------
# 4. Directorios de datos persistentes (volúmenes en compose)
# ------------------------------------------------------------------
RUN mkdir -p /app/data /app/output /app/generated

# Cache de Hugging Face persistente
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN mkdir -p ${HF_HOME}

# ------------------------------------------------------------------
# 5. Configuración de uvicorn
# ------------------------------------------------------------------
ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

# ------------------------------------------------------------------
# 6. Healthcheck
# ------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ------------------------------------------------------------------
# 7. Entrypoint
# ------------------------------------------------------------------
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app/backend
ENTRYPOINT ["/app/entrypoint.sh"]
