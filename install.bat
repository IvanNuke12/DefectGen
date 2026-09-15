@echo off
REM ============================================================
REM defectgen - Instalador Windows
REM Comprueba Docker, prepara .env y carpetas, y arranca el stack:
REM   http://localhost/ -^> portal | /hmi/ crop-hmi | /train/ defectfill
REM Uso: doble clic o "install.bat" desde la raiz de defectgen.
REM ============================================================
setlocal
cd /d "%~dp0"

echo [1/4] Comprobando Docker...
docker --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker no esta instalado o no esta en el PATH.
  echo Instala Docker Desktop: https://www.docker.com/products/docker-desktop/
  pause
  exit /b 1
)
docker compose version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] No se encontro el plugin "docker compose" (Compose v2).
  echo Actualiza Docker Desktop a una version reciente.
  pause
  exit /b 1
)
echo OK.

echo [2/4] Comprobando GPU (opcional, solo defectfill/entrenamiento)...
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo [AVISO] No se pudo usar la GPU desde Docker.
  echo El HMI de preprocesado funcionara en CPU, pero el entrenamiento
  echo requiere GPU NVIDIA + drivers + NVIDIA Container Toolkit (en WSL2
  echo viene integrado con Docker Desktop si activas soporte WSL2-GPU).
) else (
  echo OK: GPU accesible desde Docker.
)

echo [3/4] Preparando entorno (.env y carpetas de datos)...
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Creado ".env" desde ".env.example" (rellena HF_TOKEN si quieres).
) else (
  echo ".env" ya existe, no se toca.
)
if not exist "projects" mkdir "projects"
if not exist "crop-state" mkdir "crop-state"
if not exist "hf-cache" mkdir "hf-cache"
if not exist "hf-cache-torch" mkdir "hf-cache-torch"
echo OK.

echo [4/4] Construyendo y arrancando (primera vez: ~10-15 min por PyTorch)...
docker compose up -d --build
if errorlevel 1 (
  echo [ERROR] Fallo "docker compose up". Revisa los logs con: docker compose logs
  pause
  exit /b 1
)

echo.
docker compose ps
echo.
echo Instalacion completa. Abre http://localhost/ en tu navegador.
echo Si el portal tarda unos segundos, es el healthcheck inicial (espera y recarga).
pause
