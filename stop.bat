@echo off
REM defectgen - Parar stack (conserva datos en ./projects, ./crop-state, ./hf-cache*). Uso: stop.bat
REM Para eliminar contenedores huerfanos: docker compose down
setlocal
cd /d "%~dp0"
docker compose stop
docker compose ps
pause
