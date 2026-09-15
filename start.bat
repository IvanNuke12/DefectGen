@echo off
REM defectgen - Arrancar stack (sin reconstruir). Uso: start.bat
setlocal
cd /d "%~dp0"
docker compose up -d
if errorlevel 1 ( echo [ERROR] Fallo al arrancar. & pause & exit /b 1 )
docker compose ps
echo Abre http://localhost/
pause
