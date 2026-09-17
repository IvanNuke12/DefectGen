# defectgen — Inspección industrial por visión con DefectFill

Sistema de inspección de defectos por visión por computador, completo y
orquestado con Docker: un **HMI de preprocesado** (recorte, etiquetado y
fusión de imágenes), una **consola de entrenamiento e inferencia** basada en
[DefectFill](https://github.com/axelsig1/DefectFill) (CVPR 2024) y un
**portal** con entrada única a todo.

```
┌──────────────────────────────────────────────────────────────┐
│                        Nginx (portal)                        │
│                        entrada única :80                      │
│                                                              │
│   http://localhost/     → portal con pestañas (iframes)      │
│   http://localhost/hmi/ → crop-hmi (preprocesado)            │
│   http://localhost/merge/ → crop-hmi (postprocesado)         │
│   http://localhost/train/ → defectfill (entrenamiento)       │
└───────┬────────────────────────┬──────────────────┬──────────┘
        │ /hmi/                  │ /merge/          │ /train/
        ▼                        ▼                  ▼
     crop-hmi               crop-hmi            defectfill
   (Flask, CPU)             (Flask, CPU)         (FastAPI, GPU)
        │                        │                  │
        └──────────┬─────────────┘                  │
                   ▼                                ▼
            ./projects (datasets MVTec-AD) ◄────────┘
            ./crop-state (estado del HMI)           │
            ./hf-cache / ./hf-cache-torch           ▼
                                             (LoRA sobre SD2 Inpainting)
```

Las apps **no publican puertos al host**: solo son alcanzables a través del
proxy por subruta (mismo origen). Esto evita problemas de CORS y simplifica el
despliegue.

---

## Requisitos del sistema

- **GPU NVIDIA con CUDA** (recomendado ≥ 12 GB VRAM) y drivers instalados.
- **NVIDIA Container Toolkit** (para que Docker use la GPU).
- **Docker** con plugin Compose v2.
- **Internet** en la primera ejecución: se descarga
  `sd2-community/stable-diffusion-2-inpainting` (~5 GB) desde Hugging Face.
- Espacio en disco: ~6 GB adicionales para caches y checkpoints.

### Comprobación rápida

```bash
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Si el segundo comando falla, instala el
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
y reinicia Docker.

---

## Puesta en marcha

### Opción 1: Instalación rápida con scripts (recomendada)

El repositorio incluye scripts automatizados que comprueban los requisitos del sistema (Docker, Compose v2 y soporte de GPU NVIDIA), preparan el archivo `.env` a partir de la plantilla, crean las carpetas de datos persistentes y construyen/levantan el stack completo:

- **Windows**: Haz doble clic en `install.bat` (o ejecuta `.\install.bat` desde PowerShell / CMD).
- **Linux / WSL2**:
  ```bash
  chmod +x install.sh start.sh stop.sh
  ./install.sh
  ```

Una vez finalizada la construcción, abre **http://localhost/** en tu navegador.

#### Control del servicio (uso habitual)

- **Arrancar stack**: `start.bat` (Windows) o `./start.sh` (Linux).
- **Detener stack**: `stop.bat` (Windows) o `./stop.sh` (Linux) — preserva todos los datos en `./projects`, `./crop-state` y `./hf-cache*`.

---

### Opción 2: Instalación manual paso a paso

```bash
# 1. Configurar variables de entorno (token opcional)
cp .env.example .env

# 2. Crear la carpeta de datos y el estado del HMI
mkdir -p projects crop-state hf-cache hf-cache-torch

# 3. Construir (primera vez: ~10–15 min por PyTorch)
docker compose up -d --build

# 4. Ver estado
docker compose ps
```

Abre **http://localhost/** en tu navegador.

> Si `crop-hmi` y `defectfill` no arrancan al mismo tiempo, Nginx espera a su
> healthcheck (start_period) antes de dejar pasar tráfico. Espera unos segundos
> tras el primer arranque.

### Configuración de `HF_TOKEN` (opcional)

El modelo base es público y las descargas funcionan sin token. Para mayor
velocidad/fiabilidad crea un token de lectura en
https://huggingface.co/settings/tokens y añádelo a tu `.env`:

```env
HF_TOKEN=hf_xxxxx
```

---

## Estructura de datos

Todo vive en la carpeta del stack (bind mounts). No se versiona nada de esto
(ver `.gitignore`):

```
./projects/        Proyectos compartidos HMI + entrenamiento (estructura MVTec-AD)
./crop-state/      Estado del HMI (config, recientes, miniaturas)
./hf-cache/        Caché del modelo Hugging Face (~5 GB)
./hf-cache-torch/  Pesos de torchvision/LPIPS (~550 MB)
```

### Estructura de un proyecto

```
mi_proyecto/
├── images/          # imágenes de entrada (HMI)
├── labels/          # labels YOLO (.txt)
├── annotations.json # anotaciones COCO (opcional)
├── crops/           # recortes del HMI (imágenes + máscaras + crops_log.csv)
├── test/good        # imágenes buenas de validación (MVTec-AD)
├── train/defective  # defectos de entrenamiento (+ _masks)
├── output/<run>/    # checkpoints del entrenamiento
└── generated/<run>/ # imágenes sintéticas de la inferencia
```

El proyecto activo se comparte entre el HMI y la consola de entrenamiento vía
`projects/.active_project.json`.

---

## Flujo de trabajo

1. **Recorte y etiquetado** (`/hmi/`): crea/abre un proyecto, importa imágenes
   y labels, recorta parches y revisa máscaras.
2. **Entrenamiento** (`/train/`): selecciona proyecto y tipo de defecto, ajusta
   hiperparámetros (pasos, LoRA rank/alpha, learning rates, pesos de pérdida)
   y lanza el entrenamiento con pérdidas en vivo.
3. **Evaluación** (`/train/` → pestaña Evaluación): valida los checkpoints del
   run con un lote fijo determinista, midiendo **KID** e **IC-LPIPS**, y
   selecciona el mejor checkpoint.
4. **Inferencia** (`/train/` → pestaña Inferencia): genera defectos sintéticos
   sobre imágenes buenas, con selección del mejor candidato por **LPIPS
   espacial**.
5. **Fusión** (`/merge/`): incrusta crops editados externamente en la imagen
   original (pixel-perfect).

---

## Servicios y puertos

| Servicio | Contenedor | Puerto interno | Uso |
|----------|-----------|----------------|-----|
| `crop-hmi` | `defectgen-crop-hmi` | 5000 | HMI de preprocesado (CPU) |
| `defectfill` | `defectgen-defectfill` | 8000 | Entrenamiento/inferencia (GPU) |
| `portal` | `defectgen-portal` | 80 | Entrada única (Nginx) |

Para exponer una app directamente (solo desarrollo), descomenta la sección
`ports` correspondiente en `docker-compose.yml` (`CROP_PORT` / `HOST_PORT`).

### Comandos útiles

```bash
# Control con scripts
./start.sh                             # o start.bat en Windows (arranca sin reconstruir)
./stop.sh                              # o stop.bat en Windows (detiene contenedores)

# Docker Compose directo
docker compose ps                      # estado y healthchecks
docker compose logs -f crop-hmi        # logs del HMI
docker compose logs -f defectfill      # logs del entrenamiento
docker compose exec defectfill nvidia-smi
docker compose down                    # parar (conserva datos)
docker compose up -d --build defectfill  # reconstruir solo un servicio
```

---

## Componentes

| Carpeta | Descripción |
|---------|-------------|
| `crop-hmi/` | Flask + frontend: pantalla de proyectos, recorte, etiquetado y fusión. [README](crop-hmi/README.md) |
| `defectfill/` | FastAPI + frontend: dataset, entrenamiento, validación e inferencia DefectFill. [README](defectfill/README.md) |
| `portal/` | Frontend del portal (pestañas con iframes same-origin). |
| `nginx/` | `nginx.conf` con el enrutado de subrutas. |

## Créditos

- **DefectFill** — [axelsig1/DefectFill](https://github.com/axelsig1/DefectFill)
  (CVPR 2024). El código de `defectfill/backend/defectfill/` procede de ese
  repositorio con modificaciones mínimas (`write_status()` para el progreso en
  vivo).
- **Stable Diffusion 2 Inpainting** —
  [sd2-community/stable-diffusion-2-inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting).

Consulta la licencia en `LICENSE`.

## Aviso legal

Este repositorio contiene únicamente **código fuente y documentación**.
Los datasets de entrenamiento, imágenes de producto, checkpoints y caches no se
incluyen: son datos de usuario que se crean en tiempo de ejecución en las
carpetas locales (`./projects`, `./crop-state`, `./hf-cache*`), excluidas del
versionado mediante `.gitignore`.
