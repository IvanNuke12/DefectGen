# DefectFill — Consola web de entrenamiento e inferencia

Webapp (FastAPI + frontend estático) que envuelve la implementación no oficial de **DefectFill** (CVPR 2024) — [axelsig1/DefectFill](https://github.com/axelsig1/DefectFill) — para preparar datasets, entrenar el modelo (LoRA sobre Stable Diffusion 2 Inpainting) y generar defectos sintéticos desde el navegador.

`train.py`, `inference.py`, `model.py`, `data_loader.py` y `utils.py` son el código original del repositorio, casi sin cambios: solo se añadió un helper `write_status()` para que la interfaz web muestre el progreso en vivo.

---

## Requisitos importantes

- **GPU NVIDIA con CUDA** (idealmente >= 12 GB VRAM) y drivers instalados.
- **Conexión a internet** la primera vez: el backend descarga automáticamente
  `sd2-community/stable-diffusion-2-inpainting` desde Hugging Face (varios GB).
- **NVIDIA Container Toolkit** si usas Docker (ver sección Docker más abajo).

---

## Despliegue

### Dentro del stack unificado (recomendado)

Esta webapp forma parte del stack `defectgen`: HMI de preprocesado +
entrenamiento + portal con entrada única. En ese caso arranca **desde la raíz
del stack** y la accedes por subruta en `http://localhost/train/`:

```bash
# Desde la carpeta raíz del stack (defectgen)
cp .env.example .env              # token HF opcional
docker compose up -d --build
```

La webapp queda expuesta por el proxy Nginx bajo subruta `same-origin`: nada de
CORS, y los datos se montan **por proyecto** en `./projects` (compartidos con
el HMI de preprocesado). No expone puerto propio.

### Standalone (solo esta webapp)

```bash
# 1. Prerequisitos del host
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 2. Variables de entorno (opcional)
cp .env.example .env

# 3. Construir y arrancar
docker compose up -d --build
```

Abre **http://localhost:8000** en tu navegador.

Modo producción (con Nginx, compresión, cacheo de estáticos, subida hasta 500 MB):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Comandos útiles (standalone)

```bash
docker compose ps                       # estado y healthcheck
docker compose exec defectfill nvidia-smi
docker compose exec defectfill bash
docker compose down                     # parar (conserva datos)
docker compose build --no-cache && docker compose up -d
```

---

## Instalación manual (sin Docker)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Variables de entorno

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `HOST_PORT` | `8000` | Puerto expuesto en el host (solo standalone) |
| `NGINX_PORT` | `80` | Puerto del Nginx (solo prod) |
| `CORS_ORIGINS` | `*` | Orígenes permitidos (separados por coma) |
| `HOST` | `0.0.0.0` | Interfaz de escucha dentro del contenedor |
| `PORT` | `8000` | Puerto interno del contenedor |
| `HF_TOKEN` | *(vacío)* | Token de Hugging Face opcional (lectura) |
| `PROJECTS_ROOT` | `/app/projects` | Raíz de proyectos compartidos (stack unificado) |

---

## Flujo de uso en la interfaz

1. **Entrenamiento** — Selecciona el proyecto activo y el tipo de defecto,
   ajusta hiperparámetros (pasos, LoRA rank/alpha, learning rates, pesos de
   pérdidas) y pulsa "Iniciar entrenamiento". Verás pérdidas
   `L_def / L_obj / L_attn / L_total` en vivo.
2. **Evaluación** — Valida los checkpoints del run con un lote fijo
   determinista (mismas imágenes buenas × máscaras) y mide **KID** e
   **IC-LPIPS**; se selecciona el checkpoint de menor KID con IC-LPIPS por
   encima del umbral.
3. **Inferencia** — Elige un checkpoint entrenado, tipo de defecto, cuántas
   imágenes generar y candidatos por imagen (se selecciona el mejor por LPIPS
   espacial). Usa el token `<defect>` en el prompt.
4. **Resultados** — Historial de todas las tandas de entrenamiento y
   generación del proyecto.
5. **Sistema** — Estado de GPU/CUDA y registro de trabajos lanzados.

Cada pestaña tiene un icono `(i)` junto a cada hiperparámetro con una
descripción de su función (pasa el ratón o pulsa).

---

## Proyectos y datos

Los datos de entrenamiento/inferencia se organizan **por proyecto** en
`PROJECTS_ROOT` (estructura MVTec-AD). El proyecto activo se comparte con el
HMI de preprocesado mediante `projects/.active_project.json`:

```
./projects/<proyecto>/
├── images/            # imágenes de entrada
├── labels/            # anotaciones YOLO
├── test/good          # imágenes buenas de validación
├── train/defective    # defectos de entrenamiento (+ _masks)
├── output/<run>/      # checkpoints del entrenamiento
└── generated/<run>/   # imágenes generadas por inferencia
```

En el modo standalone (sin stack) estos directorios son volúmenes Docker
(`defectfill-data`, `defectfill-output`, `defectfill-generated`,
`defectfill-hf-cache`); en el stack unificado son bind mounts de `./projects`,
`./output` y `./generated`.

---

## Estructura del proyecto

```
defectfill/
├── Dockerfile                  # Imagen con CUDA 12.4 + PyTorch GPU
├── docker-compose.yml          # Stack básico standalone (FastAPI directo)
├── docker-compose.prod.yml     # Stack producción (+ Nginx)
├── .dockerignore / .env.example
├── docker/
│   ├── entrypoint.sh           # Script de entrada del contenedor
│   └── nginx.conf              # Config Nginx para producción
├── backend/
│   ├── app.py                  # API FastAPI + healthcheck + CORS + SSE
│   ├── job_manager.py          # Lanza train.py/validate/inference como subprocesos
│   ├── dataset_manager.py      # Organiza el dataset estilo MVTec-AD
│   ├── project_state.py        # Proyecto activo compartido (HMI ↔ defectfill)
│   ├── requirements.txt
│   └── defectfill/             # Código original de DefectFill
├── frontend/                   # HTML/CSS/JS estático
└── sample_dataset/             # Datasets de ejemplo
```

---

## Notas

- El entrenamiento real con difusión requiere GPU; en CPU es inviable.
- La primera ejecución de entrenamiento/inferencia descarga ~5 GB del modelo
  base desde Hugging Face.
- Los volúmenes Docker persisten al hacer `docker compose down`; usa
  `docker compose down -v` para eliminarlos.
- El endpoint `/api/health` permite monitorizar el estado del contenedor
  (usado por el healthcheck de Docker y Nginx).
- `ckpt_final.pt` y el último `ckpt_N.pt` del run **no** son duplicados: cuando
  el paso final coincide con un múltiplo de `save_steps`, solo se escribe
  `ckpt_final.pt` para representar ese estado.