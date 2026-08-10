"""
DefectFill Web App - FastAPI backend
=====================================
Wraps the original DefectFill train.py / inference.py scripts (unofficial
implementation of "DefectFill: Realistic Defect Generation with Inpainting
Diffusion Model for Visual Inspection", CVPR 2024) in a local web UI for:

  - Dataset preparation (MVTec-AD-style folder structure)
  - Launching & monitoring LoRA fine-tuning training jobs
  - Running inference (synthetic defect generation) with a trained checkpoint
  - Browsing generated triplets (original / mask / generated)

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000 in your browser.
"""
import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_manager import manager, DATA_DIR, OUTPUT_DIR, GENERATED_DIR, PROJECT_ROOT
import dataset_manager
import project_state

app = FastAPI(title="DefectFill Web")

# --- CORS: por defecto mismo origen (sin credenciales); si se define
# CORS_ORIGINS explícitamente (separado por comas) se permite con credenciales. ---
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=bool(_cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Límite de subida: 500 MB para datasets grandes (se aplica en
# dataset_manager._read_limited; override con env MAX_UPLOAD_BYTES) ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))

FRONTEND_DIR = PROJECT_ROOT / "frontend"

# --- Tiempo de arranque del servidor (para healthcheck) ---
_start_time = time.time()


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Evita que el navegador sirva versiones viejas de HTML/JS/CSS/API."""
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if ct in ("text/html", "text/css", "application/json") or "javascript" in ct:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ----------------------------------------------------------------------
# Health check (usado por Docker healthcheck y Nginx)
# ----------------------------------------------------------------------
@app.get("/api/health")
def api_health():
    uptime = round(time.time() - _start_time, 1)
    cuda = _SYSTEM_INFO.get("cuda_available")
    info = {"status": "ok", "uptime_seconds": uptime, "cuda": cuda if cuda is not None else "loading"}
    gpu = _SYSTEM_INFO.get("gpu_name")
    if gpu:
        info["gpu"] = gpu
    if _SYSTEM_INFO.get("vram_free_mb") is not None:
        info["vram_free_mb"] = _SYSTEM_INFO["vram_free_mb"]
        info["vram_total_mb"] = _SYSTEM_INFO["vram_total_mb"]
    return info


# ----------------------------------------------------------------------
# Dataset endpoints
# ----------------------------------------------------------------------
@app.get("/api/dataset/summary")
def api_dataset_summary(project: Optional[str] = None):
    if not project:
        active = project_state.get_active_project()
        if active:
            project = active["name"]
    return dataset_manager.dataset_summary(project)


@app.post("/api/dataset/load-sample")
def api_load_sample():
    return dataset_manager.load_sample_dataset()


@app.post("/api/dataset/upload-good")
async def api_upload_good(object_class: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        count = await dataset_manager.save_good_images(object_class, files)
    except dataset_manager.UploadTooLargeError as exc:
        raise HTTPException(413, str(exc))
    return {"saved": count}


@app.post("/api/dataset/upload-defect")
async def api_upload_defect(
    object_class: str = Form(...),
    defect_type: str = Form(...),
    images: List[UploadFile] = File(...),
    masks: List[UploadFile] = File(...),
):
    try:
        result = await dataset_manager.save_defect_pairs(object_class, defect_type, images, masks)
    except dataset_manager.UploadTooLargeError as exc:
        raise HTTPException(413, str(exc))
    return result


@app.delete("/api/dataset/{object_class}")
def api_delete_class(object_class: str):
    ok = dataset_manager.delete_class(object_class)
    if not ok:
        raise HTTPException(404, "Class not found")
    return {"deleted": True}


# ----------------------------------------------------------------------
# Proyecto activo (compartido con crop-hmi vía ./projects/.active.json)
# ----------------------------------------------------------------------
@app.get("/api/projects")
def api_projects():
    return {"projects": project_state.list_projects()}


@app.get("/api/project/active")
def api_project_active():
    return project_state.get_active_project() or {"name": None}


@app.post("/api/project/active")
def api_project_active_set(body: dict = Body(default={})):
    name = (body or {}).get("name")
    project = project_state.set_active_project(name or "")
    if project is None:
        raise HTTPException(400, "Proyecto no válido o no existe")
    return project


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------
@app.get("/api/checkpoints")
def api_checkpoints(project: Optional[str] = None):
    """Checkpoints de entrenamiento, que ahora viven dentro de cada carpeta de
    proyecto (<proyecto>/output/<run>/checkpoints/*.pt). Si `project` se da (o
    hay un proyecto activo), se filtran a ese proyecto únicamente."""
    if not project:
        active = project_state.get_active_project()
        if active:
            project = active["name"]
    checkpoints = []
    if not DATA_DIR.exists():
        return checkpoints
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        if project and class_dir.name != project:
            continue
        out_root = class_dir / "output"
        if not out_root.exists():
            continue
        for run_dir in sorted(out_root.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            cfg_path = run_dir / "train_config.json"
            cfg = {}
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                except Exception:
                    pass
            for ckpt_file in sorted(ckpt_dir.glob("*.pt")):
                checkpoints.append({
                    "path": str(ckpt_file),
                    "run": run_dir.name,
                    "filename": ckpt_file.name,
                    "object_class": class_dir.name,
                    "defect_type": cfg.get("defect_type"),
                    "size_mb": round(ckpt_file.stat().st_size / 1e6, 1),
                })
    return checkpoints


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
class TrainRequest(BaseModel):
    object_class: str
    class_name: Optional[str] = None
    defect_type: Optional[str] = None
    run_name: Optional[str] = None
    max_train_steps: int = 2000
    batch_size: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    text_encoder_lr: float = 4e-5
    unet_lr: float = 2e-4
    lr_warmup_steps: int = 100
    save_steps: int = 500
    gradient_accumulation_steps: int = 2
    lambda_defect: float = 0.5
    lambda_obj: float = 0.2
    lambda_attn: float = 0.05
    alpha: float = 0.3
    dilate_mask: bool = False
    mask_kernel_size: int = 3
    seed: Optional[int] = None
    resume_from: Optional[str] = None


def _resolve_class_name(object_class: str, class_name: Optional[str] = None) -> str:
    """Resuelve class_name desde el request o desde project.json del proyecto."""
    if class_name:
        return class_name
    meta_path = DATA_DIR / object_class / "project.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.get("object_class"):
                return meta["object_class"]
        except Exception:
            pass
    return object_class


@app.post("/api/train/start")
def api_train_start(req: TrainRequest):
    active = project_state.get_active_project()
    if active and req.object_class != active["name"]:
        raise HTTPException(400,
            f"El proyecto activo es '{active['name']}'. Solo se puede entrenar "
            f"sobre el proyecto activo.")
    params = req.dict()
    params["class_name"] = _resolve_class_name(req.object_class, req.class_name)
    job = manager.start_training(params)
    return job.to_summary()


class InferRequest(BaseModel):
    checkpoint: str
    object_class: str
    class_name: Optional[str] = None
    defect_type: str
    run_name: Optional[str] = None
    total_images: int = 6
    num_samples: int = 8
    steps: int = 50
    guidance_scale: float = 2.0
    batch_size: int = 4
    lora_rank: int = 8
    lora_alpha: int = 16
    dilate_mask: bool = False
    mask_kernel_size: int = 3
    use_compile: bool = False
    prompt: Optional[str] = None


@app.post("/api/infer/start")
def api_infer_start(req: InferRequest):
    active = project_state.get_active_project()
    if active and req.object_class != active["name"]:
        raise HTTPException(400,
            f"El proyecto activo es '{active['name']}'. Solo se puede inferir "
            f"sobre el proyecto activo.")
    params = req.dict()
    params["class_name"] = _resolve_class_name(req.object_class, req.class_name)
    job = manager.start_inference(params)
    return job.to_summary()


class ValidateRequest(BaseModel):
    object_class: str
    run: str
    defect_type: str
    class_name: Optional[str] = None
    num_eval_images: int = 4
    num_samples: int = 1
    steps: int = 50
    guidance_scale: float = 2.0
    batch_size: int = 1
    min_ic_lpips: float = 0.05
    lora_rank: int = 8
    lora_alpha: int = 16
    dilate_mask: bool = False
    mask_kernel_size: int = 3
    seed: int = 42
    prompt: Optional[str] = None


@app.post("/api/validate/start")
def api_validate_start(req: ValidateRequest):
    active = project_state.get_active_project()
    if active and req.object_class != active["name"]:
        raise HTTPException(400,
            f"El proyecto activo es '{active['name']}'. Solo se puede validar "
            f"sobre el proyecto activo.")
    params = req.dict()
    params["class_name"] = _resolve_class_name(req.object_class, req.class_name)
    run_dir = DATA_DIR / req.object_class / "output" / req.run
    if not (run_dir / "checkpoints").is_dir():
        raise HTTPException(400, f"El run '{req.run}' no tiene carpeta checkpoints/")
    job = manager.start_validation(params)
    return job.to_summary()


@app.get("/api/validate/result")
def api_validate_result(object_class: str, run: str):
    """Devuelve el resultado de validación más reciente de un run (si existe)."""
    base = DATA_DIR / object_class / "output" / run / "validation"
    if not base.is_dir():
        return {"ok": True, "result": None}
    val_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("val_")],
                      reverse=True)
    for vd in val_dirs:
        res_path = vd / "validation_results.json"
        if res_path.exists():
            try:
                return {"ok": True, "result": json.loads(res_path.read_text(encoding="utf-8")),
                        "dir": vd.name}
            except Exception:
                continue
    return {"ok": True, "result": None}


@app.get("/api/validate/runs")
def api_validate_runs(project: Optional[str] = None):
    """Runs de entrenamiento que tienen checkpoints, por proyecto. Si `project`
    se da (o hay un proyecto activo), se filtran a ese proyecto únicamente."""
    if not project:
        active = project_state.get_active_project()
        if active:
            project = active["name"]
    runs = []
    if not DATA_DIR.exists():
        return runs
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        if project and class_dir.name != project:
            continue
        out_root = class_dir / "output"
        if not out_root.is_dir():
            continue
        for run_dir in sorted(out_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.is_dir():
                continue
            ckpts = sorted([f.name for f in ckpt_dir.glob("*.pt")])
            if not ckpts:
                continue
            cfg = {}
            cfg_path = run_dir / "train_config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                except Exception:
                    pass
            runs.append({
                "object_class": class_dir.name,
                "run": run_dir.name,
                "checkpoints": ckpts,
                "defect_type": cfg.get("defect_type"),
                "class_name": cfg.get("class_name"),
            })
    return runs



# ----------------------------------------------------------------------
# Job monitoring (shared by train + infer)
# ----------------------------------------------------------------------
@app.get("/api/jobs")
def api_jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
def api_job_detail(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        **job.to_summary(),
        "status": job.status_json(),
        "log_tail": job.tail_log(200),
    }


@app.post("/api/jobs/{job_id}/stop")
def api_job_stop(job_id: str):
    ok = manager.stop(job_id)
    return {"stopped": ok}


@app.get("/api/jobs/{job_id}/stream")
async def api_job_stream(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_gen():
        last_total = 0
        while True:
            status = job.status_json()
            new_lines = job.lines_since(last_total)
            last_total += len(new_lines)
            payload = {
                "summary": job.to_summary(),
                "status": status,
                "new_log_lines": new_lines,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            alive = job.process is not None and job.process.poll() is None
            done = (status or {}).get("state") in ("completed", "failed")
            if not alive and (job.returncode is not None or done):
                yield f"data: {json.dumps({'summary': job.to_summary(), 'status': status, 'new_log_lines': [], 'closed': True})}\n\n"
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ----------------------------------------------------------------------
# Generated / output image browsing
# ----------------------------------------------------------------------
@app.get("/api/generated/runs")
def api_generated_runs(project: Optional[str] = None):
    """Runs de inferencia. Cada run vive dentro de su proyecto:
    <proyecto>/generated/<run>. Se devuelve la clase y los archivos para que
    el frontend construya URLs correctas (base = DATA_DIR). Si `project` se da
    (o hay un proyecto activo), se filtran a ese proyecto únicamente."""
    if not project:
        active = project_state.get_active_project()
        if active:
            project = active["name"]
    runs = []
    if not DATA_DIR.exists():
        return runs
    for class_dir in sorted(DATA_DIR.iterdir(), reverse=True):
        if not class_dir.is_dir():
            continue
        if project and class_dir.name != project:
            continue
        gen_root = class_dir / "generated"
        if not gen_root.exists():
            continue
        for run_dir in sorted(gen_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            status_path = run_dir / "status.json"
            status = None
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text())
                except Exception:
                    pass
            files = []
            for f in sorted(run_dir.rglob("*.png")):
                kind = ""
                if f.name.endswith("_generated.png"):
                    kind = "generated"
                elif f.name.endswith("_original.png"):
                    kind = "original"
                elif f.name.endswith("_mask.png"):
                    kind = "mask"
                if kind:
                    files.append({
                        "name": f.name,
                        "kind": kind,
                        # Relativo al run (el frontend antepone clase/generated/run)
                        "rel": str(f.relative_to(run_dir)).replace("\\", "/"),
                    })
            runs.append({
                "object_class": class_dir.name,
                "run": run_dir.name,
                "status": status,
                "files": files,
            })
    return runs


@app.get("/files/output/{path:path}")
def api_files_output(path: str):
    base = OUTPUT_DIR.resolve()
    target = (OUTPUT_DIR / path).resolve()
    if not target.is_relative_to(base) or not target.exists():
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/files/generated/{path:path}")
def api_files_generated(path: str):
    # Ahora los generados viven dentro de cada proyecto: <proyecto>/generated/<run>/...
    base = DATA_DIR.resolve()
    target = (DATA_DIR / path).resolve()
    if not target.is_relative_to(base) or not target.exists():
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/files/data/{path:path}")
def api_files_data(path: str):
    base = DATA_DIR.resolve()
    target = (DATA_DIR / path).resolve()
    if not target.is_relative_to(base) or not target.exists():
        raise HTTPException(404)
    return FileResponse(target)


# ----------------------------------------------------------------------
# System / environment info (helps users diagnose GPU / dependency issues)
# Se pre-importa torch en un hilo de fondo al arrancar para que el
# primer request /api/system sea inmediato y no bloquee la pill
# "comprobando GPU" en el frontend.
# ----------------------------------------------------------------------
import threading as _threading

_SYSTEM_INFO = {
    "python": sys.version,
    "cuda_available": None,   # None = aún no comprobado
    "gpu_name": None,
    "torch_version": None,
    "vram_free_mb": None,
    "vram_total_mb": None,
}


def _warmup_torch():
    """Pre-importa torch y detecta CUDA en background. Se ejecuta una sola
    vez al arrancar y almacena el resultado en _SYSTEM_INFO."""
    try:
        import torch
        _SYSTEM_INFO["torch_version"] = torch.__version__
        _SYSTEM_INFO["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            _SYSTEM_INFO["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                _SYSTEM_INFO["vram_free_mb"] = round(torch.cuda.mem_get_info()[0] / 1e6, 0)
                _SYSTEM_INFO["vram_total_mb"] = round(torch.cuda.mem_get_info()[1] / 1e6, 0)
            except Exception:
                pass
    except Exception:
        if _SYSTEM_INFO["cuda_available"] is None:
            _SYSTEM_INFO["cuda_available"] = False


# Lanza la inicialización de torch en un hilo separado para no bloquear
# el arranque de uvicorn. El endpoint /api/system devuelve el estado
# parcial hasta que termine.
_threading.Thread(target=_warmup_torch, daemon=True).start()


@app.get("/api/system")
def api_system():
    return _SYSTEM_INFO


# ----------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
