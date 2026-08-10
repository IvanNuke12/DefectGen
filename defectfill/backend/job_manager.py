"""
job_manager.py
---------------
Orchestrates DefectFill training and inference jobs as subprocesses.

Design:
- Each job runs `python defectfill/train.py ...` or `python defectfill/inference.py ...`
  exactly as documented in the original README, just launched from the web backend
  instead of a terminal.
- stdout/stderr from the subprocess is captured line-by-line into a rotating
  in-memory buffer + a log file on disk, so the web UI can show live logs.
- Fine-grained progress (step, losses, ETA, generated image previews) is read from
  the `status.json` file that the (lightly patched) train.py/inference.py scripts
  write to their own output directory. This keeps the training/inference logic
  itself completely untouched.
"""
import os
import re
import sys
import json
import time
import uuid
import shlex
import signal
import select
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFECTFILL_DIR = BASE_DIR / "defectfill"

# DATA_DIR: prioridad PROJECTS_ROOT > carpeta data del proyecto > ../projects (unified stack)
_env_root = os.getenv("PROJECTS_ROOT", "").strip()
if _env_root:
    DATA_DIR = Path(_env_root)
else:
    _local_data = PROJECT_ROOT / "data"
    _shared_projects = PROJECT_ROOT.parent / "projects"
    DATA_DIR = _local_data if _local_data.exists() and any(_local_data.iterdir()) else _shared_projects

OUTPUT_DIR = PROJECT_ROOT / "output"
GENERATED_DIR = PROJECT_ROOT / "generated"

for d in (DATA_DIR, OUTPUT_DIR, GENERATED_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_LOG_LINES = 2000

# Barras de progreso tqdm (p.ej. "Training Progress:  64%|███| 127/200 [..]")
# se filtran del log de la consola: son ruido (1 línea por paso) y con su
# actualización por \r dan sensación de consola "congelada". El progreso real
# ya se muestra en la UI desde status.json.
_TQDM_BAR_RE = re.compile(r"^\s*.*\d{1,3}%\|")


class Job:
    def __init__(self, job_id: str, kind: str, output_dir: Path, cmd: List[str], meta: Dict[str, Any]):
        self.job_id = job_id
        self.kind = kind  # "train" | "infer"
        self.output_dir = output_dir
        self.cmd = cmd
        self.meta = meta
        self.process: Optional[subprocess.Popen] = None
        self.log_lines: deque = deque(maxlen=MAX_LOG_LINES)
        self.total_lines = 0  # nº total de líneas appendeadas (puede exceder len(log_lines))
        self.log_file_path = output_dir / "web_run.log"
        self.started_at = datetime.now().isoformat()
        self.finished_at: Optional[str] = None
        self.returncode: Optional[int] = None
        self.lock = threading.Lock()

    def to_summary(self) -> Dict[str, Any]:
        try:
            rel_dir = str(self.output_dir.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_dir = str(self.output_dir)
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "meta": self.meta,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "alive": self.process is not None and self.process.poll() is None,
            "output_dir": rel_dir,
        }

    def status_json(self) -> Optional[Dict[str, Any]]:
        status_path = self.output_dir / "status.json"
        if not status_path.exists():
            return None
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def tail_log(self, n: int = 200) -> List[str]:
        with self.lock:
            return list(self.log_lines)[-n:]

    def lines_since(self, since_total: int) -> List[str]:
        """Devuelve las líneas nuevas desde un total acumulado previo.

        A diferencia de `tail_log(500)[last_len:]`, esto NO se rompe cuando el
        log supera las 500 líneas (bug que congelaba la consola SSE).
        """
        with self.lock:
            new = max(0, self.total_lines - since_total)
            if new == 0:
                return []
            return list(self.log_lines)[-new:]


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _spawn(self, kind: str, script: str, cli_args: List[str], output_dir: Path, meta: Dict[str, Any]) -> Job:
        job_id = f"{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        output_dir.mkdir(parents=True, exist_ok=True)

        python_exe = sys.executable
        cmd = [python_exe, "-u", str(DEFECTFILL_DIR / script)] + cli_args

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        log_path = output_dir / "web_run.log"
        job = Job(job_id, kind, output_dir, cmd, meta)

        try:
            job.process = subprocess.Popen(
                cmd,
                cwd=str(DEFECTFILL_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as e:
            job.log_lines.append(f"[ERROR] Failed to launch process: {e}")
            job.returncode = -1
            job.finished_at = datetime.now().isoformat()
            with self._lock:
                self.jobs[job_id] = job
            return job

        with self._lock:
            self.jobs[job_id] = job

        def _reader():
            with open(log_path, "a", buffering=1) as lf:
                lf.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
                # Lectura no bloqueante del pipe con select + os.read sobre el
                # fd real. El wrapper fd.read(4096) bloqueaba hasta acumular
                # 4096 bytes, por eso el log de train/inferencia parecía
                # "congelado" y solo volcaba en ráfagas. tqdm repinta con \r
                # (sin \n), así que las líneas del script (p.ej.
                # "Step 130/500 | ...") llegarían "pegadas" a la barra; se
                # parte por \r y \n y se filtra el bar.
                raw_fd = job.process.stdout.fileno()
                buf = ""
                while True:
                    rlist, _, _ = select.select([raw_fd], [], [], 0.5)
                    if rlist:
                        raw = os.read(raw_fd, 8192)
                        if not raw:
                            break
                        buf += raw.decode("utf-8", "replace")
                    elif job.process.poll() is not None:
                        # No hay más datos por ahora y el proceso terminó:
                        # drenar lo que pueda quedar y salir.
                        break
                    parts = re.split(r"[\r\n]", buf)
                    buf = parts.pop()
                    for line in parts:
                        line = line.strip()
                        if not line or _TQDM_BAR_RE.match(line):
                            continue
                        with job.lock:
                            job.log_lines.append(line)
                            job.total_lines += 1
                        lf.write(line + "\n")

                # Drenar cualquier resto tras el final del proceso.
                while buf:
                    parts = re.split(r"[\r\n]", buf)
                    buf = parts.pop()
                    for line in parts:
                        line = line.strip()
                        if not line or _TQDM_BAR_RE.match(line):
                            continue
                        with job.lock:
                            job.log_lines.append(line)
                            job.total_lines += 1
                        lf.write(line + "\n")

                job.process.wait()
                job.returncode = job.process.returncode
                job.finished_at = datetime.now().isoformat()
                with job.lock:
                    job.log_lines.append(f"[process exited with code {job.returncode}]")

        threading.Thread(target=_reader, daemon=True).start()
        return job

    # ------------------------------------------------------------------
    def start_training(self, params: Dict[str, Any]) -> Job:
        run_name = params.get("run_name") or f"{params['object_class']}_{params.get('defect_type') or 'all'}"
        safe_name = "".join(c for c in run_name if c.isalnum() or c in ("-", "_")).strip() or "run"
        # El output del entrenamiento se guarda dentro de la carpeta del proyecto
        # (junto a images/ y crops/), no en una carpeta global de la app.
        project_dir = DATA_DIR / params["object_class"]
        output_dir = project_dir / "output" / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cli_args = [
            "--data_dir", str(DATA_DIR),
            "--object_class", params["object_class"],
            "--class_name", params.get("class_name") or params["object_class"],
            "--output_dir", str(output_dir),
            "--lambda_defect", str(params.get("lambda_defect", 0.5)),
            "--lambda_obj", str(params.get("lambda_obj", 0.2)),
            "--lambda_attn", str(params.get("lambda_attn", 0.05)),
            "--alpha", str(params.get("alpha", 0.3)),
            "--batch_size", str(params.get("batch_size", 2)),
            "--lora_rank", str(params.get("lora_rank", 8)),
            "--lora_alpha", str(params.get("lora_alpha", 16)),
            "--text_encoder_lr", str(params.get("text_encoder_lr", 4e-5)),
            "--unet_lr", str(params.get("unet_lr", 2e-4)),
            "--max_train_steps", str(params.get("max_train_steps", 2000)),
            "--lr_warmup_steps", str(params.get("lr_warmup_steps", 100)),
            "--save_steps", str(params.get("save_steps", 500)),
            "--gradient_accumulation_steps", str(params.get("gradient_accumulation_steps", 2)),
            "--dilate_mask", "True" if params.get("dilate_mask") else "False",
            "--mask_kernel_size", str(params.get("mask_kernel_size", 3)),
            "--config_name", params.get("config_name", "web"),
        ]
        if params.get("defect_type"):
            cli_args += ["--defect_type", params["defect_type"]]
        if params.get("resume_from"):
            cli_args += ["--resume_from", params["resume_from"]]
        if params.get("seed") not in (None, "", -1):
            cli_args += ["--seed", str(params["seed"])]

        meta = {
            "object_class": params["object_class"],
            "class_name": params.get("class_name") or params["object_class"],
            "defect_type": params.get("defect_type"),
            "run_name": run_name,
            "max_train_steps": params.get("max_train_steps", 2000),
        }
        return self._spawn("train", "train.py", cli_args, output_dir, meta)

    def start_inference(self, params: Dict[str, Any]) -> Job:
        checkpoint = params["checkpoint"]
        run_name = params.get("run_name") or f"{params['object_class']}_{params.get('defect_type') or 'gen'}"
        safe_name = "".join(c for c in run_name if c.isalnum() or c in ("-", "_")).strip() or "run"
        # Los resultados de inferencia se guardan dentro de la carpeta del
        # proyecto (junto a images/ y crops/), no en una carpeta global.
        project_dir = DATA_DIR / params["object_class"]
        output_dir = project_dir / "generated" / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cli_args = [
            "--checkpoint", checkpoint,
            "--output_dir", str(output_dir),
            "--object_class", params["object_class"],
            "--class_name", params.get("class_name") or params["object_class"],
            "--defect_type", params["defect_type"],
            "--data_dir", str(DATA_DIR),
            "--num_samples", str(params.get("num_samples", 8)),
            "--steps", str(params.get("steps", 50)),
            "--guidance_scale", str(params.get("guidance_scale", 2.0)),
            "--total_images", str(params.get("total_images", 6)),
            "--batch_size", str(params.get("batch_size", 4)),
            "--lora_rank", str(params.get("lora_rank", 8)),
            "--lora_alpha", str(params.get("lora_alpha", 16)),
            "--dilate_mask", "True" if params.get("dilate_mask") else "False",
            "--mask_kernel_size", str(params.get("mask_kernel_size", 3)),
        ]
        if params.get("prompt"):
            cli_args += ["--prompt", params["prompt"]]
        if params.get("use_compile"):
            cli_args.append("--use_compile")

        meta = {
            "object_class": params["object_class"],
            "class_name": params.get("class_name") or params["object_class"],
            "defect_type": params["defect_type"],
            "checkpoint": checkpoint,
            "total_images": params.get("total_images", 6),
        }
        return self._spawn("infer", "inference.py", cli_args, output_dir, meta)

    def start_validation(self, params: Dict[str, Any]) -> Job:
        """Valida los checkpoints de un run de entrenamiento: para cada
        checkpoint genera un lote fijo determinista (validate_checkpoints.py),
        mide KID e IC-LPIPS contra el conjunto real de defectos y elige el
        mejor (menor KID sujeto a IC-LPIPS mínimo)."""
        project_dir = DATA_DIR / params["object_class"]
        run_dir = project_dir / "output" / params["run"]
        safe_run = "".join(c for c in params.get("run", "run") if c.isalnum() or c in ("-", "_")).strip() or "run"
        output_dir = run_dir / "validation" / f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cli_args = [
            "--run_dir", str(run_dir),
            "--data_dir", str(DATA_DIR),
            "--object_class", params["object_class"],
            "--class_name", params.get("class_name") or params["object_class"],
            "--defect_type", params["defect_type"],
            "--output_dir", str(output_dir),
            "--num_eval_images", str(params.get("num_eval_images", 4)),
            "--num_samples", str(params.get("num_samples", 1)),
            "--steps", str(params.get("steps", 50)),
            "--guidance_scale", str(params.get("guidance_scale", 2.0)),
            "--batch_size", str(params.get("batch_size", 1)),
            "--min_ic_lpips", str(params.get("min_ic_lpips", 0.05)),
            "--lora_rank", str(params.get("lora_rank", 8)),
            "--lora_alpha", str(params.get("lora_alpha", 16)),
            "--dilate_mask", "True" if params.get("dilate_mask") else "False",
            "--mask_kernel_size", str(params.get("mask_kernel_size", 3)),
            "--seed", str(params.get("seed", 42)),
        ]
        if params.get("prompt"):
            cli_args += ["--prompt", params["prompt"]]

        meta = {
            "object_class": params["object_class"],
            "class_name": params.get("class_name") or params["object_class"],
            "defect_type": params["defect_type"],
            "run": params["run"],
            "num_eval_images": params.get("num_eval_images", 4),
            "min_ic_lpips": params.get("min_ic_lpips", 0.05),
        }
        return self._spawn("eval", "validate_checkpoints.py", cli_args, output_dir, meta)

    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        return [j.to_summary() for j in sorted(self.jobs.values(), key=lambda j: j.started_at, reverse=True)]

    def stop(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or not job.process:
            return False
        if job.process.poll() is None:
            try:
                job.process.send_signal(signal.SIGTERM)
                time.sleep(1)
                if job.process.poll() is None:
                    job.process.kill()
                return True
            except Exception:
                return False
        return False


manager = JobManager()
