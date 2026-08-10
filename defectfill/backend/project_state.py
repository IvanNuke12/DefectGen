"""
project_state.py
-----------------
Estado compartido "proyecto activo" entre DefectFill (defectfill) y crop-hmi.

Ambas apps montan la MISMA carpeta de proyectos (./projects en el host), así que
el proyecto activo se materializa en un archivo mágico en la RAÍZ de esa carpeta:

    <projects_root>/.active_project.json   ->  {"name": "Lata"}

Webapp ve el root como /app/projects; el HMI como /data/projects. Como apuntan
al mismo bind mount, ambas escriben/leen el mismo archivo y comparten el estado
aunque sean procesos distintos.

Un "proyecto" es una subcarpeta del root que parece un dataset MVTec-AD
(tiene test/train o images/). El valor de "object_class" de DefectFill es
exactamente el nombre de esa carpeta de proyecto.
"""
import json
import os
from pathlib import Path

from job_manager import DATA_DIR

ACTIVE_PROJECT_FILENAME = ".active_project.json"


def active_project_path() -> Path:
    """Ruta del marcador de proyecto activo en la raíz de proyectos compartida."""
    return DATA_DIR / ACTIVE_PROJECT_FILENAME


def list_projects() -> list[dict]:
    """Devuelve la lista de proyectos (carpetas con estructura MVTec-AD) del root."""
    projects = []
    if not DATA_DIR.exists():
        return projects
    for child in sorted(DATA_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        # Mínimo para ser un proyecto: tiene test/ ó train/ ó images/ ó project.json
        has_meta = (child / "project.json").exists()
        has_test = (child / "test").is_dir()
        has_train = (child / "train").is_dir()
        has_images = (child / "images").is_dir()
        if not (has_meta or has_test or has_train or has_images):
            continue
        class_name = ""
        meta_path = child / "project.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and meta.get("object_class"):
                    class_name = meta["object_class"]
            except Exception:
                pass
        projects.append({
            "name": child.name,
            "class_name": class_name or child.name,
            "project_dir": str(child.resolve()),
        })
    return projects


def get_active_project() -> dict | None:
    """Proyecto activo desde el marcador compartido, o None si no hay/inexistente."""
    path = active_project_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None
    name = (data or {}).get("name")
    if not name:
        return None
    for p in list_projects():
        if p["name"] == name:
            return p
    return None


def set_active_project(name: str) -> dict | None:
    """Fija el proyecto activo en el marcador compartido. Devuelve el proyecto
    si es válido, o None si el nombre no existe."""
    if not name:
        return None
    project = next((p for p in list_projects() if p["name"] == name), None)
    if project is None:
        return None
    path = active_project_path()
    try:
        path.write_text(json.dumps({"name": name}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        return None
    return project