"""
Estación de Recorte — HMI para selección de parches sin defecto
===============================================================
Backend Flask. Sirve la interfaz web, lista imágenes de la carpeta de
entrada, genera miniaturas, ejecuta el recorte y mantiene un registro
(CSV + JSON) de cada recorte realizado.

Proyectos: cada proyecto es una carpeta con `images/` (entrada), `labels/`
(anotaciones YOLO opcionales) y `crops/` donde se guardan todos los crops,
máscaras y el log: `<proyecto>/crops/{images,masks,crops_log.csv}`.

Ejecutar:
    python app.py
Se abre automáticamente en http://127.0.0.1:5000
"""
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, render_template
from PIL import Image

from annotation_masks import (
    AnnotationError,
    annotation_dataset_summary,
    annotation_info,
    build_mask,
)

BASE_DIR = Path(__file__).resolve().parent
# Directorio de estado (config, recientes, miniaturas). En Docker se monta un
# volumen aquí (env STATE_DIR); en local se usa la carpeta de la app.
STATE_DIR = Path(os.getenv("STATE_DIR", str(BASE_DIR)))
STATE_DIR.mkdir(exist_ok=True)
CONFIG_PATH = STATE_DIR / "config.json"
RECENT_PROJECTS_PATH = STATE_DIR / "projects_recent.json"
THUMB_CACHE_DIR = STATE_DIR / ".thumb_cache"
THUMB_CACHE_DIR.mkdir(exist_ok=True)

# Generados de DefectFill (pipeline unificado: feedback de inferencia).
# En Docker lo inyecta el compose; el default apunta a la carpeta hermana webapp/.
GENERATED_ROOT = Path(os.getenv("GENERATED_ROOT", str(Path(__file__).resolve().parent.parent / "webapp" / "generated")))

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_EXT = {".txt"}
LABEL_SOURCE_EXT = {".txt", ".json"}
LOG_FIELDS = [
    "filename", "crop_name", "crop_index", "output_path", "x1", "y1", "x2", "y2",
    "crop_size", "src_width", "src_height", "mask_output_path",
    "annotation_format", "annotation_count", "timestamp",
    "dataset_kind", "dataset_defect_type", "dataset_output_path", "dataset_mask_path",
]
# Lista simple de cabecera (sin el diccionario), para comparar el esquema del CSV
# en append_log sin leer todas las filas.
LOG_FIELDS_COLS = list(LOG_FIELDS)

DEFAULT_CONFIG = {
    "input_folder": str(BASE_DIR / "input"),
    "output_folder": str(BASE_DIR / "input" / "crops"),
    "crop_size": 250,
    "annotations_enabled": False,
    "annotation_path": "",
    "annotation_format": "auto",
    "overlay_enabled": True,
    "overlay_opacity": 35,
    "amplify": False,
    "amplify_factor": 3,
}

# Subcarpetas que se crean automáticamente dentro de cada proyecto. Las tres
# últimas forman la estructura MVTec-AD que lee directamente DefectFill:
#   <proyecto>/test/good                -> imágenes buenas
#   <proyecto>/train/defective/<tipo>   -> defectuosas
#   <proyecto>/train/defective_masks/<tipo> -> máscaras (<stem>_mask.png)
PROJECT_SUBDIRS = ["images", "labels", "crops", "test/good", "train/defective", "train/defective_masks"]
PROJECT_META_NAME = "project.json"
CROP_SUBDIR = "images"
MASK_SUBDIR = "masks"
# Máscaras editadas a mano desde la pestaña Etiquetado: se guardan dentro del
# proyecto en <proyecto>/masks/<stem>_mask.png (sombra de las anotaciones).
EDIT_MASK_SUBDIR = "masks"

app = Flask(__name__)
LOG_LOCK = threading.Lock()
PROJECT_LOCK = threading.Lock()

# Prefijo de subruta cuando se sirve detrás de un proxy (nginx /hmi/).
# Vacío en local (python app.py) o si no se define: todo sigue igual.
APP_SUBPATH = os.getenv("APP_SUBPATH", "").strip().rstrip("/")


def _view_subpath():
    """Subruta activa para la petición actual.

    nginx sirve el MISMO backend crop-hmi bajo varias subrutas (/hmi/ para el
    preprocesado y /merge/ para el HMI de postprocesado) inyectando un header
    X-Forwarded-Prefix. De este modo las plantillas prefijan bien sus estáticos
    y JS y el route raíz puede elegir qué vista renderizar. Si el header no
    llega (acceso directo, sin proxy) se usa APP_SUBPATH.
    """
    prefix = request.headers.get("X-Forwarded-Prefix", "").strip().rstrip("/")
    return prefix or APP_SUBPATH


@app.context_processor
def _inject_app_subpath():
    """Expone APP_SUBPATH a las plantillas para prefijar estáticos y JS."""
    return {"APP_SUBPATH": _view_subpath()}


@app.after_request
def _no_cache_static(response):
    """Evita que el navegador sirva versiones viejas de HTML/JS/CSS/API durante
    el desarrollo, que rompían la carga de la galería tras refactors."""
    ct = response.mimetype or ""
    if ct in ("text/html", "text/css", "application/json") or "javascript" in ct:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

# Proyecto activo en memoria: { project_dir, input_folder, annotation_path,
#   annotation_format, output_folder } o None si todavía no se ha abierto.
_ACTIVE_PROJECT = None


# --------------------------------------------------------------------------
# Recents
# --------------------------------------------------------------------------
def _read_recents():
    if RECENT_PROJECTS_PATH.exists():
        try:
            with open(RECENT_PROJECTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, dict)]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _write_recents(items):
    with open(RECENT_PROJECTS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def _push_recent(project_dir, name):
    items = _read_recents()
    items = [it for it in items if it.get("path") != project_dir]
    items.insert(0, {"path": project_dir, "name": name})
    _write_recents(items[:20])


# --------------------------------------------------------------------------
# Proyecto activo COMPARTIDO con la webapp DefectFill.
# Ambas apps montan la misma raíz de proyectos; el marcador vive en la raíz:
#   <root_proyectos>/.active_project.json  ->  {"name": "Lata"}
# La webapp ve el root como /app/projects; el HMI normalmente como /data/projects.
# --------------------------------------------------------------------------
def _shared_active_path():
    """Ruta del marcador de proyecto activo en la raíz de proyectos compartida."""
    for root in _browse_roots():
        root_path = Path(root)
        if root_path.name == "projects" or str(root_path).casefold().replace("\\\\", "/").endswith("projects"):
            return root_path / ".active_project.json"
    roots = _browse_roots()
    return (Path(roots[0]) / ".active_project.json") if roots else None


def _get_shared_active_project_name():
    path = _shared_active_path()
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    name = (data or {}).get("name")
    return name or None


def _set_shared_active_project_name(name):
    path = _shared_active_path()
    if path is None:
        return
    try:
        path.write_text(json.dumps({"name": name}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        pass


def _open_project_if_valid(name):
    """Si <name> corresponde a una carpeta de proyecto dentro de una raíz de
    proyectos, abre esa carpeta en memoria."""
    for root in _browse_roots():
        candidate = Path(root) / name
        if not candidate.is_dir():
            continue
        info = _inspect_project(candidate)
        if info is not None:
            _activate_project(info)
            return True
    return False


# --------------------------------------------------------------------------
# Project helpers
# --------------------------------------------------------------------------
def _safe_dataset_name(name):
    """Nombres seguros para subcarpetas del dataset MVTec-AD (alnum, -, _)."""
    keep = "".join(c for c in str(name or "") if c.isalnum() or c in ("-", "_"))
    return keep or "clase"


# --------------------------------------------------------------------------
# Project meta — project.json (object_class, defect_types, active_defect_type)
# --------------------------------------------------------------------------
def _read_project_meta(project_dir):
    path = Path(project_dir) / PROJECT_META_NAME
    meta = {"object_class": "", "defect_types": [], "active_defect_type": ""}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k in meta:
                    if k in data and data[k] not in (None, ""):
                        meta[k] = data[k]
        except (json.JSONDecodeError, OSError):
            pass
    return meta


def _write_project_meta(project_dir, meta):
    path = Path(project_dir) / PROJECT_META_NAME
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_previous_exports(project_dir, stem, prev_records):
    """Borra del dataset test/good y train/ las copias previas (crop-1 y versión
    completa sin crop) de la imagen <stem>, para que el dataset quede SIEMPRE
    con imágenes crop.

    En lugar de escanear todas las carpetas de tipos (que se vuelve lento a
    medida que crece el dataset), borra las rutas exactas que ya registró el
    log anterior, más cualquier copia directa de <stem> en test/good.
    """
    def _clear_folders(folders, patterns):
        for folder in folders:
            if not folder.is_dir():
                continue
            for pat in patterns:
                for f in folder.glob(pat):
                    _safe_delete(f)

    # Copias exactas conocidas del registro previo en el dataset (ruta absoluta
    # o relativa al proyecto). Se intentan borrar todas; _safe_delete no falla
    # si no existen. OJO: el output_path local (crops/...) NO se borra: es la
    # misma ruta que acaba de sobreescribir el crop actual y si se eliminara
    # rompería el exportado posterior a test/good o train/.
    for rec in prev_records:
        for key in ("dataset_output_path", "dataset_mask_path"):
            p = (rec or {}).get(key)
            if p:
                _safe_delete(Path(p))

    test_good = Path(project_dir) / "test" / "good"
    _clear_folders([test_good], [f"{stem}_crop_1.*", f"{stem}.*"])

    # Por si una versión previa quedó en otra carpeta de tipo sin estar en el
    # log (p.ej. importada a mano): se limpia el stem en los tipos conocidos
    # del proyecto.
    defect_root = Path(project_dir) / "train" / "defective"
    mask_root = Path(project_dir) / "train" / "defective_masks"
    defect_dirs = [d for d in defect_root.iterdir() if d.is_dir()] if defect_root.is_dir() else []
    mask_dirs = [d for d in mask_root.iterdir() if d.is_dir()] if mask_root.is_dir() else []
    _clear_folders(
        defect_dirs,
        [f"{stem}_crop_1.*", f"{stem}.*"],
    )
    _clear_folders(
        mask_dirs,
        [f"{stem}_crop_1_mask.*", f"{stem}_mask.*"],
    )


def _import_train_folder(source, defect_type, project_dir):
    """Copia imágenes defectuosas + máscaras de source a train/."""
    defect_type = _safe_dataset_name(defect_type) or "defect"
    src = Path(source).expanduser()
    if not src.is_dir():
        raise ValueError("La carpeta TRAIN no existe o no es una carpeta.")
    dest_img = Path(project_dir) / "train" / "defective" / defect_type
    dest_mask = Path(project_dir) / "train" / "defective_masks" / defect_type
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_mask.mkdir(parents=True, exist_ok=True)

    # Indexar máscaras: <stem>_mask.<ext> → stem original
    mask_by_stem = {}
    for f in src.iterdir():
        if not f.is_file() or f.suffix.lower() not in VALID_EXT:
            continue
        stem = Path(f.name).stem
        if stem.endswith("_mask"):
            mask_by_stem[stem[:-5]] = f

    images = [
        f for f in sorted(src.iterdir())
        if f.is_file()
        and f.suffix.lower() in VALID_EXT
        and not Path(f.name).stem.endswith("_mask")
    ]
    imported, errors = [], []
    for img in images:
        stem = Path(img.name).stem
        mask = mask_by_stem.get(stem)
        if mask is None:
            errors.append(f"{img.name}: sin máscara (<stem>_mask.png)")
            continue
        try:
            shutil.copy2(img, dest_img / img.name)
            # La máscara siempre se guarda como .png (convención MVTec)
            shutil.copy2(mask, dest_mask / f"{stem}_mask.png")
            imported.append(img.name)
        except OSError as exc:
            errors.append(f"{img.name}: {exc}")
    if not imported and not errors:
        raise ValueError("No se encontraron imágenes defectuosas con su máscara en la carpeta.")
    return imported, errors


def _detect_annotations(project_dir):
    """Detecta las anotaciones del proyecto:
    1) un JSON COCO en la raíz; 2) si no, la subcarpeta labels/ (YOLO)."""
    pdir = Path(project_dir)
    candidates = sorted(pdir.glob("*.json"), key=lambda f: f.name.casefold())
    preferred = ("annotations.json", "instances_default.json", "instances.json")
    by_name = {f.name.casefold(): f for f in candidates}
    for name in preferred:
        if name in by_name:
            return str(by_name[name])
    labels_dir = pdir / "labels"
    if labels_dir.is_dir() and any(f.suffix in YOLO_EXT for f in labels_dir.iterdir()):
        return str(labels_dir)
    return ""


def _copy_into_project(source, kind):
    """Copia los archivos de `source` a la subcarpeta del proyecto activo
    correspondiente al tipo indicado. Devuelve (copied, skipped).
    - kind 'images': archivos de imagen (VALID_EXT), a <proyecto>/images/good/
      (las imágenes importadas desde test_source se consideran buenas).
    - kind 'labels': archivos .txt/.json (labels YOLO / COCO), a <proyecto>/labels."""
    _require_project()
    src = Path(source).expanduser()
    if not src.is_dir():
        raise ValueError("La carpeta de origen no existe o no es una carpeta.")
    if kind not in ("images", "labels"):
        raise ValueError("Tipo de importación inválido.")

    allowed = VALID_EXT if kind == "images" else LABEL_SOURCE_EXT
    if kind == "images":
        dest = Path(_ACTIVE_PROJECT["project_dir"]) / "images" / "good"
    else:
        dest = Path(_ACTIVE_PROJECT["project_dir"]) / "labels"
    dest.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.suffix.lower() not in allowed:
            continue
        try:
            shutil.copy2(f, dest / f.name)
            copied += 1
        except OSError:
            skipped += 1
    if copied == 0 and skipped == 0:
        raise ValueError(f"No se encontraron archivos compatibles en la carpeta de origen.")
    return copied, skipped


def _inspect_project(project_dir):
    """Devuelve un dict con la info válida de la carpeta indicada o None si no
    parece un proyecto (falta images/)."""
    pdir = Path(project_dir)
    if not pdir.is_dir():
        return None
    images_dir = pdir / "images"
    if not images_dir.is_dir():
        return None
    annotation_path = _detect_annotations(pdir)
    annotation_format = "auto"
    if annotation_path:
        if Path(annotation_path).is_dir():
            annotation_format = "yolo"
        else:
            annotation_format = "coco"
    meta = _read_project_meta(pdir)
    return {
        "project_dir": str(pdir.resolve()),
        "name": pdir.name,
        "input_folder": str(images_dir.resolve()),
        "output_folder": str(pdir.resolve()),
        "annotation_path": annotation_path,
        "annotation_format": annotation_format,
        "images_dir_exists": True,
        "object_class": meta["object_class"],
        "defect_types": meta["defect_types"],
        "active_defect_type": meta["active_defect_type"],
    }


def _activate_project(info):
    """Carga info de proyecto en memoria, lo añade a recientes y publica el
    proyecto activo en el marcador compartido con la webapp DefectFill."""
    global _ACTIVE_PROJECT
    with PROJECT_LOCK:
        _ACTIVE_PROJECT = info
    _push_recent(info["project_dir"], info["name"])
    _set_shared_active_project_name(info["name"])


def _refresh_active_project():
    """Re-inspecciona el proyecto activo (p. ej. tras importar imágenes/labels)
    y actualiza la info en memoria."""
    global _ACTIVE_PROJECT
    if _ACTIVE_PROJECT is None:
        return
    with PROJECT_LOCK:
        refreshed = _inspect_project(_ACTIVE_PROJECT["project_dir"])
        if refreshed is not None:
            _ACTIVE_PROJECT = refreshed


def _require_project():
    """Aborta 403 si no hay proyecto activo (excepto en la raíz)."""
    if _ACTIVE_PROJECT is None:
        abort(403, "No hay proyecto activo")


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------
def load_config():
    """Config base persistida en config.json (defaults de sesión/UI)."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg)
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def load_runtime_config():
    """Config efectiva: defaults + proyecto activo + anotaciones del usuario."""
    cfg = load_config()
    if _ACTIVE_PROJECT is not None:
        cfg["input_folder"] = _ACTIVE_PROJECT["input_folder"]
        cfg["output_folder"] = _ACTIVE_PROJECT["output_folder"]
        cfg["annotation_format"] = _ACTIVE_PROJECT.get("annotation_format") or "auto"
        if _ACTIVE_PROJECT.get("annotation_path"):
            cfg["annotation_path"] = _ACTIVE_PROJECT["annotation_path"]
            cfg["annotations_enabled"] = True
        elif cfg.get("annotation_path"):
            cfg["annotations_enabled"] = bool(cfg.get("annotations_enabled"))
        else:
            cfg["annotation_path"] = ""
            cfg["annotations_enabled"] = False
    return cfg


def session_dir(cfg=None):
    """Carpeta de crops del proyecto (sin crear)."""
    cfg = cfg or load_runtime_config()
    return Path(cfg["output_folder"]) / "crops"


def session_log_path(cfg=None):
    return session_dir(cfg) / "crops_log.csv"


# --------------------------------------------------------------------------
# Image / log helpers
# --------------------------------------------------------------------------
def list_image_files(folder):
    """Lista imágenes recursivamente dentro de `folder` (rutas RELATIVAS a él,
    p. ej. "good/foo.jpg" o "defect/foo.jpg"), ordenadas por subcarpeta y nombre."""
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    out = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in VALID_EXT:
            rel = f.relative_to(p)
            out.append(rel.as_posix())
    return sorted(out)


def load_log(cfg=None):
    """dict: filename -> list[record] (crops del proyecto)."""
    records = {}
    log_path = session_log_path(cfg)
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records.setdefault(row["filename"], []).append(row)
    return records


def load_crop_index(cfg=None):
    """dict: crop_name -> registro (crops del proyecto)."""
    index = {}
    log_path = session_log_path(cfg)
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("crop_name"):
                    index[row["crop_name"]] = row
    return index


def next_crop_index(log_records_for_image):
    """Devuelve el siguiente índice 1..N para una imagen dada."""
    max_idx = 0
    for rec in log_records_for_image:
        try:
            max_idx = max(max_idx, int(rec.get("crop_index") or 0))
        except (TypeError, ValueError):
            continue
    return max_idx + 1


def _safe_delete(path):
    """Borra un archivo si existe. Devuelve True si se eliminó."""
    try:
        if path.is_file():
            path.unlink()
            return True
    except OSError:
        pass
    return False


def _remove_log_entries(filename, cfg=None):
    """Elimina del crops_log.csv las entradas de una imagen. Devuelve cuántas
    entradas se quitaron. No-op si el log no existe."""
    log_path = session_log_path(cfg)
    if not log_path.exists():
        return 0
    with LOG_LOCK:
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or LOG_FIELDS
            rows = list(reader)
        kept = [row for row in rows if row.get("filename") != filename]
        removed = len(rows) - len(kept)
        if removed == 0:
            return 0
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def append_log(record, cfg=None):
    log_path = session_log_path(cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        # Si el CSV ya existe pero con un esquema distinto (LOG_FIELDS cambió
        # entre versiones), se reescribe completo con el nuevo esquema. En el
        # caso común (mismo esquema) solo se comprueban los campos de cabecera
        # SIN leer todas las filas, que es lo que hacía más lento el crop a
        # medida que crecía el log.
        needs_migration = False
        existing = log_path.exists()
        if existing:
            with open(log_path, newline="", encoding="utf-8") as f:
                header_reader = csv.reader(f)
                try:
                    header = next(header_reader, [])
                except StopIteration:
                    header = []
            if header != LOG_FIELDS_COLS:
                needs_migration = True

        if needs_migration:
            with open(log_path, newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f, fieldnames=LOG_FIELDS))
            temp_path = log_path.with_suffix(".csv.tmp")
            with open(temp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(old_rows)
            temp_path.replace(log_path)

        is_new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(record)


def safe_input_path(filename):
    """Resolve filename inside the configured input folder, blocking traversal."""
    cfg = load_runtime_config()
    root = Path(cfg["input_folder"]).resolve()
    candidate = (root / filename).resolve()
    if root not in candidate.parents and candidate != root:
        abort(400, "Ruta inválida")
    if not candidate.exists():
        abort(404, "Imagen no encontrada")
    return candidate, cfg


def safe_path_in_roots(filename, roots):
    """Resolve filename inside any of the given roots, blocking traversal."""
    roots = [Path(r).resolve() for r in roots]
    for root in roots:
        candidate = (root / filename).resolve()
        if candidate == root or root in candidate.parents:
            if candidate.exists():
                return candidate
            continue
    abort(404, "Archivo no encontrado")


def _json_error(message, status=400):
    return jsonify({"ok": False, "message": message}), status


# --------------------------------------------------------------------------
# Selector de carpetas servido por web (sustituye a los diálogos tkinter,
# que requerían escritorio y bloqueaban el uso en contenedores headless).
# --------------------------------------------------------------------------
def _browse_roots():
    """Raíces de arranque del navegador (env PROJECTS_ROOTS separado por ';')."""
    raw = os.getenv("PROJECTS_ROOTS") or ";".join([str(BASE_DIR), str(Path.home())])
    return [r.strip() for r in raw.split(";") if r.strip()]


def _safe_utf8(s):
    """Limpia nombres de archivo con bytes no UTF-8 (fs compartido con Windows)
    para que jsonify no falle al serializar surrogates."""
    try:
        return s.encode("utf-8").decode("utf-8")
    except UnicodeError:
        return s.encode("utf-8", "replace").decode("utf-8", "replace")


@app.route("/api/browse")
def api_browse():
    path = (request.args.get("path") or "").strip()
    mode = request.args.get("mode", "folder")
    if mode not in {"folder", "json"}:
        return _json_error("Tipo de selector no válido.")
    roots = _browse_roots()
    if not path:
        return jsonify({
            "roots": roots,
            "path": "",
            "parent": None,
            "entries": [{"name": r, "is_dir": True, "path": r} for r in roots],
        })
    try:
        current = Path(path).expanduser().resolve()
    except OSError as exc:
        return _json_error(f"No se puede acceder a la ruta: {exc}")
    if not current.is_dir():
        return _json_error("La ruta no es una carpeta.", 404)
    entries = []
    try:
        with os.scandir(current) as it:
            children = []
            for child in it:
                try:
                    is_dir = child.is_dir()
                except OSError:
                    # Entradas del sistema sin permiso de stat (p. ej.
                    # DumpStack.log.tmp en C:\ ): se muestran como archivo.
                    is_dir = False
                children.append((child.name, is_dir, child.path))
    except OSError as exc:
        return _json_error(f"No se puede leer la carpeta: {exc}")
    children.sort(key=lambda c: (not c[1], c[0].lower()))
    for name, is_dir, path in children:
        name = _safe_utf8(name)
        path = _safe_utf8(path)
        if is_dir:
            entries.append({"name": name, "is_dir": True, "path": path})
        else:
            # En modo explorador se listan también los archivos (solo se
            # seleccionan en modo 'json'); el frontend los marca como no
            # seleccionables.
            entries.append({"name": name, "is_dir": False, "path": path})
    parent = str(current.parent) if current.parent != current else None
    return jsonify({"roots": roots, "path": str(current), "parent": parent, "entries": entries})


# --------------------------------------------------------------------------
# Routes — pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Si no hay proyecto activo en memoria, intenta autoabrir el proyecto
    activo compartido (puede haberlo fijado la webapp DefectFill). Si aun así
    no hay proyecto, muestra la pantalla de proyectos."""
    if _ACTIVE_PROJECT is None:
        shared = _get_shared_active_project_name()
        if shared and _open_project_if_valid(shared):
            pass
    if _ACTIVE_PROJECT is None:
        return render_template("projects.html")
    if _view_subpath() == "/merge":
        return render_template("postprocess.html")
    return render_template("index.html")


# --------------------------------------------------------------------------
# Routes — project / API
# --------------------------------------------------------------------------
@app.route("/api/session")
def api_session():
    return jsonify({
        "has_project": _ACTIVE_PROJECT is not None,
        "project": _ACTIVE_PROJECT,
    })


@app.route("/api/projects/list")
def api_projects_list():
    """Lista los proyectos visibles en las raíces compartidas de proyectos."""
    items = []
    for root in _browse_roots():
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            info = _inspect_project(child)
            if info is not None:
                items.append({"name": info["name"], "path": info["project_dir"]})
    return jsonify({"projects": items})


@app.route("/api/project/active", methods=["GET", "POST"])
def api_project_active():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return _json_error("Falta 'name'")
        if not _open_project_if_valid(name):
            return _json_error("Proyecto no válido o no existe.", 404)
        return jsonify({"ok": True, "project": _ACTIVE_PROJECT})
    # GET
    name = _get_shared_active_project_name()
    return jsonify({"name": name, "project": _ACTIVE_PROJECT})


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Lee (GET) o guarda (POST) la configuración efectiva de la app."""
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        cfg = load_config()
        for key in ("crop_size", "annotations_enabled", "annotation_format",
                    "annotation_path", "overlay_enabled", "overlay_opacity",
                    "amplify", "amplify_factor"):
            if key in data:
                cfg[key] = data[key]
        save_config(cfg)
    return jsonify(load_runtime_config())


@app.route("/api/projects/recents")
def api_projects_recents():
    return jsonify({"recents": _read_recents()})


@app.route("/api/project/open", methods=["POST"])
def api_project_open():
    data = request.get_json(force=True, silent=True) or {}
    path = (data.get("path") or "").strip()
    if not path:
        return _json_error("Falta 'path'")
    info = _inspect_project(path)
    if info is None:
        # Tolerancia: si el usuario selecciona una carpeta interna del proyecto
        # (p. ej. .../Proyecto/images), subimos hasta encontrar la raíz con images/.
        candidate = Path(path)
        if candidate.is_dir():
            for anc in candidate.parents:
                info = _inspect_project(anc)
                if info is not None:
                    break
    if info is None:
        return _json_error(
            "La carpeta no parece un proyecto (falta la subcarpeta 'images/').", 404
        )
    _activate_project(info)
    return jsonify({"ok": True, "project": _ACTIVE_PROJECT})


@app.route("/api/project/create", methods=["POST"])
def api_project_create():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    parent = (data.get("parent") or "").strip()
    if not name or not parent:
        return _json_error("Faltan 'name' o 'parent'.")
    parent_path = Path(parent).expanduser()
    if not parent_path.is_dir():
        return _json_error("La carpeta padre no existe.", 404)
    # Evita path traversal y rutas absolutas: se neutralizan los separadores
    # de ruta y solo se usa el nombre base (robusto en Windows y Linux).
    name = name.replace("\\", "_").replace("/", "_")
    name = Path(name).name
    if name in ("", ".", ".."):
        return _json_error("Nombre de proyecto no válido.")
    project_dir = parent_path / name
    if project_dir.exists():
        return _json_error("Ya existe una carpeta con ese nombre.", 409)
    try:
        project_dir.mkdir(parents=True, exist_ok=False)
        for subdir in PROJECT_SUBDIRS:
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _json_error(f"No se pudo crear el proyecto: {exc}", 500)
    info = _inspect_project(project_dir)
    _activate_project(info)

    imported = {"images": 0, "labels": 0, "test": 0, "train": 0, "errors": []}

    # --- Copiar labels (etiquetas) ---
    labels_source = (data.get("labels_source") or "").strip()
    if labels_source:
        try:
            copied, _ = _copy_into_project(labels_source, "labels")
            imported["labels"] += copied
        except ValueError as exc:
            imported["errors"].append(f"labels: {exc}")

    # --- Copiar TEST: imágenes → images/ + test/good/ ---
    test_source = (data.get("test_source") or "").strip()
    if test_source:
        try:
            copied, _ = _copy_into_project(test_source, "images")
            imported["images"] += copied
            # Copiar también a test/good/
            good_dir = project_dir / "test" / "good"
            good_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(Path(test_source).expanduser().iterdir()):
                if f.is_file() and f.suffix.lower() in VALID_EXT:
                    shutil.copy2(f, good_dir / f.name)
            imported["test"] += copied
        except ValueError as exc:
            imported["errors"].append(f"test: {exc}")

    # --- Copiar TRAIN: defectuosas + máscaras → train/defective/<tipo>/ ---
    train_source = (data.get("train_source") or "").strip()
    defect_type_raw = (data.get("defect_type") or "").strip()
    if train_source:
        defect_type = defect_type_raw or "defect"
        try:
            train_images, train_errors = _import_train_folder(
                train_source, defect_type, project_dir
            )
            imported["train"] = len(train_images)
            if train_errors:
                imported["errors"].extend(train_errors)
        except ValueError as exc:
            imported["errors"].append(f"train: {exc}")

    # --- Guardar project.json ---
    object_class_raw = (data.get("object_class") or "").strip()
    object_class = _safe_dataset_name(object_class_raw) if object_class_raw else ""
    defect_types = []
    active_defect_type = ""
    if imported["train"] > 0:
        dt = _safe_dataset_name(defect_type_raw or "defect")
        defect_types = [dt]
        active_defect_type = dt
    elif defect_type_raw:
        dt = _safe_dataset_name(defect_type_raw)
        defect_types = [dt]
        active_defect_type = dt
    _write_project_meta(project_dir, {
        "object_class": object_class,
        "defect_types": defect_types,
        "active_defect_type": active_defect_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    _refresh_active_project()

    if imported["errors"]:
        return jsonify({
            "ok": True,
            "project": _ACTIVE_PROJECT,
            "imported": imported,
            "message": "Proyecto creado con avisos: " + " · ".join(imported["errors"]),
        })

    return jsonify({"ok": True, "project": _ACTIVE_PROJECT, "imported": imported})


@app.route("/api/project/import", methods=["POST"])
def api_project_import():
    """Copia imágenes o labels de una carpeta de origen al proyecto activo."""
    _require_project()
    data = request.get_json(force=True, silent=True) or {}
    kind = (data.get("kind") or "").strip()
    source = (data.get("source") or "").strip()
    if kind not in ("images", "labels"):
        return _json_error("Falta 'kind' ('images' o 'labels').")
    if not source:
        return _json_error("Falta 'source'.")
    try:
        copied, skipped = _copy_into_project(source, kind)
    except ValueError as exc:
        return _json_error(str(exc))
    _refresh_active_project()
    return jsonify({"ok": True, "kind": kind, "copied": copied, "skipped": skipped,
                    "project": _ACTIVE_PROJECT})


@app.route("/api/project/upload-images", methods=["POST"])
def api_project_upload_images():
    """Guarda imágenes arrastradas/solteadas en la carpeta de imágenes del
    proyecto activo (multipart, campo 'files').

    El campo opcional `dataset_kind` clasifica la subida directamente en el
    dataset MVTec del proyecto:
      - "good":   copia además a test/good/
      - "defect": copia además a train/defective/<tipo activo>/
    Si se omite, solo se guarda en images/ (comportamiento previo).

    Si el archivo ya existe, se SOBRESCRIBE (misma ruta, sin sufijos _N)."""
    _require_project()
    dataset_kind = (request.form.get("dataset_kind") or "").strip()
    if dataset_kind not in ("", "good", "defect"):
        return _json_error("dataset_kind inválido (good|defect|vacío).")

    cfg = load_runtime_config()
    dest_root = Path(cfg["input_folder"]).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    files = request.files.getlist("files")
    if not files:
        return _json_error("No se recibieron archivos.")

    project_dir = Path(_ACTIVE_PROJECT["project_dir"])
    ds_good = project_dir / "test" / "good"
    ds_defect = project_dir / "train" / "defective"
    meta = _read_project_meta(project_dir)
    tipo = _safe_dataset_name(
        (request.form.get("defect_type") or "").strip()
        or meta.get("active_defect_type")
    ) or "defect"
    if dataset_kind == "defect" and tipo != _safe_dataset_name(meta.get("active_defect_type")):
        meta["active_defect_type"] = tipo
        _write_project_meta(project_dir, meta)

    # Las imágenes viven en subcarpetas de images/: images/good/ o images/defect/.
    # Si no se indica dataset_kind se guardan en la raíz de images/ (legacy).
    sub = {"good": "good", "defect": "defect"}.get(dataset_kind, "")
    dest = dest_root / sub if sub else dest_root
    dest.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    errors = []
    for f in files:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower() not in VALID_EXT:
            skipped += 1
            continue
        target = dest / name
        # Si ya existe, se sobrescribe (misma ruta, sin sufijo _N).
        try:
            f.save(target)
            copied += 1
            # Copiar al dataset según la clasificación elegida
            if dataset_kind == "good":
                ds_good.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, ds_good / target.name)
            elif dataset_kind == "defect":
                dst = ds_defect / tipo
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, dst / target.name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    _refresh_active_project()
    return jsonify({"ok": True, "copied": copied, "skipped": skipped,
                    "errors": errors, "project": _ACTIVE_PROJECT})


@app.route("/api/project/upload-labels", methods=["POST"])
def api_project_upload_labels():
    """Guarda labels (JSON/TXT) arrastrados o solteados en <proyecto>/labels.
    Si el archivo ya existe, se sobrescribe (misma ruta, sin sufijo _N)."""
    _require_project()
    project_dir = Path(_ACTIVE_PROJECT["project_dir"]).resolve()
    dest = project_dir / "labels"
    dest.mkdir(parents=True, exist_ok=True)
    files = request.files.getlist("files")
    if not files:
        return _json_error("No se recibieron archivos.")
    copied = skipped = 0
    errors = []
    for f in files:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower() not in LABEL_SOURCE_EXT:
            skipped += 1
            continue
        target = dest / name
        # Si ya existe, se sobrescribe (misma ruta, sin sufijo _N).
        try:
            f.save(target)
            copied += 1
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    _refresh_active_project()
    return jsonify({"ok": True, "copied": copied, "skipped": skipped,
                    "errors": errors, "project": _ACTIVE_PROJECT})


@app.route("/api/project/upload-coco", methods=["POST"])
def api_project_upload_coco():
    """Guarda un JSON COCO arrastrado en la raíz del proyecto como
    annotations.json y lo activa como fuente de anotaciones."""
    _require_project()
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return _json_error("No se recibió el archivo JSON.")
    if Path(upload.filename).suffix.lower() != ".json":
        return _json_error("Solo se aceptan archivos JSON COCO.")
    project_dir = Path(_ACTIVE_PROJECT["project_dir"]).resolve()
    target = project_dir / "annotations.json"
    try:
        upload.save(target)
    except OSError as exc:
        return _json_error(f"No se pudo guardar el archivo: {exc}", 500)
    try:
        with open(target, "r", encoding="utf-8-sig") as fh:
            json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        _safe_delete(target)
        return _json_error(f"El archivo no es un JSON válido: {exc}")
    _ACTIVE_PROJECT["annotation_path"] = str(target)
    _ACTIVE_PROJECT["annotation_format"] = "coco"
    cfg = load_config()
    cfg["annotation_path"] = str(target)
    cfg["annotation_format"] = "coco"
    cfg["annotations_enabled"] = True
    save_config(cfg)
    return jsonify({"ok": True, "path": str(target), "project": _ACTIVE_PROJECT})


@app.route("/api/project/delete-image", methods=["POST"])
def api_project_delete_image():
    """Borra una imagen del proyecto (y sus recortes/máscaras/dataset asociados).

    Parámetros:
      - filename: nombre del archivo dentro de images/ del proyecto activo.
    """
    _require_project()
    data = request.get_json(force=True, silent=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return _json_error("Falta 'filename'.")

    cfg = load_runtime_config()
    root = Path(cfg["input_folder"]).resolve()
    fp = (root / filename).resolve()
    if root not in fp.parents or fp == root:
        return _json_error("Ruta inválida", 400)
    if not fp.exists():
        return _json_error("La imagen no existe.", 404)

    removed = {"image": False, "crops": [], "masks": [], "dataset": [], "log": 0}
    removed["image"] = _safe_delete(fp)

    project_dir = Path(_ACTIVE_PROJECT["project_dir"])
    log_entries = load_log(cfg).get(filename, [])

    # Limpiar recortes y máscaras del proyecto
    sdir = session_dir(cfg)
    for rec in log_entries:
        if rec.get("output_path"):
            removed["crops"].append(str(rec["output_path"]))
            _safe_delete(Path(rec["output_path"]))
        if rec.get("mask_output_path"):
            removed["masks"].append(str(rec["mask_output_path"]))
            _safe_delete(Path(rec["mask_output_path"]))
        # Limpiar exports del dataset (campos nuevos o fallback glob)
        for key in ("dataset_output_path", "dataset_mask_path"):
            p = rec.get(key)
            if p:
                removed["dataset"].append(p)
                _safe_delete(Path(p))
    # Fallback: si registros antiguos no tienen campos dataset, limpiar por glob
    img_stem = Path(filename).stem
    _clear_previous_exports(project_dir, img_stem, log_entries)

    removed["log"] = _remove_log_entries(filename, cfg)

    _refresh_active_project()
    return jsonify({"ok": True, "filename": filename, "removed": removed})


@app.route("/api/project/close", methods=["POST"])
def api_project_close():
    global _ACTIVE_PROJECT
    with PROJECT_LOCK:
        _ACTIVE_PROJECT = None
    # Al cerrar el proyecto también se limpia el marcador compartido con la
    # webapp, para que al volver a /hmi/ se muestre el panel de proyectos.
    path = _shared_active_path()
    if path is not None and path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return jsonify({"ok": True})


@app.route("/api/project/meta", methods=["GET", "POST"])
def api_project_meta():
    """Lee (GET) o actualiza (POST) los metadatos del proyecto activo:
    object_class, defect_types, active_defect_type."""
    _require_project()
    project_dir = Path(_ACTIVE_PROJECT["project_dir"])
    meta = _read_project_meta(project_dir)
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        if "object_class" in data:
            value = (data["object_class"] or "").strip()
            meta["object_class"] = _safe_dataset_name(value) if value else ""
        if "active_defect_type" in data:
            value = (data["active_defect_type"] or "").strip()
            meta["active_defect_type"] = _safe_dataset_name(value) if value else ""
        if "defect_types" in data and isinstance(data["defect_types"], list):
            meta["defect_types"] = list(dict.fromkeys(
                _safe_dataset_name(t) for t in data["defect_types"] if str(t).strip()
            ))
        _write_project_meta(project_dir, meta)
        _refresh_active_project()
    return jsonify({"ok": True, "meta": meta})


@app.route("/api/annotations/summary")
def api_annotations_summary():
    cfg = load_runtime_config()
    if not cfg["annotations_enabled"]:
        return jsonify({"ok": True, "enabled": False})
    try:
        summary = annotation_dataset_summary(cfg["annotation_path"], cfg["annotation_format"])
    except AnnotationError as exc:
        return _json_error(str(exc))
    return jsonify({"enabled": True, **summary})


@app.route("/api/images")
def api_images():
    _require_project()
    cfg = load_runtime_config()
    files = list_image_files(cfg["input_folder"])
    log = load_log(cfg)  # crops del proyecto
    root = Path(cfg["input_folder"]).resolve()

    # La clasificación good/defect se deriva de la subcarpeta de la imagen:
    # images/good/<f> -> good; images/defect/<f> -> defect.
    def _image_dataset_kind(relname):
        parts = relname.split("/")
        if parts and parts[0] == "good":
            return "good"
        if parts and parts[0] == "defect":
            return "defect"
        return ""

    result = []
    errors = []
    annotation_error = None
    for relname in files:
        fp = root / relname
        try:
            with Image.open(fp) as im:
                w, h = im.size
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{relname}: {exc}")
            continue
        recs = log.get(relname, [])
        crop_box = None
        if recs:
            last = recs[-1]
            try:
                crop_box = {
                    "x1": int(last.get("x1") or 0),
                    "y1": int(last.get("y1") or 0),
                    "x2": int(last.get("x2") or 0),
                    "y2": int(last.get("y2") or 0),
                }
            except (TypeError, ValueError):
                crop_box = None
        annotation = {"matched": False, "annotation_count": 0, "format": None}
        if cfg["annotations_enabled"] and annotation_error is None:
            try:
                annotation = annotation_info(cfg["annotation_path"], cfg["annotation_format"], relname)
            except AnnotationError as exc:
                annotation_error = str(exc)
        result.append({
            "filename": relname,
            "width": w,
            "height": h,
            "done": len(recs) > 0,
            "crop_count": len(recs),
            "crop_box": crop_box,
            "has_annotation": bool(annotation["matched"]),
            "annotation_count": int(annotation["annotation_count"]),
            "annotation_format": annotation["format"],
            "dataset_kind": _image_dataset_kind(relname),
        })

    return jsonify({
        "input_folder": cfg["input_folder"],
        "output_folder": cfg["output_folder"],
        "images": result,
        "errors": errors,
        "folder_exists": Path(cfg["input_folder"]).exists(),
        "annotations_enabled": cfg["annotations_enabled"],
        "annotation_error": annotation_error,
    })


@app.route("/api/image/<path:filename>")
def api_image(filename):
    _require_project()
    cfg = load_runtime_config()
    # Las imágenes fusionadas viven en <proyecto>/merge/, fuera de images/;
    # se permiten ambas raíces para servir el resultado de api_merge.
    roots = [cfg["input_folder"]]
    merge_root = Path(cfg["output_folder"]) / "merge"
    if merge_root.is_dir():
        roots.append(str(merge_root))
    fp = safe_path_in_roots(filename, roots)
    return send_file(fp)


@app.route("/api/thumbnail/<path:filename>")
def api_thumbnail(filename):
    _require_project()
    fp, _ = safe_input_path(filename)
    # Hash determinista de la ruta (md5), a diferencia de hash() que es salado
    # por proceso y hacía que la caché nunca se reutilizara entre reinicios.
    thumb_name = f"{fp.stem}_{hashlib.md5(str(fp).encode('utf-8')).hexdigest()[:12]}.jpg"
    thumb_path = THUMB_CACHE_DIR / thumb_name

    if not thumb_path.exists() or thumb_path.stat().st_mtime < fp.stat().st_mtime:
        with Image.open(fp) as im:
            im = im.convert("RGB")
            im.thumbnail((220, 220))
            im.save(thumb_path, "JPEG", quality=85)

    return send_file(thumb_path)


@app.route("/api/mask/<path:filename>")
def api_mask(filename):
    _require_project()
    fp, cfg = safe_input_path(filename)
    if not cfg["annotations_enabled"]:
        abort(404, "Las anotaciones no estan activadas")

    try:
        with Image.open(fp) as source:
            mask, metadata = build_mask(
                cfg["annotation_path"], cfg["annotation_format"], filename, source.size
            )
    except AnnotationError as exc:
        return _json_error(str(exc))

    overlay = Image.new("RGBA", mask.size, (239, 83, 80, 0))
    overlay.putalpha(mask)
    buffer = io.BytesIO()
    overlay.save(buffer, "PNG")
    buffer.seek(0)
    response = send_file(buffer, mimetype="image/png", max_age=0)
    response.headers["X-Annotation-Count"] = str(metadata["annotation_count"])
    response.headers["X-Annotation-Format"] = metadata["format"]
    return response


def _edited_mask_path(project_dir, stem):
    """Ruta donde se guarda la máscara editada a mano de una imagen."""
    return Path(project_dir) / EDIT_MASK_SUBDIR / f"{stem}_mask.png"


@app.route("/api/mask/raw/<path:filename>")
def api_mask_raw(filename):
    """Máscara binaria (L, 0/255) para editar en la pestaña Etiquetado.

    Prioridad: máscara editada guardada > anotaciones > máscara vacía.
    """
    _require_project()
    fp, cfg = safe_input_path(filename)
    stem = Path(filename).stem
    project_dir = Path(_ACTIVE_PROJECT["project_dir"])

    edited = _edited_mask_path(project_dir, stem)
    if edited.exists():
        response = send_file(edited, mimetype="image/png", max_age=0)
        response.headers["X-Mask-Source"] = "saved"
        return response

    mask = None
    if cfg["annotations_enabled"]:
        try:
            with Image.open(fp) as source:
                mask, _ = build_mask(
                    cfg["annotation_path"], cfg["annotation_format"], filename, source.size
                )
        except AnnotationError:
            mask = None
    if mask is None:
        with Image.open(fp) as source:
            mask = Image.new("L", source.size, 0)

    buffer = io.BytesIO()
    mask.save(buffer, "PNG")
    buffer.seek(0)
    response = send_file(buffer, mimetype="image/png", max_age=0)
    response.headers["X-Mask-Source"] = "annotations"
    return response


@app.route("/api/mask/save", methods=["POST"])
def api_mask_save():
    """Guarda una máscara editada (base64 PNG en 'mask') para la imagen dada.

    Se guarda SOLO en <proyecto>/masks/<stem>_mask.png como fuente editable.
    NO se copia al dataset: el dataset solo recibe crops (imagen + máscara
    cropeada), que es lo que consume el entrenamiento.
    """
    _require_project()
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename")
    mask_b64 = data.get("mask")
    if not filename or not mask_b64:
        return _json_error("Faltan 'filename' o 'mask'.")
    try:
        mask = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
    except Exception as exc:
        return _json_error(f"Máscara inválida: {exc}")

    fp, _ = safe_input_path(filename)
    with Image.open(fp) as source:
        size = source.size
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)

    stem = Path(filename).stem
    project_dir = Path(_ACTIVE_PROJECT["project_dir"])
    masks_dir = Path(project_dir) / EDIT_MASK_SUBDIR
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{stem}_mask.png"
    mask.save(mask_path, "PNG")

    return jsonify({
        "ok": True,
        "path": str(mask_path),
        "mask_source": "saved",
    })


@app.route("/api/crop", methods=["POST"])
def api_crop():
    _require_project()
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename")
    if not filename:
        abort(400, "Falta 'filename'")

    try:
        cx = int(round(float(data["cx"])))
        cy = int(round(float(data["cy"])))
        crop_size = int(round(float(data["crop_size"])))
    except (KeyError, TypeError, ValueError):
        abort(400, "Parámetros de recorte inválidos")

    fp, cfg = safe_input_path(filename)
    sdir = session_dir(cfg)
    crop_dir = sdir / CROP_SUBDIR
    mask_dir = sdir / MASK_SUBDIR
    crop_dir.mkdir(parents=True, exist_ok=True)

    log_records = load_log(cfg).get(filename, [])
    # Un solo recorte por imagen: si ya existía, se sobreescribe con el índice 1
    # (elimina la posibilidad de recortes múltiples / etiqueta "x2").
    crop_index = 1
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    crop_name = f"{stem}_crop_{crop_index}{suffix}"

    mask_crop = None
    mask_output_path = ""
    annotation_format = ""
    annotation_count = 0
    project_dir = Path(_ACTIVE_PROJECT["project_dir"])
    with Image.open(fp) as im:
        im = im.convert("RGB")
        w, h = im.size
        cs = max(1, min(crop_size, w, h))

        # Modo panorámico: el cliente envía el box real (rectángulo más ancho
        # que alto o viceversa). Mientras tanto, modo cuadrado = cs×cs centrado.
        amplify = bool(data.get("amplify"))
        if amplify and all(k in data for k in ("box_x", "box_y", "box_w", "box_h")):
            cw = max(1, min(int(round(float(data["box_w"]))), w))
            ch = max(1, min(int(round(float(data["box_h"]))), h))
            bx = max(0, min(int(round(float(data["box_x"]))), w - cw))
            by = max(0, min(int(round(float(data["box_y"]))), h - ch))
            x1, y1, x2, y2 = bx, by, bx + cw, by + ch
        else:
            half = cs // 2
            x1 = max(0, min(cx - half, w - cs))
            y1 = max(0, min(cy - half, h - cs))
            x2, y2 = x1 + cs, y1 + cs
        crop = im.crop((x1, y1, x2, y2))

        # Fuente de la máscara: prioridad a la máscara editada a mano
        # (<proyecto>/masks/<stem>_mask.png); si no existe, se usan las
        # anotaciones COCO/YOLO.
        mask = None
        edited = _edited_mask_path(project_dir, stem)
        if edited.exists():
            mask = Image.open(edited).convert("L")
            if mask.size != (w, h):
                mask = mask.resize((w, h), Image.Resampling.NEAREST)
            annotation_format = "edited"
            annotation_count = 1
        elif cfg["annotations_enabled"]:
            try:
                mask, metadata = build_mask(
                    cfg["annotation_path"], cfg["annotation_format"], filename, (w, h)
                )
            except AnnotationError as exc:
                return _json_error(f"No se guardo el recorte: {exc}")
            annotation_format = metadata["format"]
            annotation_count = metadata["annotation_count"]

        if mask is not None:
            mask_crop = mask.crop((x1, y1, x2, y2))
        out_path = crop_dir / crop_name
        crop.save(out_path)

    mask_output = None
    if mask_crop is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{stem}_mask_{crop_index}.png"
        mask_output = mask_dir / mask_name
        mask_crop.save(mask_output, "PNG")
        mask_output_path = str(mask_output)

    # --- Auto-export al dataset MVTec del proyecto ---
    meta = _read_project_meta(project_dir)
    tipo = _safe_dataset_name(meta.get("active_defect_type")) or "defect"
    has_mask = mask_crop is not None and mask_output is not None
    has_pixels = False
    if has_mask:
        has_pixels = mask_crop.convert("L").getextrema()[1] > 0
    dataset_kind = "defect" if has_mask and has_pixels else "good"
    dataset_output_path = ""
    dataset_mask_path = ""

    # Limpiar copias anteriores de este crop en el dataset
    _clear_previous_exports(project_dir, stem, log_records)

    ds_good = project_dir / "test" / "good"
    ds_defect = project_dir / "train" / "defective"
    ds_masks = project_dir / "train" / "defective_masks"

    if dataset_kind == "defect":
        ds_dest = ds_defect / tipo / crop_name
        ds_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, ds_dest)
        dataset_output_path = str(ds_dest)
        mask_dest_name = f"{stem}_crop_{crop_index}_mask.png"
        ds_mask_dest = ds_masks / tipo / mask_dest_name
        ds_mask_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask_output, ds_mask_dest)
        dataset_mask_path = str(ds_mask_dest)
    else:
        ds_dest = ds_good / crop_name
        ds_dest.mkdir(parents=True, exist_ok=True) if not ds_dest.parent.exists() else None
        ds_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, ds_dest)
        dataset_output_path = str(ds_dest)

    # Borrar registros anteriores de esta imagen
    if log_records:
        _remove_log_entries(filename, cfg)

    record = {
        "filename": filename,
        "crop_name": crop_name,
        "crop_index": crop_index,
        "output_path": str(out_path),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "crop_size": cs,
        "src_width": w, "src_height": h,
        "mask_output_path": mask_output_path,
        "annotation_format": annotation_format,
        "annotation_count": annotation_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_kind": dataset_kind,
        "dataset_defect_type": tipo if dataset_kind == "defect" else "",
        "dataset_output_path": dataset_output_path,
        "dataset_mask_path": dataset_mask_path,
    }
    append_log(record, cfg)

    return jsonify({"ok": True, **record, "crop_w": x2 - x1, "crop_h": y2 - y1})


@app.route("/api/crops")
def api_crops():
    """Lista plana de los crops del proyecto, SOLO los de imágenes buenas.

    En la ventana Fusionar solo tienen sentido los crops de imágenes
    categorizadas como good/test. Se excluyen los crops de defectos
    (train/defective), que no deben aparecer en el selector.
    """
    log = load_log()
    flat = []
    for recs in log.values():
        flat.extend(recs)
    flat = [rec for rec in flat if rec.get("dataset_kind") == "good"]
    return jsonify({"crops": flat, "count": len(flat)})


@app.route("/api/preview/crop/<path:crop_name>")
def api_preview_crop(crop_name):
    """Sirve un crop guardado por su nombre (lookup en crops/ del proyecto)."""
    _require_project()
    cfg = load_runtime_config()
    sdir = session_dir(cfg)
    candidate = (sdir / CROP_SUBDIR / Path(crop_name)).resolve()
    crop_root = (sdir / CROP_SUBDIR).resolve()
    if crop_root not in candidate.parents and candidate != crop_root:
        abort(400, "Ruta inválida")
    if not candidate.exists():
        abort(404, "Crop no encontrado")
    return send_file(candidate)


@app.route("/api/merge", methods=["POST"])
def api_merge():
    """Incrusta un crop editado en la imagen original, en sus coordenadas.

    Parámetros:
      - crop_name: nombre del crop original (lookup en el log del proyecto)
      - file: archivo de imagen subido (el crop editado)
      - dest (opcional): 'overwrite' (sobrescribe la entrada) | 'new' (nuevo archivo)
    """
    _require_project()
    crop_name = (request.form.get("crop_name") or "").strip()
    if not crop_name:
        return _json_error("Falta 'crop_name'")

    if "file" not in request.files:
        return _json_error("Falta el archivo editado")
    upload = request.files["file"]
    if not upload or not upload.filename:
        return _json_error("Archivo vacío")

    cfg = load_runtime_config()
    index = load_crop_index(cfg)
    rec = index.get(crop_name)
    if rec is None:
        return _json_error(f"No hay registro de crop para '{crop_name}'", 404)

    root = Path(cfg["input_folder"]).resolve()
    src_filename = rec["filename"]
    src_path = (root / src_filename).resolve()
    if root not in src_path.parents and src_path != root:
        return _json_error("Ruta de origen inválida")
    if not src_path.exists():
        return _json_error("La imagen original ya no existe en la carpeta de entrada")

    try:
        x1, y1, x2, y2 = (int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"]))
    except (KeyError, TypeError, ValueError):
        return _json_error("El registro del crop tiene coordenadas inválidas")
    cs = x2 - x1
    if cs <= 0 or (y2 - y1) <= 0:
        return _json_error("Coordenadas del crop inválidas")

    try:
        edited = Image.open(upload.stream)
    except Exception as exc:  # noqa: BLE001
        return _json_error(f"No se pudo leer la imagen editada: {exc}")

    try:
        dest_path = _merge_one(cfg, rec, edited, dest_mode="new")
    except ValueError as exc:
        return _json_error(str(exc))

    rel_root = merge_dir_of(cfg)
    rel = rel_path(dest_path.relative_to(rel_root) if rel_root in dest_path.parents or dest_path == rel_root else dest_path)
    return jsonify({
        "ok": True,
        "merged_path": str(dest_path),
        "merged_url": "/api/image/" + rel,
        "merged_name": rel,
    })


def merge_dir_of(cfg):
    return (Path(cfg["output_folder"]) / "merge").resolve()


def rel_path(p):
    return str(p).replace("\\", "/")


# Regex para extraer el índice de una imagen generada (*_generated.png).
_GEN_NAME_RE = None
_LEGACY_GEN_RE = None


def _generated_index(fname):
    """Índice (output_idx) de una *_generated.png a partir de su nombre."""
    global _GEN_NAME_RE, _LEGACY_GEN_RE
    import re as _re
    if _GEN_NAME_RE is None:
        _GEN_NAME_RE = _re.compile(r"^0*(\d+)_generated\.png$")
        _LEGACY_GEN_RE = _re.compile(r"^.*?_gen(\d+)_generated\.png$")
    m = _GEN_NAME_RE.match(fname) or _LEGACY_GEN_RE.match(fname)
    return int(m.group(1)) if m else None


def _run_source_by_idx(run_dir):
    """Mapea output_idx -> basename de la imagen original (input_image) del run
    leyendo su inference_log.json. Fuente de verdad para resolver el crop de origen."""
    source_by_idx = {}
    log_path = Path(run_dir) / "inference_log.json"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                log = json.load(fh)
            for res in log.get("results", []):
                if res.get("input_image"):
                    source_by_idx[res.get("output_idx")] = Path(res["input_image"]).name
        except (json.JSONDecodeError, OSError):
            source_by_idx = {}
    return source_by_idx


def _merge_one_generated(cfg, run_dir, gen_file, source_by_idx, index):
    """Fusiona una única imagen *_generated.png en su imagen original.

    Resuelve el crop de origen (por índice del run, con fallback al nombre del
    archivo '<crop>_genNNNN_generated.png' para runs sin inference_log.json) y
    lo incrusta. Devuelve un dict en el mismo esquema que un ítem del batch
    (ok/error + rutas)."""
    crop_name = ""
    rec = None
    idx = _generated_index(gen_file.name)
    if idx is not None:
        src_name = source_by_idx.get(idx) or ""
        crop_name = Path(src_name).name if src_name else ""
        rec = index.get(crop_name) if crop_name else None

    if rec is None:
        import re as _re
        # Fallback sin log: el nombre incrusta el stem del crop de origen
        # ('<crop>_genNNNN_generated.png'), ya que el crop bueno alimenta la
        # inferencia. Válido para runs interrumpidos o en curso.
        m = _re.match(r"^(.+)_gen\d+_generated\.png$", gen_file.name)
        if m:
            stem = m.group(1)
            for cname, crec in index.items():
                if Path(cname).stem == stem:
                    crop_name, rec = cname, crec
                    break

    if rec is None:
        return {
            "name": gen_file.name,
            "source": crop_name or "",
            "ok": False,
            "error": "No hay crop registrado de origen para esta generada",
            "merged_path": None,
            "merged_url": None,
            "merged_name": None,
        }
    try:
        with Image.open(gen_file) as edited:
            edited.load()
        stem = Path(gen_file).stem
        suffix = Path(gen_file).suffix or ".png"
        # Organiza los resultados por run (como en generated/), dentro de merge:
        # <proyecto>/merge/<run>/<archivo>_merged_<crop>.png. Así no se mezclan
        # ni se sobreescriben los merges de distintos runs.
        run_name = Path(run_dir).name or "run"
        dest_path = merge_dir_of(cfg) / run_name / f"{stem}_merged_{Path(crop_name).stem}{suffix}"
        merged = _merge_one(cfg, rec, edited, dest_mode="new", dest_path=dest_path)
        rel = rel_path(
            merged.relative_to(merge_dir_of(cfg))
            if merge_dir_of(cfg) in merged.parents or merged == merge_dir_of(cfg)
            else merged
        )
        return {
            "name": gen_file.name,
            "source": crop_name,
            "ok": True,
            "merged_path": str(merged),
            "merged_url": "/api/image/" + rel,
            "merged_name": rel,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": gen_file.name,
            "source": crop_name or "",
            "ok": False,
            "error": str(exc),
            "merged_path": None,
            "merged_url": None,
            "merged_name": None,
        }


def _merge_one(cfg, rec, edited, dest_mode="new", dest_path=None):
    """Incrusta `edited` (PIL Image) en la imagen original según el registro.

    Devuelve la ruta del archivo resultante. Lanza ValueError con mensaje legible.
    - dest_mode 'overwrite': sobrescribe la imagen original de entrada.
    - dest_mode 'new': guarda en <proyecto>/merge/ preservando la original
      (usa dest_path si se indica, si no lo deriva del nombre del crop).
    """
    if edited.mode in ("RGBA", "LA") or (edited.mode == "P" and "transparency" in edited.info):
        bg = Image.new("RGB", edited.size, (0, 0, 0))
        if edited.mode == "P":
            edited = edited.convert("RGBA")
        bg.paste(edited, (0, 0), edited if edited.mode in ("RGBA", "LA") else None)
        edited = bg
    else:
        edited = edited.convert("RGB")

    root = Path(cfg["input_folder"]).resolve()
    src_path = (root / rec["filename"]).resolve()
    if root not in src_path.parents and src_path != root:
        raise ValueError("Ruta de origen inválida")
    if not src_path.exists():
        raise ValueError("La imagen original ya no existe en la carpeta de entrada")
    try:
        x1, y1, x2, y2 = (int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("El registro del crop tiene coordenadas inválidas")
    expected_w, expected_h = x2 - x1, y2 - y1
    if expected_w <= 0 or expected_h <= 0:
        raise ValueError("Coordenadas del crop inválidas")

    ew, eh = edited.size
    if (ew, eh) != (expected_w, expected_h):
        edited = edited.resize((expected_w, expected_h), Image.Resampling.NEAREST)
        ew, eh = edited.size

    with Image.open(src_path) as base:
        base = base.convert("RGB")
        if base.size != (int(rec["src_width"]), int(rec["src_height"])):
            raise ValueError(
                "La imagen original cambió de tamaño desde el recorte; no se puede incrustar."
            )
        base.paste(edited, (x1, y1, x1 + ew, y1 + eh))
        if dest_mode == "overwrite":
            out_path = src_path
        else:
            out_path = dest_path or (merge_dir_of(cfg) / f"{src_path.stem}_merged_{Path(rec['crop_name']).stem}{src_path.suffix or '.png'}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(out_path)
    return out_path


@app.route("/api/merge/batch", methods=["POST"])
def api_merge_batch():
    """Fusiona en bloque todas las imágenes generadas de un run de inferencia.

    Parámetros (JSON):
      - run: nombre del run (carpeta en <proyecto>/generated/<run>).
      - dest (opcional): 'new' (default) | 'overwrite'.

    Para cada entrada del inference_log.json del run:
      - localiza el crop de origen (input_image) en el registro del proyecto,
      - localiza la imagen *_generated.png correspondiente,
      - la incrusta en la imagen original y guarda en <proyecto>/merge/.
    Devuelve el resumen por imagen (ok | error + ruta).
    """
    _require_project()
    data = request.get_json(silent=True) or {}
    run = (data.get("run") or "").strip()
    if not run:
        return _json_error("Falta 'run'")

    cfg = load_runtime_config()
    gen_root = (Path(_ACTIVE_PROJECT["project_dir"]) / "generated").resolve()
    run_dir = (gen_root / run).resolve()
    if gen_root not in run_dir.parents:
        return _json_error("Run inválido")
    if not run_dir.is_dir():
        return _json_error(f"No existe el run '{run}'", 404)

    dest_mode = (data.get("dest") or "new").lower()
    if dest_mode not in ("new", "overwrite"):
        dest_mode = "new"

    index = load_crop_index(cfg)

    source_by_idx = _run_source_by_idx(run_dir)

    results = []
    for f in sorted(run_dir.rglob("*.png")):
        if not f.name.endswith("_generated.png"):
            continue
        res = _merge_one_generated(cfg, run_dir, f, source_by_idx, index)
        results.append(res)

    ok_count = sum(1 for r in results if r["ok"])
    return jsonify({
        "ok": True,
        "run": run,
        "total": len(results),
        "success": ok_count,
        "failed": len(results) - ok_count,
        "results": results,
    })


@app.route("/api/merge/generated", methods=["POST"])
def api_merge_generated():
    """Fusiona UNA imagen generada (flujo individual inverso).

    Parámetros (JSON):
      - path: ruta relativa a <proyecto>/generated/ del archivo *_generated.png
        (p. ej. "run1/Good_crop_1_gen7_generated.png").

    Resuelve automáticamente el crop de origen (por índice del run en el
    inference_log.json) y lo incrusta en la imagen original. Devuelve el
    resultado en el esquema de un ítem del batch.
    """
    _require_project()
    data = request.get_json(silent=True) or {}
    rel = (data.get("file") or "").strip()
    if not rel:
        return _json_error("Falta 'file'")

    gen_root = (Path(_ACTIVE_PROJECT["project_dir"]) / "generated").resolve()
    target = (gen_root / rel).resolve()
    if gen_root not in target.parents or target == gen_root:
        return _json_error("Ruta inválida")
    if not target.is_file() or not target.name.endswith("_generated.png"):
        return _json_error("No es una imagen generada (*_generated.png)", 404)

    run_dir = target.parent
    while run_dir != gen_root and not (
        (run_dir / "inference_log.json").is_file()
    ):
        run_dir = run_dir.parent
    index = load_crop_index()
    source_by_idx = _run_source_by_idx(run_dir)
    result = _merge_one_generated(load_runtime_config(), run_dir, target, source_by_idx, index)
    if not result["ok"]:
        return jsonify(result), 404 if result.get("error", "").startswith("No hay") else 400
    return jsonify(result)


# --------------------------------------------------------------------------
# Exportación de máscaras editadas (pestaña Etiquetado) a COCO JSON / YOLO
# --------------------------------------------------------------------------
def _resolve_mask_images(input_folder):
    """stem -> lista de rutas relativas de imágenes en `input_folder`."""
    by_stem = {}
    for rel in list_image_files(input_folder):
        by_stem.setdefault(Path(rel).stem, []).append(rel)
    return by_stem


def _mask_binary(mask):
    """Devuelve la máscara como array uint8 0/1 y (w, h)."""
    import numpy as np

    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8), arr.shape[1], arr.shape[0]


def _mask_coco_segmentation(mask):
    """Segmentación COCO RLE comprimida + area de una máscara binaria."""
    import numpy as np
    from pycocotools import mask as mask_utils

    binary, width, height = _mask_binary(mask)
    rle = mask_utils.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"counts": rle["counts"], "size": [height, width]}, int(binary.sum())


def _mask_bbox(mask):
    """Bounding box [x, y, w, h] de los píxeles de defecto."""
    import numpy as np

    binary, width, height = _mask_binary(mask)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return [0, 0, width, height], False
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1], True


def _mask_yolo_line(mask, width, height):
    """Línea YOLO normalizada `0 cx cy w h` de la bbox de la máscara."""
    import numpy as np

    binary, _, _ = _mask_binary(mask)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return None
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w = (x2 - x1 + 1) / width
    h = (y2 - y1 + 1) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _edited_masks():
    """(mask, stem, image_rel, meta) por cada máscara editada del proyecto activo."""
    project_dir = Path(_ACTIVE_PROJECT["project_dir"])
    masks_dir = project_dir / EDIT_MASK_SUBDIR
    images_by_stem = _resolve_mask_images(_ACTIVE_PROJECT["input_folder"])
    meta = _read_project_meta(project_dir)
    category = meta.get("active_defect_type") or "defect"
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
        stem = mask_path.name[: -len("_mask.png")]
        try:
            with Image.open(mask_path) as im:
                mask = im.copy()
        except OSError as exc:
            continue
        matches = images_by_stem.get(stem, [])
        image_rel = matches[0] if matches else None
        yield mask, stem, image_rel, category, mask_path


@app.route("/api/export/masks")
def api_export_masks():
    """Exporta las máscaras editadas en la pestaña Etiquetado.

    format=coco (por defecto): JSON COCO con segmentación RLE + bbox,
    reimportable vía POST /api/project/upload-coco.
    format=yolo: ZIP con un <stem>.txt por máscara (bbox normalizada),
    reimportable vía POST /api/project/upload-labels.
    """
    _require_project()
    fmt = request.args.get("format", "coco").casefold()
    if fmt not in ("coco", "yolo"):
        return _json_error("Formato no soportado (coco|yolo).", 400)

    base = _ACTIVE_PROJECT["name"] if _ACTIVE_PROJECT else "proyecto"
    masks = list(_edited_masks())
    if not masks:
        return _json_error("Todavía no hay máscaras editadas en este proyecto.", 404)

    warnings = []
    if fmt == "coco":
        try:
            import numpy as np  # noqa: F401
            from pycocotools import mask as mask_utils  # noqa: F401
        except ImportError:
            return _json_error("Falta numpy/pycocotools para exportar COCO.", 500)

        images = []
        annotations = []
        categories = []
        category_ids = {}
        image_id = 0
        ann_id = 0
        for mask, stem, image_rel, category, _path in masks:
            if not image_rel:
                warnings.append(f"{stem}: no se encontró su imagen en images/.")
                continue
            image_id += 1
            images.append({"id": image_id, "file_name": image_rel,
                           "width": mask.width, "height": mask.height})
            segmentation, area = _mask_coco_segmentation(mask)
            bbox, _has_defect = _mask_bbox(mask)
            if category not in category_ids:
                category_ids[category] = len(category_ids) + 1
                categories.append({"id": category_ids[category], "name": category, "supercategory": ""})
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": category_ids[category],
                "segmentation": segmentation, "area": area, "bbox": bbox,
                "iscrowd": 0,
            })
        payload = {
            "licenses": [],
            "info": {"description": "Máscaras editadas del proyecto %s" % base},
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        buf = io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=f"{base}_masks_annotations.json",
                         mimetype="application/json")

    # YOLO: ZIP con un .txt por imagen (bbox normalizada).
    import zipfile

    buf = io.BytesIO()
    written = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for mask, stem, image_rel, _category, _path in masks:
            if not image_rel:
                warnings.append(f"{stem}: no se encontró su imagen en images/.")
                continue
            line = _mask_yolo_line(mask, mask.width, mask.height)
            if line is None:
                warnings.append(f"{stem}: máscara vacía; se omite.")
                continue
            zf.writestr(f"{stem}.txt", line + "\n")
            written += 1
    if written == 0:
        return _json_error("No se pudo exportar: ninguna máscara válida.", 404)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"{base}_labels.zip",
                     mimetype="application/zip")


@app.route("/api/export/<fmt>")
def api_export(fmt):
    _require_project()
    log_path = session_log_path()
    base = _ACTIVE_PROJECT["name"] if _ACTIVE_PROJECT else "crops"
    if fmt == "csv":
        if not log_path.exists():
            abort(404, "Todavía no hay recortes guardados")
        return send_file(log_path, as_attachment=True, download_name=f"{base}_crops.csv")

    if fmt == "json":
        log = load_log()
        flat = [rec for recs in log.values() for rec in recs]
        buf = io.BytesIO(json.dumps(flat, indent=2, ensure_ascii=False).encode("utf-8"))
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=f"{base}_crops.json",
                          mimetype="application/json")

    abort(404)


# --------------------------------------------------------------------------
# Pipeline unificado — generados de DefectFill
# --------------------------------------------------------------------------


@app.route("/api/generated/runs")
def api_generated_runs():
    """Runs de inferencia con sus imágenes generadas (*_generated.png).

    Los runs viven dentro de cada proyecto (<proyecto>/generated/<run>),
    no en una carpeta global. Se escanean los runs del proyecto activo.

    Cada imagen generada expone su `source`: el nombre del crop bueno del que
    procede (p. ej. Good_..._crop_1.jpg). Se obtiene del inference_log.json del
    run (input_image), que es la fuente de verdad, con fallback al patrón de
    nombre '<bueno>_genNNNN_generated.png' para runs sin log (p. ej.
    interrumpidos), resolviendo contra el índice de crops del proyecto. Así la
    fusión puede ofrecer solo las generadas a partir del crop seleccionado.
    """
    import re as _re
    runs = []
    if _ACTIVE_PROJECT is not None:
        gen_root = Path(_ACTIVE_PROJECT["project_dir"]) / "generated"
    else:
        gen_root = GENERATED_ROOT
    # output_idx -> basename de la imagen original (input_image) por run.
    _gen_name_re = _re.compile(r"^.*?_gen(\d+)_generated\.png$")
    _legacy_gen_re = _re.compile(r"^(\d+)_generated\.png$")
    if gen_root.exists():
        index = load_crop_index()
        for run_dir in sorted(gen_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            source_by_idx = {}
            log_path = run_dir / "inference_log.json"
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as fh:
                        log = json.load(fh)
                    for res in log.get("results", []):
                        if res.get("input_image"):
                            source_by_idx[res.get("output_idx")] = Path(res["input_image"]).name
                except (json.JSONDecodeError, OSError):
                    source_by_idx = {}
            files = []
            for f in sorted(run_dir.rglob("*.png")):
                if f.name.endswith("_generated.png"):
                    source = ""
                    m = _gen_name_re.match(f.name) or _legacy_gen_re.match(f.name)
                    if m:
                        idx = int(m.group(1))
                        source = source_by_idx.get(idx) or ""
                    if not source:
                        # Fallback sin log: nombre '<crop>_genNNNN_generated.png'
                        # incrusta el stem del crop de origen; se resuelve contra
                        # el índice de crops del proyecto.
                        sm = _re.match(r"^(.+)_gen\d+_generated\.png$", f.name)
                        if sm:
                            stem = sm.group(1)
                            for cname, crec in index.items():
                                if Path(cname).stem == stem:
                                    source = cname
                                    break
                    files.append({
                        # Ruta relativa a gen_root (incluye el run):
                        # es la que api_generated_file resuelve contra el root.
                        "path": str(f.relative_to(gen_root)).replace("\\", "/"),
                        "name": f.name,
                        "source": source,
                    })
            runs.append({"run": run_dir.name, "files": files})
    return jsonify({"root": str(gen_root), "exists": gen_root.exists(), "runs": runs})


@app.route("/api/generated/file")
def api_generated_file():
    path = request.args.get("path", "")
    if _ACTIVE_PROJECT is not None:
        base = (Path(_ACTIVE_PROJECT["project_dir"]) / "generated").resolve()
    else:
        base = GENERATED_ROOT.resolve()
    target = (base / path).resolve()
    if base not in target.parents and target != base:
        abort(400, "Ruta inválida")
    if not target.is_file():
        abort(404, "Archivo no encontrado")
    return send_file(target)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))

    # Solo se abre el navegador en local y si no se desactiva (OPEN_BROWSER=0).
    if os.getenv("OPEN_BROWSER", "1") == "1" and host in ("127.0.0.1", "localhost"):
        def _open_browser():
            webbrowser.open(f"http://{host}:{port}")

        threading.Timer(1.0, _open_browser).start()
    app.run(host=host, port=port, debug=False)
