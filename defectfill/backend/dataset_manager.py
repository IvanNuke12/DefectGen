"""
dataset_manager.py
-------------------
Handles building the MVTec-AD-style folder structure that data_loader.py expects:

data/<object_class>/
    train/
        defective/<defect_type>/*.png
        defective_masks/<defect_type>/<same_name>_mask.png
    test/
        good/*.png
"""
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

from job_manager import DATA_DIR, PROJECT_ROOT

SAMPLE_DATASET_DIR = PROJECT_ROOT / "sample_dataset"
ALLOWED_EXT = (".png", ".jpg", ".jpeg")

# Límite máximo por fichero subido (500 MB por defecto, override con env).
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))


class UploadTooLargeError(Exception):
    """Se lanza cuando un fichero subido supera MAX_UPLOAD_BYTES."""


def _safe_name(name: str) -> str:
    keep = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
    return keep or "class"


def _safe_filename(name: str) -> str:
    """Devuelve solo el nombre base (sin rutas) con caracteres seguros.

    Elimina cualquier componente de directorio del nombre subido para
    impedir path traversal (p. ej. '../../backend/app.py')."""
    base = Path(name or "").name
    clean = "".join(c for c in base if c.isalnum() or c in ("-", "_", "."))
    return clean or "file"


async def _read_limited(upload: UploadFile) -> bytes:
    """Lee el fichero por chunks aplicando MAX_UPLOAD_BYTES."""
    total = 0
    chunks = []
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError(
                f"File exceeds maximum upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def save_good_images(object_class: str, files: List[UploadFile]) -> int:
    object_class = _safe_name(object_class)
    target = DATA_DIR / object_class / "test" / "good"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files:
        fname = _safe_filename(f.filename or "")
        if not fname.lower().endswith(ALLOWED_EXT):
            continue
        dest = target / fname
        with open(dest, "wb") as out:
            out.write(await _read_limited(f))
        count += 1
    return count


async def save_defect_pairs(
    object_class: str, defect_type: str, images: List[UploadFile], masks: List[UploadFile]
) -> Dict[str, int]:
    """Saves defective images and their binary masks. Images and masks are matched
    by their position in the two upload lists (upload image #1 with mask #1, etc.),
    and masks are renamed to the `<basename>_mask.png` convention data_loader.py expects."""
    object_class = _safe_name(object_class)
    defect_type = _safe_name(defect_type)
    img_dir = DATA_DIR / object_class / "train" / "defective" / defect_type
    mask_dir = DATA_DIR / object_class / "train" / "defective_masks" / defect_type
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(images), len(masks))
    saved = 0
    for i in range(n):
        img_file = images[i]
        mask_file = masks[i]
        img_name = _safe_filename(img_file.filename or "")
        if not img_name.lower().endswith(ALLOWED_EXT):
            continue
        base_name = Path(img_name).stem
        ext = Path(img_name).suffix
        img_dest = img_dir / f"{base_name}{ext}"
        with open(img_dest, "wb") as out:
            out.write(await _read_limited(img_file))

        mask_dest = mask_dir / f"{base_name}_mask.png"
        with open(mask_dest, "wb") as out:
            out.write(await _read_limited(mask_file))
        saved += 1

    return {"pairs_saved": saved, "images_provided": len(images), "masks_provided": len(masks)}


def load_sample_dataset() -> Dict[str, Any]:
    """Copies the bundled MVTec2 sample dataset (concrete/crack, hazelnut/hole)
    from the original DefectFill repo into data/, so users can try the app
    immediately without preparing their own dataset."""
    if not SAMPLE_DATASET_DIR.exists():
        return {"copied": False, "reason": "sample_dataset directory not found"}
    copied_classes = []
    for cls_dir in SAMPLE_DATASET_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        dest = DATA_DIR / cls_dir.name
        if dest.exists():
            continue
        shutil.copytree(cls_dir, dest)
        copied_classes.append(cls_dir.name)
    return {"copied": True, "classes": copied_classes}


def dataset_summary(project: Optional[str] = None) -> List[Dict[str, Any]]:
    summary = []
    if not DATA_DIR.exists():
        return summary
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        if project and cls_dir.name != project:
            continue
        good_dir = cls_dir / "test" / "good"
        good_count = len(list(good_dir.glob("*"))) if good_dir.exists() else 0

        defect_root = cls_dir / "train" / "defective"
        defect_types = []
        if defect_root.exists():
            for dt_dir in sorted(defect_root.iterdir()):
                if not dt_dir.is_dir():
                    continue
                mask_dir = cls_dir / "train" / "defective_masks" / dt_dir.name
                img_count = len([f for f in dt_dir.glob("*") if f.suffix.lower() in ALLOWED_EXT])
                mask_count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
                defect_types.append({
                    "defect_type": dt_dir.name,
                    "image_count": img_count,
                    "mask_count": mask_count,
                })

        # Leer class_name desde project.json si existe
        class_name = ""
        meta_path = cls_dir / "project.json"
        if meta_path.exists():
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta_data, dict) and meta_data.get("object_class"):
                    class_name = meta_data["object_class"]
            except Exception:
                pass

        summary.append({
            "object_class": cls_dir.name,
            "class_name": class_name or cls_dir.name,
            "good_count": good_count,
            "defect_types": defect_types,
        })
    return summary


def delete_class(object_class: str) -> bool:
    object_class = _safe_name(object_class)
    target = DATA_DIR / object_class
    if target.exists():
        shutil.rmtree(target)
        return True
    return False
