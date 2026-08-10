"""Carga anotaciones COCO/YOLO y las convierte en máscaras binarias."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw


class AnnotationError(ValueError):
    """Error de configuración o lectura de las anotaciones."""


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").casefold()


def _choose_coco_json(path: Path) -> Path:
    if path.is_file() and path.suffix.casefold() == ".json":
        return path
    if not path.is_dir():
        raise AnnotationError("La ruta de anotaciones no existe.")

    json_files = sorted(path.glob("*.json"), key=lambda item: item.name.casefold())
    if not json_files:
        raise AnnotationError("No se encontró ningún JSON COCO en la carpeta.")

    preferred_names = ("instances_default.json", "annotations.json", "instances.json")
    by_name = {item.name.casefold(): item for item in json_files}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]
    return json_files[0]


def resolve_annotation_source(annotation_path: str, requested_format: str) -> tuple[str, Path]:
    """Devuelve (formato, fuente) validando una configuración de anotaciones."""
    if not annotation_path.strip():
        raise AnnotationError("Selecciona una ruta de anotaciones.")

    path = Path(annotation_path).expanduser()
    fmt = (requested_format or "auto").casefold()
    if fmt not in {"auto", "coco", "yolo"}:
        raise AnnotationError(f"Formato de anotaciones no soportado: {requested_format}")

    if fmt == "coco":
        return "coco", _choose_coco_json(path)
    if fmt == "yolo":
        folder = path.parent if path.is_file() else path
        if not folder.is_dir():
            raise AnnotationError("La carpeta de etiquetas YOLO no existe.")
        return "yolo", folder

    if path.is_file():
        if path.suffix.casefold() == ".json":
            return "coco", path
        if path.suffix.casefold() == ".txt":
            return "yolo", path.parent
        raise AnnotationError("No se reconoce el formato del archivo de anotaciones.")

    if not path.is_dir():
        raise AnnotationError("La ruta de anotaciones no existe.")
    if any(path.glob("*.json")):
        return "coco", _choose_coco_json(path)
    if any(path.glob("*.txt")):
        return "yolo", path
    raise AnnotationError("La carpeta no contiene JSON COCO ni etiquetas TXT de YOLO.")


@lru_cache(maxsize=8)
def _load_coco_index(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns  # Forma parte de la clave para invalidar la caché al cambiar el JSON.
    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"No se pudo leer el JSON COCO: {exc}") from exc

    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise AnnotationError("El JSON no contiene las listas 'images' y 'annotations' de COCO.")

    by_exact: dict[str, dict[str, Any]] = {}
    by_basename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_info in images:
        if not isinstance(image_info, dict) or "id" not in image_info or "file_name" not in image_info:
            continue
        normalised = _normalise_name(str(image_info["file_name"]))
        by_exact[normalised] = image_info
        by_basename[Path(normalised).name].append(image_info)

    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if isinstance(annotation, dict) and "image_id" in annotation:
            annotations_by_image[annotation["image_id"]].append(annotation)

    return {
        "images": images,
        "annotations": annotations,
        "by_exact": by_exact,
        "by_basename": by_basename,
        "annotations_by_image": annotations_by_image,
    }


def _coco_index(json_path: Path) -> dict[str, Any]:
    try:
        modified_ns = json_path.stat().st_mtime_ns
    except OSError as exc:
        raise AnnotationError(f"No se puede acceder al JSON COCO: {exc}") from exc
    return _load_coco_index(str(json_path.resolve()), modified_ns)


def _find_coco_image(index: dict[str, Any], filename: str) -> dict[str, Any] | None:
    normalised = _normalise_name(filename)
    exact = index["by_exact"].get(normalised)
    if exact is not None:
        return exact
    matches = index["by_basename"].get(Path(normalised).name, [])
    return matches[0] if len(matches) == 1 else None


def _polygon_groups(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if all(isinstance(value, (int, float)) for value in segmentation):
        return [segmentation]
    return [group for group in segmentation if isinstance(group, list)]


def _draw_coco_annotation(mask: Image.Image, annotation: dict[str, Any], warnings: list[str]) -> bool:
    segmentation = annotation.get("segmentation")
    polygons = _polygon_groups(segmentation)
    drew_segmentation = False

    if polygons:
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                warnings.append(f"Anotación {annotation.get('id', '?')}: polígono no válido.")
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)
            drew_segmentation = True
        if drew_segmentation:
            return True

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils  # Dependencia opcional para RLE comprimido.

            rle = dict(segmentation)
            counts = rle.get("counts")
            if isinstance(counts, list):
                rle = mask_utils.frPyObjects(rle, mask.height, mask.width)
            elif isinstance(counts, str):
                rle["counts"] = counts.encode("ascii")
            decoded = mask_utils.decode(rle)
            if getattr(decoded, "ndim", 2) == 3:
                decoded = decoded.max(axis=2)
            decoded_mask = Image.fromarray((decoded > 0).astype("uint8") * 255, mode="L")
            if decoded_mask.size != mask.size:
                decoded_mask = decoded_mask.resize(mask.size, Image.Resampling.NEAREST)
            mask.paste(ImageChops.lighter(mask, decoded_mask))
            return True
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            warnings.append(
                f"Anotación {annotation.get('id', '?')}: no se pudo decodificar RLE ({exc}); se usa bbox."
            )

    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            x, y, width, height = (float(value) for value in bbox[:4])
            ImageDraw.Draw(mask).rectangle((x, y, x + width, y + height), fill=255)
            return True
        except (TypeError, ValueError):
            pass
    warnings.append(f"Anotación {annotation.get('id', '?')}: no contiene segmentación ni bbox válida.")
    return False


def _coco_mask(json_path: Path, filename: str, size: tuple[int, int]) -> tuple[Image.Image, dict[str, Any]]:
    index = _coco_index(json_path)
    image_info = _find_coco_image(index, filename)
    if image_info is None:
        return Image.new("L", size, 0), {
            "format": "coco",
            "source": str(json_path),
            "annotation_count": 0,
            "matched": False,
            "warnings": ["La imagen no figura en el JSON COCO."],
        }

    source_size = (
        max(1, int(image_info.get("width") or size[0])),
        max(1, int(image_info.get("height") or size[1])),
    )
    mask = Image.new("L", source_size, 0)
    warnings: list[str] = []
    annotations = index["annotations_by_image"].get(image_info["id"], [])
    drawn = sum(_draw_coco_annotation(mask, annotation, warnings) for annotation in annotations)
    if mask.size != size:
        warnings.append(
            f"Dimensiones COCO {mask.width}×{mask.height} ajustadas a la imagen {size[0]}×{size[1]}."
        )
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask, {
        "format": "coco",
        "source": str(json_path),
        "annotation_count": len(annotations),
        "drawn_count": drawn,
        "matched": True,
        "warnings": warnings,
    }


def _yolo_label_path(folder: Path, filename: str) -> Path:
    return folder / f"{Path(filename).stem}.txt"


def _valid_yolo_rows(path: Path) -> list[tuple[int, list[str]]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise AnnotationError(f"No se pudo leer {path.name}: {exc}") from exc
    return [(number, line.split()) for number, line in enumerate(lines, 1) if line.strip()]


def _yolo_mask(folder: Path, filename: str, size: tuple[int, int]) -> tuple[Image.Image, dict[str, Any]]:
    label_path = _yolo_label_path(folder, filename)
    rows = _valid_yolo_rows(label_path)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    warnings: list[str] = []
    drawn = 0
    image_width, image_height = size

    for line_number, parts in rows:
        if len(parts) < 5:
            warnings.append(f"{label_path.name}:{line_number}: se esperaban al menos 5 valores.")
            continue
        try:
            center_x, center_y, box_width, box_height = map(float, parts[1:5])
        except ValueError:
            warnings.append(f"{label_path.name}:{line_number}: coordenadas no numéricas.")
            continue

        left = max(0.0, (center_x - box_width / 2) * image_width)
        top = max(0.0, (center_y - box_height / 2) * image_height)
        right = min(float(image_width), (center_x + box_width / 2) * image_width)
        bottom = min(float(image_height), (center_y + box_height / 2) * image_height)
        if right <= left or bottom <= top:
            warnings.append(f"{label_path.name}:{line_number}: bbox vacía o fuera de la imagen.")
            continue
        draw.rectangle((left, top, right, bottom), fill=255)
        drawn += 1

    return mask, {
        "format": "yolo",
        "source": str(label_path),
        "annotation_count": len(rows),
        "drawn_count": drawn,
        "matched": label_path.exists(),
        "warnings": warnings,
    }


def build_mask(
    annotation_path: str,
    annotation_format: str,
    filename: str,
    size: tuple[int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    """Construye una máscara L (0/255) y devuelve metadatos de la conversión."""
    resolved_format, source = resolve_annotation_source(annotation_path, annotation_format)
    if resolved_format == "coco":
        return _coco_mask(source, filename, size)
    return _yolo_mask(source, filename, size)


def annotation_info(annotation_path: str, annotation_format: str, filename: str) -> dict[str, Any]:
    """Consulta rápida para la galería sin rasterizar la máscara."""
    resolved_format, source = resolve_annotation_source(annotation_path, annotation_format)
    if resolved_format == "coco":
        index = _coco_index(source)
        image_info = _find_coco_image(index, filename)
        annotations = [] if image_info is None else index["annotations_by_image"].get(image_info["id"], [])
        return {
            "format": "coco",
            "source": str(source),
            "annotation_count": len(annotations),
            "matched": image_info is not None,
        }

    label_path = _yolo_label_path(source, filename)
    rows = _valid_yolo_rows(label_path)
    return {
        "format": "yolo",
        "source": str(label_path),
        "annotation_count": len(rows),
        "matched": label_path.exists(),
    }


def annotation_dataset_summary(annotation_path: str, annotation_format: str) -> dict[str, Any]:
    """Resume la fuente seleccionada para mostrar validación inmediata en la UI."""
    resolved_format, source = resolve_annotation_source(annotation_path, annotation_format)
    if resolved_format == "coco":
        index = _coco_index(source)
        return {
            "ok": True,
            "format": "coco",
            "source": str(source),
            "image_count": len(index["images"]),
            "annotation_count": len(index["annotations"]),
        }

    txt_files = list(source.glob("*.txt"))
    annotation_count = sum(len(_valid_yolo_rows(path)) for path in txt_files)
    return {
        "ok": True,
        "format": "yolo",
        "source": str(source),
        "image_count": len(txt_files),
        "annotation_count": annotation_count,
    }
