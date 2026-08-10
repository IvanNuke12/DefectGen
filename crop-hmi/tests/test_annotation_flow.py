import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

import app as crop_app
from annotation_masks import annotation_dataset_summary, build_mask


class AnnotationFlowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.image_path = self.images / "sample.png"
        Image.new("RGB", (100, 80), "white").save(self.image_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_coco(self):
        path = self.root / "instances_default.json"
        path.write_text(json.dumps({
            "images": [{"id": 7, "file_name": "nested/sample.png", "width": 100, "height": 80}],
            "annotations": [{
                "id": 11,
                "image_id": 7,
                "category_id": 1,
                "segmentation": [[20, 10, 80, 10, 80, 70, 20, 70]],
                "bbox": [20, 10, 60, 60],
                "area": 3600,
                "iscrowd": 0,
            }],
            "categories": [{"id": 1, "name": "defect"}],
        }), encoding="utf-8")
        return path

    def test_coco_polygon_becomes_binary_mask(self):
        coco_path = self.make_coco()
        mask, metadata = build_mask(str(coco_path), "coco", "sample.png", (100, 80))

        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getpixel((50, 40)), 255)
        self.assertEqual(mask.getpixel((5, 5)), 0)
        self.assertEqual(metadata["annotation_count"], 1)
        self.assertTrue(metadata["matched"])

    def test_yolo_detection_becomes_rectangle_mask(self):
        labels = self.root / "labels"
        labels.mkdir()
        (labels / "sample.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")

        mask, metadata = build_mask(str(labels), "yolo", "sample.png", (100, 80))

        self.assertEqual(mask.getpixel((50, 40)), 255)
        self.assertEqual(mask.getpixel((10, 10)), 0)
        self.assertEqual(metadata["drawn_count"], 1)
        self.assertEqual(annotation_dataset_summary(str(labels), "auto")["format"], "yolo")

    def test_coco_compressed_rle_is_supported(self):
        binary = np.zeros((80, 100), dtype=np.uint8)
        binary[15:45, 30:70] = 1
        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("ascii")
        coco_path = self.root / "rle.json"
        coco_path.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "sample.png", "width": 100, "height": 80}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": rle}],
            "categories": [{"id": 1, "name": "defect"}],
        }), encoding="utf-8")

        mask, metadata = build_mask(str(coco_path), "coco", "sample.png", (100, 80))
        self.assertEqual(mask.getpixel((50, 30)), 255)
        self.assertEqual(mask.getpixel((10, 10)), 0)
        self.assertEqual(metadata["drawn_count"], 1)

    def _make_config(self, output, **overrides):
        cfg = {
            "input_folder": str(self.images),
            "output_folder": str(output),
            "output_folder_auto": False,
            "crop_size": 40,
            "annotations_enabled": False,
            "annotation_path": "",
            "annotation_format": "auto",
            "overlay_enabled": True,
            "overlay_opacity": 35,
        }
        cfg.update(overrides)
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        return config_path

    def test_crop_api_saves_rgb_and_aligned_mask(self):
        coco_path = self.make_coco()
        output = self.root / "output"
        config_path = self._make_config(output, crop_size=40, annotations_enabled=True,
                                        annotation_path=str(coco_path), annotation_format="coco")
        active_project = {
            "project_dir": str(self.root),
            "name": "test_project",
            "input_folder": str(self.images),
            "output_folder": str(output),
            "annotation_path": str(coco_path),
            "annotation_format": "auto",
            "images_dir_exists": True,
        }

        with mock.patch.object(crop_app, "CONFIG_PATH", config_path), mock.patch.object(
            crop_app, "_ACTIVE_PROJECT", active_project
        ):
            client = crop_app.app.test_client()
            image_list = client.get("/api/images").get_json()
            self.assertEqual(image_list["images"][0]["annotation_count"], 1)

            overlay_response = client.get("/api/mask/sample.png")
            self.assertEqual(overlay_response.status_code, 200)
            self.assertEqual(overlay_response.mimetype, "image/png")

            response = client.post("/api/crop", json={
                "filename": "sample.png", "cx": 50, "cy": 40, "crop_size": 40,
            })
            payload = response.get_json()
            self.assertEqual(response.status_code, 200, payload)
            self.assertEqual(payload["annotation_count"], 1)

            session = output / "crops"
            self.assertTrue((session / "crops_log.csv").exists())
            self.assertTrue((session / "images" / "sample_crop_1.png").exists())
            self.assertTrue((session / "masks" / "sample_mask_1.png").exists())

            with Image.open(session / "images" / "sample_crop_1.png") as rgb_crop, Image.open(
                session / "masks" / "sample_mask_1.png"
            ) as mask_crop:
                self.assertEqual(rgb_crop.size, (40, 40))
                self.assertEqual(mask_crop.size, (40, 40))
                self.assertEqual(mask_crop.getpixel((20, 20)), 255)

            self.assertEqual(payload["crop_index"], 1)
            self.assertEqual(payload["crop_name"], "sample_crop_1.png")

    def test_old_csv_header_is_migrated_without_losing_rows(self):
        output = self.root / "output"
        config_path = self._make_config(output)
        log_path = output / "crops" / "crops_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        old_fields = [
            "filename", "output_path", "x1", "y1", "x2", "y2",
            "crop_size", "src_width", "src_height", "timestamp",
        ]
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow({field: "old" for field in old_fields})

        record = {field: "new" for field in crop_app.LOG_FIELDS}
        with mock.patch.object(crop_app, "CONFIG_PATH", config_path):
            crop_app.append_log(record)

        with log_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, crop_app.LOG_FIELDS)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["filename"], "old")
        self.assertEqual(rows[1]["filename"], "new")


class CropsAndMergeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.images = self.root / "images"
        self.images.mkdir()
        # Imagen original de 60x40 con esquina superior izquierda blanca.
        src = Image.new("RGB", (60, 40), (10, 20, 30))
        for x in range(0, 20):
            for y in range(0, 20):
                src.putpixel((x, y), (255, 255, 255))
        self.image_path = self.images / "src.png"
        src.save(self.image_path)

        config_path = self.root / "config.json"
        output = self.root / "output"
        config_path.write_text(json.dumps({
            "input_folder": str(self.images),
            "output_folder": str(output),
            "output_folder_auto": False,
            "crop_size": 20,
            "annotations_enabled": False,
            "annotation_path": "",
            "annotation_format": "auto",
            "overlay_enabled": True,
            "overlay_opacity": 35,
        }), encoding="utf-8")

        self.config_path = config_path
        self.output = output
        self.active_project = {
            "project_dir": str(self.root),
            "name": "test_project",
            "input_folder": str(self.images),
            "output_folder": str(self.output),
            "annotation_path": "",
            "annotation_format": "auto",
            "images_dir_exists": True,
        }
        self.patches = [
            mock.patch.object(crop_app, "CONFIG_PATH", config_path),
            mock.patch.object(crop_app, "_ACTIVE_PROJECT", self.active_project),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tempdir.cleanup()

    def _crop(self, client, cx, cy):
        return client.post("/api/crop", json={
            "filename": "src.png", "cx": cx, "cy": cy, "crop_size": 20,
        }).get_json()

    def test_crop_listed_in_api_crops(self):
        client = crop_app.app.test_client()
        rec = self._crop(client, 10, 10)
        self.assertEqual(rec["crop_name"], "src_crop_1.png")
        listing = client.get("/api/crops").get_json()
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["crops"][0]["crop_name"], "src_crop_1.png")

    def test_preview_endpoint_serves_crop(self):
        client = crop_app.app.test_client()
        self._crop(client, 10, 10)
        r = client.get("/api/preview/crop/src_crop_1.png")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "image/png")

    def test_merge_incrusta_crop_editado_en_la_original(self):
        client = crop_app.app.test_client()
        rec = self._crop(client, 10, 10)
        # Crop editado: todo rojo, mismo tamaño (20x20).
        edited = Image.new("RGB", (20, 20), (220, 30, 30))
        buf = io.BytesIO()
        edited.save(buf, "PNG")
        buf.seek(0)
        buf.name = "src_crop_1_edit.png"

        resp = client.post(
            "/api/merge",
            data={"crop_name": rec["crop_name"], "file": (buf, "edit.png")},
            content_type="multipart/form-data",
        )
        payload = resp.get_json()
        self.assertEqual(resp.status_code, 200, payload)
        self.assertIn("merged_path", payload)

        # La imagen resultante debe ser nueva (no sobrescribe la entrada por defecto).
        self.assertTrue(payload["merged_path"].endswith("src_merged_src_crop_1.png"))
        with Image.open(payload["merged_path"]) as result:
            self.assertEqual(result.size, (60, 40))
            self.assertEqual(result.getpixel((10, 10)), (220, 30, 30))
            # Fuera de la región del crop mantiene el color original.
            self.assertEqual(result.getpixel((40, 30)), (10, 20, 30))

    def test_merge_resize_si_el_crop_editado_cambia_de_tamano(self):
        client = crop_app.app.test_client()
        rec = self._crop(client, 10, 10)
        edited = Image.new("RGB", (40, 40), (0, 200, 0))  # más grande
        buf = io.BytesIO()
        edited.save(buf, "PNG")
        buf.seek(0)
        resp = client.post(
            "/api/merge",
            data={"crop_name": rec["crop_name"], "file": (buf, "edit.png")},
            content_type="multipart/form-data",
        )
        payload = resp.get_json()
        self.assertEqual(resp.status_code, 200, payload)
        with Image.open(payload["merged_path"]) as result:
            self.assertEqual(result.size, (60, 40))
            self.assertEqual(result.getpixel((10, 10)), (0, 200, 0))

    def test_merge_png_alpha_se_flattena_sin_bordes(self):
        client = crop_app.app.test_client()
        rec = self._crop(client, 10, 10)
        # Crop RGBA con contenido opaco rojo y una región transparente —
        # el flatten debe pintar todo el crop de rojo sólido, sin bleed.
        edited = Image.new("RGBA", (20, 20), (220, 30, 30, 255))
        edited.putpixel((0, 0), (220, 30, 30, 0))      # transparente
        edited.putpixel((19, 19), (220, 30, 30, 0))    # transparente
        buf = io.BytesIO()
        edited.save(buf, "PNG")
        buf.seek(0)
        resp = client.post(
            "/api/merge",
            data={"crop_name": rec["crop_name"], "file": (buf, "edit.png")},
            content_type="multipart/form-data",
        )
        payload = resp.get_json()
        self.assertEqual(resp.status_code, 200, payload)
        with Image.open(payload["merged_path"]) as result:
            # Todas las esquinas del crop pegado deben ser rojo sólido:
            # el flatten sobre fondo negro pinta (220,30,30) en opacos y
            # (0,0,0) en transparentes. Lo importante: ningún bleed blando.
            for x in range(20):
                for y in range(20):
                    px = result.getpixel((x, y))
                    self.assertIn(px, {(220, 30, 30), (0, 0, 0)},
                                  f"bleed en ({x},{y}): {px}")

    def test_merge_pixel_perfect_en_las_esquinas_del_crop(self):
        client = crop_app.app.test_client()
        # Controlamos el tamaño y las coordenadas exactas: crop en (0,0)→(20,20).
        rec = self._crop(client, 10, 10)
        self.assertEqual((rec["x1"], rec["y1"], rec["x2"], rec["y2"]), (0, 0, 20, 20))
        # Crop editado: patrón de esquinas distinguible.
        edited = Image.new("RGB", (20, 20), (0, 0, 0))
        edited.putpixel((0, 0), (255, 1, 1))
        edited.putpixel((19, 0), (2, 255, 2))
        edited.putpixel((0, 19), (3, 3, 255))
        edited.putpixel((19, 19), (255, 255, 255))
        buf = io.BytesIO()
        edited.save(buf, "PNG")
        buf.seek(0)
        resp = client.post(
            "/api/merge",
            data={"crop_name": rec["crop_name"], "file": (buf, "edit.png")},
            content_type="multipart/form-data",
        )
        payload = resp.get_json()
        self.assertEqual(resp.status_code, 200, payload)
        with Image.open(payload["merged_path"]) as result:
            self.assertEqual(result.getpixel((0, 0)), (255, 1, 1))
            self.assertEqual(result.getpixel((19, 0)), (2, 255, 2))
            self.assertEqual(result.getpixel((0, 19)), (3, 3, 255))
            self.assertEqual(result.getpixel((19, 19)), (255, 255, 255))
            # Justo fuera del crop, no se toca:
            self.assertEqual(result.getpixel((20, 20)), (10, 20, 30))

    def test_merge_sin_registro_devuelve_error(self):
        client = crop_app.app.test_client()
        buf = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(buf, "PNG")
        buf.seek(0)
        resp = client.post(
            "/api/merge",
            data={"crop_name": "inexistente_crop_1.png", "file": (buf, "edit.png")},
            content_type="multipart/form-data",
        )
        payload = resp.get_json()
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(payload.get("ok", False))


class ProjectFlowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.project = self.root / "mi_proyecto"
        self._prev_patches = [
            mock.patch.object(crop_app, "_ACTIVE_PROJECT", None),
            mock.patch.object(crop_app, "RECENT_PROJECTS_PATH",
                              self.root / "projects_recent.json"),
        ]
        for p in self._prev_patches:
            p.start()

    def tearDown(self):
        for p in self._prev_patches:
            p.stop()
        self.tempdir.cleanup()

    def test_sin_proyecto_las_apis_dan_403(self):
        client = crop_app.app.test_client()
        self.assertEqual(client.post("/api/crop", json={}).status_code, 403)
        self.assertEqual(client.get("/api/images").status_code, 403)
        self.assertEqual(client.get("/api/export/csv").status_code, 403)

    def test_crear_abrir_cerrar_proyecto(self):
        client = crop_app.app.test_client()
        r = client.post("/api/project/create", json={"name": "mi_proyecto", "parent": str(self.root)})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertTrue((self.project / "images").is_dir())
        self.assertTrue((self.project / "labels").is_dir())
        self.assertTrue(crop_app._ACTIVE_PROJECT)
        self.assertEqual(crop_app._ACTIVE_PROJECT["name"], "mi_proyecto")
        self.assertEqual(crop_app._ACTIVE_PROJECT["input_folder"], str(self.project / "images"))

        recents = client.get("/api/projects/recents").get_json()["recents"]
        self.assertEqual(recents[0]["name"], "mi_proyecto")

        client.post("/api/project/close")
        self.assertIsNone(crop_app._ACTIVE_PROJECT)
        self.assertFalse(client.get("/api/session").get_json()["has_project"])

    def test_crear_proyecto_copia_imagenes_y_labels_de_origen(self):
        src_images = self.root / "src_imgs"
        src_labels = self.root / "src_labels"
        src_images.mkdir()
        src_labels.mkdir()
        Image.new("RGB", (20, 20), "white").save(src_images / "a.png")
        Image.new("RGB", (20, 20), "black").save(src_images / "b.jpg")
        (src_images / "notes.txt").write_text("no es imagen", encoding="utf-8")
        (src_labels / "a.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
        (src_labels / "b.txt").write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        client = crop_app.app.test_client()
        r = client.post("/api/project/create", json={
            "name": "mi_proyecto", "parent": str(self.root),
            "test_source": str(src_images), "labels_source": str(src_labels),
        })
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["imported"]["images"], 2)
        self.assertEqual(r.get_json()["imported"]["labels"], 2)
        self.assertTrue((self.project / "images" / "a.png").exists())
        self.assertTrue((self.project / "images" / "b.jpg").exists())
        self.assertFalse((self.project / "images" / "notes.txt").exists())
        self.assertTrue((self.project / "labels" / "a.txt").exists())
        # Las anotaciones se detectan automáticamente (labels/ → YOLO).
        self.assertTrue(crop_app._ACTIVE_PROJECT["annotation_path"].endswith("labels"))
        self.assertEqual(crop_app._ACTIVE_PROJECT["annotation_format"], "yolo")

    def test_import_imagenes_y_labels_al_proyecto_activo(self):
        client = crop_app.app.test_client()
        client.post("/api/project/create", json={"name": "mi_proyecto", "parent": str(self.root)})
        src = self.root / "origen"
        src.mkdir()
        Image.new("RGB", (10, 10), "red").save(src / "img1.png")
        Image.new("RGB", (10, 10), "green").save(src / "img2.png")
        r = client.post("/api/project/import", json={"kind": "images", "source": str(src)})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["copied"], 2)
        self.assertTrue((self.project / "images" / "img1.png").exists())

        labels_src = self.root / "labels_origen"
        labels_src.mkdir()
        (labels_src / "img1.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
        r = client.post("/api/project/import", json={"kind": "labels", "source": str(labels_src)})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertTrue((self.project / "labels" / "img1.txt").exists())
        self.assertEqual(crop_app._ACTIVE_PROJECT["annotation_format"], "yolo")

    def test_import_requiere_proyecto_y_carpeta_valida(self):
        client = crop_app.app.test_client()
        self.assertEqual(client.post("/api/project/import", json={
            "kind": "images", "source": str(self.root),
        }).status_code, 403)
        client.post("/api/project/create", json={"name": "mi_proyecto", "parent": str(self.root)})
        no_src = self.root / "inexistente"
        r = client.post("/api/project/import", json={"kind": "images", "source": str(no_src)})
        self.assertEqual(r.status_code, 400)
        empty = self.root / "vacia"
        empty.mkdir()
        r = client.post("/api/project/import", json={"kind": "images", "source": str(empty)})
        self.assertEqual(r.status_code, 400)

    def test_abrir_carpeta_no_proyecto_falla(self):
        not_a_project = self.root / "nope"
        not_a_project.mkdir()
        client = crop_app.app.test_client()
        r = client.post("/api/project/open", json={"path": str(not_a_project)})
        self.assertEqual(r.status_code, 404)
        self.assertIsNone(crop_app._ACTIVE_PROJECT)


class PipelineDatasetTests(unittest.TestCase):
    """Pipeline unificado: auto-export de crops al dataset MVTec del proyecto,
    clasificación de subidas (buenas/malas) y generados."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.generated = self.root / "generated"
        self.project = self.root / "proyecto"
        self.config_path = self.root / "config.json"
        self._prev_patches = [
            mock.patch.object(crop_app, "_ACTIVE_PROJECT", None),
            mock.patch.object(crop_app, "CONFIG_PATH", self.config_path),
            mock.patch.object(crop_app, "RECENT_PROJECTS_PATH",
                              self.root / "projects_recent.json"),
            mock.patch.object(crop_app, "GENERATED_ROOT", self.generated),
        ]
        for p in self._prev_patches:
            p.start()
        self.client = crop_app.app.test_client()

    def tearDown(self):
        for p in self._prev_patches:
            p.stop()
        self.tempdir.cleanup()

    def _make_project_with_crop(self, enable_labels=True):
        self.client.post("/api/project/create",
                         json={"name": "proyecto", "parent": str(self.root)})
        Image.new("RGB", (100, 80), "white").save(self.project / "images" / "img.png")
        if enable_labels:
            labels = self.project / "labels"
            labels.mkdir(exist_ok=True)
            (labels / "img.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
            # El proyecto con labels/ detecta anotaciones automáticamente (YOLO).
            self.client.post("/api/project/open", json={"path": str(self.project)})
        r = self.client.post("/api/crop", json={
            "filename": "img.png", "cx": 50, "cy": 40, "crop_size": 40,
        })
        self.assertEqual(r.status_code, 200, r.get_json())
        return r.get_json()

    def test_export_good_y_defect_crean_estructura_mvtec(self):
        rec = self._make_project_with_crop(enable_labels=True)
        self.assertEqual(rec["crop_name"], "img_crop_1.png")
        self.assertTrue(rec["mask_output_path"])
        # Con máscara → defect → auto-export a train/defective/clase/ + mask.
        self.assertEqual(rec["dataset_kind"], "defect")
        self.assertTrue((self.project / "train" / "defective" / "clase"
                         / "img_crop_1.png").is_file())
        self.assertTrue((self.project / "train" / "defective_masks" / "clase"
                         / "img_crop_1_mask.png").is_file())

        # Sin labels no hay máscara → good ya se comprueba en el otro test.
        self.assertTrue(rec["dataset_output_path"].startswith(str(self.project)))

    def test_export_defect_sin_mascara_se_omite(self):
        rec = self._make_project_with_crop(enable_labels=False)
        self.assertFalse(rec["mask_output_path"])
        # Sin máscara → good (no defect) → test/good/.
        self.assertEqual(rec["dataset_kind"], "good")
        self.assertTrue((self.project / "test" / "good"
                         / "img_crop_1.png").is_file())

    def test_export_requiere_proyecto_y_rechaza_crops_inexistentes(self):
        client = crop_app.app.test_client()
        self.assertEqual(client.post("/api/crop", json={
            "filename": "fake.png", "cx": 50, "cy": 40, "crop_size": 40,
        }).status_code, 403)
        r = client.get("/api/images")
        self.assertEqual(r.status_code, 403)

    def test_generated_runs_y_file_con_proteccion_de_traversal(self):
        run = self.generated / "run_01" / "hole"
        run.mkdir(parents=True)
        Image.new("RGB", (10, 10), "blue").save(run / "0000_generated.png")
        Image.new("RGB", (10, 10), "red").save(run / "0000_original.png")

        runs = self.client.get("/api/generated/runs").get_json()
        self.assertEqual(runs["exists"], True)
        self.assertEqual(runs["runs"][0]["run"], "run_01")
        self.assertEqual([f["name"] for f in runs["runs"][0]["files"]], ["0000_generated.png"])

        ok = self.client.get("/api/generated/file?path=run_01/hole/0000_generated.png")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.mimetype, "image/png")
        ok.close()  # libera el manejador de archivo en Windows

        bad = self.client.get("/api/generated/file?path=../fuera.png")
        self.assertIn(bad.status_code, (400, 404))

        secret = self.root / "secret.txt"
        secret.write_text("secreto", encoding="utf-8")
        bad2 = self.client.get("/api/generated/file?path=" + secret.name)
        self.assertIn(bad2.status_code, (400, 404))

    def test_upload_classifica_buenas_y_malas(self):
        """Subidas con dataset_kind van a images/ + test/good (buenas) o
        train/defective/<tipo> (malas), y /api/images las clasifica."""
        self.client.post("/api/project/create",
                         json={"name": "proyecto", "parent": str(self.root)})
        Image.new("RGB", (10, 10), "white").save(self.root / "buena.png")
        Image.new("RGB", (10, 10), "black").save(self.root / "mala.png")
        Image.new("RGB", (10, 10), "gray").save(self.root / "neutra.png")

        # Subida buena → images/ + test/good/
        with open(self.root / "buena.png", "rb") as fh:
            r = self.client.post("/api/project/upload-images",
                                 data={"files": (fh, "buena.png"), "dataset_kind": "good"},
                                 content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertTrue((self.project / "images" / "buena.png").exists())
        self.assertTrue((self.project / "test" / "good" / "buena.png").exists())

        # Subida mala → images/ + train/defective/clase/  (tipo activo por defecto)
        with open(self.root / "mala.png", "rb") as fh:
            r = self.client.post("/api/project/upload-images",
                                 data={"files": (fh, "mala.png"), "dataset_kind": "defect"},
                                 content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertTrue((self.project / "train" / "defective" / "clase" / "mala.png").exists())

        # Sin clasificar → solo images/
        with open(self.root / "neutra.png", "rb") as fh:
            r = self.client.post("/api/project/upload-images",
                                 data={"files": (fh, "neutra.png")},
                                 content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertFalse((self.project / "test" / "good" / "neutra.png").exists())

        # /api/images clasifica cada imagen.
        imgs = self.client.get("/api/images").get_json()["images"]
        kinds = {im["filename"]: im["dataset_kind"] for im in imgs}
        self.assertEqual(kinds["buena.png"], "good")
        self.assertEqual(kinds["mala.png"], "defect")
        self.assertEqual(kinds["neutra.png"], "")


if __name__ == "__main__":
    unittest.main()

