/* Orquestador: proyecto activo, tabs, glue entre módulos. */
(() => {
  "use strict";

  const el = {
    app: document.getElementById("app"),
    tabsNav: document.getElementById("tabs"),

    projectNameRow: document.getElementById("projectNameRow"),
    projectInputRow: document.getElementById("projectInputRow"),
    folderWarning: document.getElementById("folderWarning"),
    importGoodInput: document.getElementById("importGoodInput"),
    importGoodDropzoneLabel: document.getElementById("importGoodDropzoneLabel"),
    importDefectInput: document.getElementById("importDefectInput"),
    importDefectDropzoneLabel: document.getElementById("importDefectDropzoneLabel"),
    defectTypeInput: document.getElementById("defectTypeInput"),
    defectTypePath: document.getElementById("defectTypePath"),
    dragOverlay: document.getElementById("dragOverlay"),
    deleteImageBtn: document.getElementById("deleteImageBtn"),

    annotationControls: document.getElementById("annotationControls"),
    annotationFormat: document.getElementById("annotationFormat"),
    annotationPath: document.getElementById("annotationPath"),
    browseAnnotationsFolderBtn: document.getElementById("browseAnnotationsFolderBtn"),
    labelsDropzone: document.getElementById("labelsDropzone"),
    labelsDropzoneLabel: document.getElementById("labelsDropzoneLabel"),
    labelsFileInput: document.getElementById("labelsFileInput"),
    annotationStatus: document.getElementById("annotationStatus"),
    annotationsEnabled: document.getElementById("annotationsEnabled"),
    applyLabelsBtn: document.getElementById("applyLabelsBtn"),
    overlayEnabled: document.getElementById("overlayEnabled"),
    opacitySlider: document.getElementById("opacitySlider"),
    opacityValue: document.getElementById("opacityValue"),

    exportCsvBtn: document.getElementById("exportCsvBtn"),
    exportMasksCocoBtn: document.getElementById("exportMasksCocoBtn"),
    exportMasksYoloBtn: document.getElementById("exportMasksYoloBtn"),
    statusMessage: document.getElementById("statusMessage"),
  };

  const state = {
    annotationsEnabled: false,
    annotationFormat: "auto",
    project: null,
  };

  const { bus, responseJson } = window.HMI;
  const setStatus = (msg, isError) => {
    el.statusMessage.textContent = msg;
    el.statusMessage.style.color = isError ? "var(--danger)" : "var(--text-faint)";
  };

  function setAnnotationStatus(message, kind) {
    el.annotationStatus.textContent = message;
    el.annotationStatus.classList.toggle("ok", kind === "ok");
    el.annotationStatus.classList.toggle("error", kind === "error");
  }

  function updateAnnotationControls() {
    const enabled = el.annotationsEnabled.checked;
    el.annotationControls.classList.toggle("disabled", !enabled);
    el.annotationControls.querySelectorAll("input, select, button").forEach((c) => { c.disabled = !enabled; });
    if (!enabled) setAnnotationStatus("Activa el etiquetado y selecciona las anotaciones.");
  }

  function updateOverlayStyle() {
    el.opacityValue.textContent = `${Number(el.opacitySlider.value)}%`;
    bus.emit("labels-opacity", Number(el.opacitySlider.value));
    bus.emit("crop-overlay-visible", el.overlayEnabled.checked);
  }

  async function openNativeDialog(mode, title, initialPath) {
    setStatus("Abriendo selector…");
    const selected = await window.HMI.pickPath(mode, title, initialPath);
    setStatus(selected ? "Ruta seleccionada." : "Selección cancelada.");
    return selected;
  }

  async function loadSession() {
    const r = await fetch(window.HMI.api("/api/session"));
    const d = await r.json();
    if (!d.has_project) {
      window.location.href = window.HMI.api("/");
      return;
    }
    state.project = d.project;
    if (d.project) {
      el.projectNameRow.textContent = d.project.name || "—";
      el.projectInputRow.textContent = d.project.input_folder || "—";
      el.projectInputRow.title = d.project.input_folder || "";
      el.defectTypeInput.value = d.project.active_defect_type || "";
      renderDefectType();
    }
  }

  function renderDefectType() {
    const tipo = el.defectTypeInput.value.trim() || "defect";
    el.defectTypePath.textContent = `train/defective/${tipo}`;
  }

  async function saveDefectType() {
    const value = el.defectTypeInput.value.trim();
    renderDefectType();
    try {
      const res = await fetch(window.HMI.api("/api/project/meta"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active_defect_type: value }),
      });
      const data = await responseJson(res);
      if (!res.ok) throw new Error(data.message || "No se pudo guardar el tipo de defecto.");
      setStatus(value
        ? `Tipo de defecto guardado: "${value}".`
        : "Tipo de defecto vacío (se usará \"defect\").");
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  async function loadConfig() {
    const res = await fetch(window.HMI.api("/api/config"));
    const cfg = await responseJson(res);
    if (!res.ok) throw new Error(cfg.message || "No se pudo leer la configuración.");
    state.annotationsEnabled = Boolean(cfg.annotations_enabled);
    state.annotationFormat = cfg.annotation_format || "auto";
    el.annotationsEnabled.checked = state.annotationsEnabled;
    el.annotationPath.value = cfg.annotation_path || "";
    el.annotationFormat.value = state.annotationFormat;
    el.overlayEnabled.checked = Boolean(cfg.overlay_enabled);
    el.opacitySlider.value = String(cfg.overlay_opacity ?? 35);
    updateAnnotationControls();
    applyConfigToTabs(cfg);
    updateOverlayStyle();
    return cfg;
  }

  function applyConfigToTabs(cfg) {
    window.HMI.crop.applyConfig(cfg);
    window.HMI.labels.applyConfig(cfg);
  }

  function postConfig(partial) {
    return fetch(window.HMI.api("/api/config"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(partial),
    });
  }

  async function applyLabelsConfig() {
    setStatus("Aplicando etiquetado…");
    try {
      const res = await postConfig({
        annotations_enabled: el.annotationsEnabled.checked,
        annotation_format: el.annotationFormat.value,
        annotation_path: el.annotationPath.value,
        overlay_enabled: el.overlayEnabled.checked,
        overlay_opacity: Number(el.opacitySlider.value),
      });
      if (!res.ok) throw new Error("No se pudo guardar la configuración");
      const cfg = await res.json();
      state.annotationsEnabled = Boolean(cfg.annotations_enabled);
      state.annotationFormat = cfg.annotation_format;
      applyConfigToTabs(cfg);
      await updateOverlayStyle();
      await refreshAnnotationSummary();
      await loadImages();
      setStatus("Etiquetado aplicado.");
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  async function refreshAnnotationSummary() {
    if (!state.annotationsEnabled) {
      setAnnotationStatus("Etiquetado desactivado.");
      return;
    }
    try {
      const res = await fetch(window.HMI.api("/api/annotations/summary"));
      const data = await responseJson(res);
      if (!res.ok) throw new Error(data.message || "No se pudieron validar las anotaciones.");
      const formatName = String(data.format || "").toUpperCase();
      setAnnotationStatus(
        `${formatName}: ${data.image_count} imágenes y ${data.annotation_count} anotaciones.`,
        "ok"
      );
    } catch (err) {
      setAnnotationStatus(err.message, "error");
    }
  }

  async function loadImages() {
    setStatus("Cargando imágenes…");
    try {
      const res = await fetch(window.HMI.api("/api/images"));
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        el.folderWarning.style.display = "block";
        el.folderWarning.textContent = `Error ${res.status} al pedir imágenes${txt ? ": " + txt.slice(0, 200) : ""}`;
        window.HMI.crop.setImages([]);
        setStatus(`Error ${res.status} al cargar imágenes.`, true);
        return;
      }
      const data = await res.json();
      if (!data.folder_exists) {
        el.folderWarning.style.display = "block";
        el.folderWarning.textContent = "La carpeta de imágenes del proyecto no existe o está vacía: " + data.input_folder;
      } else {
        el.folderWarning.style.display = "none";
      }
      const imgs = Array.isArray(data.images) ? data.images : [];
      window.HMI.crop.setImages(imgs);
      if (!data.folder_exists || imgs.length === 0) {
        setStatus(data.folder_exists ? `No se encontraron imágenes (carpeta: ${data.input_folder}).` : "La carpeta de imágenes del proyecto no existe o está vacía.", true);
        return;
      }
      setStatus(`Listo · ${imgs.length} imágenes.`);
    } catch (err) {
      setStatus("Error: " + err.message, true);
    }
  }

  function switchTab(tab) {
    document.querySelectorAll(".tab").forEach((n) => n.classList.toggle("active", n.dataset.tab === tab));
    document.querySelectorAll(".tab-pane").forEach((n) => n.classList.toggle("active", n.dataset.tab === tab));
    document.querySelectorAll(".params-pane").forEach((n) => n.classList.toggle("active", n.dataset.tab === tab));
    if (tab === "labels" && window.HMI.crop.currentImageMeta()) {
      window.HMI.labels.setImage(window.HMI.crop.currentImageMeta());
    }
    // El panel ya es visible: re-sincroniza el wrap por si se redimensionó o
    // cambió de imagen con el panel oculto (allí no hay layout que medir).
    if (window.HMI.crop.syncWrap) window.HMI.crop.syncWrap();
    if (window.HMI.labels.syncWrap) window.HMI.labels.syncWrap();
  }

  // Subida de imágenes por dropzone (arrastrar o clic) al proyecto activo.
  // `datasetKind`: "" (solo images/), "good" (→ images/ + test/good) o
  // "defect" (→ images/ + train/defective/<tipo>).
  async function uploadImages(fileList, datasetKind) {
    const files = Array.from(fileList || []).filter((f) => /^image\//.test(f.type));
    if (!files.length) {
      const hasJson = Array.from(fileList || []).some((f) => /\.(json|txt)$/i.test(f.name));
      setStatus(hasJson
        ? "Suelta los JSON/TXT en el recuadro de labels o anotaciones."
        : "No hay archivos de imagen válidos.", true);
      return;
    }
    setStatus(`Subiendo ${files.length} imagen(es)…`);
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    if (datasetKind) fd.append("dataset_kind", datasetKind);
    if (datasetKind === "defect") {
      const tipo = el.defectTypeInput.value.trim();
      if (tipo) fd.append("defect_type", tipo);
    }
    try {
      const res = await fetch(window.HMI.api("/api/project/upload-images"), {
        method: "POST",
        body: fd,
      });
      const data = await responseJson(res);
      if (!res.ok) throw new Error(data.message || "No se pudo subir.");
      state.project = data.project;
      if (data.project) {
        el.projectInputRow.textContent = data.project.input_folder || "—";
        el.projectInputRow.title = data.project.input_folder || "";
      }
      await loadImages();
      const extra = data.errors && data.errors.length ? ` · ${data.errors.length} error(es)` : "";
      const label = datasetKind === "good" ? "buenas" : datasetKind === "defect" ? "defectuosas" : "";
      setStatus(`Subidas ${data.copied} imagen(es)${label ? " (" + label + ")" : ""}.` + extra);
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  // Subida de labels (JSON/TXT) al proyecto — dropzone del panel Etiquetado.
  async function uploadLabels(fileList) {
    const files = Array.from(fileList || []).filter((f) => /\.(json|txt)$/i.test(f.name));
    if (!files.length) {
      setStatus("Solo se aceptan archivos .json o .txt.", true);
      return;
    }
    setStatus(`Subiendo ${files.length} label(s)…`);
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    try {
      const res = await fetch(window.HMI.api("/api/project/upload-labels"), {
        method: "POST",
        body: fd,
      });
      const data = await responseJson(res);
      if (!res.ok) throw new Error(data.message || "No se pudo subir.");
      const cfg = await loadConfig();
      await refreshAnnotationSummary();
      if (cfg.annotations_enabled) await loadImages();
      const extra = data.errors && data.errors.length ? ` · ${data.errors.length} error(es)` : "";
      setStatus(`Subidos ${data.copied} label(s).` + extra);
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  // Subida de labels o JSON COCO desde un único panel (dropzone compartido).
  // Un único archivo .json se trata como anotaciones COCO; el resto
  // (varios .json, o .txt) se envía como labels.
  function handleLabelsOrCoco(fileList) {
    const files = Array.from(fileList || []);
    const singleJson = files.length === 1 && /\.json$/i.test(files[0].name);
    if (singleJson) {
      uploadCoco(files[0]);
    } else {
      uploadLabels(files);
    }
  }

  // Subida de un JSON COCO como anotaciones del proyecto.
  async function uploadCoco(file) {
    if (!file) return;
    if (!/\.json$/i.test(file.name)) {
      setStatus("Solo se aceptan archivos JSON.", true);
      return;
    }
    setStatus("Subiendo JSON COCO…");
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const res = await fetch(window.HMI.api("/api/project/upload-coco"), {
        method: "POST",
        body: fd,
      });
      const data = await responseJson(res);
      if (!res.ok) throw new Error(data.message || "No se pudo subir.");
      el.annotationPath.value = data.path;
      el.annotationFormat.value = "coco";
      el.annotationsEnabled.checked = true;
      updateAnnotationControls();
      setAnnotationStatus("JSON COCO subido. Pulsa Aplicar y recargar.", "ok");
    } catch (err) {
      setAnnotationStatus(err.message, "error");
    }
  }

  function isFileDrag(e) {
    return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  }

  // Permite arrastrar y soltar directamente sobre un dropzone concreto.
  function bindDropzone(dz, onFiles) {
    if (!dz) return;
    dz.addEventListener("dragover", (e) => {
      if (isFileDrag(e)) {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.add("dragover");
      }
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      e.stopPropagation();
      dz.classList.remove("dragover");
      onFiles(e.dataTransfer.files);
    });
  }

  function initDragAndDrop() {
    let dragDepth = 0;
    document.addEventListener("dragenter", (e) => {
      if (!isFileDrag(e) || e.target.closest(".picker-overlay")) return;
      dragDepth += 1;
      el.dragOverlay.style.display = "flex";
    });
    document.addEventListener("dragleave", (e) => {
      if (!isFileDrag(e)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) el.dragOverlay.style.display = "none";
    });
    document.addEventListener("dragover", (e) => { if (isFileDrag(e)) e.preventDefault(); });
    document.addEventListener("drop", (e) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      dragDepth = 0;
      el.dragOverlay.style.display = "none";
      if (e.target.closest(".picker-overlay")) return;
      // Soltar sobre un dropzone concreto (buenas/malas) respeta su clase.
      const dzLabel = e.target.closest("label.dropzone");
      if (dzLabel && dzLabel.htmlFor === "importGoodInput") {
        uploadImages(e.dataTransfer.files, "good");
        return;
      }
      if (dzLabel && dzLabel.htmlFor === "importDefectInput") {
        uploadImages(e.dataTransfer.files, "defect");
        return;
      }
      uploadImages(e.dataTransfer.files);
    });
  }

  async function deleteCurrentImage() {
    const im = window.HMI.crop.currentImageMeta();
    if (!im) {
      setStatus("No hay imagen seleccionada.", true);
      return;
    }
    if (!window.confirm(`¿Borrar "${im.filename}" y sus recortes?`)) return;
    setStatus("Borrando imagen…");
    try {
      const res = await fetch(window.HMI.api("/api/project/delete-image"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: im.filename }),
      });
      const data = await responseJson(res);
      if (!res.ok) throw new Error(data.message || "No se pudo borrar la imagen.");
      setStatus(`Borrada ${data.filename}.`);
      await loadImages();
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  async function init() {
    el.tabsNav.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab");
      if (btn && btn.dataset.tab) switchTab(btn.dataset.tab);
    });
    el.importGoodInput.addEventListener("change", () => {
      if (el.importGoodInput.files.length) {
        uploadImages(el.importGoodInput.files, "good");
        el.importGoodInput.value = "";
      }
    });
    el.importDefectInput.addEventListener("change", () => {
      if (el.importDefectInput.files.length) {
        uploadImages(el.importDefectInput.files, "defect");
        el.importDefectInput.value = "";
      }
    });
    el.defectTypeInput.addEventListener("change", saveDefectType);
    el.deleteImageBtn.addEventListener("click", deleteCurrentImage);
    initDragAndDrop();

    el.applyLabelsBtn.addEventListener("click", applyLabelsConfig);

    el.labelsFileInput.addEventListener("change", () => {
      const n = el.labelsFileInput.files.length;
      if (n) {
        handleLabelsOrCoco(el.labelsFileInput.files);
        el.labelsDropzoneLabel.textContent = `${n} archivo(s) seleccionados`;
        el.labelsFileInput.value = "";
      }
    });
    bindDropzone(el.labelsDropzone, (files) => {
      handleLabelsOrCoco(files);
      el.labelsDropzoneLabel.textContent = `${files.length} archivo(s) seleccionados`;
    });

    el.annotationsEnabled.addEventListener("change", () => {
      updateAnnotationControls();
      if (el.annotationsEnabled.checked) {
        setAnnotationStatus("Selecciona las anotaciones y pulsa Aplicar y recargar.");
      }
    });

    el.browseAnnotationsFolderBtn.addEventListener("click", async () => {
      try {
        const s = await openNativeDialog("folder", "Seleccionar carpeta de anotaciones", el.annotationPath.value);
        if (s) { el.annotationPath.value = s; setAnnotationStatus("Ruta preparada. Pulsa Aplicar y recargar."); }
      } catch (err) { setAnnotationStatus(err.message, "error"); }
    });

    el.annotationFormat.addEventListener("change", () => {
      setAnnotationStatus("Formato cambiado. Pulsa Aplicar y recargar.");
    });

    el.overlayEnabled.addEventListener("change", updateOverlayStyle);
    el.opacitySlider.addEventListener("input", updateOverlayStyle);

    el.exportCsvBtn.addEventListener("click", () => { window.location.href = window.HMI.api("/api/export/csv"); });

    el.exportMasksCocoBtn.addEventListener("click", () => { window.location.href = window.HMI.api("/api/export/masks?format=coco"); });
    el.exportMasksYoloBtn.addEventListener("click", () => { window.location.href = window.HMI.api("/api/export/masks?format=yolo"); });

    bus.on("status", setStatus);

    window.HMI.crop.init();
    window.HMI.labels.init();

    (async () => {
      await loadSession();
      await loadConfig();
      await loadImages();
      await refreshAnnotationSummary();
    })();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
