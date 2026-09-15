/* Tab Fusionar (Individual) — proceso inverso: seleccionas una imagen
   generada por DefectFill, se previsualiza en pequeño y se fusiona automáticamente
   sobre su crop registrado, mostrando el resultado. */
window.HMI.merge = (() => {
  "use strict";

  const el = {
    runSelect: document.getElementById("mergeRunSelect"),
    genFileSelect: document.getElementById("mergeGenFileSelect"),
    previewWrap: document.getElementById("mergePreviewWrap"),
    previewImg: document.getElementById("mergePreviewImg"),
    previewNote: document.getElementById("mergePreviewNote"),

    meta: document.getElementById("mergeMeta"),
    sourceCrop: document.getElementById("mergeSourceCrop"),
    filename: document.getElementById("mergeFilename"),
    coords: document.getElementById("mergeCoords"),
    size: document.getElementById("mergeSize"),

    resultEmpty: document.getElementById("mergeResultEmpty"),
    resultWrap: document.getElementById("mergeResultWrap"),
    resultImg: document.getElementById("mergeResultImg"),
    resultActions: document.getElementById("mergeResultActions"),
    resultPath: document.getElementById("mergeResultPath"),
    downloadBtn: document.getElementById("mergeDownloadBtn"),
  };

  const state = {
    generatedRuns: [],
    crops: [],        // registro del proyecto (para meta del crop de origen)
    genBlobUrl: null,
    resultBlobUrl: null,
    busy: false,
  };

  const { bus } = window.HMI;

  // Parsea la respuesta como JSON de forma segura. Cuando el backend aborta
  // (p. ej. 403 "No hay proyecto activo") devuelve una página HTML de error,
  // que rompía res.json() con "Unexpected token '<'". Aquí se captura y se
  // devuelve un objeto con un mensaje legible.
  async function safeJson(res) {
    const text = await res.text();
    let data = null;
    try { data = JSON.parse(text); } catch (_) { data = null; }
    if (!data || typeof data !== "object") {
      data = { ok: false, error: `No se pudo fusionar (HTTP ${res.status}). ¿Hay un proyecto abierto?` };
    }
    return data;
  }

  function cacheBust(url) {
    if (!url) return url;
    const sep = url.includes("?") ? "&" : "?";
    return url + sep + "t=" + Date.now();
  }

  // ------------------------------------------------------------------ Result
  function clearResult() {
    el.resultEmpty.style.display = "flex";
    el.resultWrap.style.display = "none";
    el.resultActions.style.display = "none";
    el.resultImg.removeAttribute("src");
    el.resultPath.textContent = "";
    if (state.resultBlobUrl) { URL.revokeObjectURL(state.resultBlobUrl); state.resultBlobUrl = null; }
  }

  function showResult(data) {
    const url = cacheBust(window.HMI.api(data.merged_url));
    el.resultImg.removeAttribute("src");
    el.resultImg.onload = () => {
      el.resultEmpty.style.display = "none";
      el.resultWrap.style.display = "flex";
      el.resultActions.style.display = "flex";
      el.resultPath.textContent = data.merged_path;
      el.downloadBtn.href = url;
      el.downloadBtn.download = (data.merged_name || "").split("/").pop();
    };
    el.resultImg.onerror = () => {
      el.resultEmpty.style.display = "none";
      el.resultWrap.style.display = "none";
      el.resultActions.style.display = "flex";
      el.resultPath.textContent = data.merged_path;
      el.downloadBtn.href = url;
      el.downloadBtn.download = (data.merged_name || "").split("/").pop();
    };
    el.resultImg.src = url;
  }

  function clearPreview() {
    el.previewWrap.style.display = "none";
    el.previewImg.removeAttribute("src");
    el.previewNote.textContent = "";
    if (state.genBlobUrl) { URL.revokeObjectURL(state.genBlobUrl); state.genBlobUrl = null; }
  }

  function clearMeta() {
    el.meta.style.display = "none";
    el.sourceCrop.textContent = "—";
    el.filename.textContent = "—";
    el.coords.textContent = "—";
    el.size.textContent = "—";
  }

  // ------------------------------------------------------------------ Runs
  async function loadCrops() {
    try {
      const res = await fetch(window.HMI.api("/api/crops"));
      const data = await res.json();
      state.crops = data.crops || [];
    } catch (err) {
      bus.emit("status", "No se pudo listar los crops: " + err.message, true);
    }
  }

  async function loadGeneratedRuns() {
    try {
      const res = await fetch(window.HMI.api("/api/generated/runs"));
      const data = await res.json();
      state.generatedRuns = (data && data.runs) || [];
      el.runSelect.innerHTML = '<option value="">— cargar runs —</option>';
      if (!state.generatedRuns.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Sin runs de inferencia todavía";
        el.runSelect.appendChild(opt);
      }
      for (const r of state.generatedRuns) {
        const opt = document.createElement("option");
        opt.value = r.run;
        opt.textContent = `${r.run}  (${r.files.length} generadas)`;
        el.runSelect.appendChild(opt);
      }
      el.genFileSelect.innerHTML = '<option value="">— selecciona run —</option>';
      el.genFileSelect.disabled = true;
    } catch (err) {
      bus.emit("status", "No se pudieron cargar los runs: " + err.message, true);
    }
  }

  function onRunChange() {
    const run = el.runSelect.value;
    const r = state.generatedRuns.find((x) => x.run === run);
    el.genFileSelect.innerHTML = '<option value="">— selecciona imagen —</option>';
    el.genFileSelect.disabled = !r;
    for (const f of (r && r.files) || []) {
      const opt = document.createElement("option");
      opt.value = f.path;
      opt.textContent = f.name;
      el.genFileSelect.appendChild(opt);
    }
    if ((r && !r.files.length) || (!r && state.generatedRuns.length)) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Sin imágenes generadas en este run";
      el.genFileSelect.appendChild(opt);
    }
    clearPreview();
    clearMeta();
    clearResult();
  }

  // ------------------------------------------------------------- Selección
  function findGeneration(path) {
    for (const r of state.generatedRuns) {
      const f = (r.files || []).find((x) => x.path === path);
      if (f) return { run: r.run, file: f };
    }
    return null;
  }

  async function onFileChange() {
    const path = el.genFileSelect.value;
    clearPreview();
    clearMeta();
    clearResult();
    if (!path) return;

    // Previsualizar la generada en pequeño
    const gen = findGeneration(path);
    const pngUrl = window.HMI.api("/api/generated/file?path=") + encodeURIComponent(path);
    if (state.genBlobUrl) URL.revokeObjectURL(state.genBlobUrl);
    try {
      const pr = await fetch(pngUrl);
      if (!pr.ok) { bus.emit("status", "No se pudo leer la imagen generada.", true); return; }
      const blob = await pr.blob();
      state.genBlobUrl = URL.createObjectURL(blob);
      el.previewImg.src = state.genBlobUrl;
      el.previewWrap.style.display = "flex";
      el.previewNote.textContent = gen && gen.file.source
        ? `Crop de origen: ${gen.file.source}`
        : "Documenta el crop de origen si se registró.";
    } catch (_) { bus.emit("status", "No se pudo leer la imagen generada.", true); }

    // Rellenar meta con el crop de origen registrado
    const rec = (gen && gen.file.source)
      ? state.crops.find((c) => c.crop_name === gen.file.source || c.filename === gen.file.source) || null
      : null;
    renderMeta(rec);
    if (rec) el.sourceCrop.textContent = rec.crop_name;

    // Fusión automática
    await doMerge(path);
  }

  function renderMeta(rec) {
    if (!rec) { el.meta.style.display = "none"; return; }
    el.meta.style.display = "block";
    el.filename.textContent = rec.filename;
    el.coords.textContent = `(${rec.x1}, ${rec.y1}) → (${rec.x2}, ${rec.y2})`;
    el.size.textContent = `${rec.crop_size}×${rec.crop_size}`;
  }

  async function doMerge(path) {
    if (state.busy) return;
    state.busy = true;
    bus.emit("status", "Fusionando…");
    try {
      const res = await fetch(window.HMI.api("/api/merge/generated"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: path }),
      });
      const data = await safeJson(res);
      if (!res.ok || !data.ok) {
        bus.emit("status", (data && data.error) || (data && data.message) || `No se pudo fusionar (HTTP ${res.status}).`, true);
        return;
      }
      showResult(data);
      bus.emit("status", `Fusionado: ${data.merged_path}`);
      bus.emit("merge-done", data);
    } catch (err) {
      bus.emit("status", err.message, true);
    } finally {
      state.busy = false;
    }
  }

  function init() {
    el.runSelect.addEventListener("change", onRunChange);
    el.genFileSelect.addEventListener("change", onFileChange);
    bus.on("crop-saved", loadCrops);
    loadGeneratedRuns();
  }

  return { init, onFileChange };
})();