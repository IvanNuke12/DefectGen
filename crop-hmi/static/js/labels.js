/* Tab Etiquetado — visualización y edición interactiva de la máscara.

   La máscara editable se mantiene en un canvas a resolución natural
   (maskCanvas). Sobre la imagen se muestra un canvas (labelsEditCanvas) que
   pinta esa máscara en rojo translúcido, escalada al tamaño visible. Los
   trazos del pincel/goma/rectángulo se dibujan en resolución natural y se
   guardan con POST /api/mask/save (PNG base64). */
window.HMI.labels = (() => {
  "use strict";

  const el = {
    filename: document.getElementById("labelsFilename"),
    stageEmpty: document.getElementById("labelsStageEmpty"),
    imageWrap: document.getElementById("labelsImageWrap"),
    image: document.getElementById("labelsImage"),
    prevCrop: document.getElementById("labelsPrevCrop"),
    maskOverlay: document.getElementById("labelsMaskOverlay"),
    editCanvas: document.getElementById("labelsEditCanvas"),
    toolbar: document.getElementById("maskToolbar"),
    brushSize: document.getElementById("brushSizeSlider"),
    brushSizeValue: document.getElementById("brushSizeValue"),
    saveBtn: document.getElementById("saveMaskBtn"),
    resetBtn: document.getElementById("resetMaskBtn"),
    status: document.getElementById("maskEditorStatus"),
  };

  const state = {
    annotationsEnabled: false,
    overlayEnabled: true,
    opacity: 35,
    current: null,
    tool: "brush",
    brushSize: 20,
    natural: null,           // {w, h} resolución natural de la imagen actual
    maskCanvas: null,        // canvas a resolución natural con la máscara
    drawing: false,
    lastPos: null,           // {x, y} en coords naturales (para trazos continuos)
    rectStart: null,         // {x, y} display — esquina inicial del rectángulo
    rectCurrent: null,       // {x, y} display — esquina actual durante el arrastre
    cursor: null,            // {x, y} display — posición para el anillo del pincel
    saveInFlight: false,
  };

  const { bus } = window.HMI;

  let lastStageSize = { w: 0, h: 0 };

  function updateStageCache() {
    const stage = el.image.closest(".viewer-stage");
    if (!stage) return;
    const r = stage.getBoundingClientRect();
    if (r.width > 0) lastStageSize = { w: r.width, h: r.height };
  }

  function setWrapFromMeta(im) {
    if (!im || !im.width || !im.height) return;
    // Mide el stage ahora (si está visible) antes de fijar el wrap; de lo
    // contrario lastStageSize arranca en {0,0} y el wrap se fija a 10px,
    // mostrando la imagen diminuta la primera vez que se abre el proyecto.
    updateStageCache();
    const availW = Math.max(10, lastStageSize.w - 56);
    const maxH = Math.max(10, Math.min(lastStageSize.h, window.innerHeight * 0.7));
    let w = Math.min(im.width, availW, 1000);
    let h = w * (im.height / im.width);
    if (h > maxH) {
      h = maxH;
      w = h * (im.width / im.height);
    }
    el.imageWrap.style.width = Math.max(1, Math.round(w)) + "px";
    el.imageWrap.style.height = Math.max(1, Math.round(h)) + "px";
  }

  function syncWrapSize() {
    if (el.imageWrap.style.display === "none") return;
    updateStageCache();
    const rect = el.image.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      el.imageWrap.style.width = rect.width + "px";
      el.imageWrap.style.height = rect.height + "px";
    }
    syncCanvasBuffer();
  }

  function syncCanvasBuffer() {
    const rect = el.editCanvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      if (el.editCanvas.width !== w || el.editCanvas.height !== h) {
        el.editCanvas.width = w;
        el.editCanvas.height = h;
      }
    }
    renderView();
  }

  // Muestra el rectángulo discontínuo del recorte anterior (si existe),
  // escalado al tamaño visible de la imagen.
  function renderPrevCrop() {
    const box = state.current && state.current.crop_box;
    const rect = el.image.getBoundingClientRect();
    if (!el.prevCrop || !state.current || !box ||
        rect.width === 0 || rect.height === 0 || !state.current.width || !state.current.height) {
      if (el.prevCrop) el.prevCrop.style.display = "none";
      return;
    }
    const scaleX = rect.width / state.current.width;
    const scaleY = rect.height / state.current.height;
    el.prevCrop.style.display = "block";
    el.prevCrop.style.left = box.x1 * scaleX + "px";
    el.prevCrop.style.top = box.y1 * scaleY + "px";
    el.prevCrop.style.width = (box.x2 - box.x1) * scaleX + "px";
    el.prevCrop.style.height = (box.y2 - box.y1) * scaleY + "px";
  }

  // ---------------------------------------------------------------------
  // Máscara a resolución natural
  // ---------------------------------------------------------------------
  function setStatus(text, isError) {
    el.status.textContent = text;
    el.status.classList.toggle("error", Boolean(isError));
    el.status.classList.toggle("ok", !isError && text === "Máscara guardada.");
  }

  function initMaskCanvas(w, h) {
    const c = document.createElement("canvas");
    c.width = Math.max(1, w);
    c.height = Math.max(1, h);
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    return c;
  }

  // Convierte una imagen L (0/255) en un canvas con blanco opaco = defecto y
  // transparencia en el fondo (black -> alpha 0).
  function drawMaskFromL(img, naturalW, naturalH) {
    const c = state.maskCanvas;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(img, 0, 0, naturalW, naturalH);
    const data = ctx.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const lum = data.data[i];
      data.data[i] = data.data[i + 1] = data.data[i + 2] = 255;
      data.data[i + 3] = lum;
    }
    ctx.putImageData(data, 0, 0);
  }

  function loadEditableMask(im) {
    const expected = im.filename;
    if (!state.maskCanvas) return;
    fetch(window.HMI.api("/api/mask/raw/") + window.HMI.rel(im.filename) + "?t=" + Date.now())
      .then((res) => (res.ok ? res.blob() : Promise.reject(new Error("HTTP " + res.status))))
      .then((blob) => {
        if (state.current && state.current.filename !== expected) return;
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          if (state.current && state.current.filename !== expected) return;
          drawMaskFromL(img, state.natural.w, state.natural.h);
          URL.revokeObjectURL(url);
          renderView();
        };
        img.onerror = () => { URL.revokeObjectURL(url); };
        img.src = url;
      })
      .catch(() => {
        if (state.current && state.current.filename !== expected) return;
        state.maskCanvas.getContext("2d").clearRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
        renderView();
      });
  }

  // ---------------------------------------------------------------------
  // Render del canvas visible (rojo translúcido)
  // ---------------------------------------------------------------------
  function renderView() {
    const v = el.editCanvas;
    const vctx = v.getContext("2d");
    vctx.clearRect(0, 0, v.width, v.height);
    if (!state.maskCanvas || !state.natural) return;

    if (state.overlayEnabled) {
      const dispW = v.width;
      const dispH = v.height;
      vctx.globalCompositeOperation = "source-over";
      vctx.fillStyle = `rgba(239, 83, 80, ${state.opacity / 100})`;
      vctx.fillRect(0, 0, dispW, dispH);
      vctx.globalCompositeOperation = "destination-in";
      vctx.drawImage(state.maskCanvas, 0, 0, state.natural.w, state.natural.h, 0, 0, dispW, dispH);
      vctx.globalCompositeOperation = "source-over";
    }

    drawRectPreview(vctx);
    drawCursor(vctx);
  }

  function drawRectPreview(vctx) {
    if (!state.rectStart || !state.rectCurrent) return;
    const sx = state.rectStart;
    const ex = state.rectCurrent;
    vctx.save();
    vctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    vctx.lineWidth = 1.5;
    vctx.setLineDash([5, 4]);
    vctx.strokeRect(Math.min(sx.x, ex.x), Math.min(sx.y, ex.y),
      Math.abs(ex.x - sx.x), Math.abs(ex.y - sx.y));
    vctx.restore();
  }

  function drawCursor(vctx) {
    if (!state.cursor) return;
    // En modo rectángulo se usa la cruceta nativa del navegador (cursor:
    // crosshair); solo el pincel/goma dibujan el anillo de tamaño.
    if (state.tool === "rect") return;
    vctx.save();
    vctx.strokeStyle = "rgba(255,255,255,0.85)";
    vctx.fillStyle = "rgba(255,255,255,0.12)";
    vctx.lineWidth = 1.25;
    vctx.beginPath();
    vctx.arc(state.cursor.x, state.cursor.y, state.brushSize, 0, Math.PI * 2);
    vctx.fill();
    vctx.stroke();
    vctx.restore();
  }

  // ---------------------------------------------------------------------
  // Dibujo sobre la máscara natural
  // ---------------------------------------------------------------------
  function displayToNatural(clientX, clientY) {
    const r = el.editCanvas.getBoundingClientRect();
    return {
      x: ((clientX - r.left) * state.natural.w) / r.width,
      y: ((clientY - r.top) * state.natural.h) / r.height,
    };
  }

  function naturalBrushRadius() {
    const r = el.editCanvas.getBoundingClientRect();
    return state.brushSize * (state.natural.w / r.width);
  }

  function paintStroke(from, to) {
    const ctx = state.maskCanvas.getContext("2d");
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(1, naturalBrushRadius() * 2);
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    if (state.tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#fff";
    }
    ctx.stroke();
    ctx.restore();
  }

  function paintDot(x, y) {
    const ctx = state.maskCanvas.getContext("2d");
    ctx.save();
    ctx.globalCompositeOperation = state.tool === "eraser" ? "destination-out" : "source-over";
    ctx.fillStyle = state.tool === "eraser" ? "rgba(0,0,0,1)" : "#fff";
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1, naturalBrushRadius()), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function finishRect(startNat, endNat) {
    const ctx = state.maskCanvas.getContext("2d");
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "#fff";
    ctx.fillRect(
      Math.min(startNat.x, endNat.x), Math.min(startNat.y, endNat.y),
      Math.abs(endNat.x - startNat.x), Math.abs(endNat.y - startNat.y)
    );
    ctx.restore();
  }

  // ---------------------------------------------------------------------
  // Eventos del canvas
  // ---------------------------------------------------------------------
  function onPointerDown(e) {
    if (!state.maskCanvas || !state.natural) return;
    e.preventDefault();
    el.editCanvas.setPointerCapture(e.pointerId);
    const r = el.editCanvas.getBoundingClientRect();
    const disp = { x: (e.clientX - r.left) * (el.editCanvas.width / r.width), y: (e.clientY - r.top) * (el.editCanvas.height / r.height) };
    const nat = displayToNatural(e.clientX, e.clientY);

    if (state.tool === "rect") {
      state.rectStart = disp;
      state.rectCurrent = disp;
    } else {
      state.drawing = true;
      state.lastPos = nat;
      paintDot(nat.x, nat.y);
    }
    state.cursor = disp;
    renderView();
  }

  function onPointerMove(e) {
    if (!state.natural || !state.maskCanvas) return;
    const r = el.editCanvas.getBoundingClientRect();
    const disp = { x: (e.clientX - r.left) * (el.editCanvas.width / r.width), y: (e.clientY - r.top) * (el.editCanvas.height / r.height) };
    state.cursor = disp;

    if (state.rectStart) {
      state.rectCurrent = disp;
      renderView();
      return;
    }
    if (!state.drawing) { renderView(); return; }

    const nat = displayToNatural(e.clientX, e.clientY);
    if (state.lastPos) {
      paintStroke(state.lastPos, nat);
      state.lastPos = nat;
    } else {
      paintDot(nat.x, nat.y);
      state.lastPos = nat;
    }
    renderView();
  }

  function onPointerUp(e) {
    if (state.rectStart) {
      const endNat = displayToNatural(e.clientX, e.clientY);
      const startNat = displayToNaturalRectStart();
      finishRect(startNat, endNat);
      state.rectStart = null;
      state.rectCurrent = null;
      renderView();
    }
    state.drawing = false;
    state.lastPos = null;
    try { el.editCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
  }

  function displayToNaturalRectStart() {
    const r = el.editCanvas.getBoundingClientRect();
    const sx = state.rectStart;
    return {
      x: (sx.x * state.natural.w) / el.editCanvas.width,
      y: (sx.y * state.natural.h) / el.editCanvas.height,
    };
  }

  // ---------------------------------------------------------------------
  // Guardar / descartar
  // ---------------------------------------------------------------------
  function saveMask() {
    if (!state.current || !state.maskCanvas || state.saveInFlight) return;
    state.saveInFlight = true;
    el.saveBtn.disabled = true;
    setStatus("Guardando…");
    const dataUrl = state.maskCanvas.toDataURL("image/png");
    const payload = { filename: state.current.filename, mask: dataUrl.split(",")[1] };

    fetch(window.HMI.api("/api/mask/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
        return data;
      })
      .then((data) => {
        setStatus("Máscara guardada. Pulsa Recortar para generar el crop con esta máscara.");
      })
      .catch((err) => {
        setStatus("No se pudo guardar: " + err.message, true);
      })
      .finally(() => {
        state.saveInFlight = false;
        el.saveBtn.disabled = false;
      });
  }

  function resetMask() {
    if (!state.current) return;
    setStatus("Descartando cambios…");
    loadEditableMask(state.current);
    setStatus("Dibuja sobre la imagen para corregir la máscara.");
  }

  // ---------------------------------------------------------------------
  // Cambio de imagen
  // ---------------------------------------------------------------------
  function setImage(im) {
    state.current = im;
    state.rectStart = null;
    state.rectCurrent = null;
    state.drawing = false;
    state.lastPos = null;
    state.cursor = null;

    if (!im) {
      el.stageEmpty.style.display = "flex";
      el.stageEmpty.querySelector("p").textContent = "Sin imagen seleccionada.";
      el.stageEmpty.querySelector(".stage-empty-sub").textContent = "Pulsa una miniatura de la galería.";
      el.imageWrap.style.display = "none";
      el.filename.textContent = "—";
      el.maskOverlay.removeAttribute("src");
      el.maskOverlay.style.display = "none";
      el.editCanvas.style.display = "none";
      if (el.prevCrop) el.prevCrop.style.display = "none";
      state.natural = null;
      state.maskCanvas = null;
      return;
    }

    el.stageEmpty.style.display = "none";
    el.imageWrap.style.display = "inline-block";
    setWrapFromMeta(im);
    el.image.style.visibility = "hidden";
    el.maskOverlay.style.display = "none";
    el.filename.textContent = `${im.filename}  (${im.width}×${im.height})`;

    el.image.onload = () => {
      syncWrapSize();
      el.image.style.visibility = "visible";
      el.editCanvas.style.display = "block";
      state.natural = { w: im.width, h: im.height };
      state.maskCanvas = initMaskCanvas(im.width, im.height);
      syncCanvasBuffer();
      renderPrevCrop();
      loadEditableMask(im);
    };
    el.image.onerror = () => {
      el.imageWrap.style.display = "none";
      el.maskOverlay.style.display = "none";
      el.maskOverlay.removeAttribute("src");
      el.editCanvas.style.display = "none";
      el.stageEmpty.style.display = "flex";
      el.stageEmpty.querySelector("p").textContent = "No se pudo cargar la imagen.";
      el.stageEmpty.querySelector(".stage-empty-sub").textContent = "Revisa la conexión o recarga la página.";
      el.filename.textContent = `${im.filename}  (error de carga)`;
      state.natural = null;
      state.maskCanvas = null;
    };
    el.image.src = window.HMI.api("/api/image/") + window.HMI.rel(im.filename);
  }

  function applyConfig(cfg) {
    state.annotationsEnabled = Boolean(cfg.annotations_enabled);
    state.overlayEnabled = Boolean(cfg.overlay_enabled);
    state.opacity = Number(cfg.overlay_opacity ?? state.opacity);
    if (state.current) loadEditableMask(state.current);
    renderView();
  }

  // ---------------------------------------------------------------------
  // Toolbar
  // ---------------------------------------------------------------------
  function setTool(tool) {
    state.tool = tool;
    el.toolbar.querySelectorAll(".tool-btn").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.tool === tool);
    });
    setStatus("Dibuja sobre la imagen para corregir la máscara.");
  }

  function bindEvents() {
    el.editCanvas.addEventListener("pointerdown", onPointerDown);
    el.editCanvas.addEventListener("pointermove", onPointerMove);
    el.editCanvas.addEventListener("pointerup", onPointerUp);
    el.editCanvas.addEventListener("pointercancel", onPointerUp);
    el.editCanvas.addEventListener("pointerleave", () => {
      state.cursor = null;
      renderView();
    });

    el.toolbar.addEventListener("click", (e) => {
      const btn = e.target.closest(".tool-btn");
      if (btn) setTool(btn.dataset.tool);
    });

    el.brushSize.addEventListener("input", () => {
      state.brushSize = Number(el.brushSize.value);
      el.brushSizeValue.textContent = state.brushSize + "px";
      renderView();
    });

    el.saveBtn.addEventListener("click", saveMask);
    el.resetBtn.addEventListener("click", resetMask);
  }

  function init() {
    bindEvents();
    bus.on("image-selected", setImage);
    bus.on("labels-opacity", (v) => { state.opacity = Number(v); renderView(); });
    bus.on("crop-overlay-visible", (v) => { state.overlayEnabled = Boolean(v); renderView(); });
    window.addEventListener("resize", () => {
      if (el.imageWrap.style.display === "none") return;
      updateStageCache();
      const rect = el.image.getBoundingClientRect();
      if (rect.width > 0) {
        syncWrapSize();
      } else if (state.current) {
        setWrapFromMeta(state.current);
        syncCanvasBuffer();
      }
      renderPrevCrop();
    });
  }

  return { init, setImage, applyConfig, syncWrap: syncWrapSize };
})();
