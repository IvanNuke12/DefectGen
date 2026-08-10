/* Tab Recortar — galería + viewer con retícula + crop. */
window.HMI.crop = (() => {
  "use strict";

  const el = {
    searchInput: document.getElementById("searchInput"),
    galleryList: document.getElementById("galleryList"),
    galleryEmpty: document.getElementById("galleryEmpty"),

    stageEmpty: document.getElementById("stageEmpty"),
    imageWrap: document.getElementById("imageWrap"),
    mainImage: document.getElementById("mainImage"),
    prevCrop: document.getElementById("prevCrop"),
    reticle: document.getElementById("reticle"),
    hud: document.getElementById("hud"),
    savedFlash: document.getElementById("savedFlash"),
    currentFilename: document.getElementById("currentFilename"),

    progressFill: document.getElementById("progressFill"),
    progressLabel: document.getElementById("progressLabel"),
    totalCrops: document.getElementById("totalCrops"),

    cropSlider: document.getElementById("cropSlider"),
    cropSizeInput: document.getElementById("cropSizeInput"),
  };

  const state = {
    images: [],
    filterText: "",
    currentIndex: -1,
    cropSize: 250,
    lastNatural: null,
    busy: false,
  };

  const { bus, clampBox } = window.HMI;

  function currentImageMeta() { return state.images[state.currentIndex] || null; }

  function renderGallery() {
    const filter = state.filterText.toLowerCase();
    el.galleryList.innerHTML = "";
    const filtered = state.images
      .map((im, idx) => ({ im, idx }))
      .filter(({ im }) => im.filename.toLowerCase().includes(filter));

    if (filtered.length === 0) {
      const empty = document.createElement("div");
      empty.className = "gallery-empty";
      empty.textContent = state.images.length === 0 ? "Sin imágenes." : "Sin coincidencias.";
      el.galleryList.appendChild(empty);
      return;
    }

    // Separa buenas y malas con una línea: primero las defectuosas, luego un
    // separador, y después las buenas. Las sin clasificar quedan al final.
    const order = { defect: 0, good: 1, "": 2 };
    const sorted = filtered
      .slice()
      .sort((a, b) => (order[a.im.dataset_kind] ?? 2) - (order[b.im.dataset_kind] ?? 2) || a.idx - b.idx);

    let lastKind = null;
    for (const { im, idx } of sorted) {
      const kind = im.dataset_kind || "";
      if (lastKind !== null && kind !== lastKind) {
        const divider = document.createElement("div");
        divider.className = "gallery-divider";
        divider.textContent = kind === "good" ? "Buenas" : kind === "defect" ? "Malas" : "Sin clasificar";
        el.galleryList.appendChild(divider);
      }
      lastKind = kind;

      const item = document.createElement("div");
      item.className = "gallery-item" + (idx === state.currentIndex ? " active" : "") + (kind ? " kind-" + kind : "");
      item.dataset.idx = String(idx);

      const thumb = document.createElement("img");
      thumb.className = "gallery-thumb";
      thumb.loading = "lazy";
      thumb.onerror = () => { thumb.style.visibility = "hidden"; };
      thumb.src = window.HMI.api("/api/thumbnail/") + window.HMI.rel(im.filename);
      thumb.alt = "";

      const info = document.createElement("div");
      info.className = "gallery-info";

      const dot = document.createElement("span");
      dot.className = "gallery-dot" + (im.done ? " done" : "");

      const name = document.createElement("div");
      name.className = "gallery-name mono";
      name.textContent = im.filename;

      const status = document.createElement("div");
      status.className = "gallery-status" + (im.done ? " done" : "");
      status.textContent = im.done ? "recortada" : (kind === "good" ? "buena" : kind === "defect" ? "mala" : "pendiente");

      info.appendChild(name);
      info.appendChild(status);
      item.appendChild(dot);
      item.appendChild(thumb);
      item.appendChild(info);

      item.addEventListener("click", () => selectImage(idx));
      el.galleryList.appendChild(item);
    }
  }

  function updateGalleryItemDom(idx) {
    const item = el.galleryList.querySelector(`.gallery-item[data-idx="${idx}"]`);
    if (!item) return;
    const im = state.images[idx];
    item.querySelector(".gallery-dot").classList.toggle("done", im.done);
    const status = item.querySelector(".gallery-status");
    status.classList.toggle("done", im.done);
    status.textContent = im.done ? "recortada" : "pendiente";
  }

  function highlightActiveGalleryItem() {
    el.galleryList.querySelectorAll(".gallery-item").forEach((node) => {
      node.classList.toggle("active", Number(node.dataset.idx) === state.currentIndex);
    });
    const active = el.galleryList.querySelector(".gallery-item.active");
    if (active) active.scrollIntoView({ block: "nearest" });
  }

  function updateProgress() {
    const total = state.images.length;
    const done = state.images.filter((im) => im.done).length;
    const totalCrops = state.images.reduce((sum, im) => sum + (im.crop_count || 0), 0);
    el.progressLabel.textContent = `${done} / ${total}`;
    el.progressFill.style.width = total ? `${(done / total) * 100}%` : "0%";
    el.totalCrops.textContent = String(totalCrops);
  }

  function showEmptyStage() {
    el.stageEmpty.style.display = "flex";
    el.imageWrap.style.display = "none";
    el.reticle.style.display = "none";
    el.hud.style.display = "none";
    if (el.prevCrop) el.prevCrop.style.display = "none";
    el.currentFilename.textContent = "—";
    el.stageEmpty.querySelector("p").textContent = "Sin imágenes cargadas todavía.";
    el.stageEmpty.querySelector(".stage-empty-sub").textContent =
      "Asegúrate de que el proyecto tenga imágenes en su carpeta images/.";
  }

  function selectImage(idx) {
    if (idx < 0 || idx >= state.images.length) return;
    state.currentIndex = idx;
    const im = state.images[idx];

    el.stageEmpty.style.display = "none";
    el.imageWrap.style.display = "inline-block";
    // Fija el wrap de inmediato con los metadatos (funciona aunque el panel
    // esté oculto, donde no hay layout que medir) y oculta la imagen anterior
    // para que no se vea estirada/cortada durante la carga.
    setWrapFromMeta(im);
    el.mainImage.style.visibility = "hidden";
    el.currentFilename.textContent = `${im.filename}  (${im.width}×${im.height})`;

    el.reticle.style.display = "none";
    el.hud.style.display = "none";

    el.mainImage.onload = () => {
      syncWrapSize();
      el.mainImage.style.visibility = "visible";
      renderPrevCrop();
      const maxSlider = Math.max(1, Math.min(im.width, im.height));
      el.cropSlider.max = String(maxSlider);
      el.cropSizeInput.max = String(maxSlider);
      if (state.cropSize > maxSlider) {
        state.cropSize = maxSlider;
        el.cropSlider.value = maxSlider;
        el.cropSizeInput.value = maxSlider;
      }
    };
    el.mainImage.onerror = () => {
      // Estado limpio si la imagen no se pudo cargar (evita el icono de imagen
      // rota sobre una banda oscura en el visor).
      el.imageWrap.style.display = "none";
    el.reticle.style.display = "none";
    el.hud.style.display = "none";
    if (el.prevCrop) el.prevCrop.style.display = "none";
      el.stageEmpty.style.display = "flex";
      el.stageEmpty.querySelector("p").textContent = "No se pudo cargar la imagen.";
      el.stageEmpty.querySelector(".stage-empty-sub").textContent = "Revisa la conexión o recarga la página.";
      el.currentFilename.textContent = `${im.filename}  (error de carga)`;
      bus.emit("status", `No se pudo cargar ${im.filename}`, true);
    };
    // La imagen base (images/) no cambia al cropear, así que permitimos la
    // caché del navegador (URL estable) para que pasar a la siguiente imagen
    // tras un crop sea inmediato en vez de re-descargar la foto cada vez.
    el.mainImage.src = window.HMI.api("/api/image/") + window.HMI.rel(im.filename);

    highlightActiveGalleryItem();
    bus.emit("image-selected", im);
  }

  function nextImage() {
    if (state.currentIndex < state.images.length - 1) selectImage(state.currentIndex + 1);
    else bus.emit("status", "No hay más imágenes.");
  }
  function prevImage() {
    if (state.currentIndex > 0) selectImage(state.currentIndex - 1);
    else bus.emit("status", "Esta es la primera imagen.");
  }

  function displayRectOfImage() { return el.mainImage.getBoundingClientRect(); }

  // Tamaño del stage (caché): el panel puede estar oculto (display:none) al
  // cambiar de imagen, momento en que el stage mide 0 y no hay layout que medir.
  let lastStageSize = { w: 0, h: 0 };

  function updateStageCache() {
    const stage = el.mainImage.closest(".viewer-stage");
    if (!stage) return;
    const r = stage.getBoundingClientRect();
    if (r.width > 0) lastStageSize = { w: r.width, h: r.height };
  }

  // Reproduce el cálculo CSS del img (max-width: min(100%, 1000px);
  // max-height: 70vh) para fijar el wrap de inmediato con los metadatos,
  // sin esperar a la carga (el wrap inline-block no respeta el max-width
  // porcentual de su contenido: quedaría 88px más ancho que la imagen).
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

  // Ajuste fino con el tamaño real ya medido (imagen visible).
  function syncWrapSize() {
    if (el.imageWrap.style.display === "none") return;
    updateStageCache();
    const rect = el.mainImage.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      el.imageWrap.style.width = rect.width + "px";
      el.imageWrap.style.height = rect.height + "px";
    }
  }

  // Muestra el rectángulo discontínuo del recorte anterior (si existe),
  // escalado al tamaño visible de la imagen, para ayudar a recortar de nuevo.
  function renderPrevCrop() {
    const im = currentImageMeta();
    if (!el.prevCrop || !im) return;
    const box = im.crop_box;
    const rect = displayRectOfImage();
    if (!box || rect.width === 0 || rect.height === 0 || !im.width || !im.height) {
      el.prevCrop.style.display = "none";
      return;
    }
    const scaleX = rect.width / im.width;
    const scaleY = rect.height / im.height;
    el.prevCrop.style.display = "block";
    el.prevCrop.style.left = box.x1 * scaleX + "px";
    el.prevCrop.style.top = box.y1 * scaleY + "px";
    el.prevCrop.style.width = (box.x2 - box.x1) * scaleX + "px";
    el.prevCrop.style.height = (box.y2 - box.y1) * scaleY + "px";
  }

  function renderReticleAtNatural(nx, ny) {
    const im = currentImageMeta();
    if (!im) return;
    const rect = displayRectOfImage();
    if (rect.width === 0 || rect.height === 0) return;

    const box = clampBox(nx, ny, state.cropSize, im.width, im.height);
    const scaleX = rect.width / im.width;
    const scaleY = rect.height / im.height;

    el.reticle.style.display = "block";
    el.reticle.style.left = box.x1 * scaleX + "px";
    el.reticle.style.top = box.y1 * scaleY + "px";
    el.reticle.style.width = box.cs * scaleX + "px";
    el.reticle.style.height = box.cs * scaleY + "px";

    el.hud.style.display = "block";
    el.hud.style.left = (box.x1 + box.cs / 2) * scaleX + "px";
    el.hud.style.top = box.y1 * scaleY + "px";
    el.hud.textContent = `${box.x1},${box.y1} · ${box.cs}×${box.cs}px`;

    state.lastNatural = { nx, ny };
    return box;
  }

  function flashSaved() {
    el.savedFlash.classList.add("show");
    window.clearTimeout(flashSaved._t);
    flashSaved._t = window.setTimeout(() => el.savedFlash.classList.remove("show"), 700);
  }

  function setImages(images, preferFirstNotDone = true) {
    state.images = images || [];
    renderGallery();
    updateProgress();
    if (state.images.length === 0) {
      state.currentIndex = -1;
      showEmptyStage();
      return;
    }
    let startIdx = preferFirstNotDone ? state.images.findIndex((im) => !im.done) : 0;
    if (startIdx === -1) startIdx = 0;
    selectImage(startIdx);
  }

  function init() {
    el.mainImage.addEventListener("mousemove", (e) => {
      const im = currentImageMeta();
      if (!im) return;
      const rect = displayRectOfImage();
      const dx = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const dy = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
      const nx = (dx / rect.width) * im.width;
      const ny = (dy / rect.height) * im.height;
      renderReticleAtNatural(nx, ny);
    });

    window.addEventListener("resize", () => {
      if (el.imageWrap.style.display === "none") return;
      updateStageCache();
      const rect = el.mainImage.getBoundingClientRect();
      if (rect.width > 0) {
        syncWrapSize();
      } else {
        const im = currentImageMeta();
        if (im) setWrapFromMeta(im);
      }
      renderPrevCrop();
      if (state.lastNatural) renderReticleAtNatural(state.lastNatural.nx, state.lastNatural.ny);
    });

    el.mainImage.addEventListener("click", async (e) => {
      const im = currentImageMeta();
      if (!im || state.busy) return;
      const rect = displayRectOfImage();
      const dx = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const dy = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
      const nx = (dx / rect.width) * im.width;
      const ny = (dy / rect.height) * im.height;

      state.busy = true;
      try {
        const res = await fetch(window.HMI.api("/api/crop"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: im.filename, cx: nx, cy: ny, crop_size: state.cropSize }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.description || data.message || "Error al recortar");

        im.done = true;
        im.crop_count = 1;
        im.crop_box = { x1: data.x1, y1: data.y1, x2: data.x2, y2: data.y2 };
        updateGalleryItemDom(state.currentIndex);
        updateProgress();
        flashSaved();
        const dsLabel = data.dataset_kind === "defect"
          ? ` train/defective/${data.dataset_defect_type}`
          : " test/good";
        bus.emit("status", `Guardado →${dsLabel}: ${im.filename}`);
        bus.emit("crop-saved", { image: im, record: data });
        nextImage();
      } catch (err) {
        bus.emit("status", err.message, true);
      } finally {
        state.busy = false;
      }
    });

    function setCropSize(value) {
      const parsed = Math.max(1, Math.min(Number(value) || state.cropSize, Number(el.cropSlider.max) || 500));
      state.cropSize = parsed;
      el.cropSlider.value = String(parsed);
      el.cropSizeInput.value = String(parsed);
      if (state.lastNatural) renderReticleAtNatural(state.lastNatural.nx, state.lastNatural.ny);
    }

    el.cropSlider.addEventListener("input", () => {
      el.cropSizeInput.value = el.cropSlider.value;
      state.cropSize = Number(el.cropSlider.value);
      if (state.lastNatural) renderReticleAtNatural(state.lastNatural.nx, state.lastNatural.ny);
    });

    el.cropSizeInput.addEventListener("input", () => {
      const parsed = Number(el.cropSizeInput.value);
      if (Number.isFinite(parsed) && parsed >= 1) {
        el.cropSlider.value = String(parsed);
        state.cropSize = parsed;
        if (state.lastNatural) renderReticleAtNatural(state.lastNatural.nx, state.lastNatural.ny);
      }
    });

    el.cropSizeInput.addEventListener("change", () => {
      setCropSize(Number(el.cropSizeInput.value));
      fetch(window.HMI.api("/api/config"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_size: state.cropSize }),
      }).catch(() => {});
    });

    el.cropSlider.addEventListener("change", () => {
      setCropSize(Number(el.cropSlider.value));
      fetch(window.HMI.api("/api/config"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_size: state.cropSize }),
      }).catch(() => {});
    });

    el.searchInput.addEventListener("input", () => {
      state.filterText = el.searchInput.value;
      renderGallery();
    });

    document.addEventListener("keydown", (e) => {
      if (document.activeElement && ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      if (!document.querySelector('.tab.active[data-tab="crop"], .tab.active[data-tab="labels"]')) return;
      if (e.code === "ArrowRight") { e.preventDefault(); nextImage(); }
      else if (e.code === "ArrowLeft") { e.preventDefault(); prevImage(); }
      else if (e.code === "Space") {
        e.preventDefault();
        const im = currentImageMeta();
        if (im) bus.emit("status", `Saltada: ${im.filename}`);
        nextImage();
      }
    });
  }

  function applyConfig(cfg) {
    state.cropSize = cfg.crop_size || 250;
    el.cropSlider.value = state.cropSize;
    el.cropSizeInput.value = state.cropSize;
  }

  return { init, setImages, applyConfig, selectImage, currentImageMeta, updateProgress, syncWrap: syncWrapSize };
})();
