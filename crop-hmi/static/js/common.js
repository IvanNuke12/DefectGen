/* Utilidades compartidas por todos los módulos del HMI. */
window.HMI = window.HMI || (() => {
  "use strict";

  // Prefijo de subruta cuando la app se sirve detrás de nginx (APP_SUBPATH,
  // inyectado en las plantillas como window.APP_BASE, p. ej. "/hmi").
  const APP_BASE = (window.APP_BASE || "").replace(/\/+$/, "");
  function api(path) { return APP_BASE + path; }

  // Codifica un nombre de archivo RELATIVO (p. ej. "good/foo.jpg") segmento a
  // segmento, preservando las "/" para que las rutas Flask <path:filename>
  // sigan funcionando con subcarpetas.
  function rel(name) {
    return String(name || "")
      .split("/")
      .map((s) => encodeURIComponent(s))
      .join("/");
  }

  const listeners = new Map();

  async function responseJson(res) {
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return res.json();
    const text = await res.text();
    return { message: text || `Error HTTP ${res.status}` };
  }

  const bus = {
    on(event, fn) {
      if (!listeners.has(event)) listeners.set(event, new Set());
      listeners.get(event).add(fn);
      return () => listeners.get(event).delete(fn);
    },
    emit(event, payload) {
      const set = listeners.get(event);
      if (set) set.forEach((fn) => fn(payload));
    },
  };

  // ----------------------------------------------------------------------
  // pickPath — navegador de carpetas servido por web (sustituye al selector
  // nativo tkinter del servidor, que no funciona en contenedores headless).
  // Devuelve una promesa con la ruta elegida o null si se cancela.
  // mode: "folder" | "json"
  // ----------------------------------------------------------------------
  let pickerEls = null;
  let pickerResolve = null;
  let pickerMode = "folder";
  let pickerCurrent = "";

  function ensurePickerDom() {
    if (pickerEls) return;
    const overlay = document.createElement("div");
    overlay.className = "picker-overlay";
    overlay.innerHTML = `
      <div class="picker" role="dialog" aria-modal="true">
        <header class="picker-head">
          <span class="picker-title"></span>
          <button class="picker-close" type="button" aria-label="Cerrar">&#10005;</button>
        </header>
        <div class="picker-roots"></div>
        <div class="picker-nav">
          <input type="text" class="picker-path" spellcheck="false" autocomplete="off">
          <button class="picker-go" type="button">Ir</button>
        </div>
        <ul class="picker-list"></ul>
        <footer class="picker-foot">
          <button class="picker-cancel" type="button">Cancelar</button>
          <button class="picker-select" type="button">Seleccionar</button>
        </footer>
      </div>`;
    document.body.appendChild(overlay);

    const q = (sel) => overlay.querySelector(sel);
    pickerEls = {
      overlay,
      title: q(".picker-title"),
      roots: q(".picker-roots"),
      path: q(".picker-path"),
      list: q(".picker-list"),
      go: q(".picker-go"),
      close: q(".picker-close"),
      cancel: q(".picker-cancel"),
      select: q(".picker-select"),
    };

    const close = (result) => {
      pickerEls.overlay.hidden = true;
      if (pickerResolve) { pickerResolve(result); pickerResolve = null; }
    };

    pickerEls.close.addEventListener("click", () => close(null));
    pickerEls.cancel.addEventListener("click", () => close(null));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });
    pickerEls.go.addEventListener("click", () => load(pickerEls.path.value.trim() || ""));
    pickerEls.path.addEventListener("keydown", (e) => { if (e.key === "Enter") load(pickerEls.path.value.trim() || ""); });
    pickerEls.select.addEventListener("click", () => {
      if (pickerMode === "json") {
        const sel = pickerEls.list.querySelector(".picker-file.selected");
        if (sel) close(sel.dataset.path);
      } else if (pickerCurrent) {
        close(pickerCurrent);
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && pickerEls.overlay && !pickerEls.overlay.hidden) close(null);
    });
  }

  async function load(path) {
    if (!pickerEls) return;
    try {
      const url = api("/api/browse?mode=") + encodeURIComponent(pickerMode)
        + (path ? "&path=" + encodeURIComponent(path) : "");
      const res = await fetch(url);
      const data = await responseJson(res);
      if (!res.ok) {
        pickerEls.path.setCustomValidity(data.message || "Error al navegar");
        pickerEls.path.reportValidity();
        return;
      }
      render(data);
    } catch (err) {
      pickerEls.path.setCustomValidity("Error al conectar: " + err.message);
      pickerEls.path.reportValidity();
    }
  }

  function render(data) {
    pickerEls.path.setCustomValidity("");
    pickerEls.path.value = data.path || "";
    pickerCurrent = data.path || "";

    pickerEls.roots.innerHTML = "";
    (data.roots || []).forEach((r) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "picker-root";
      btn.textContent = r;
      btn.addEventListener("click", () => load(r));
      pickerEls.roots.appendChild(btn);
    });

    pickerEls.list.innerHTML = "";
    const addEntry = (name, isDir, path, cls) => {
      const li = document.createElement("li");
      li.className = "picker-entry" + (cls ? " " + cls : "");
      const icon = document.createElement("span");
      icon.className = "picker-icon";
      icon.textContent = isDir ? "\u25B8" : "\u2022";
      const label = document.createElement("span");
      label.className = "picker-name";
      label.textContent = name;
      label.title = path;
      li.appendChild(icon);
      li.appendChild(label);
      if (isDir) {
        li.addEventListener("click", () => load(path));
      } else if (pickerMode === "json") {
        li.addEventListener("click", () => {
          pickerEls.list.querySelectorAll(".picker-file.selected").forEach((n) => n.classList.remove("selected"));
          li.classList.add("selected");
          pickerEls.select.disabled = false;
        });
        li.addEventListener("dblclick", () => close(path));
      } else {
        // Modo carpeta: los archivos se muestran (explorador) pero no se seleccionan.
        li.classList.add("picker-file");
      }
      pickerEls.list.appendChild(li);
    };

    if (data.parent) addEntry("..", true, data.parent, "picker-parent");
    (data.entries || []).forEach((e) => {
      if (e.is_dir) addEntry(e.name, true, e.path);
      else addEntry(e.name, false, e.path, "picker-file");
    });

    const anyFiles = (data.entries || []).some((e) => !e.is_dir);
    pickerEls.select.disabled = !(pickerMode === "folder" && data.path) && !anyFiles;
    pickerEls.select.textContent = pickerMode === "json" ? "Seleccionar archivo" : "Seleccionar carpeta";
  }

  function pickPath(mode, title, initialPath) {
    ensurePickerDom();
    pickerMode = mode === "json" ? "json" : "folder";
    pickerEls.title.textContent = title || (pickerMode === "json" ? "Seleccionar archivo JSON" : "Seleccionar carpeta");
    pickerEls.overlay.hidden = false;
    pickerResolve = null;
    const promise = new Promise((resolve) => { pickerResolve = resolve; });
    load((initialPath && initialPath.trim()) || "");
    return promise;
  }

  // ----------------------------------------------------------------------
  // initProjectSwitcher — selecciona el proyecto activo (compartido con la
  // webapp DefectFill). Al cambiar, se abre el proyecto y se recarga la página.
  // ----------------------------------------------------------------------
  async function initProjectSwitcher() {
    const sel = document.getElementById("projectSwitcher");
    if (!sel) return;
    try {
      const res = await fetch(api("/api/projects/list"));
      const data = await responseJson(res);
      const projects = (data && data.projects) || [];
      sel.innerHTML = "";
      if (!projects.length) {
        const o = document.createElement("option");
        o.value = ""; o.textContent = "— sin proyectos —";
        sel.appendChild(o);
        return;
      }
      let activeName = "";
      try {
        const ar = await fetch(api("/api/project/active"));
        const ad = await responseJson(ar);
        activeName = (ad && ad.name) || "";
      } catch (_) { /* ignore */ }
      for (const p of projects) {
        const o = document.createElement("option");
        o.value = p.path;
        o.textContent = p.name;
        if (p.name === activeName) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener("change", async () => {
        const value = sel.value;
        if (!value) return;
        const res = await fetch(api("/api/project/open"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: value }),
        });
        const data = await responseJson(res);
        if (!res.ok || !data.ok) {
          if (window.HMI.bus) window.HMI.bus.emit("status", (data && data.message) || "No se pudo abrir el proyecto.", true);
          return;
        }
        window.location.reload();
      });
    } catch (err) {
      sel.innerHTML = "";
      const o = document.createElement("option");
      o.value = ""; o.textContent = "—";
      sel.appendChild(o);
    }
  }

  return {
    bus,
    responseJson,
    pickPath,
    api,
    rel,
    initProjectSwitcher,
    clampBox(nx, ny, cropSize, w, h) {
      const cs = Math.max(1, Math.min(Math.round(cropSize), w, h));
      const half = Math.floor(cs / 2);
      const x1 = Math.max(0, Math.min(Math.round(nx) - half, w - cs));
      const y1 = Math.max(0, Math.min(Math.round(ny) - half, h - cs));
      return { x1, y1, x2: x1 + cs, y2: y1 + cs, cs };
    },
  };
})();
