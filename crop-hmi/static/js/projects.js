/* Pantalla de proyectos — abrir o crear un proyecto antes de entrar al HMI. */
(() => {
  "use strict";

  const el = {
    openProjectBtn: document.getElementById("openProjectBtn"),
    recentBlock: document.getElementById("recentBlock"),
    recentList: document.getElementById("recentList"),
    projectName: document.getElementById("projectName"),
    projectParent: document.getElementById("projectParent"),
    browseParentBtn: document.getElementById("browseParentBtn"),
    objectClass: document.getElementById("objectClass"),
    createProjectBtn: document.getElementById("createProjectBtn"),
    status: document.getElementById("status"),
  };

  // Ubicación por defecto de los proyectos (data/projects si está disponible).
  const DEFAULT_PROJECT_PARENT = "/data/projects";
  let parentFolder = "";
  let recents = [];

  const setStatus = (msg, kind) => {
    el.status.textContent = msg;
    el.status.className = "status" + (kind ? " " + kind : "");
  };

  async function jsonRequest(url, method, body) {
    const opts = method ? { method, headers: { "Content-Type": "application/json" } } : {};
    if (body) opts.body = JSON.stringify(body);
    try {
      const res = await fetch(url, opts);
      let data = null;
      try { data = await res.json(); } catch (_) { /* ignore */ }
      return { res, data };
    } catch (err) {
      console.error("Error en " + url, err);
      return { res: { ok: false, status: 0 }, data: { message: "Error al conectar: " + err.message } };
    }
  }

  async function openDialog(mode, title, initialPath) {
    return window.HMI.pickPath(mode, title, initialPath || "");
  }

  function renderRecents() {
    if (!recents.length) { el.recentBlock.style.display = "none"; return; }
    el.recentBlock.style.display = "block";
    el.recentList.innerHTML = "";
    for (const r of recents) {
      const li = document.createElement("li");
      li.className = "recent-item";
      li.title = r.path;
      const name = document.createElement("span");
      name.className = "recent-name";
      name.textContent = r.name;
      const path = document.createElement("span");
      path.className = "recent-path";
      path.textContent = r.path;
      li.appendChild(name);
      li.appendChild(path);
      li.addEventListener("click", () => openProject(r.path));
      el.recentList.appendChild(li);
    }
  }

  async function loadRecents() {
    const { data } = await jsonRequest(window.HMI.api("/api/projects/recents"));
    recents = (data && data.recents) || [];
    renderRecents();
  }

  async function openProject(path) {
    setStatus("Abriendo proyecto…");
    const { res, data } = await jsonRequest(window.HMI.api("/api/project/open"), "POST", { path });
    if (!res.ok || !data || !data.ok) {
      setStatus((data && data.message) || "No se pudo abrir el proyecto.", "error");
      return;
    }
    window.location.href = window.HMI.api("/");
  }

  function validateCreate() {
    const name = el.projectName.value.trim();
    const valid = name && parentFolder;
    el.createProjectBtn.disabled = !valid;
  }

  async function createProject() {
    const name = el.projectName.value.trim();
    if (!name) { setStatus("Pon un nombre al proyecto.", "error"); return; }
    if (!parentFolder) { setStatus("Selecciona la carpeta donde crearlo.", "error"); return; }
    setStatus("Creando proyecto…");
    const { res, data } = await jsonRequest(window.HMI.api("/api/project/create"), "POST", {
      name, parent: parentFolder,
      object_class: el.objectClass.value.trim(),
    });
    if (!res.ok || !data || !data.ok) {
      setStatus((data && data.message) || "No se pudo crear el proyecto.", "error");
      return;
    }
    window.location.href = window.HMI.api("/");
  }

  // Aplica la ubicación por defecto (data/projects). Si no existe en las raíces
  // del navegador, usa la primera raíz disponible o deja la elección al usuario.
  async function applyDefaultParent() {
    const { data } = await jsonRequest(window.HMI.api("/api/browse"));
    const roots = (data && data.roots) || [];
    const target = DEFAULT_PROJECT_PARENT
      || roots.find((r) => /projects/i.test(r))
      || roots[0]
      || "";
    if (target && target.trim()) {
      parentFolder = target.trim();
      el.projectParent.value = parentFolder;
      validateCreate();
    }
  }

  function init() {
    el.openProjectBtn.addEventListener("click", async () => {
      const selected = await openDialog("folder", "Seleccionar carpeta de proyecto");
      if (selected) openProject(selected);
    });

    el.browseParentBtn.addEventListener("click", async () => {
      const selected = await openDialog("folder", "Carpeta donde crear el proyecto");
      if (selected) {
        parentFolder = selected;
        el.projectParent.value = selected;
        validateCreate();
      }
    });

    el.projectName.addEventListener("input", validateCreate);
    el.objectClass.addEventListener("input", validateCreate);
    el.createProjectBtn.addEventListener("click", createProject);

    loadRecents();
    applyDefaultParent();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
