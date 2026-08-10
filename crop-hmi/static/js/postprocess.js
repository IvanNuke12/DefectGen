/* HMI de postprocesado — orquesta la ventana Fusionar de ancho completo.
   Reutiliza el módulo merge.js (misma funcionalidad) pero sin pestañas. */
(() => {
  "use strict";

  const el = {
    statusMessage: document.getElementById("statusMessage"),
  };

  const { bus } = window.HMI;

  const setStatus = (msg, isError) => {
    el.statusMessage.textContent = msg;
    el.statusMessage.style.color = isError ? "var(--danger)" : "var(--text-faint)";
  };

  async function loadSession() {
    const r = await fetch(window.HMI.api("/api/session"));
    const d = await r.json();
    if (!d.has_project) {
      window.location.href = window.HMI.api("/");
      return;
    }
  }

  async function init() {
    bus.on("status", setStatus);
    const tabs = document.querySelectorAll("#postprocTabs .tab");
    const panes = {
      single: document.getElementById("viewSingle"),
      batch: document.getElementById("viewBatch"),
    };
    const switchTab = (name) => {
      for (const t of tabs) {
        const active = t.dataset.view === name;
        t.classList.toggle("active", active);
        t.setAttribute("aria-selected", String(active));
      }
      panes.single.style.display = name === "single" ? "flex" : "none";
      panes.batch.style.display = name === "batch" ? "flex" : "none";
    };
    for (const t of tabs) {
      t.addEventListener("click", () => switchTab(t.dataset.view));
    }

    window.HMI.merge.init();
    window.HMI.batchmerge.init();
    await loadSession();
    window.HMI.merge.loadCrops();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
