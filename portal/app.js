/* Portal unificado — gestión de pestañas e iframes same-origin.
   Las apps se sirven por subruta en el MISMO origen (nginx):
     crop  -> /hmi/    (preprocesado)
     train -> /train/  (entrenamiento DefectFill)
     merge -> /merge/  (postprocesado Fusionar)
   No se usan puertos adicionales ni ventanas emergentes. */
(() => {
  "use strict";

  const urls = {
    crop: "/hmi/",
    train: "/train/",
    merge: "/merge/",
  };

  const el = {
    tabs: Array.from(document.querySelectorAll(".tab")),
    frames: {
      crop: document.getElementById("frame-crop"),
      train: document.getElementById("frame-train"),
      merge: document.getElementById("frame-merge"),
    },
    status: document.getElementById("statusText"),
    projPill: document.getElementById("portalProject"),
    projBtn: document.getElementById("portalProjectsBtn"),
  };

  let current = "crop";

  function label(tab) {
    return tab === "crop" ? "preprocesado" : tab === "train" ? "entrenamiento" : "postprocesado";
  }

  // ---- Proyecto activo (compartido con defectfill/HMI) ----
  function refreshActiveProject() {
    fetch("/hmi/api/project/active")
      .then((r) => r.json())
      .then((data) => {
        const name = (data && data.name) || "";
        el.projPill.textContent = name || "Sin proyecto";
        el.projPill.setAttribute("data-active", name ? "1" : "");
      })
      .catch(() => {
        el.projPill.textContent = "Sin proyecto";
        el.projPill.setAttribute("data-active", "");
      });
  }

  function goToProjects() {
    fetch("/hmi/api/project/close", { method: "POST" })
      .catch(() => {})
      .finally(() => {
        refreshActiveProject();
        activate("crop");
        // Forzar recarga del iframe de Preprocesado: sin proyecto activo, el
        // HMI mostrará el panel de crear/cargar proyectos dentro del iframe.
        const f = el.frames.crop;
        f.src = "";
        f.setAttribute("src", urls["crop"]);
      });
  }

  function activate(tab) {
    current = tab;
    el.tabs.forEach((t) => {
      const active = t.dataset.tab === tab;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", String(active));
    });
    Object.entries(el.frames).forEach(([key, frame]) => {
      frame.classList.toggle("hidden", key !== tab);
      // OJO: frame.src con atributo vacío devuelve la URL base (truthy),
      // así que se comprueba el atributo real para cargar una sola vez.
      if (key === tab && !frame.getAttribute("src")) frame.setAttribute("src", urls[key]);
    });
    el.status.textContent = `Cargando ${label(tab)}: ${urls[tab]}`;
  }

  el.tabs.forEach((t) => t.addEventListener("click", () => activate(t.dataset.tab)));
  el.projBtn.addEventListener("click", goToProjects);

  Object.entries(el.frames).forEach(([key, frame]) => {
    frame.addEventListener("load", () => {
      refreshActiveProject();
      if (key === current) {
        el.status.textContent = `Conectado: ${urls[key]}`;
        el.status.classList.add("ok");
      }
    });
  });

  activate("crop");
  refreshActiveProject();
})();
