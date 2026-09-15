/* Tab Fusionar — vista "Por lote": fusiona todas las imágenes generadas de
   un run de inferencia de una en una, mostrando una progress bar con cuántas
   quedan por fusionar y cuántas se han completado. */
window.HMI.batchmerge = (() => {
  "use strict";

  const el = {
    runSelect: document.getElementById("batchRunSelect"),
    info: document.getElementById("batchInfo"),
    mergeBtn: document.getElementById("batchMergeBtn"),
    progress: document.getElementById("batchProgress"),
    progressFill: document.getElementById("batchProgressFill"),
    progressText: document.getElementById("batchProgressText"),
    resultEmpty: document.getElementById("batchResultEmpty"),
    results: document.getElementById("batchResults"),
  };

  const state = { runs: [], running: false };

  const { bus } = window.HMI;

  // Ver nota en merge.js: el backend puede responder con HTML en errores
  // (p. ej. 403 "No hay proyecto activo"), lo que rompía res.json().
  async function safeJson(res) {
    const text = await res.text();
    let data = null;
    try { data = JSON.parse(text); } catch (_) { data = null; }
    if (!data || typeof data !== "object") {
      data = { ok: false, error: `No se pudo fusionar (HTTP ${res.status}). ¿Hay un proyecto abierto?` };
    }
    return data;
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function summarize(run) {
    if (!run || !run.files.length) {
      return "Este run no tiene imágenes generadas (*_generated.png).";
    }
    const withSource = run.files.filter((f) => f.source).length;
    return (
      `El run <strong>${escapeHtml(run.run)}</strong> contiene ` +
      `<strong>${run.files.length}</strong> imagen(es) generada(s), de las que ` +
      `<strong>${withSource}</strong> tienen crop de origen registrado. ` +
      "Se fusionarán todas las que tengan registro."
    );
  }

  async function loadRuns() {
    try {
      const res = await fetch(window.HMI.api("/api/generated/runs"));
      const data = await res.json();
      state.runs = (data && data.runs) || [];
      el.runSelect.innerHTML = '<option value="">— cargar runs —</option>';
      if (!state.runs.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Sin runs de inferencia todavía";
        el.runSelect.appendChild(opt);
      }
      for (const r of state.runs) {
        const opt = document.createElement("option");
        opt.value = r.run;
        opt.textContent = `${r.run}  (${r.files.length} generadas)`;
        el.runSelect.appendChild(opt);
      }
      onRunChange();
    } catch (err) {
      bus.emit("status", "No se pudieron cargar los runs: " + err.message, true);
    }
  }

  function onRunChange() {
    const run = state.runs.find((x) => x.run === el.runSelect.value) || null;
    el.info.innerHTML = run ? summarize(run) : "Selecciona un run para ver cuántas imágenes se fusionarán.";
    el.mergeBtn.disabled = !run || !run.files.length;
    resetResults();
  }

  function setProgress(done, total) {
    el.progress.style.display = "flex";
    const pct = total ? Math.round((done / total) * 100) : 0;
    el.progressFill.style.width = pct + "%";
    el.progressText.textContent = `${done} / ${total}`;
  }

  function resetResults() {
    el.resultEmpty.style.display = "flex";
    el.results.style.display = "none";
    el.results.innerHTML = "";
    el.progress.style.display = "none";
    el.progressFill.style.width = "0%";
  }

  function addResultRow(r) {
    el.resultEmpty.style.display = "none";
    el.results.style.display = "flex";
    const row = document.createElement("div");
    row.className = "batch-result-row " + (r.ok ? "ok" : "err");

    const badge = document.createElement("span");
    badge.className = "batch-result-badge " + (r.ok ? "ok" : "err");
    badge.textContent = r.ok ? "OK" : "Error";

    const info = document.createElement("div");
    info.style.minWidth = "0";
    info.style.flex = "1";
    const name = document.createElement("div");
    name.className = "batch-result-name";
    name.textContent = r.name;
    const detail = document.createElement("div");
    detail.className = "batch-result-detail";
    detail.textContent = r.ok ? `→ ${r.merged_path}` : (r.error || "Error desconocido");
    info.appendChild(name);
    info.appendChild(detail);

    row.appendChild(badge);
    row.appendChild(info);

    if (r.ok) {
      const link = document.createElement("a");
      link.className = "btn btn-secondary";
      link.href = window.HMI.api(r.merged_url);
      link.setAttribute("download", r.merged_name.split("/").pop());
      link.textContent = "Descargar";
      row.appendChild(link);
    }

    el.results.appendChild(row);
    el.results.scrollTop = el.results.scrollHeight;
  }

  async function runBatch() {
    const run = el.runSelect.value;
    if (!run || state.running) return;
    const files = (state.runs.find((x) => x.run === run) || {}).files || [];
    const todo = files.filter((f) => f.source);
    if (!todo.length) { bus.emit("status", "Ninguna generada de este run tiene crop de origen registrado.", true); return; }

    state.running = true;
    el.mergeBtn.disabled = true;
    resetResults();
    setProgress(0, todo.length);
    bus.emit("status", `Fusionando ${todo.length} imagen(es) del run…`);

    let done = 0;
    let failed = 0;
    const totals = { total: todo.length, success: 0, failed: 0 };
    for (const f of todo) {
      try {
        const res = await fetch(window.HMI.api("/api/merge/generated"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file: f.path }),
        });
        const data = await safeJson(res);
        if (!res.ok || !data.ok) {
          data.ok = false;
          data.error = data.error || `HTTP ${res.status}`;
        }
        addResultRow(data);
        if (data.ok) totals.success += 1; else totals.failed += 1;
      } catch (err) {
        totals.failed += 1;
        addResultRow({ ok: false, name: f.name, source: f.source, error: err.message });
      }
      done += 1;
      setProgress(done, todo.length);
    }
    totals.total = todo.length;
    bus.emit("status", `Lote fusionado: ${totals.success}/${totals.total} correctas.`);
    bus.emit("merge-done", totals);
    state.running = false;
    el.mergeBtn.disabled = false;
  }

  function init() {
    el.runSelect.addEventListener("change", onRunChange);
    el.mergeBtn.addEventListener("click", runBatch);
    loadRuns();
  }

  return { init, loadRuns };
})();