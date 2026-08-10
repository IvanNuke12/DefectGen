/* DefectFill — Consola de inspección.
   Prefijo de subruta cuando la app se sirve detrás de nginx
   (window.APP_BASE = "/train" definido en index.html). */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const APP_BASE = (window.APP_BASE || "").replace(/\/+$/, "");
const api = (p) => APP_BASE + p;

// ---------------------------------------------------------------- Tabs
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#panel-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "dataset") loadDatasetSummary();
    if (tab.dataset.tab === "results") loadResults();
    if (tab.dataset.tab === "system") { loadSystem(); loadJobs(); }
    if (tab.dataset.tab === "train") drawAllCharts();
    if (tab.dataset.tab === "train" || tab.dataset.tab === "infer") {
      refreshSelectors();
    }
  });
});

// ---------------------------------------------------------------- Helpers
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}
function corners() {
  return `<span class="vf-corner vf-tl"></span><span class="vf-corner vf-tr"></span><span class="vf-corner vf-bl"></span><span class="vf-corner vf-br"></span>`;
}
function fmtBytes(n) {
  if (!Number.isFinite(n)) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + " KB";
  return String(n) + " B";
}

// ---------------------------------------------------------------- Ayuda (i) en hiperparámetros
// Descripción de cada parámetro de los formularios Entrenamiento/Evaluación/Inferencia.
const HYPERPARAM_HELP = {
  "max_train_steps":  "Número total de pasos de optimización del entrenamiento. Más pasos suele mejorar el resultado, pero alarga la duración y consume más GPU.",
  "batch_size":       "Imágenes que el modelo procesa por paso (lote). Valores altos aceleran pero consumen más VRAM; si la GPU va escasa, baja a 1–2.",
  "lora_rank":        "Rango de las matrices LoRA. Controla la capacidad de adaptación al nuevo concepto: más rank = más flexibilidad, pero más parámetros y riesgo de sobreajuste.",
  "lora_alpha":       "Escala de la actualización LoRA. Con alpha > rank se intensifica el efecto del ajuste fino.",
  "text_encoder_lr":  "Tasa de aprendizaje del codificador de texto (CLIP). Ajusta lo rápido que aprende el prompt/embedding de texto.",
  "unet_lr":          "Tasa de aprendizaje de la UNet (el modelo de difusión). Suele ser mayor que la del text encoder.",
  "lr_warmup_steps":  "Pasos de calentamiento: la tasa de aprendizaje sube progresivamente desde 0 hasta su valor objetivo durante estos pasos iniciales.",
  "save_steps":       "Cada cuántos pasos se guarda un checkpoint del modelo. Los checkpoints son los candidatos que luego se validan.",
  "lambda_defect":    "Peso de la pérdida de defecto (reconstrucción del área defectuosa). Mayor valor prioriza corregir bien el defecto.",
  "lambda_obj":       "Peso de la pérdida de objeto (preservar la identidad del objeto). Mayor valor mantiene mejor la forma/estilo del objeto.",
  "lambda_attn":      "Peso de la pérdida de atención (consistencia estructural entre las features de atención).",
  "dilate_mask":      "Expande las máscaras de defecto unos píxeles antes de entrenar, dando más contexto al modelo en los bordes.",
  "num_eval_images":  "Número de imágenes buenas del lote fijo de evaluación. Se recomienda 2–5: mismo lote determinista para todos los checkpoints.",
  "min_ic_lpips":     "Diversidad mínima exigida entre candidatos (IC-LPIPS). Evita que el generador colapse a una única salida.",
  "steps":            "Pasos de difusión (denoising). Más pasos = mejor calidad, pero más lento. 50 es el estándar.",
  "guidance_scale":   "Fuerza con la que la generación se adhiere al prompt. Más alto = más fiel al texto, pero menos diverso; valores 1.5–3 típicos.",
  "prompt":           "Descripción textual del defecto a generar. Debe ir en inglés y el token <defect> es obligatorio (embedding aprendido en el entrenamiento).",
  "total_images":     "Número total de imágenes sintéticas defectuosas que se generarán.",
  "num_samples":      "Candidatos generados por cada imagen de origen; se selecciona el mejor por LPIPS espacial.",
  "run_name":         "Nombre identificativo de la ejecución. Se usa para la carpeta de salida y los resultados.",
  "run":              "Ejecución de entrenamiento cuyos checkpoints se van a validar.",
  "checkpoint":       "Checkpoint entrenado que se usará para la generación.",
  "defect_type":      "Tipo de defecto a entrenar, validar o generar. Vacío en entrenamiento = usa todos los tipos presentes."
};

function initHyperparamHelp() {
  Object.entries(HYPERPARAM_HELP).forEach(([name, tip]) => {
    $$(`input[name="${name}"], select[name="${name}"]`).forEach((input) => {
      const label = input.closest(".field")?.querySelector("label");
      if (!label || label.querySelector(".help-btn")) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "help-btn";
      btn.dataset.tip = tip;
      btn.setAttribute("aria-label", "Ayuda: " + tip);
      btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>';
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const open = btn.classList.toggle("open");
        document.querySelectorAll(".help-btn.open").forEach((b) => {
          if (b !== btn) b.classList.remove("open");
        });
        if (open) {
          const close = (ev) => {
            btn.classList.remove("open");
            document.removeEventListener("click", close);
            document.removeEventListener("keydown", esc);
          };
          const esc = (ev) => { if (ev.key === "Escape") close(); };
          setTimeout(() => document.addEventListener("click", close), 0);
          document.addEventListener("keydown", esc);
        }
      });
      label.appendChild(btn);
    });
  });
}

// ---------------------------------------------------------------- Training charts (real-time subplots)
const TRAIN_METRICS = [
  { key: "def",   canvas: "chartLossDef",   value: "chartValDef",   color: "#ff6b6b" },
  { key: "obj",   canvas: "chartLossObj",   value: "chartValObj",   color: "#ffb86b" },
  { key: "attn",  canvas: "chartLossAttn",  value: "chartValAttn",  color: "#5fd897" },
  { key: "total", canvas: "chartLossTotal", value: "chartValTotal", color: "#5cc8ff" },
];
const LOSS_FIELD = { def: "loss_defect", obj: "loss_object", attn: "loss_attention", total: "loss_total" };
let trainHistory = {};

function resetTrainCharts() {
  trainHistory = {};
  TRAIN_METRICS.forEach((m) => {
    trainHistory[m.key] = [];
    const span = $("#" + m.value);
    if (span) span.textContent = "—";
    drawLossChart(m, []);
  });
}

function drawAllCharts() {
  TRAIN_METRICS.forEach((m) => drawLossChart(m, trainHistory[m.key] || []));
}

function recordTrainStatus(s) {
  if (s.loss_total == null) return;
  const step = s.step ?? 0;
  TRAIN_METRICS.forEach((m) => {
    const raw = s[LOSS_FIELD[m.key]];
    if (raw == null) return;
    const hist = trainHistory[m.key];
    if (hist.length && hist[hist.length - 1].step >= step) return;
    hist.push({ step, value: raw });
    drawLossChart(m, hist);
    const span = $("#" + m.value);
    if (span) span.textContent = raw.toFixed(4);
  });
}

function drawLossChart(m, data) {
  const canvas = $("#" + m.canvas);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(40, rect.width), H = Math.max(40, rect.height);
  const pw = Math.round(W * dpr), ph = Math.round(H * dpr);
  if (canvas.width !== pw) canvas.width = pw;
  if (canvas.height !== ph) canvas.height = ph;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const padL = 30, padR = 8, padT = 8, padB = 4;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textBaseline = "middle";

  if (!data.length) {
    ctx.fillStyle = "#6b7280";
    ctx.textAlign = "center";
    ctx.fillText("esperando datos…", W / 2, H / 2);
    return;
  }

  const xMin = data[0].step, xMax = data[data.length - 1].step;
  let vMin = Infinity, vMax = -Infinity;
  for (const d of data) { if (d.value < vMin) vMin = d.value; if (d.value > vMax) vMax = d.value; }
  if (vMin === vMax) { vMin -= 1; vMax += 1; }
  const p = (vMax - vMin) * 0.08; vMin -= p; vMax += p;
  const xspan = (xMax - xMin) || 1;

  ctx.strokeStyle = "#2c3140";
  ctx.fillStyle = "#6b7280";
  ctx.lineWidth = 1;
  const gridLines = 3;
  for (let i = 0; i <= gridLines; i++) {
    const y = padT + plotH - (i / gridLines) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const v = vMin + (i / gridLines) * (vMax - vMin);
    ctx.textAlign = "right";
    ctx.fillText(v.toFixed(2), padL - 4, y);
  }

  ctx.strokeStyle = m.color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = padL + ((d.step - xMin) / xspan) * plotW;
    const y = padT + plotH - ((d.value - vMin) / (vMax - vMin)) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "left";
  ctx.fillText(String(xMin), padL, H - padB + 7);
  ctx.textAlign = "right";
  ctx.fillText(String(xMax), W - padR, H - padB + 7);
}

// ---------------------------------------------------------------- Boot
async function checkSystemPill() {
  const dot = $("#sysDot"), label = $("#sysLabel");
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    const res = await fetch(api("/api/system"), { signal: controller.signal });
    clearTimeout(timeoutId);
    const info = await res.json();
    if (info.cuda_available === null || info.cuda_available === undefined) {
      dot.className = "dot off";
      label.textContent = "GPU: comprobando…";
      setTimeout(checkSystemPill, 3000);
    } else if (info.cuda_available) {
      dot.className = "dot on";
      label.textContent = `GPU: ${info.gpu_name || "detectada"}`;
    } else {
      dot.className = "dot off";
      label.textContent = "sin GPU CUDA disponible";
    }
  } catch {
    dot.className = "dot off";
    label.textContent = "backend no disponible";
  }
}

// ---------------------------------------------------------------- Dataset tab
let datasetSummary = [];
let projects = [];
let activeProject = null;   // { name, class_name, ... } o null
const ACTIVE_PROJECT_KEY = "defectfill.active_project";

function populateSelect(select, options, placeholder) {
  select.innerHTML = "";
  if (placeholder != null) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = placeholder;
    select.appendChild(opt);
  }
  for (const { value, text } of options) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = text;
    select.appendChild(opt);
  }
}

function populateDefectTypes(kind) {
  const defSel = kind === "train" ? $("#trainDefectType") : $("#inferDefectType");
  if (!defSel) return;
  const cls = (datasetSummary || []).find((c) => c.object_class === (activeProject && activeProject.name && kind === "infer" ? activeProject.name : activeProject && activeProject.name));
  const types = (cls && cls.defect_types) || [];
  populateSelect(defSel, types.map((t) => ({ value: t.defect_type, text: `${t.defect_type} (${t.image_count} img / ${t.mask_count} masc)` })), "— todos —");
  if (kind === "infer") defSel.disabled = !types.length;
  else defSel.disabled = false;
}

async function refreshSelectors() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);
  const proj = activeProject ? `?project=${encodeURIComponent(activeProject.name)}` : "";
  const [summRes, ckptRes] = await Promise.all([
    fetch(api(`/api/dataset/summary${proj}`), { signal: controller.signal }).catch(() => null),
    fetch(api(`/api/checkpoints${proj}`), { signal: controller.signal }).catch(() => null),
  ]);
  clearTimeout(timeoutId);
  if (summRes && summRes.ok) {
    try { datasetSummary = await summRes.json(); } catch (_) { datasetSummary = []; }
  } else {
    datasetSummary = [];
  }
  populateDefectTypes("train");
  populateDefectTypes("infer");

  let checkpoints = [];
  if (ckptRes && ckptRes.ok) {
    try { checkpoints = await ckptRes.json(); } catch (_) { checkpoints = []; }
  }
  populateSelect($("#inferCheckpoint"), (Array.isArray(checkpoints) ? checkpoints : []).map((c) => ({
    value: c.path,
    text: `${c.run} / ${c.filename} (${fmtBytes(c.size_mb * 1e6)})`,
  })), "— sin checkpoints —");
}

async function loadDatasetSummary() {
  const box = $("#datasetSummary");
  if (!box) return;
  try {
    const res = await fetch(api("/api/dataset/summary"));
    const classes = await res.json();
    datasetSummary = Array.isArray(classes) ? classes : [];
    refreshSelectors();
    box.innerHTML = "";
    if (!datasetSummary.length) {
      box.innerHTML = '<p class="empty-state">Sin clases todavía. Sube imágenes o carga el dataset de ejemplo.</p>';
      return;
    }
    for (const c of datasetSummary) {
      const row = document.createElement("div");
      row.className = "dataset-row";
      const info = document.createElement("div");
      const name = document.createElement("div");
      name.className = "dr-name";
      name.textContent = c.class_name || c.object_class;
      const meta = document.createElement("div");
      meta.className = "dr-meta";
      meta.textContent = `${c.object_class} · ${c.good_count} buenas en test/good`;
      info.appendChild(name);
      info.appendChild(meta);
      if (c.defect_types && c.defect_types.length) {
        const types = document.createElement("div");
        types.className = "dr-types";
        c.defect_types.forEach((t) => {
          const tag = document.createElement("span");
          tag.className = "type-tag";
          tag.textContent = `${t.defect_type} · ${t.image_count}/${t.mask_count}`;
          types.appendChild(tag);
        });
        info.appendChild(types);
      }
      const del = document.createElement("button");
      del.type = "button";
      del.className = "dr-delete";
      del.textContent = "eliminar";
      del.addEventListener("click", async () => {
        if (!window.confirm(`¿Eliminar la clase "${c.object_class}" completa?`)) return;
        try {
          await fetch(api(`/api/dataset/${encodeURIComponent(c.object_class)}`), { method: "DELETE" });
          await loadDatasetSummary();
        } catch (err) {
          window.alert("No se pudo eliminar: " + err.message);
        }
      });
      row.appendChild(info);
      row.appendChild(del);
      box.appendChild(row);
    }
  } catch (err) {
    box.innerHTML = `<p class="empty-state">Error al cargar el dataset: ${err.message}</p>`;
  }
}

// ---------------------------------------------------------------- Proyecto activo (global compartido)
function renderActiveProject() {
  const sel = $("#projectSelect");
  if (sel && activeProject) sel.value = activeProject.name;
  const boxTrain = $("#trainActiveProject");
  if (boxTrain) boxTrain.textContent = activeProject ? activeProject.class_name : "— sin proyecto —";
  const boxInfer = $("#inferActiveProject");
  if (boxInfer) boxInfer.textContent = activeProject ? activeProject.class_name : "— sin proyecto —";
}

function populateProjectSelect() {
  const sel = $("#projectSelect");
  if (!sel) return;
  const opts = Array.isArray(projects) ? projects : [];
  sel.innerHTML = "";
  const ph = document.createElement("option");
  ph.value = "";
  ph.textContent = opts.length ? "— elegir proyecto —" : "— sin proyectos —";
  sel.appendChild(ph);
  for (const p of opts) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.class_name || p.name;
    sel.appendChild(opt);
  }
  if (activeProject) sel.value = activeProject.name;
}

async function loadProjectState() {
  try {
    const res = await fetch(api("/api/projects"));
    const data = await res.json();
    projects = (data && data.projects) || [];
  } catch (_) { projects = []; }
  // proyecto activo: prioridad a la fuente compartida; si no hay, a localStorage
  let stored = null;
  try { stored = localStorage.getItem(ACTIVE_PROJECT_KEY) || null; } catch (_) { /* ignore */ }
  let active = null;
  try {
    const res = await fetch(api("/api/project/active"));
    const data = await res.json();
    if (data && data.name) active = data;
  } catch (_) { /* ignore */ }
  if (!active && stored) {
    const match = (projects || []).find((p) => p.name === stored);
    if (match) active = match;
  }
  activeProject = active;
  populateProjectSelect();
  renderActiveProject();
  refreshSelectors();
  loadValidateRuns();
}

async function setActiveProject(name) {
  if (!name) return;
  try {
    const res = await fetch(api("/api/project/active"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    activeProject = await res.json();
  } catch (err) {
    window.alert("No se pudo fijar el proyecto activo: " + err.message);
    return;
  }
  try { localStorage.setItem(ACTIVE_PROJECT_KEY, activeProject.name); } catch (_) { /* ignore */ }
  renderActiveProject();
  refreshSelectors();
  loadValidateRuns();
}

function populateProjectTree() { populateProjectSelect(); }

async function loadSampleDataset() {
  const btn = $("#btnLoadSample");
  if (btn) { btn.disabled = true; btn.textContent = "Cargando…"; }
  try {
    const res = await fetch(api("/api/dataset/load-sample"), { method: "POST" });
    const data = await res.json();
    window.alert(data.copied
      ? `Dataset de ejemplo listo: ${(data.classes || []).join(", ") || "ver resumen"}.`
      : `No se pudo cargar: ${data.reason || "desconocido"}`);
    await loadDatasetSummary();
  } catch (err) {
    window.alert("Error al cargar el dataset de ejemplo: " + err.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Cargar dataset de ejemplo"; }
  }
}

async function uploadGood() {
  const cls = $("#dsGoodClass").value.trim();
  const files = $("#dsGoodFiles").files;
  if (!cls) { window.alert("Indica la clase de objeto."); return; }
  if (!files.length) { window.alert("Selecciona imágenes."); return; }
  const fd = new FormData();
  fd.append("object_class", cls);
  for (const f of files) fd.append("files", f, f.name);
  try {
    const res = await fetch(api("/api/dataset/upload-good"), { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    window.alert(`Guardadas ${data.saved} imagen(es) buena(s).`);
    $("#dsGoodFiles").value = "";
    await loadDatasetSummary();
  } catch (err) {
    window.alert("Error: " + err.message);
  }
}

async function uploadDefect() {
  const cls = $("#dsDefectClass").value.trim();
  const def = $("#dsDefectType").value.trim();
  const images = $("#dsDefectImages").files;
  const masks = $("#dsDefectMasks").files;
  if (!cls || !def) { window.alert("Indica clase y tipo de defecto."); return; }
  if (!images.length || !masks.length) { window.alert("Selecciona imágenes y máscaras."); return; }
  const fd = new FormData();
  fd.append("object_class", cls);
  fd.append("defect_type", def);
  for (const f of images) fd.append("images", f, f.name);
  for (const f of masks) fd.append("masks", f, f.name);
  try {
    const res = await fetch(api("/api/dataset/upload-defect"), { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    window.alert(`Guardados ${data.pairs_saved} pares (recibidos ${data.images_provided} img / ${data.masks_provided} masc).`);
    $("#dsDefectImages").value = "";
    $("#dsDefectMasks").value = "";
    await loadDatasetSummary();
  } catch (err) {
    window.alert("Error: " + err.message);
  }
}

// ---------------------------------------------------------------- Job monitoring (SSE)
const monitors = { train: null, infer: null, eval: null };
const RUNNING = { train: false, infer: false, eval: false };

// Prefijo DOM para cada tipo de job: train→train, infer→infer, eval→val.
function domId(kind, base) {
  const prefix = kind === "train" ? "train" : kind === "infer" ? "infer" : "val";
  return document.getElementById(prefix + base);
}

function setBadge(kind, state, label) {
  const badge = domId(kind, "StatusBadge");
  if (!badge) return;
  badge.className = "status-badge " + (state || "idle");
  badge.textContent = label || (state === "running" ? "ejecutando" : state || "sin ejecución");
}

function setProgress(kind, step, total) {
  const label = domId(kind, "StepLabel");
  const fill = domId(kind, "ProgressFill");
  if (label) label.textContent = `${step} / ${total}`;
  if (fill) fill.style.width = total ? `${Math.max(0, Math.min(100, (step / total) * 100))}%` : "0%";
}

function appendLog(kind, lines) {
  if (!lines || !lines.length) return;
  const box = domId(kind, "Log");
  if (!box) return;
  if (box.querySelector(".log-placeholder")) box.innerHTML = "";
  lines.forEach((line) => {
    const p = document.createElement("p");
    p.className = "log-line" + (/error|traceback|exception|failed|out of memory|no space left|nan/i.test(line) ? " err" : "");
    p.textContent = line;
    box.appendChild(p);
  });
  while (box.children.length > 800) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

function fmtEta(secs) {
  if (secs == null || !Number.isFinite(secs) || secs < 0) return "ETA —";
  const m = Math.floor(secs / 60), s = Math.round(secs % 60);
  return `ETA ${m}m ${String(s).padStart(2, "0")}s`;
}

function runFromSummary(summary) {
  if (!summary || !summary.output_dir) return { cls: "", run: "" };
  const parts = String(summary.output_dir).split("/").filter(Boolean);
  // output_dir es relativo a PROJECT_ROOT, p.ej. "projects/Prueba/output/run_20260805_..."
  const cls = parts.length >= 3 ? parts[parts.length - 3] : "";
  const run = parts.length ? parts[parts.length - 1] : "";
  return { cls, run };
}

function renderInferPreviews(meta, images) {
  const grid = $("#inferPreviewGrid");
  if (!grid) return;
  grid.innerHTML = "";
  const cls = meta && meta.cls ? meta.cls : "";
  const run = meta && meta.run ? meta.run : "";
  if (!run || !images || !images.length) {
    grid.innerHTML = '<p class="empty-state">Aún no hay generaciones en esta sesión.</p>';
    return;
  }
  for (const im of images) {
    if (!im.generated) continue;
    const gen = api(`/files/generated/${cls}/generated/${run}/${im.generated}`);
    const orig = im.original ? api(`/files/generated/${cls}/generated/${run}/${im.original}`) : "";
    const mask = im.mask ? api(`/files/generated/${cls}/generated/${run}/${im.mask}`) : "";
    const trip = el(`
      <div class="triplet">
        <div class="triplet-labels"><span>Original</span><span>Máscara</span><span>Generado</span></div>
        <div class="triplet-imgs">
          <div class="viewfinder">${corners()}<img src="${orig || gen}" alt="original"></div>
          <div class="viewfinder">${corners()}<img src="${mask || gen}" alt="máscara"></div>
          <div class="viewfinder">${corners()}<img src="${gen}" alt="generado"></div>
        </div>
        <div class="triplet-caption"><span>#${String(im.output_idx ?? "").padStart(4, "0")}</span><span>LPIPS ${im.lpips_score != null ? im.lpips_score.toFixed(3) : "—"}</span></div>
      </div>`);
    grid.appendChild(trip);
  }
}

function handleStream(kind, d) {
  if (d.new_log_lines && d.new_log_lines.length) appendLog(kind, d.new_log_lines);
  const status = d.status;
  if (status) {
    const state = status.state || "running";
    setBadge(kind, state === "running" ? "running" : state, state === "running" ? "ejecutando" : state);
    if (status.step != null && status.total_steps != null) setProgress(kind, status.step, status.total_steps);
    if (kind === "train") {
      recordTrainStatus(status);
      const eta = $("#trainEtaLabel");
      if (eta && status.eta_seconds != null) eta.textContent = fmtEta(status.eta_seconds);
      if (status.error) appendLog("train", [`[error] ${status.error}`]);
    } else if (kind === "infer") {
      if (status.images) renderInferPreviews(runFromSummary(d.summary), status.images);
      if (status.error) appendLog("infer", [`[error] ${status.error}`]);
    } else if (kind === "eval") {
      if (d.closed) loadValidationResult();
      if (status.error) appendLog("eval", [`[error] ${status.error}`]);
    }
  }
  if (d.closed) {
    setBadge(kind, "idle", "finalizado");
    const summary = d.summary || {};
    const rc = summary.returncode;
    if (rc != null && rc !== 0) {
      setBadge(kind, "failed", "fallido");
      appendLog(kind, [`[proceso terminado con código ${rc}]`]);
    } else if (summary.alive === false) {
      setBadge(kind, "completed", "completado");
    }
  }
}

function stopMonitor(kind) {
  const es = monitors[kind];
  if (es) { es.close(); monitors[kind] = null; }
}

function setRunningUi(kind, running) {
  RUNNING[kind] = running;
  if (kind === "train") {
    const start = $("#btnStartTrain"), stop = $("#btnStopTrain");
    if (start) start.disabled = running;
    if (stop) stop.disabled = !running;
  } else if (kind === "infer") {
    const start = $("#btnStartInfer"), stop = $("#btnStopInfer");
    if (start) start.disabled = running;
    if (stop) stop.disabled = !running;
  } else if (kind === "eval") {
    const start = $("#btnStartValidate"), stop = $("#btnStopValidate");
    if (start) start.disabled = running;
    if (stop) stop.disabled = !running;
  }
}

function monitorJob(kind, jobId) {
  stopMonitor(kind);
  setBadge(kind, "running", "ejecutando");
  setProgress(kind, 0, 0);
  appendLog(kind, [`[job ${jobId}] lanzado…`]);
  const es = new EventSource(api(`/api/jobs/${jobId}/stream`));
  monitors[kind] = es;
  setRunningUi(kind, true);
  es.onmessage = (e) => {
    let d;
    try { d = JSON.parse(e.data); } catch (_) { return; }
    handleStream(kind, d);
    if (d.closed) {
      es.close();
      if (monitors[kind] === es) monitors[kind] = null;
      setRunningUi(kind, false);
    }
  };
  es.onerror = () => {
    es.close();
    if (monitors[kind] === es) monitors[kind] = null;
    setRunningUi(kind, false);
  };
}

// ---------------------------------------------------------------- Train
function readTrainForm() {
  const fd = new FormData($("#formTrain"));
  const o = Object.fromEntries(fd.entries());
  const num = (k, d) => (o[k] === undefined || o[k] === "") ? d : Number(o[k]);
  return {
    object_class: (activeProject && activeProject.name) || "",
    defect_type: o.defect_type || "",
    run_name: o.run_name || undefined,
    max_train_steps: num("max_train_steps", 2000),
    batch_size: num("batch_size", 2),
    lora_rank: num("lora_rank", 8),
    lora_alpha: num("lora_alpha", 16),
    text_encoder_lr: num("text_encoder_lr", 4e-5),
    unet_lr: num("unet_lr", 2e-4),
    lr_warmup_steps: num("lr_warmup_steps", 100),
    save_steps: num("save_steps", 500),
    gradient_accumulation_steps: num("gradient_accumulation_steps", 2),
    lambda_defect: num("lambda_defect", 0.5),
    lambda_obj: num("lambda_obj", 0.2),
    lambda_attn: num("lambda_attn", 0.05),
    alpha: num("alpha", 0.3),
    dilate_mask: o.dilate_mask === "on",
  };
}

async function startTrain(ev) {
  ev.preventDefault();
  const payload = readTrainForm();
  if (!payload.object_class) { window.alert("Selecciona un proyecto activo (esquina superior derecha)."); return; }
  try {
    const res = await fetch(api("/api/train/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.job_id) throw new Error(data.detail || data.message || ("HTTP " + res.status));
    resetTrainCharts();
    const eta = $("#trainEtaLabel");
    if (eta) eta.textContent = "ETA —";
    $("#btnStopTrain").dataset.jobId = data.job_id;
    monitorJob("train", data.job_id);
  } catch (err) {
    window.alert("No se pudo iniciar el entrenamiento: " + err.message);
  }
}

async function stopTrain() {
  if (!monitors.train) return;
  const es = monitors.train;
  // job_id no está disponible una vez cerrado el monitor; se guarda en el atributo.
  const jobId = $("#btnStopTrain").dataset.jobId;
  if (!jobId) return;
  try { await fetch(api(`/api/jobs/${jobId}/stop`), { method: "POST" }); } catch (_) { /* ignore */ }
  es.close();
  monitors.train = null;
  setRunningUi("train", false);
}

// ---------------------------------------------------------------- Infer
function readInferForm() {
  const fd = new FormData($("#formInfer"));
  const o = Object.fromEntries(fd.entries());
  const num = (k, d) => (o[k] === undefined || o[k] === "") ? d : Number(o[k]);
  return {
    checkpoint: o.checkpoint || "",
    object_class: (activeProject && activeProject.name) || "",
    defect_type: o.defect_type || "",
    total_images: num("total_images", 6),
    num_samples: num("num_samples", 8),
    steps: num("steps", 50),
    guidance_scale: num("guidance_scale", 2.0),
    batch_size: num("batch_size", 4),
    lora_rank: num("lora_rank", 8),
    lora_alpha: num("lora_alpha", 16),
    dilate_mask: o.dilate_mask === "on",
    prompt: o.prompt || undefined,
  };
}

async function startInfer(ev) {
  ev.preventDefault();
  const payload = readInferForm();
  if (!payload.checkpoint) { window.alert("Selecciona un checkpoint entrenado."); return; }
  if (!payload.object_class) { window.alert("Selecciona un proyecto activo (esquina superior derecha)."); return; }
  try {
    const res = await fetch(api("/api/infer/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.job_id) throw new Error(data.detail || data.message || ("HTTP " + res.status));
    renderInferPreviews("", []);
    $("#btnStopInfer").dataset.jobId = data.job_id;
    monitorJob("infer", data.job_id);
  } catch (err) {
    window.alert("No se pudo iniciar la generación: " + err.message);
  }
}

async function stopInfer() {
  if (!monitors.infer) return;
  const es = monitors.infer;
  const jobId = $("#btnStopInfer").dataset.jobId;
  if (!jobId) return;
  try { await fetch(api(`/api/jobs/${jobId}/stop`), { method: "POST" }); } catch (_) { /* ignore */ }
  es.close();
  monitors.infer = null;
  setRunningUi("infer", false);
}

// ---------------------------------------------------------------- Validate checkpoints
let valJobId = null;
let valRuns = [];

async function loadValidateRuns() {
  const sel = $("#valRun");
  if (!sel) return;
  try {
    const proj = activeProject ? `?project=${encodeURIComponent(activeProject.name)}` : "";
    const res = await fetch(api(`/api/validate/runs${proj}`));
    valRuns = await res.json();
    const list = Array.isArray(valRuns) ? valRuns : [];
    sel.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = "— selecciona run —";
    sel.appendChild(ph);
    for (const r of list) {
      const opt = document.createElement("option");
      opt.value = `${r.object_class}::${r.run}`;
      opt.textContent = `${r.run} · ${r.object_class} (${r.checkpoints.length} ckpt)`;
      sel.appendChild(opt);
    }
    onValidateRunChange();
  } catch (err) {
    sel.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = "Error cargando runs";
    sel.appendChild(ph);
  }
}

function onValidateRunChange() {
  const sel = $("#valRun");
  const defSel = $("#valDefectType");
  if (!sel || !defSel) return;
  const key = sel.value;
  if (!key) {
    defSel.innerHTML = "";
    $("#valResults").innerHTML = '<p class="empty-state">Selecciona un run de entrenamiento.</p>';
    return;
  }
  const [objectClass, run] = key.split("::");
  const item = valRuns.find((r) => r.object_class === objectClass && r.run === run);
  defSel.innerHTML = "";
  const opts = item && item.defect_type ? [item.defect_type] : [];
  const cls = (datasetSummary || []).find((c) => c.object_class === objectClass);
  if (cls && cls.defect_types) {
    for (const t of cls.defect_types) if (!opts.includes(t.defect_type)) opts.push(t.defect_type);
  }
  if (!opts.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "— sin tipos —";
    defSel.appendChild(opt);
  } else {
    for (const t of opts) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      defSel.appendChild(opt);
    }
  }
  loadValidationResult();
}

async function loadValidationResult() {
  const box = $("#valResults");
  const sel = $("#valRun");
  if (!box || !sel || !sel.value) return;
  const [objectClass, run] = sel.value.split("::");
  try {
    const res = await fetch(api(`/api/validate/result?object_class=${encodeURIComponent(objectClass)}&run=${encodeURIComponent(run)}`));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    renderValidationResult(box, data.result);
  } catch (err) {
    box.innerHTML = `<p class="empty-state">Error cargando resultado: ${err.message}</p>`;
  }
}

function renderValidationResult(box, result) {
  if (!result || !Array.isArray(result.results) || !result.results.length) {
    box.innerHTML = '<p class="empty-state">Todavía no hay validaciones para este run.</p>';
    return;
  }
  const rows = result.results.slice().sort((a, b) => (a.step === b.step ? 0 : a.step < b.step ? -1 : 1));
  const html = `
    <div class="card-head-row">
      <h4>Resultado de validación</h4>
      <span class="tag">mejor: ${result.best ? result.best.split("/").pop() : "—"}</span>
    </div>
    ${result.warning ? `<p class="hint warn">${result.warning}</p>` : ""}
    <div class="validate-table">
      <div class="vt-head"><span>Checkpoint</span><span>KID ↓</span><span>IC-LPIPS ↑</span><span>Generadas</span><span></span></div>
      ${rows.map((r) => `
        <div class="vt-row ${r.best ? "vt-best" : ""}">
          <span class="mono">${r.filename || r.checkpoint.split("/").pop()}</span>
          <span class="mono">${Number.isFinite(r.kid_mean) ? r.kid_mean.toFixed(5) : "—"}</span>
          <span class="mono">${Number.isFinite(r.ic_lpips_mean) ? r.ic_lpips_mean.toFixed(4) : "—"}</span>
          <span class="mono">${r.num_generated ?? "—"}</span>
          <span>${r.best ? "★ mejor" : ""}${r.error ? `<span class="err">${r.error}</span>` : ""}</span>
        </div>`).join("")}
    </div>`;
  box.innerHTML = html;
}

function readValidateForm() {
  const fd = new FormData($("#formValidate"));
  const o = Object.fromEntries(fd.entries());
  const num = (k, d) => (o[k] === undefined || o[k] === "") ? d : Number(o[k]);
  const key = o.run || "";
  const [objectClass, run] = key.split("::");
  return {
    object_class: objectClass || "",
    run: run || "",
    defect_type: o.defect_type || "",
    num_eval_images: num("num_eval_images", 4),
    min_ic_lpips: num("min_ic_lpips", 0.05),
    steps: num("steps", 50),
    guidance_scale: num("guidance_scale", 2.0),
    prompt: o.prompt || undefined,
  };
}

async function startValidate(ev) {
  ev.preventDefault();
  const payload = readValidateForm();
  if (!payload.object_class || !payload.run) { window.alert("Selecciona un run de entrenamiento."); return; }
  if (!payload.defect_type) { window.alert("Selecciona un tipo de defecto."); return; }
  try {
    const res = await fetch(api("/api/validate/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.job_id) throw new Error(data.detail || data.message || ("HTTP " + res.status));
    valJobId = data.job_id;
    $("#btnStopValidate").dataset.jobId = data.job_id;
    monitorJob("eval", data.job_id);
  } catch (err) {
    window.alert("No se pudo iniciar la validación: " + err.message);
  }
}

async function stopValidate() {
  if (!monitors.eval) return;
  const es = monitors.eval;
  const jobId = valJobId || $("#btnStopValidate").dataset.jobId;
  if (jobId) {
    try { await fetch(api(`/api/jobs/${jobId}/stop`), { method: "POST" }); } catch (_) { /* ignore */ }
  }
  es.close();
  monitors.eval = null;
  setRunningUi("eval", false);
}

// ---------------------------------------------------------------- Results
async function loadResults() {
  const box = $("#resultsRuns");
  if (!box) return;
  try {
    const proj = activeProject ? `?project=${encodeURIComponent(activeProject.name)}` : "";
    const res = await fetch(api(`/api/generated/runs${proj}`));
    const runs = await res.json();
    box.innerHTML = "";
    if (!Array.isArray(runs) || !runs.length) {
      box.innerHTML = '<p class="empty-state">Sin resultados todavía.</p>';
      return;
    }
    const visibleRuns = runs
      .filter((r) => !activeProject || r.object_class === activeProject.name);
    for (const r of visibleRuns) {
      const block = document.createElement("div");
      block.className = "run-block";
      const head = document.createElement("div");
      head.className = "run-head";
      const name = document.createElement("span");
      name.className = "run-name";
      name.textContent = r.run;
      const state = (r.status && r.status.state) || "—";
      const tag = document.createElement("span");
      tag.className = "chip " + (state === "completed" ? "good" : state === "failed" ? "defect" : "");
      tag.textContent = state;
      head.appendChild(name);
      head.appendChild(tag);
      block.appendChild(head);

      const files = Array.isArray(r.files) ? r.files : [];
      const groups = {};
      for (const f of files) {
        const stem = f.name.replace(/\.png$/i, "").replace(/_(generated|original|mask)$/i, "");
        groups[stem] = groups[stem] || {};
        groups[stem][f.kind] = f;
      }
      const stems = Object.keys(groups).sort();
      if (!stems.length) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "Run sin imágenes (*_generated.png).";
        block.appendChild(empty);
      }
      const grid = document.createElement("div");
      grid.className = "triplet-grid";
      for (const stem of stems) {
        const g = groups[stem];
        const gen = g.generated && api(`/files/generated/${r.object_class}/generated/${r.run}/${g.generated.rel}`);
        if (!gen) continue;
        const orig = g.original ? api(`/files/generated/${r.object_class}/generated/${r.run}/${g.original.rel}`) : gen;
        const mask = g.mask ? api(`/files/generated/${r.object_class}/generated/${r.run}/${g.mask.rel}`) : gen;
        const lpips = (r.status && r.status.images || []).find((im) => String(im.output_idx) === String(parseInt(stem, 10)))?.lpips_score;
        grid.appendChild(el(`
          <div class="triplet">
            <div class="triplet-labels"><span>Original</span><span>Máscara</span><span>Generado</span></div>
            <div class="triplet-imgs">
              <div class="viewfinder">${corners()}<img src="${orig}" alt="original"></div>
              <div class="viewfinder">${corners()}<img src="${mask}" alt="máscara"></div>
              <div class="viewfinder">${corners()}<img src="${gen}" alt="generado"></div>
            </div>
            <div class="triplet-caption"><span>${stem}</span><span>LPIPS ${lpips != null ? lpips.toFixed(3) : "—"}</span></div>
          </div>`));
      }
      if (grid.children.length) block.appendChild(grid);
      else {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "Run sin tríos completos.";
        block.appendChild(empty);
      }
      box.appendChild(block);
    }
  } catch (err) {
    box.innerHTML = `<p class="empty-state">Error al cargar resultados: ${err.message}</p>`;
  }
}

// ---------------------------------------------------------------- System / jobs
async function loadSystem() {
  const box = $("#systemSpecs");
  if (!box) return;
  try {
    const info = await (await fetch(api("/api/system"))).json();
    box.innerHTML = `
      <div><dt>Python</dt><dd class="mono">${info.python || "—"}</dd></div>
      <div><dt>PyTorch</dt><dd class="mono">${info.torch_version || "—"}</dd></div>
      <div><dt>CUDA disponible</dt><dd class="mono">${info.cuda_available ? "sí" : "no"}</dd></div>
      <div><dt>GPU</dt><dd class="mono">${info.gpu_name || "—"}</dd></div>`;
  } catch (err) {
    box.innerHTML = `<div><dt>Error</dt><dd class="mono">${err.message}</dd></div>`;
  }
}

async function loadJobs() {
  const box = $("#jobsList");
  if (!box) return;
  try {
    const res = await fetch(api("/api/jobs"));
    const jobs = await res.json();
    box.innerHTML = "";
    if (!Array.isArray(jobs) || !jobs.length) {
      box.innerHTML = '<p class="empty-state">Ningún trabajo lanzado todavía.</p>';
      return;
    }
    for (const j of jobs) {
      const row = document.createElement("div");
      row.className = "job-row";
      const left = document.createElement("div");
      const kind = document.createElement("div");
      kind.textContent = `${j.kind} · ${j.meta && (j.meta.run_name || j.meta.object_class || "")}`;
      const id = document.createElement("div");
      id.className = "job-id";
      id.textContent = j.job_id;
      left.appendChild(kind);
      left.appendChild(id);
      const state = document.createElement("span");
      const st = j.alive ? "running" : (j.returncode == null ? "—" : j.returncode === 0 ? "completed" : "failed");
      state.className = "chip " + (st === "completed" ? "good" : st === "failed" || st === "running" ? "defect" : "");
      state.textContent = st;
      row.appendChild(left);
      row.appendChild(state);
      box.appendChild(row);
    }
  } catch (err) {
    box.innerHTML = `<p class="empty-state">Error al cargar trabajos: ${err.message}</p>`;
  }
}

// ---------------------------------------------------------------- Wire-up
function initTrainEvents() {
  const form = $("#formTrain");
  if (!form) return;
  form.addEventListener("submit", startTrain);
  $("#btnStopTrain").addEventListener("click", stopTrain);
  const toc = $("#trainObjectClass");
  if (toc) toc.addEventListener("change", () => populateDefectTypes("train"));
}

function initInferEvents() {
  const form = $("#formInfer");
  if (!form) return;
  form.addEventListener("submit", startInfer);
  $("#btnStopInfer").addEventListener("click", stopInfer);
  $("#inferDefectType").addEventListener("change", () => {});
}

function initDatasetEvents() {
  const goodBtn = $("#btnUploadGood");
  if (goodBtn) goodBtn.addEventListener("click", uploadGood);
  const defectBtn = $("#btnUploadDefect");
  if (defectBtn) defectBtn.addEventListener("click", uploadDefect);
  const sampleBtn = $("#btnLoadSample");
  if (sampleBtn) sampleBtn.addEventListener("click", loadSampleDataset);
}

function initValidateEvents() {
  const form = $("#formValidate");
  if (!form) return;
  form.addEventListener("submit", startValidate);
  const btn = $("#btnStartValidate");
  if (btn) btn.addEventListener("click", startValidate);
  const stop = $("#btnStopValidate");
  if (stop) stop.addEventListener("click", stopValidate);
  const runSel = $("#valRun");
  if (runSel) runSel.addEventListener("change", onValidateRunChange);
}

function initProjectEvents() {
  const sel = $("#projectSelect");
  if (!sel) return;
  sel.addEventListener("change", () => {
    setActiveProject(sel.value).then(() => {
      if (window.location.hash) {
        // nada especial: refresca lo ya abierto
      }
    });
  });
}

loadProjectState();
checkSystemPill();
resetTrainCharts();
initTrainEvents();
initInferEvents();
initValidateEvents();
initDatasetEvents();
initProjectEvents();
initHyperparamHelp();
window.addEventListener("resize", drawAllCharts);
