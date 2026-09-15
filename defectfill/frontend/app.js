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
    if (tab.dataset.tab === "train" || tab.dataset.tab === "infer" || tab.dataset.tab === "manual") {
      refreshSelectors();
    }
    if (tab.dataset.tab === "manual") loadManualSources();
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
  const defSel = kind === "train" ? $("#trainDefectType") : kind === "infer" ? $("#inferDefectType") : $("#manualDefectType");
  if (!defSel) return;
  const cls = (datasetSummary || []).find((c) => c.object_class === (activeProject && activeProject.name));
  const types = (cls && cls.defect_types) || [];
  populateSelect(defSel, types.map((t) => ({ value: t.defect_type, text: `${t.defect_type} (${t.image_count} img / ${t.mask_count} masc)` })), "— todos —");
  if (kind === "infer" || kind === "manual") defSel.disabled = !types.length;
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
  const ckptOptions = (Array.isArray(checkpoints) ? checkpoints : []).map((c) => ({
    value: c.path,
    text: `${c.run} / ${c.filename} (${fmtBytes(c.size_mb * 1e6)})`,
  }));
  populateSelect($("#inferCheckpoint"), ckptOptions, "— sin checkpoints —");
  populateSelect($("#manualCheckpoint"), ckptOptions, "— sin checkpoints —");
  populateDefectTypes("manual");
  loadManualSources();
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
  const boxManual = $("#manualActiveProject");
  if (boxManual) boxManual.textContent = activeProject ? activeProject.class_name : "— sin proyecto —";
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
const monitors = { train: null, infer: null, eval: null, manual: null };
const RUNNING = { train: false, infer: false, eval: false, manual: false };

// Prefijo DOM para cada tipo de job: train→train, infer→infer, eval→val, manual→manual.
const DOM_PREFIX = { train: "train", infer: "infer", eval: "val", manual: "manual" };
function domId(kind, base) {
  const prefix = DOM_PREFIX[kind] || "val";
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

function renderTripletGrid(grid, meta, images) {
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

function renderInferPreviews(meta, images) {
  renderTripletGrid($("#inferPreviewGrid"), meta, images);
}

function renderManualPreviews(meta, images) {
  renderTripletGrid($("#manualPreviewGrid"), meta, images);
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
    } else if (kind === "manual") {
      if (status.images) renderManualPreviews(runFromSummary(d.summary), status.images);
      if (status.error) appendLog("manual", [`[error] ${status.error}`]);
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
  } else if (kind === "manual") {
    const start = $("#btnStartManual"), stop = $("#btnStopManual");
    if (start) start.disabled = running || !manualMask.hasSource();
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

// ---------------------------------------------------------------- Inferencia manual
let manualSources = [];

async function loadManualSources() {
  const sel = $("#manualSourceImage");
  if (!sel) return;
  const proj = activeProject ? `?project=${encodeURIComponent(activeProject.name)}` : "";
  const prev = sel.value;
  try {
    const res = await fetch(api(`/api/infer/manual/sources${proj}`));
    const data = await res.json();
    manualSources = (data && data.sources) || [];
  } catch (_) { manualSources = []; }
  sel.innerHTML = "";
  const ph = document.createElement("option");
  ph.value = "";
  ph.textContent = manualSources.length ? "— elige imagen —" : "— sin imágenes disponibles —";
  sel.appendChild(ph);
  // Solo crops de good: agrupados por tamaño es innecesario (homogéneo)
  for (const s of manualSources) {
    const opt = document.createElement("option");
    opt.value = s.rel;
    opt.textContent = `${s.name}  (${s.width}×${s.height})`;
    sel.appendChild(opt);
  }
  if (prev && manualSources.some((s) => s.rel === prev)) sel.value = prev;
  onManualSourceChange();
}

// Estado del editor de máscara (pincel/goma/rect, a resolución natural).
const manualMask = {
  natural: null,       // {w, h} resolución natural de la imagen actual
  maskCanvas: null,    // canvas a resolución natural (blanco opaco = defecto)
  current: null,       // {rel, name, kind, width, height}
  tool: "brush",
  brushSize: 20,
  drawing: false,
  lastPos: null,
  rectStart: null,
  rectCurrent: null,
  cursor: null,
  token: 0,

  hasSource() { return !!this.current; },

  hasPixels() {
    if (!this.maskCanvas) return false;
    const ctx = this.maskCanvas.getContext("2d");
    const d = ctx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    for (let i = 3; i < d.data.length; i += 4) if (d.data[i] > 0) return true;
    return false;
  },

  asDataUrlB64() {
    return this.maskCanvas ? (this.maskCanvas.toDataURL("image/png").split(",")[1] || "") : "";
  },
};

function manualStageStatus(text, isError) {
  const st = $("#manualMaskStatus");
  if (!st) return;
  st.textContent = text;
  st.classList.toggle("error", Boolean(isError));
  st.classList.toggle("ok", !isError && text !== "—");
}

function manualSetImage(src) {
  const im = manualSources.find((s) => s.rel === src) || null;
  const empty = $("#manualStageEmpty"), wrap = $("#manualImageWrap");
  const img = $("#manualImage"), canvas = $("#manualEditCanvas");
  const btn = $("#btnStartManual");
  const tok = ++manualMask.token;
  manualMask.current = im;
  manualMask.natural = im ? { w: im.width, h: im.height } : null;
  manualMask.maskCanvas = null;
  manualMask.rectStart = null;
  manualMask.rectCurrent = null;
  manualMask.drawing = false;
  manualMask.lastPos = null;
  manualMask.cursor = null;
  if (btn) btn.disabled = !im || RUNNING.manual;
  if (!im) {
    empty.style.display = "flex";
    empty.querySelector("p").textContent = "Selecciona una imagen de origen para pintar la máscara.";
    wrap.style.display = "none";
    img.removeAttribute("src");
    canvas.style.display = "none";
    manualStageStatus("—");
    return;
  }
  empty.style.display = "none";
  wrap.style.display = "block";
  img.style.visibility = "hidden";
  img.onload = () => {
    if (manualMask.token !== tok) return;
    img.style.visibility = "visible";
    canvas.style.display = "block";
    manualSyncCanvasBuffer();
    const c = document.createElement("canvas");
    c.width = Math.max(1, im.width);
    c.height = Math.max(1, im.height);
    manualMask.maskCanvas = c;
    manualRender();
    manualStageStatus(`Máscara lista para ${im.name} (${im.width}×${im.height}).`);
  };
  img.onerror = () => {
    if (manualMask.token !== tok) return;
    empty.style.display = "flex";
    empty.querySelector("p").textContent = "No se pudo cargar la imagen.";
    wrap.style.display = "none";
    manualStageStatus("No se pudo cargar la imagen de origen.", true);
  };
  img.src = api(`/files/data/${activeProject.name}/${src}`) + "?t=" + Date.now();
}

function manualSyncCanvasBuffer() {
  const canvas = $("#manualEditCanvas");
  const rect = canvas.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  manualRender();
}

function manualRender() {
  const v = $("#manualEditCanvas");
  if (!v) return;
  const vctx = v.getContext("2d");
  vctx.clearRect(0, 0, v.width, v.height);
  if (!manualMask.maskCanvas || !manualMask.natural) return;

  // Máscara en rojo translúcido (destino-in recorta al área pintada)
  vctx.globalCompositeOperation = "source-over";
  vctx.fillStyle = "rgba(239, 83, 80, 0.35)";
  vctx.fillRect(0, 0, v.width, v.height);
  vctx.globalCompositeOperation = "destination-in";
  vctx.drawImage(manualMask.maskCanvas, 0, 0, manualMask.natural.w, manualMask.natural.h, 0, 0, v.width, v.height);
  vctx.globalCompositeOperation = "source-over";

  // Rectángulo en curso
  if (manualMask.rectStart && manualMask.rectCurrent) {
    const sx = manualMask.rectStart, ex = manualMask.rectCurrent;
    vctx.save();
    vctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    vctx.lineWidth = 1.5;
    vctx.setLineDash([5, 4]);
    vctx.strokeRect(Math.min(sx.x, ex.x), Math.min(sx.y, ex.y), Math.abs(ex.x - sx.x), Math.abs(ex.y - sx.y));
    vctx.restore();
  }

  // Anillo del pincel — igual que en el HMI (preprocesado), con corrección DPR
  // para que el anillo coincida exactamente con lo que se pinta.
  if (manualMask.cursor && manualMask.tool !== "rect") {
    const r = v.getBoundingClientRect();
    const dpr = r.width > 0 ? v.width / r.width : (window.devicePixelRatio || 1);
    vctx.save();
    vctx.strokeStyle = "rgba(255,255,255,0.85)";
    vctx.fillStyle = "rgba(255,255,255,0.12)";
    vctx.lineWidth = 1.25;
    vctx.beginPath();
    vctx.arc(manualMask.cursor.x, manualMask.cursor.y, manualMask.brushSize * dpr, 0, Math.PI * 2);
    vctx.fill();
    vctx.stroke();
    vctx.restore();
  }
}

function manualDisplayToNatural(clientX, clientY) {
  const r = $("#manualEditCanvas").getBoundingClientRect();
  return {
    x: ((clientX - r.left) * manualMask.natural.w) / r.width,
    y: ((clientY - r.top) * manualMask.natural.h) / r.height,
  };
}

function manualNaturalBrushRadius() {
  const r = $("#manualEditCanvas").getBoundingClientRect();
  return manualMask.brushSize * (manualMask.natural.w / r.width);
}

function manualPaintStroke(from, to) {
  const ctx = manualMask.maskCanvas.getContext("2d");
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(1, manualNaturalBrushRadius() * 2);
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  if (manualMask.tool === "eraser") {
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
  } else {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "#fff";
  }
  ctx.stroke();
  ctx.restore();
}

function manualPaintDot(x, y) {
  const ctx = manualMask.maskCanvas.getContext("2d");
  ctx.save();
  ctx.globalCompositeOperation = manualMask.tool === "eraser" ? "destination-out" : "source-over";
  ctx.fillStyle = manualMask.tool === "eraser" ? "rgba(0,0,0,1)" : "#fff";
  ctx.beginPath();
  ctx.arc(x, y, Math.max(1, manualNaturalBrushRadius()), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function manualFinishRect(startNat, endNat) {
  const ctx = manualMask.maskCanvas.getContext("2d");
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "#fff";
  ctx.fillRect(
    Math.min(startNat.x, endNat.x), Math.min(startNat.y, endNat.y),
    Math.abs(endNat.x - startNat.x), Math.abs(endNat.y - startNat.y)
  );
  ctx.restore();
}

function manualDispPoint(clientX, clientY) {
  const r = $("#manualEditCanvas").getBoundingClientRect();
  return {
    x: (clientX - r.left) * ($("#manualEditCanvas").width / r.width),
    y: (clientY - r.top) * ($("#manualEditCanvas").height / r.height),
  };
}

function manualOnPointerDown(e) {
  if (!manualMask.maskCanvas || !manualMask.natural) return;
  e.preventDefault();
  const canvas = $("#manualEditCanvas");
  canvas.setPointerCapture(e.pointerId);
  const disp = manualDispPoint(e.clientX, e.clientY);
  const nat = manualDisplayToNatural(e.clientX, e.clientY);
  if (manualMask.tool === "rect") {
    manualMask.rectStart = disp;
    manualMask.rectCurrent = disp;
  } else {
    manualMask.drawing = true;
    manualMask.lastPos = nat;
    manualPaintDot(nat.x, nat.y);
  }
  manualMask.cursor = disp;
  manualRender();
}

function manualOnPointerMove(e) {
  if (!manualMask.natural || !manualMask.maskCanvas) return;
  const disp = manualDispPoint(e.clientX, e.clientY);
  manualMask.cursor = disp;
  if (manualMask.rectStart) {
    manualMask.rectCurrent = disp;
    manualRender();
    return;
  }
  if (!manualMask.drawing) { manualRender(); return; }
  const nat = manualDisplayToNatural(e.clientX, e.clientY);
  if (manualMask.lastPos) {
    manualPaintStroke(manualMask.lastPos, nat);
    manualMask.lastPos = nat;
  } else {
    manualPaintDot(nat.x, nat.y);
    manualMask.lastPos = nat;
  }
  manualRender();
}

function manualOnPointerUp(e) {
  if (manualMask.rectStart) {
    const r = $("#manualEditCanvas").getBoundingClientRect();
    const sx = manualMask.rectStart;
    const startNat = {
      x: (sx.x * manualMask.natural.w) / $("#manualEditCanvas").width,
      y: (sx.y * manualMask.natural.h) / $("#manualEditCanvas").height,
    };
    const endNat = manualDisplayToNatural(e.clientX, e.clientY);
    manualFinishRect(startNat, endNat);
    manualMask.rectStart = null;
    manualMask.rectCurrent = null;
    manualRender();
  }
  manualMask.drawing = false;
  manualMask.lastPos = null;
  try { $("#manualEditCanvas").releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
}

function manualSetTool(tool) {
  manualMask.tool = tool;
  document.querySelectorAll("#manualMaskToolbar .tool-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.tool === tool);
  });
  const st = $("#manualMaskStatus");
  if (st) st.textContent = "Dibuja sobre la imagen para marcar el área del defecto.";
}

function manualClearMask() {
  if (!manualMask.maskCanvas || !manualMask.current) return;
  manualMask.maskCanvas.getContext("2d").clearRect(0, 0, manualMask.maskCanvas.width, manualMask.maskCanvas.height);
  manualRender();
  manualStageStatus("Máscara vaciada. Pinta el área donde quieres el defecto.");
}

// Convierte el brillo de la máscara importada (blanco/negro) a la
// representación interna del editor: píxel blanco opaco = defecto,
// transparente = zona vacía. Así queda lista para retocar con pincel/goma.
function manualApplyImportedMask(draw) {
  const W = manualMask.maskCanvas.width;
  const H = manualMask.maskCanvas.height;
  const ctx = manualMask.maskCanvas.getContext("2d");
  const tmp = document.createElement("canvas");
  tmp.width = W;
  tmp.height = H;
  const tctx = tmp.getContext("2d");
  tctx.imageSmoothingEnabled = false;
  tctx.drawImage(draw, 0, 0, W, H);
  const src = tctx.getImageData(0, 0, W, H).data;
  const out = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const r = src[i * 4], g = src[i * 4 + 1], b = src[i * 4 + 2];
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;   // brillo = intensidad del defecto
    out.data[i * 4] = 255;
    out.data[i * 4 + 1] = 255;
    out.data[i * 4 + 2] = 255;
    out.data[i * 4 + 3] = lum;                          // alfa = brillo (0=vacío, 255=defecto)
  }
  ctx.clearRect(0, 0, W, H);
  ctx.putImageData(out, 0, 0);
  manualRender();
}

function manualImportMask(file) {
  if (!manualMask.maskCanvas || !manualMask.current) {
    manualStageStatus("Selecciona una imagen de origen antes de importar.", true);
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const im = new Image();
    im.onload = () => manualApplyImportedMask(im);
    im.onerror = () => manualStageStatus("No se pudo leer la máscara importada.", true);
    im.src = reader.result;
  };
  reader.onerror = () => manualStageStatus("No se pudo leer el archivo.", true);
  reader.readAsDataURL(file);
}

function manualOnSourceChange() {
  const sel = $("#manualSourceImage");
  manualSetImage(sel ? sel.value : "");
}

function readManualForm() {
  const fd = new FormData($("#formManual"));
  const o = Object.fromEntries(fd.entries());
  const num = (k, d) => (o[k] === undefined || o[k] === "") ? d : Number(o[k]);
  return {
    checkpoint: o.checkpoint || "",
    object_class: (activeProject && activeProject.name) || "",
    defect_type: o.defect_type || "",
    run_name: o.run_name || undefined,
    image: o.image || "",
    mask_b64: manualMask.asDataUrlB64() || "",
    total_images: num("total_images", 4),
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

async function startManual(ev) {
  ev.preventDefault();
  const payload = readManualForm();
  if (!payload.checkpoint) { window.alert("Selecciona un checkpoint entrenado."); return; }
  if (!payload.object_class) { window.alert("Selecciona un proyecto activo."); return; }
  if (!payload.image) { window.alert("Selecciona una imagen de origen."); return; }
  if (!payload.mask_b64) { window.alert("Pinta una máscara sobre la imagen antes de generar."); return; }
  if (!manualMask.hasPixels()) { window.alert("La máscara está vacía: pinta con el pincel el área del defecto."); return; }
  try {
    const res = await fetch(api("/api/infer/manual/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.job_id) throw new Error(data.detail || data.message || ("HTTP " + res.status));
    renderManualPreviews("", []);
    $("#btnStopManual").dataset.jobId = data.job_id;
    monitorJob("manual", data.job_id);
  } catch (err) {
    window.alert("No se pudo iniciar la inferencia manual: " + err.message);
  }
}

async function stopManual() {
  if (!monitors.manual) return;
  const es = monitors.manual;
  const jobId = $("#btnStopManual").dataset.jobId;
  if (!jobId) return;
  try { await fetch(api(`/api/jobs/${jobId}/stop`), { method: "POST" }); } catch (_) { /* ignore */ }
  es.close();
  monitors.manual = null;
  setRunningUi("manual", false);
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
      const stems = Object.keys(groups).sort().reverse();
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
        const stemIdx = (stem.match(/_gen(\d+)$/) || [])[1];
        const lpips = stemIdx == null ? null : (r.status && r.status.images || []).find((im) => String(im.output_idx) === String(parseInt(stemIdx, 10)))?.lpips_score;
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

function initManualEvents() {
  const form = $("#formManual");
  if (!form) return;
  form.addEventListener("submit", startManual);
  $("#btnStopManual").addEventListener("click", stopManual);
  $("#manualSourceImage").addEventListener("change", manualOnSourceChange);
  const canvas = $("#manualEditCanvas");
  if (canvas) {
    canvas.addEventListener("pointerdown", manualOnPointerDown);
    canvas.addEventListener("pointermove", manualOnPointerMove);
    canvas.addEventListener("pointerup", manualOnPointerUp);
    canvas.addEventListener("pointercancel", manualOnPointerUp);
    canvas.addEventListener("pointerleave", () => {
      manualMask.cursor = null;
      manualRender();
    });
  }
  const toolbar = $("#manualMaskToolbar");
  if (toolbar) toolbar.addEventListener("click", (e) => {
    const btn = e.target.closest(".tool-btn");
    if (!btn) return;
    if (btn.dataset.tool === "clear") manualClearMask();
    else if (btn.dataset.tool === "import") {
      const file = $("#manualImportFile");
      if (file) file.click();
    }
    else manualSetTool(btn.dataset.tool);
  });
  const importFile = $("#manualImportFile");
  if (importFile) importFile.addEventListener("change", () => {
    if (importFile.files && importFile.files[0]) manualImportMask(importFile.files[0]);
    importFile.value = "";
  });
  const brush = $("#manualBrushSize");
  if (brush) {
    brush.addEventListener("input", () => {
      manualMask.brushSize = Number(brush.value);
      const val = $("#manualBrushSizeValue");
      if (val) val.textContent = brush.value + "px";
      manualRender();
    });
  }
  window.addEventListener("resize", () => {
    if (manualMask.current) manualSyncCanvasBuffer();
  });
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
initManualEvents();
initValidateEvents();
initDatasetEvents();
initProjectEvents();
initHyperparamHelp();
window.addEventListener("resize", drawAllCharts);
