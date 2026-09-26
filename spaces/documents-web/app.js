// solvi documents: the page. Extraction runs in extractor.worker.js (onnxruntime-web), decisions in Pyodide (solvi_docs.py).
import { USE_CASES } from "./usecases/index.js";

const P = new URLSearchParams(location.search);
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const MODELS = {
  base: { id: "solvi-ai/extract-base", name: "extract-base", label: "extract-base: general, fields by description" },
  receipts: { id: "solvi-ai/extract-receipts", name: "extract-receipts", label: "extract-receipts: fine-tuned on receipts (+790 MB)" },
};
const FP16 = "onnx/model_fp16.onnx";
const Q4 = "onnx/model_q4.onnx";
const repoOf = (k) => P.get(k + "_repo") || (k === "base" && P.get("repo")) || `https://huggingface.co/${MODELS[k].id}/resolve/main/`;
const PALETTE = ["#e8590c", "#2f9e44", "#1c7ed6", "#c2255c", "#7048e8", "#0c8599", "#d9480f", "#5c940d", "#ae3ec9", "#1098ad", "#f08c00", "#364fc7"];
const PY_KEYWORDS = new Set(("False None True and as assert async await break class continue def del elif else except finally for from " +
  "global if import in is lambda nonlocal not or pass raise return try while with yield match case").split(" "));
const RESERVED = new Set(["doc", "today", "cat", "Answer", "Question", "Quote", "date", "datetime", "timedelta", "re", "math",
  "money", "number", "parse_date", "days", "hours", "found", "need", "mentions", "iban_ok", "QUESTIONS", "INPUTS"]);

const $ = (id) => document.getElementById(id);
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("--")) n.style.setProperty(k, v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) n.append(c.nodeType ? c : String(c));
  return n;
}
const fmtMs = (ms) => ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`;
const fmtMB = (b) => `${(b / 1e6).toFixed(0)} MB`;
function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0).toString(36) + ":" + s.length; }
const todayIso = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);

// ------------------------------------------------------------------------------------------------------------ state
const st = {
  ucIdx: 0, docIdx: 0, today: todayIso(),
  models: Object.fromEntries(Object.keys(MODELS).map((k) => [k, { status: "idle" }])),
  onnx: P.get("onnx") || FP16,
  cache: new Map(),              // extraction key -> result
  job: null, jobSeq: 0, runToken: 0,
  py: null, pyStatus: "loading", pyPromise: null,
  last: null, lastError: null, tab: "flow",
  gpu: null, metrics: [],
};
window.__solvi = st;                               // for debugging and automated tests
const uc = () => USE_CASES[st.ucIdx];

function storeKey(u) { return "solvi-documents:" + u.id; }
function loadEdits(u) { try { return JSON.parse(localStorage.getItem(storeKey(u)) || "{}"); } catch { return {}; } }
function saveEdits(u, patch) {
  const e = { ...loadEdits(u), ...patch };
  try { localStorage.setItem(storeKey(u), JSON.stringify(e)); } catch { /* storage unavailable: edits live for this visit only */ }
  memEdits[u.id] = e;
}
const memEdits = {};
const edits = (u = uc()) => memEdits[u.id] ?? (memEdits[u.id] = loadEdits(u));
const fields = () => (edits().fields ?? uc().fields).map(([n, d]) => [n, d]);
const code = () => edits().code ?? uc().code;
const modelKey = () => (uc().altModel && edits().model === uc().altModel) ? uc().altModel : "base";
function currentText() {
  if (st.docIdx === "own") return (edits().own || "").normalize("NFC");
  return uc().docs[st.docIdx].text.normalize("NFC");
}
function fieldError(name, i, list) {
  if (!name) return "name needed";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return "letters, digits, _";
  if (PY_KEYWORDS.has(name) || RESERVED.has(name)) return "reserved name";
  if (list.findIndex(([n]) => n === name) !== i) return "duplicate";
  return "";
}
const validFields = () => fields().filter(([n, d], i, all) => d.trim() && !fieldError(n, i, all));
const colorOf = (name) => { const i = fields().findIndex(([n]) => n === name); return PALETTE[(i < 0 ? 0 : i) % PALETTE.length]; };
const cacheKey = (key, text, name, desc) => `${key}|${st.onnx}|${hashStr(text)}|${name}|${desc}`;
const hitFor = (name, desc, text = currentText()) => st.cache.get(cacheKey(modelKey(), text, name, desc));

// ------------------------------------------------------------------------------------------------------------ worker
const worker = new Worker(new URL("./extractor.worker.js", import.meta.url), { type: "module" });
worker.onmessage = (ev) => onWorker(ev.data);
worker.onerror = (e) => { showNote(`The extractor worker failed to start: ${e.message || e}. Try a current Chrome, Edge or Firefox.`); };
const pending = new Map();
let reqSeq = 0;
function ask(msg) { const id = "r" + (++reqSeq); return new Promise((res) => { pending.set(id, res); worker.postMessage({ ...msg, id }); }); }

function onWorker(m) {
  if ((m.type === "probe" || m.type === "cached") && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.type === "hello") { st.workerInfo = m; return; }
  if (m.type === "warn") { console.warn("[extractor]", m.message); return; }
  if (m.type === "unloaded") { if (st.models[m.key]) st.models[m.key] = { status: "idle" }; renderEngine(); return; }
  if (m.type === "progress" || m.type === "ready" || (m.type === "error" && !m.id)) return onModelMsg(m);
  const job = st.job;
  if (!job || m.id !== job.id) return;
  if (m.type === "window") { job.onWindow?.(m); return; }
  if (m.type === "field") { job.onField?.(m); return; }
  if (m.type === "done") { job.resolve(m); return; }
  if (m.type === "error") { job.reject(new Error(m.message)); }
}

// ------------------------------------------------------------------------------------------------------------ model
function onModelMsg(m) {
  const md = st.models[m.key];
  if (!md) return;
  if (m.type === "progress") { md.status = "loading"; md.progress = m; }
  else if (m.type === "ready") {
    md.status = "ready"; md.info = m; md.resolve?.(m);
    st.metrics.push({ what: "model-ready", key: m.key, backend: m.backend, fromCache: m.fromCache, totalMs: m.totalMs, downloadMs: m.downloadMs, sessionMs: m.sessionMs });
    console.info(`[solvi documents] ${MODELS[m.key].name} ready on ${m.backend} (${m.fromCache ? "cache" : "download"}): ` +
      `download/read ${fmtMs(m.downloadMs)}, session ${fmtMs(m.sessionMs)}, total ${fmtMs(m.totalMs)}`);
    if (m.key === modelKey()) analyze({ auto: true });
  } else if (m.type === "error") {
    md.status = "error"; md.error = m; md.reject?.(new Error(m.message));
    handleModelError(m);
  }
  renderEngine();
}

async function handleModelError(m) {
  if (m.code === "fp16-wasm") {
    const probe = await ask({ type: "probe", repo: repoOf(m.key), onnx: Q4 });
    const q4 = probe.ok && st.onnx !== Q4;
    showNote(`This browser has no WebGPU, and the fp16 model could not run on the CPU backend (${m.message.slice(0, 160)}). ` +
      `Use Chrome or Edge 113+ (desktop) with WebGPU enabled.` + (q4 ? " Or run the smaller 4-bit model on the CPU (422 MB, " +
      "same span as the full model on about 84% of fields):" : ""), q4 ? el("button", { class: "btn small", onclick: () => switchToQ4() }, "Use the 4-bit model") : null);
  } else {
    showNote(`The model could not be loaded: ${m.message.slice(0, 300)}`, el("button", { class: "btn small", onclick: () => { st.models[m.key] = { status: "idle" }; loadModel(m.key); } }, "Try again"));
  }
}

function switchToQ4() {
  st.onnx = Q4;
  for (const k of Object.keys(st.models)) st.models[k] = { status: "idle" };
  hideNote();
  loadModel(modelKey());
}

function loadModel(key) {
  const md = st.models[key];
  if (md.status === "loading" || md.status === "ready") return md.promise;
  md.status = "loading";
  md.progress = { phase: "files", loaded: 0, total: 0 };
  md.promise = new Promise((res, rej) => { md.resolve = res; md.reject = rej; });
  md.promise.catch(() => {});
  worker.postMessage({ type: "init", key, repo: repoOf(key), onnx: st.onnx, backend: P.get("backend") || undefined });
  renderEngine();
  return md.promise;
}

async function detectGpu() {
  try {
    const a = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    st.gpu = !!a && (a.features.has("shader-f16") || !/fp16/.test(st.onnx));
  } catch { st.gpu = false; }
  if (P.get("backend") === "wasm") st.gpu = false;
  renderEngine();
}

function renderEngine() {
  const key = modelKey();
  const md = st.models[key];
  const name = MODELS[key].name + (st.onnx === Q4 ? " (4-bit)" : "");
  const dot = $("model-dot"), line = $("model-line"), sub = $("model-sub"), bar = $("model-bar"), btn = $("model-load");
  dot.className = "dot " + ({ loading: "busy", ready: "ok", error: "bad" }[md.status] || "");
  bar.hidden = md.status !== "loading";
  btn.hidden = md.status === "loading" || md.status === "ready";
  const size = st.onnx === Q4 ? "422 MB" : "790 MB";
  const where = st.gpu === null ? "" : st.gpu ? "Runs on your GPU (WebGPU)." :
    "No WebGPU in this browser: it will run on the CPU (WebAssembly), which is several times slower.";
  if (md.status === "idle") {
    line.textContent = `Extractor ${name}: not loaded`;
    sub.textContent = `${size}, downloaded once from Hugging Face and cached by your browser. ${where}`;
    btn.textContent = `Load the model (${size})`;
  } else if (md.status === "loading") {
    const p = md.progress || {};
    const pct = p.total ? Math.min(100, 100 * p.loaded / p.total) : 0;
    $("model-bar-fill").style.width = (p.phase === "session" ? 100 : pct) + "%";
    line.textContent = p.phase === "download" ? `Downloading ${name}: ${fmtMB(p.loaded)}${p.total ? " / " + fmtMB(p.total) : ""}` :
      p.phase === "cache" ? `Reading ${name} from the browser cache: ${fmtMB(p.loaded)} / ${fmtMB(p.total)}` :
      p.phase === "session" ? `Starting ${name} on ${p.backend === "webgpu" ? "WebGPU" : "WebAssembly (CPU)"}…` : `Loading ${name}: tokenizer and settings…`;
    sub.textContent = p.phase === "download" ? "Only once: later visits read it from the browser cache." : where;
  } else if (md.status === "ready") {
    const i = md.info;
    line.textContent = `Extractor ${name}: ready on ${i.backend === "webgpu" ? "WebGPU (GPU)" : `WebAssembly (CPU, ${i.threads} thread${i.threads > 1 ? "s" : ""})`}`;
    sub.textContent = `${i.fromCache ? "Loaded from the browser cache" : "Downloaded and cached"} in ${fmtMs(i.totalMs)}. Your document is read in this tab.`;
  } else {
    line.textContent = `Extractor ${name}: failed to load`;
    sub.textContent = md.error?.message?.slice(0, 160) || "";
    btn.hidden = false; btn.textContent = "Retry";
  }
  const pd = $("py-dot");
  pd.className = "dot " + ({ loading: "busy", ready: "ok", error: "bad" }[st.pyStatus] || "");
  if (md.status === "ready" && md.info.backend === "wasm") {
    const iframe = window.top !== window;
    const why = md.info.gpuState === "no-f16" ? "Your GPU is visible to WebGPU but without 16-bit float shaders (shader-f16), which this fp16 model needs, so it runs on the CPU: " :
      md.info.gpuFailed ? "WebGPU could not run the model here, so it runs on the CPU: " : "Running on the CPU: ";
    const msg = why + `about 2 s per field on a receipt and 5–9 s per 1 000-token window with 8 threads` +
      (md.info.threads <= 1 ? "; this tab runs single-threaded (no cross-origin isolation), so expect several times more." : ".") +
      " A browser with WebGPU (Chrome or Edge on desktop) is about 5× faster.";
    showNote(msg, md.info.threads <= 1 && iframe ? el("a", { class: "btn small", href: location.href, target: "_blank", rel: "noopener" }, "Open in its own tab (multi-threaded)") : null, true);
  }
}
let noteSticky = false;
function showNote(text, extra = null, soft = false) {
  if (soft && noteSticky) return;
  const n = $("engine-note");
  n.replaceChildren(text, extra ? " " : "", extra || "");
  n.hidden = false;
  noteSticky = !soft;
}
function hideNote() { $("engine-note").hidden = true; noteSticky = false; }

// ------------------------------------------------------------------------------------------------------------ python
async function loadPython() {
  const t0 = performance.now();
  try {
    $("py-line").textContent = "Python (Pyodide) + solvi: downloading Python…";
    const { loadPyodide } = await import(PYODIDE + "pyodide.mjs");
    const py = await loadPyodide({ indexURL: PYODIDE });
    $("py-line").textContent = "Python (Pyodide) + solvi: installing numpy and solvi…";
    await py.loadPackage(["numpy", "micropip"]);
    try {                         // solvi's own code needs only numpy at import time: skip scipy (another ~10 MB)
      await py.runPythonAsync("import micropip\nawait micropip.install('solvi>=0.2.1', deps=False)\nimport solvi");
    } catch (e) {
      console.warn("solvi without dependencies failed, installing with dependencies", e);
      await py.runPythonAsync("import micropip\nawait micropip.install('solvi>=0.2.1')\nimport solvi");
    }
    const src = await (await fetch(new URL("solvi_docs.py", import.meta.url))).text();
    py.FS.writeFile("/home/pyodide/solvi_docs.py", src);
    py.runPython("import sys\nif '/home/pyodide' not in sys.path: sys.path.insert(0, '/home/pyodide')");
    st.py = py.pyimport("solvi_docs");
    const ver = py.runPython("from importlib.metadata import version\nversion('solvi')");
    st.pyStatus = "ready";
    st.pyVersion = ver;
    $("py-line").textContent = `Python ${py.version ? "(Pyodide " + py.version + ")" : ""} + solvi ${ver}: ready`;
    $("py-sub").textContent = `Loaded in ${fmtMs(performance.now() - t0)}. Rules, checks, the strategist and the trace replay run here.`;
    st.metrics.push({ what: "python-ready", ms: performance.now() - t0, solvi: ver });
    console.info(`[solvi documents] Python + solvi ${ver} ready in ${fmtMs(performance.now() - t0)}`);
  } catch (e) {
    st.pyStatus = "error";
    $("py-line").textContent = "Python failed to load";
    $("py-sub").textContent = String(e).slice(0, 200);
    console.error(e);
  }
  renderEngine();
}

// ------------------------------------------------------------------------------------------------------------ run
function cancelJob() {
  if (st.job) { worker.postMessage({ type: "cancel", id: st.job.id }); st.job.resolve({ cancelled: true }); st.job = null; }
}

async function extractMissing(key, text, list, token) {
  cancelJob();
  const id = "j" + (++st.jobSeq);
  const t0 = performance.now();
  let current = 0;
  const status = $("extract-status");
  const job = { id };
  const done = new Promise((res, rej) => { job.resolve = res; job.reject = rej; });
  job.onWindow = (m) => {
    status.replaceChildren(`Reading for “${m.name}” (${current + 1}/${list.length})` + (m.n > 1 ? `, window ${m.w}/${m.n}` : "") + "… ",
      el("button", { class: "btn small", onclick: () => { cancelJob(); status.textContent = "Cancelled."; } }, "Cancel"));
  };
  job.onField = (m) => {
    current++;
    st.cache.set(cacheKey(key, text, m.name, m.desc), { ...m.result, ms: m.ms });
    st.metrics.push({ what: "field", name: m.name, ms: m.ms, windows: m.result.windows, tokens: m.result.tokens, backend: st.models[key].info?.backend });
    if (token === st.runToken) { updateFound(m.name); renderDoc(); }
  };
  st.job = job;
  status.textContent = `Reading the document for ${list.length} field${list.length > 1 ? "s" : ""}…`;
  worker.postMessage({ type: "extract", id, key, text, fields: list.map(([name, desc]) => ({ name, desc })) });
  const r = await done;
  if (st.job === job) st.job = null;
  if (!r.cancelled) {
    const ms = performance.now() - t0;
    status.textContent = `Extracted ${list.length} field${list.length > 1 ? "s" : ""} in ${fmtMs(ms)} (${fmtMs(ms / list.length)} per field) on ` +
      `${st.models[key].info.backend === "webgpu" ? "WebGPU" : "WebAssembly"}.`;
  }
  return !r.cancelled;
}

async function analyze({ auto = false } = {}) {
  const token = ++st.runToken;
  const text = currentText();
  renderDoc();
  if (!text.trim()) { renderAnswersEmpty("Paste a document first."); return; }
  const key = modelKey();
  const list = validFields();
  const missing = list.filter(([n, d]) => !st.cache.has(cacheKey(key, text, n, d)));
  if (missing.length) {
    const md = st.models[key];
    if (md.status !== "ready") {
      if (auto && md.status !== "loading") {
        renderAnswersEmpty(`Load the model (top bar) to read this document. The ${MODELS[key].name} extractor finds each field described in the table, then the rules decide.`);
        return;
      }
      try { await loadModel(key); } catch { return; }
      if (token !== st.runToken) return;
    }
    const ok = await extractMissing(key, text, missing, token);
    if (!ok || token !== st.runToken) return;
  } else if (list.length) {
    $("extract-status").textContent = "All fields already extracted for this text (cached in this tab).";
  }
  renderDoc();
  for (const [n] of fields()) updateFound(n);
  await decide(token);
}

async function decide(token = st.runToken) {
  if (st.pyStatus !== "ready") {
    renderAnswersEmpty(st.pyStatus === "error" ? "Python failed to load; see the top bar." : "Waiting for Python (Pyodide) to finish loading…");
    await st.pyPromise;
    if (st.pyStatus !== "ready" || token !== st.runToken) return;
  }
  const text = currentText();
  const payload = {
    doc: text, today: st.today, code: code(),
    fields: validFields().map(([name, desc]) => {
      const h = hitFor(name, desc, text);
      return { name, desc, hit: h ? { present: h.present, start: h.start, end: h.end, score: h.score } : null };
    }),
  };
  const t0 = performance.now();
  let res;
  try { res = JSON.parse(st.py.run(JSON.stringify(payload))); } catch (e) { res = { error: String(e) }; }
  const wall = performance.now() - t0;
  if (token !== st.runToken) return;
  st.last = res;
  $("code-err").textContent = res.error ? res.error + (res.traceback ? "\n" + res.traceback : "") : "";
  if (res.error) { renderAnswersEmpty("The rules did not run: " + res.error); renderTraceEmpty(); return; }
  st.metrics.push({ what: "decide", ms: res.ms, wall });
  renderAnswers(res, wall);
  renderTrace(res);
}

// ------------------------------------------------------------------------------------------------------------ render
function renderLibrary() {
  const q = $("lib-filter").value.trim().toLowerCase();
  const ol = $("lib-list");
  ol.replaceChildren(...USE_CASES.map((u, i) => {
    if (q && !(u.title + " " + u.domain + " " + u.why).toLowerCase().includes(q)) return null;
    return el("li", {}, el("button", { "aria-current": i === st.ucIdx ? "true" : "false", onclick: () => selectUseCase(i) },
      el("span", { class: "lib-title", text: u.title }),
      el("span", { class: "lib-meta" }, el("span", { class: "tag", text: u.domain }),
        el("span", { class: "sub", text: `${u.fields.length} fields · ${u.docs.length} doc${u.docs.length > 1 ? "s" : ""}` }))));
  }));
}

function renderUseCase() {
  const u = uc();
  $("uc-title").textContent = u.title;
  $("uc-domain").textContent = u.domain;
  $("uc-why").textContent = u.why;
  const mc = $("model-choice");
  $("model-choice-wrap").hidden = !u.altModel;
  if (u.altModel) {
    mc.replaceChildren(el("option", { value: "base", text: MODELS.base.label }), el("option", { value: u.altModel, text: MODELS[u.altModel].label }));
    mc.value = modelKey();
  }
  renderDocTabs();
  $("today").value = st.today;
  renderFields();
  $("code").value = code();
  $("code-err").textContent = "";
  renderEngine();
}

function renderDocTabs() {
  const u = uc();
  const tabs = u.docs.map((d, i) => el("button", { role: "tab", "aria-selected": st.docIdx === i ? "true" : "false", onclick: () => selectDoc(i) }, d.name));
  tabs.push(el("button", { role: "tab", "aria-selected": st.docIdx === "own" ? "true" : "false", onclick: () => selectDoc("own") }, "✎ Paste your own"));
  $("doc-tabs").replaceChildren(...tabs);
  const own = st.docIdx === "own";
  $("own-box").hidden = !own;
  if (own) { $("own-text").value = edits().own || ""; ownCount(); }
}
function ownCount() { const t = $("own-text").value; $("own-count").textContent = t ? `${t.length.toLocaleString()} characters` : ""; }

function renderDoc() {
  const text = currentText();
  const pre = $("doc");
  const found = [];
  const legend = [];
  for (const [name, desc] of validFields()) {
    const h = hitFor(name, desc, text);
    const c = colorOf(name);
    legend.push(el("button", { class: "chip" + (h && !h.present ? " absent" : ""), "--c": c, title: h ? (h.present ? `found, score ${h.score.toFixed(2)}` : `absent (best ${h.score.toFixed(2)} < ${h.threshold})`) : "not extracted yet",
      onclick: () => jump(name) }, name));
    if (h && h.present && h.end16 > h.start16) found.push({ name, s: h.start16, e: h.end16, c });
  }
  $("legend").replaceChildren(...legend);
  if (!text) { pre.replaceChildren(el("span", { class: "sub", text: "No text yet." })); $("doc-foot").textContent = ""; return; }
  const cuts = [...new Set([0, text.length, ...found.flatMap((f) => [f.s, f.e])])].sort((a, b) => a - b);
  const nodes = [];
  const first = new Set();
  for (let i = 0; i < cuts.length - 1; i++) {
    const a = cuts[i], b = cuts[i + 1];
    const cov = found.filter((f) => f.s <= a && b <= f.e);
    const seg = text.slice(a, b);
    if (!cov.length) { nodes.push(seg); continue; }
    const m = el("mark", { "--c": cov[0].c, "--c2": cov[1]?.c, class: cov.length > 1 ? "multi" : null, title: cov.map((f) => f.name).join(" + "),
      "data-fields": cov.map((f) => f.name).join(" "), onclick: () => flashRow(cov[0].name) }, seg);
    for (const f of cov) if (!first.has(f.name)) { first.add(f.name); m.classList.add("first-" + f.name); }
    nodes.push(m);
  }
  pre.replaceChildren(...nodes);
  const words = text.split(/\s+/).filter(Boolean).length;
  $("doc-foot").textContent = `${text.length.toLocaleString()} characters · ${words.toLocaleString()} words · about ${Math.max(1, Math.round(words / 500))} page${words > 750 ? "s" : ""}`;
}

function jump(name) {
  const m = $("doc").querySelector("mark.first-" + CSS.escape(name));
  if (!m) { flashRow(name); return; }
  const pre = $("doc");
  pre.scrollTo({ top: m.offsetTop - pre.clientHeight / 3, behavior: "smooth" });
  m.scrollIntoView({ block: "nearest", behavior: "smooth" });
  pre.querySelectorAll("mark").forEach((x) => { if ((x.dataset.fields || "").split(" ").includes(name)) { x.classList.remove("flash"); void x.offsetWidth; x.classList.add("flash"); } });
}
function flashRow(name) {
  const row = document.querySelector(`tr[data-name="${CSS.escape(name)}"]`);
  if (row) { row.scrollIntoView({ block: "nearest", behavior: "smooth" }); row.animate([{ background: "var(--accent-soft)" }, { background: "transparent" }], { duration: 1200 }); }
}

function renderFields() {
  const tb = $("fields").querySelector("tbody");
  const list = fields();
  tb.replaceChildren(...list.map(([name, desc], i) => {
    const err = fieldError(name, i, list);
    const nameIn = el("input", { value: name, "aria-label": "field name", spellcheck: "false", placeholder: "field_name",
      onchange: (e) => setField(i, e.target.value.trim(), null) });
    const descIn = el("textarea", { rows: 2, "aria-label": "field description", placeholder: "Describe the field in plain English, e.g. “the date the contract ends”",
      onchange: (e) => setField(i, null, e.target.value) }, desc);
    descIn.value = desc;
    return el("tr", { "data-name": name },
      el("td", { class: "sw" }, el("span", { "--c": colorOf(name) })),
      el("td", {}, nameIn, err ? el("div", { class: "name-err", text: err }) : null),
      el("td", {}, descIn),
      el("td", { class: "found", id: "found-" + i, onclick: () => jump(name) }),
      el("td", {}, el("button", { class: "x", title: "Remove field", "aria-label": "remove field " + name, onclick: () => removeField(i) }, "×")));
  }));
  list.forEach(([n]) => updateFound(n));
}

function updateFound(name) {
  const list = fields();
  const i = list.findIndex(([n]) => n === name);
  if (i < 0) return;
  const cell = $("found-" + i);
  if (!cell) return;
  const [, desc] = list[i];
  const h = desc.trim() ? hitFor(name, desc) : null;
  if (!h) {
    cell.className = "found pending";
    cell.replaceChildren(el("div", { class: "val", text: desc.trim() ? "not extracted yet" : "add a description" }));
    return;
  }
  cell.className = "found" + (h.present ? "" : " absent");
  cell.replaceChildren(
    el("div", { class: "val", title: h.present ? h.value : "", text: h.present ? h.value : "absent" }),
    el("div", { class: "meta", text: h.present ? `score ${h.score.toFixed(2)} · chars ${h.start}–${h.end}` :
      `best span scored ${h.score.toFixed(2)} < threshold ${h.threshold}` }));
}

function setField(i, name, desc) {
  const list = fields();
  if (name !== null) {
    let n = name.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
    if (/^\d/.test(n)) n = "f_" + n;
    list[i][0] = n;
  }
  if (desc !== null) {
    list[i][1] = desc;
    if (!list[i][0] && desc.trim()) list[i][0] = autoName(desc, list);
  }
  saveEdits(uc(), { fields: list });
  renderFields();
  renderDoc();
  analyze({ auto: true });
}
function autoName(desc, list) {
  const stop = new Set("the a an of that which who is are to in on for and or by with when where how what this any it its be was were does do".split(" "));
  let base = desc.toLowerCase().match(/[a-z0-9]+/g)?.filter((w) => !stop.has(w)).slice(0, 3).join("_") || "field";
  if (/^\d/.test(base)) base = "f_" + base;
  let n = base, k = 2;
  while (list.some(([x]) => x === n) || RESERVED.has(n) || PY_KEYWORDS.has(n)) n = `${base}_${k++}`;
  return n;
}
function removeField(i) {
  const list = fields();
  list.splice(i, 1);
  saveEdits(uc(), { fields: list });
  renderFields(); renderDoc();
  analyze({ auto: true });
}
function addField() {
  const list = fields();
  list.push(["", ""]);
  saveEdits(uc(), { fields: list });
  renderFields();
  const rows = $("fields").querySelectorAll("tbody tr");
  rows[rows.length - 1]?.querySelector("textarea")?.focus();
}

function renderAnswersEmpty(msg) {
  $("answers").replaceChildren(el("p", { class: "empty", text: msg }));
  $("answers-meta").textContent = "";
}

function answerCard(a) {
  const cls = a.status === "ok" ? "" : a.status;
  const label = a.answer === null || a.answer === undefined ? "no answer" : String(a.answer);
  const statusText = a.status === "forced" ? "forced by a hard check" : a.status === "abstain" ? "abstained" : "";
  return el("div", { class: "ans " + cls },
    el("div", { class: "ans-top" },
      el("span", { class: "ans-q", text: a.text }),
      el("span", {}, el("span", { class: "ans-a", text: label }), statusText ? el("span", { class: "ans-status", text: statusText }) : null)),
    el("div", { class: "conf" }, `confidence ${a.confidence.toFixed(2)}`,
      el("div", { class: "bar" }, el("div", { style: { width: (100 * a.confidence).toFixed(0) + "%" } })),
      el("span", { class: "sub", text: `options: ${a.options.join(" / ")}` })),
    el("div", { class: "why-line", text: "why: " + a.why }),
    a.checks.length ? el("div", { class: "checks" }, a.checks.map((c) => el("span", {
      class: "check" + (c.value === "False" ? " fail" : c.value === null ? " na" : ""), title: c.doc || "",
    }, `${c.value === "False" ? "✗" : c.value === null ? "–" : "✓"} ${c.name}${c.hard ? " (hard)" : ""}`))) : null,
    a.cites.length ? el("div", { class: "cites" }, el("span", { class: "sub", text: "cites:" }), a.cites.map((n) => {
      const d = fields().find(([x]) => x === n);
      const h = d ? hitFor(n, d[1]) : null;
      return el("button", { class: "chip" + (h && !h.present ? " absent" : ""), "--c": colorOf(n), title: h?.present ? h.value : "absent in the document", onclick: () => jump(n) },
        h?.present ? `${n}: “${h.value.length > 40 ? h.value.slice(0, 38) + "…" : h.value}”` : `${n}: absent`);
    })) : null);
}

function renderAnswers(res, wall) {
  const main = res.answers.filter((a) => !a.auto), auto = res.answers.filter((a) => a.auto);
  const kids = main.map(answerCard);
  if (auto.length) {
    kids.push(el("details", { class: "auto", open: main.length ? null : true },
      el("summary", {}, `Field presence: ${auto.length} automatic question${auto.length > 1 ? "s" : ""} for fields no rule reads`), auto.map(answerCard)));
  }
  $("answers").replaceChildren(...kids);
  const n = { ok: 0, forced: 0, abstain: 0 };
  res.answers.forEach((a) => n[a.status]++);
  $("answers-meta").textContent = `decided in ${res.ms.toFixed(1)} ms · ${n.ok} answered, ${n.forced} forced, ${n.abstain} abstained`;
}

function renderTraceEmpty() { $("trace-body").replaceChildren(el("p", { class: "empty", text: "No trace." })); $("replay").replaceChildren(); }

function renderTrace(res) {
  const rp = res.replay;
  $("replay").replaceChildren(
    el("span", { class: "badge " + (rp.ok ? "ok" : "bad"), text: rp.ok ? "Replay OK" : `Replay: ${rp.mismatches.length} mismatch(es)` }),
    el("span", { class: "sub", text: rp.ok ? `All ${rp.steps} steps were re-executed from the recorded inputs; values, quotes and the hash chain match.` : rp.mismatches.map((m) => m.join(" ")).join("; ") }),
    el("button", { class: "btn small", onclick: tamper, title: "Edit one computed fact in a copy of the trace, re-hash it, and replay" }, "Tamper with a copy"));
  document.querySelectorAll("#trace-tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === st.tab ? "true" : "false"));
  const body = $("trace-body");
  if (st.tab === "flow") {
    const rows = res.flow.map((s) => el("tr", {},
      el("td", { text: s.i }), el("td", {}, el("span", { class: "kind " + s.kind, text: s.kind + (s.hard ? " · hard" : "") })),
      el("td", { class: "mono", text: `${s.name} ← ${s.inputs.join(", ") || "—"}` }), el("td", { class: "sub", text: s.reasons.join("; ") })));
    body.replaceChildren(...[
      el("p", { class: "sub", text: "The strategist planned this flow for the questions: only the parts they need, hard checks and their inputs first." }),
      el("table", { class: "flow" }, el("thead", {}, el("tr", {}, ["#", "kind", "part ← reads", "why it is in the flow"].map((h) => el("th", { text: h })))), el("tbody", {}, rows)),
      res.skipped_at_run.length ? el("div", { class: "skipped", text: "Skipped at run time: " + res.skipped_at_run.map(([n, w]) => `${n} (${w})`).join(", ") }) : null,
      res.not_taken.length ? el("div", { class: "skipped", text: "Not taken from the catalog: " + res.not_taken.map(([n, w]) => `${n} (${w})`).join(", ") }) : null].filter(Boolean));
  } else if (st.tab === "state") {
    body.replaceChildren(el("p", { class: "sub", text: "Every fact that was computed, its value, the quote offsets of extracted fields (Python string indices) and errors." }),
      el("pre", { class: "state", text: res.computed_state }));
  } else {
    body.replaceChildren(el("p", { class: "sub", text: `Hash-chained records: each record hashes its inputs, value, quote and the previous record. Initial state hash ${res.init_hash}.` }),
      el("table", { class: "flow" }, el("thead", {}, el("tr", {}, ["step", "kind", "name", "value", "quote", "hash", "prev"].map((h) => el("th", { text: h })))),
        el("tbody", {}, res.records.map((r) => el("tr", {},
          el("td", { text: r.step }), el("td", {}, el("span", { class: "kind " + r.kind, text: r.kind })), el("td", { class: "mono", text: r.name }),
          el("td", { class: "mono", text: (r.value ?? "—") + (r.error ? `  ERROR: ${r.error}` : "") }),
          el("td", { class: "mono", text: r.quote ? `[${r.quote[0]}:${r.quote[1]}]` : "" }),
          el("td", { class: "mono", text: r.hash }), el("td", { class: "mono", text: r.prev.slice(0, 16) }))))));
  }
}

function tamper() {
  if (!st.py) return;
  const r = JSON.parse(st.py.tamper());
  const box = el("div", { class: "tamper" });
  if (r.error) box.append(r.error);
  else {
    box.append(el("b", {}, `Changed step ${r.step} (${r.name}) from ${r.from} to ${r.to} in a copy of the trace and re-hashed that record. `),
      r.replay.ok ? "Replay did not notice (unexpected)." : `Replay caught it: ${r.replay.mismatches.length} problem(s):`,
      el("ul", {}, r.replay.mismatches.map((m) => el("li", { text: m.join(" · ") }))),
      el("span", { class: "sub", text: "The real trace is unchanged." }));
  }
  document.querySelectorAll(".tamper").forEach((x) => x.remove());
  $("replay").after(box);
}

// ------------------------------------------------------------------------------------------------------------ navigation
function selectUseCase(i, docIdx = 0) {
  cancelJob();
  st.ucIdx = i;
  st.docIdx = docIdx;
  const d = docIdx === "own" ? null : uc().docs[docIdx];
  st.today = d?.today || todayIso();
  st.last = null;
  document.querySelectorAll(".tamper").forEach((x) => x.remove());
  renderLibrary();
  renderUseCase();
  renderAnswersEmpty("Answers appear here after the fields are extracted.");
  renderTraceEmpty();
  $("extract-status").textContent = "";
  history.replaceState(null, "", `#${uc().id}${docIdx ? "/" + docIdx : ""}`);
  analyze({ auto: true });
}
function selectDoc(i) {
  cancelJob();
  st.docIdx = i;
  if (i !== "own") st.today = uc().docs[i].today || todayIso();
  $("today").value = st.today;
  document.querySelectorAll(".tamper").forEach((x) => x.remove());
  renderDocTabs();
  history.replaceState(null, "", `#${uc().id}${i ? "/" + i : ""}`);
  for (const [n] of fields()) updateFound(n);
  renderAnswersEmpty(i === "own" && !(edits().own || "").trim() ? "Paste a document above and press “Read this document”." : "");
  renderTraceEmpty();
  analyze({ auto: true });
}

// ------------------------------------------------------------------------------------------------------------ wiring
function wire() {
  $("lib-filter").addEventListener("input", renderLibrary);
  $("model-load").addEventListener("click", () => {
    const k = modelKey();
    if (st.models[k].status === "error") st.models[k] = { status: "idle" };
    hideNote();
    analyze();
  });
  $("run").addEventListener("click", () => analyze());
  $("run-rules").addEventListener("click", () => { saveEdits(uc(), { code: $("code").value }); analyze(); });
  $("reset-rules").addEventListener("click", () => { const e = { ...edits() }; delete e.code; replaceEdits(e); $("code").value = code(); analyze({ auto: true }); });
  $("reset-uc").addEventListener("click", () => { replaceEdits({}); selectUseCase(st.ucIdx); });
  $("add-field").addEventListener("click", addField);
  $("today").addEventListener("change", (e) => { st.today = e.target.value || todayIso(); if (st.last) decide(); });
  $("model-choice").addEventListener("change", (e) => { saveEdits(uc(), { model: e.target.value }); renderEngine(); analyze({ auto: true }); });
  $("own-text").addEventListener("input", ownCount);
  $("own-use").addEventListener("click", () => { saveEdits(uc(), { own: $("own-text").value.normalize("NFC") }); analyze(); });
  $("own-file").addEventListener("change", async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    $("own-text").value = await f.text(); ownCount();
    saveEdits(uc(), { own: $("own-text").value.normalize("NFC") });
    analyze();
  });
  const ta = $("code");
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      const s = ta.selectionStart;
      ta.setRangeText("    ", s, ta.selectionEnd, "end");
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault(); $("run-rules").click();
    }
  });
  ta.addEventListener("change", () => saveEdits(uc(), { code: ta.value }));
  document.querySelectorAll("#trace-tabs button").forEach((b) => b.addEventListener("click", () => { st.tab = b.dataset.tab; if (st.last && !st.last.error) renderTrace(st.last); }));
}
function replaceEdits(e) {
  memEdits[uc().id] = e;
  try { localStorage.setItem(storeKey(uc()), JSON.stringify(e)); } catch { /* ignore */ }
}

function fromHash() {
  const [id, d] = decodeURIComponent(location.hash.slice(1)).split("/");
  const i = Math.max(0, USE_CASES.findIndex((u) => u.id === id));
  const di = d === "own" ? "own" : Math.min(+d || 0, USE_CASES[i].docs.length - 1);
  return [i, di];
}

async function start() {
  wire();
  st.pyPromise = loadPython();
  detectGpu();
  selectUseCase(...fromHash());
  window.addEventListener("hashchange", () => {
    const [i, di] = fromHash();
    if (i !== st.ucIdx) selectUseCase(i, di); else if (di !== st.docIdx) selectDoc(di);
  });
  // the model is large: load it automatically only when it is already in the browser cache (or ?autoload=1)
  const key = modelKey();
  const c = await ask({ type: "cached", url: repoOf(key) + st.onnx });
  if (c.hit || P.get("autoload") === "1") loadModel(key);
}
start();
