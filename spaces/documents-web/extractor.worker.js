// Field extractor in a Web Worker: onnxruntime-web (WebGPU, WASM fallback) + a byte-level BPE tokenizer with offsets.
// Reproduces solvi.extract_long.LongSpanExtractor.predict exactly:
//   windows "[CLS] description(<=64 tokens) [SEP] chunk [SEP]", room = max_len - len(desc) - 3, step = room - stride;
//   per window softmax of the start and end logits over the window; best span start <= end < start + max_span inside the
//   chunk by ps[s] * pe[e]; best over windows; leading whitespace stripped; present if score >= threshold.
// Messages in:  {type: "init", key, repo, onnx, backend?}  {type: "extract", id, key, text, fields: [{name, desc}]}
//               {type: "cancel", id}  {type: "probe", repo, onnx}
// Messages out: progress / ready / error / field / window / done.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { ByteLevelBPE } from "./tokenizer.js";

const ORT_VERSION = "1.30.0";
const CACHE = "solvi-documents-v1";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 1)) : 1;
ort.env.logLevel = "error";

const models = new Map();          // key -> {session, tok, cfg, backend, file}
const cancelled = new Set();
let queue = Promise.resolve();
const post = (m) => self.postMessage(m);

// ------------------------------------------------------------------------------------------------ downloads + cache
async function openCache() {
  try { return await caches.open(CACHE); } catch { return null; }   // no Cache API (e.g. some private modes): download each visit
}

async function readBody(resp, total, onProgress) {
  const reader = resp.body.getReader();
  let buf = total > 0 ? new Uint8Array(total) : null;
  const parts = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && loaded + value.length <= buf.length) buf.set(value, loaded);
    else { if (buf) { parts.push(buf.subarray(0, loaded)); buf = null; } parts.push(value); }
    loaded += value.length;
    const now = performance.now();
    if (onProgress && now - last > 120) { last = now; onProgress(loaded, total); }
  }
  if (onProgress) onProgress(loaded, total);
  if (buf) return loaded === buf.length ? buf : buf.subarray(0, loaded);
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// -> {bytes, fromCache}
async function fetchCached(url, onProgress) {
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const total = +hit.headers.get("x-solvi-size") || +hit.headers.get("content-length") || 0;
      const bytes = await readBody(hit, total, onProgress && ((l, t) => onProgress(l, t, true)));
      if (!total || bytes.length === total) return { bytes, fromCache: true };
      await cache.delete(url);                                      // truncated entry: download again
    }
  }
  const resp = await fetch(url, { mode: "cors" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  const total = +resp.headers.get("content-length") || 0;
  let body = resp;
  let put = null;
  if (cache && resp.body) {
    const [a, b] = resp.body.tee();
    body = new Response(a);
    const headers = { "content-type": "application/octet-stream", "x-solvi-size": String(total) };
    put = cache.put(url, new Response(b, { headers })).catch((e) => post({ type: "warn", message: "Could not cache " + url + ": " + e }));
  }
  const bytes = await readBody(body, total, onProgress && ((l, t) => onProgress(l, t, false)));
  if (put) await put;
  return { bytes, fromCache: false };
}

async function fetchJson(url) {
  const { bytes } = await fetchCached(url);
  return JSON.parse(new TextDecoder().decode(bytes));
}

// ------------------------------------------------------------------------------------------------------ init
// -> "none" | "no-f16" | "ok". An fp16 graph needs the shader-f16 feature (missing e.g. on some Linux/Vulkan setups).
async function webGPUState() {
  try {
    if (!self.navigator?.gpu) return "none";
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!a) return "none";
    return a.features.has("shader-f16") ? "ok" : "no-f16";
  } catch { return "none"; }
}

async function init(msg) {
  const { key, repo, onnx } = msg;
  if (models.has(key) && models.get(key).file === onnx) { post({ type: "ready", key, ...models.get(key).info }); return; }
  const t0 = performance.now();
  post({ type: "progress", key, phase: "files", loaded: 0, total: 0 });
  const [cfg, tj] = await Promise.all([fetchJson(repo + "solvi_extract.json"), fetchJson(repo + "tokenizer.json")]);
  const tok = new ByteLevelBPE(tj);
  const gpuState = msg.backend === "wasm" ? "off" : await webGPUState();
  const needsF16 = /fp16/.test(onnx);
  let backend = gpuState === "ok" || (gpuState === "no-f16" && !needsF16) ? "webgpu" : "wasm";
  const t1 = performance.now();
  const { bytes, fromCache } = await fetchCached(repo + onnx, (loaded, total, cached) =>
    post({ type: "progress", key, phase: cached ? "cache" : "download", loaded, total }));
  const t2 = performance.now();
  post({ type: "progress", key, phase: "session", backend, loaded: bytes.length, total: bytes.length });
  // one model at a time: two ModernBERT-large sessions do not fit in one WebAssembly heap (4 GB)
  for (const [k, m] of models) {
    try { await m.session.release(); } catch { /* ignore */ }
    models.delete(k);
    if (k !== key) post({ type: "unloaded", key: k });
  }
  const create = async (be) => {
    try {
      return await ort.InferenceSession.create(bytes, { executionProviders: [be], graphOptimizationLevel: "all" });
    } catch (e) {
      // right after a reload the previous page's heap may not be freed yet: wait and try once more
      if (!/bad_alloc|out of memory|memory access out of bounds/i.test(String(e?.message || e))) throw e;
      await new Promise((r) => setTimeout(r, 4000));
      return await ort.InferenceSession.create(bytes, { executionProviders: [be], graphOptimizationLevel: "all" });
    }
  };
  // create + warm-up on a short input (compiles kernels/shaders; fails early if an operator is missing on this backend)
  const start = async (be) => {
    const s = await create(be);
    const tw = performance.now();
    await runWindow(s, [tok.cls, tok.sep, 100, tok.sep]);
    return { s, warm: performance.now() - tw };
  };
  let session, warmupMs, gpuFailed = null;
  try {
    ({ s: session, warm: warmupMs } = await start(backend));
  } catch (e) {
    if (backend !== "webgpu") {
      const text = String(e?.message || e);
      const fp16 = /float16|fp16|MLFloat16|not implemented|Could not find an implementation/i.test(text);
      post({ type: "error", key, code: fp16 ? "fp16-wasm" : "session", backend, message: text.slice(0, 600) });
      return;
    }
    gpuFailed = String(e?.message || e).slice(0, 300);          // WebGPU could not run this graph: fall back to the CPU
    backend = "wasm";
    post({ type: "progress", key, phase: "session", backend, loaded: bytes.length, total: bytes.length });
    try {
      ({ s: session, warm: warmupMs } = await start(backend));
    } catch (e2) {
      post({ type: "error", key, code: "session", backend, message: String(e2?.message || e2).slice(0, 600) });
      return;
    }
  }
  const t4 = performance.now();
  const t3 = t4 - warmupMs;
  const info = { backend, gpuState, gpuFailed, fromCache, threads: ort.env.wasm.numThreads, file: onnx, size: bytes.length, cfg,
                 filesMs: t1 - t0, downloadMs: t2 - t1, sessionMs: t3 - t2, warmupMs: t4 - t3, totalMs: t4 - t0 };
  models.set(key, { session, tok, cfg, backend, file: onnx, info, encCache: new Map() });
  post({ type: "ready", key, ...info });
}

// ------------------------------------------------------------------------------------------------------ predict
async function runWindow(session, seq) {
  const n = seq.length;
  const ids = new BigInt64Array(n), att = new BigInt64Array(n);
  for (let i = 0; i < n; i++) { ids[i] = BigInt(seq[i]); att[i] = 1n; }
  const out = await session.run({
    input_ids: new ort.Tensor("int64", ids, [1, n]),
    attention_mask: new ort.Tensor("int64", att, [1, n]),
  });
  const t = out.logits ?? out[session.outputNames[0]];
  const data = t.data;                  // Float32Array [1, n, 2]
  const res = data.slice ? data.slice(0, n * 2) : Float32Array.from(data);
  t.dispose?.();
  return res;
}

function softmaxCol(lg, n, col) {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) mx = Math.max(mx, lg[2 * i + col]);
  const p = new Float64Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) { p[i] = Math.exp(lg[2 * i + col] - mx); s += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= s;
  return p;
}

function encodeText(m, text) {
  let e = m.encCache.get(text);
  if (!e) {
    e = m.tok.encode(text);
    if (m.encCache.size > 16) m.encCache.clear();
    m.encCache.set(text, e);
  }
  return e;
}

async function predict(m, text, desc, onWindow, isCancelled) {
  const { tok, cfg, session } = m;
  const enc = encodeText(m, text);
  const ids = enc.ids;
  const d = tok.encode(desc.normalize("NFC")).ids.slice(0, 64);
  const room = cfg.max_len - d.length - 3;
  const step = Math.max(1, room - cfg.stride);
  const nWin = ids.length <= room ? 1 : 1 + Math.ceil((ids.length - room) / step);
  let best = { s: 0, e: 0, score: -1 }, nul = 0, w = 0;
  for (let a = 0; a < Math.max(1, ids.length); a += step) {
    if (isCancelled()) return null;
    const chunk = ids.slice(a, a + room);
    if (chunk.length) {
      const seq = [tok.cls, ...d, tok.sep, ...chunk, tok.sep];
      const n = seq.length;
      const lg = await runWindow(session, seq);
      const ps = softmaxCol(lg, n, 0), pe = softmaxCol(lg, n, 1);
      nul = Math.max(nul, ps[0] * pe[0]);
      const c0 = d.length + 2, c1 = c0 + chunk.length;
      let bs = -1, be = -1, bsc = -1;
      for (let s = c0; s < c1; s++) {
        const lim = Math.min(c1, s + cfg.max_span);
        const pss = ps[s];
        for (let e = s; e < lim; e++) {
          const sc = pss * pe[e];
          if (sc > bsc) { bsc = sc; bs = s; be = e; }
        }
      }
      if (bsc > best.score) best = { s: a + bs - c0, e: a + be - c0, score: bsc };
    }
    w++;
    onWindow?.(w, nWin);
    if (a + room >= ids.length) break;
  }
  let st = 0, en = 0, st16 = 0, en16 = 0;
  if (best.score >= 0 && ids.length) {
    st = enc.offsets[best.s][0]; en = enc.offsets[best.e][1];
    st16 = enc.offsets16[best.s][0]; en16 = enc.offsets16[best.e][1];
    while (st < en && /\s/.test(text[st16])) { st++; st16++; }        // BPE offsets include the leading space
  }
  return { start: st, end: en, start16: st16, end16: en16, score: best.score, nullScore: nul, windows: w, tokens: ids.length,
           value: text.slice(st16, en16) };
}

async function extract(msg) {
  const { id, key, fields } = msg;
  const text = msg.text;
  const m = models.get(key);
  if (!m) { post({ type: "error", id, key, code: "not-ready", message: "model not loaded" }); return; }
  const t0 = performance.now();
  for (const f of fields) {
    if (cancelled.has(id)) break;
    const t = performance.now();
    const r = await predict(m, text, f.desc, (w, n) => post({ type: "window", id, name: f.name, w, n }), () => cancelled.has(id));
    if (!r) break;
    const thr = m.cfg.thr?.[f.name] ?? m.cfg.thr_default;
    post({ type: "field", id, key, name: f.name, desc: f.desc, result: { ...r, threshold: thr, present: r.score >= thr },
           ms: performance.now() - t });
  }
  post({ type: "done", id, key, cancelled: cancelled.has(id), ms: performance.now() - t0 });
  cancelled.delete(id);
}

async function probe(msg) {                   // does a file exist? (HEAD, follows redirects)
  try {
    const r = await fetch(msg.repo + msg.onnx, { method: "HEAD" });
    post({ type: "probe", id: msg.id, ok: r.ok, size: +r.headers.get("content-length") || +r.headers.get("x-linked-size") || 0 });
  } catch { post({ type: "probe", id: msg.id, ok: false }); }
}

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "cancel") { cancelled.add(msg.id); return; }
  if (msg.type === "probe") { probe(msg); return; }
  if (msg.type === "cached") {
    (async () => {
      const c = await openCache();
      const hit = c ? !!(await c.match(msg.url)) : false;
      post({ type: "cached", id: msg.id, hit });
    })();
    return;
  }
  queue = queue.then(async () => {
    try {
      if (msg.type === "init") await init(msg);
      else if (msg.type === "extract") await extract(msg);
    } catch (e) {
      post({ type: "error", id: msg.id, key: msg.key, code: "exception", message: String(e?.stack || e).slice(0, 800) });
    }
  });
};
post({ type: "hello", ort: ORT_VERSION, crossOriginIsolated: self.crossOriginIsolated, threads: ort.env.wasm.numThreads });
