"""The segment model behind solvi.strategy.ModelStrategist. Experimental: no checkpoint is published — it reads one you trained yourself (docs/strategist.md has the format).

The typed decomposer from research,
compressed (the first 2 layers of ModernBERT-base, a vocabulary cut to 8192 tokens), without the graph level (the compressed
decomposer did not need it), reading one short task per segment and pointing at 1–4 catalog parts.

A segment (solvi.strategy.segments) becomes cells — short texts, each encoded on its own:
  GOAL   "produce <fact>: <type> | for: <question>"
  FACT   "<fact>: <type> | given" / "| computed"         (only available facts that some candidate reads)
  FUNC   "<kind> <name> -> <type> | <docstring> | reads <param>: <type>, … | provides <fact> | cost <c>"
plus per cell its kind, value type, depth (1: a producer of the segment's fact; 2: of an input of one; …) and, for a FUNC
cell, whether all its inputs are available (computed by code). Output: N_NODES query slots; slot i → a node type (EMPTY /
STEP) and a pointer to a FUNC cell. The proposal is the slots up to the first EMPTY.

Backends: "torch" (`solvi[model]`: torch + transformers) and "onnx" (`solvi[onnx]`: onnxruntime + tokenizers, no torch).
The checkpoint format is in docs/strategist.md."""
from __future__ import annotations

import json
import math
import os

import numpy as np

FORMAT = "solvi_strategist v1"
CELL_KINDS = ["GOAL", "FACT", "FUNC", "CONST", "REG"]          # the typed decomposer's cell kinds (weights are shared)
VTYPES = ["num", "bool", "str", "entity", "list<entity>", "list<num>", "func", "goal", "none"]
NODE_TYPES = ["EMPTY", "STEP", "CHECK", "BRANCH", "FOREACH", "RETRY", "STOP", "ESCALATE", "GOTO"]
N_NODES = 8
CELL_TOK = 48


# ---------------------------------------------------------------------------------------------------------------- cells
def vtype(t):
    t = (t or "any").lower()
    if t in ("float", "int", "decimal") or t.startswith(("annotated[float", "annotated[int", "decimal")):
        return "num"
    if t == "bool":
        return "bool"
    if t in ("str",) or t.startswith("annotated[str"):
        return "str"
    if t.startswith(("list[", "set[", "tuple[")):
        return "list<num>" if any(x in t for x in ("float", "int")) else "list<entity>"
    return "entity"


def _num(c):
    return f"{c:g}" if isinstance(c, (int, float)) else str(c)


def func_text(c):
    """A candidate part as one cell: name, docstring first (cells are cut at CELL_TOK tokens), then what it reads."""
    s = f"{c['kind']} {c['name']} -> {c['returns']}"
    if c.get("doc"):
        s += " | " + " ".join(str(c["doc"]).split())
    if c.get("params"):
        s += " | reads " + ", ".join(f"{x}: {t}" for x, t in c["params"])
    if c.get("provides"):
        s += f" | provides {c['provides']}"
    if c.get("cost") is not None:
        s += f" | cost {_num(c['cost'])}"
    return s


def seg_cells(seg):
    """A segment → (texts, kinds, vts, depths, ready, fn names). ready: 0 not a function, 1 all inputs available, 2 not."""
    avail = {x for x, _, _ in seg["available"]}
    reads = {x for c in seg["candidates"] for x, _ in c["params"]}
    texts = [f"produce {seg['fact']}: {seg['type']} | for: {seg.get('question', '')}"]
    kinds, vts, depth, ready = [0], [VTYPES.index("goal")], [0], [0]
    for x, t, how in seg["available"]:
        if x in reads:
            texts.append(f"{x}: {t} | {how}")
            kinds.append(1)
            vts.append(VTYPES.index(vtype(t)))
            depth.append(0)
            ready.append(0)
    names = []
    for c in seg["candidates"]:
        texts.append(func_text(c))
        kinds.append(2)
        vts.append(VTYPES.index("func"))
        depth.append(min(int(c.get("depth", 1)), 255))
        ready.append(1 if all(x in avail for x, _ in c["params"]) else 2)
        names.append(c["name"])
    return texts, kinds, vts, depth, ready, names


def seg_labels(names, nodes, n=N_NODES):
    """Gold nodes (part names in order) → (type labels, fn labels) over n slots."""
    ty = [NODE_TYPES.index("STEP")] * len(nodes) + [0] * (n - len(nodes))
    fn = [names.index(x) for x in nodes] + [-100] * (n - len(nodes))
    return ty[:n], fn[:n]


def _auto_backend(has_onnx_file, what):
    """backend="auto": onnx when onnxruntime is installed and the checkpoint has the onnx files, else torch; with
    neither runtime installed, an ImportError that names both extras (not "No module named 'torch'")."""
    import importlib.util
    ort, torch = (importlib.util.find_spec(m) is not None for m in ("onnxruntime", "torch"))
    if ort and has_onnx_file:
        return "onnx"
    if torch:
        return "torch"
    raise ImportError(f"{what} needs a runtime: pip install 'solvi[onnx]' (onnxruntime, small) or 'solvi[model]' "
                      "(torch)" + ("" if has_onnx_file else " — this checkpoint has no onnx files: 'solvi[model]'"))


class Tok:
    """ModernBERT's tokenizer (tokenizers), cells truncated to CELL_TOK tokens, with a cache of encoded cell texts."""

    def __init__(self, path, cell_tok=CELL_TOK):
        from .loader import optional
        Tokenizer = optional("tokenizers", "onnx", "the strategist's tokenizer").Tokenizer
        self.t = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.t.enable_truncation(cell_tok)
        self.t.no_padding()
        self.cache = {}
        pid = self.t.token_to_id("[PAD]")
        self.pad = 0 if pid is None else pid

    def __call__(self, texts):
        out = []
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            for t, e in zip(todo, self.t.encode_batch(todo)):
                self.cache[t] = e.ids
        for t in texts:
            out.append(self.cache[t])
        if len(self.cache) > 200_000:
            self.cache.clear()
        return out


def batch_arrays(segs, tok, labels=None, n_nodes=N_NODES):
    """Segments → numpy arrays (the model's inputs; the ONNX graph takes exactly these):
    ids, mask [B·C, T] (padding cells hold one pad token, masked in cmask), kind, vt, depth, ready, cmask [B, C],
    fn_idx, fn_mask [B, F]; with labels ([nodes] per segment) also type_lab, fn_lab [B, N]."""
    cs = [seg_cells(s) for s in segs]
    B = len(segs)
    C = max(len(c[0]) for c in cs)
    F = max(1, max(len(c[5]) for c in cs))
    ids_l = []
    for c in cs:
        ids_l += tok(c[0]) + [[tok.pad]] * (C - len(c[0]))
    T = max(len(x) for x in ids_l)
    ids = np.full((B * C, T), tok.pad, dtype=np.int64)
    mask = np.zeros((B * C, T), dtype=np.int64)
    for i, x in enumerate(ids_l):
        ids[i, :len(x)] = x
        mask[i, :len(x)] = 1
    a = {"ids": ids, "mask": mask}
    for k in ("kind", "vt", "depth", "ready"):
        a[k] = np.zeros((B, C), dtype=np.int64)
    a["cmask"] = np.zeros((B, C), dtype=bool)
    a["fn_idx"] = np.zeros((B, F), dtype=np.int64)
    a["fn_mask"] = np.zeros((B, F), dtype=bool)
    for i, (texts, kinds, vts, depth, ready, names) in enumerate(cs):
        n = len(texts)
        a["kind"][i, :n], a["vt"][i, :n], a["depth"][i, :n], a["ready"][i, :n] = kinds, vts, depth, ready
        a["cmask"][i, :n] = True
        fi = [j for j, k in enumerate(kinds) if k == 2]
        a["fn_idx"][i, :len(fi)] = fi
        a["fn_mask"][i, :len(fi)] = True
    if labels is not None:
        a["type_lab"] = np.zeros((B, n_nodes), dtype=np.int64)
        a["fn_lab"] = np.full((B, n_nodes), -100, dtype=np.int64)
        for i, (nodes, c) in enumerate(zip(labels, cs)):
            ty, fn = seg_labels(c[5], nodes, n_nodes)
            a["type_lab"][i], a["fn_lab"][i] = ty, fn
    return a, [c[5] for c in cs]


def decode(type_logits, fn_logits, names_list, n_second=True):
    """Last-pass logits → per segment {"nodes", "score" (log-probability of the proposal), "second" (the runner-up for the
    least sure pointer, or None)}."""
    out = []
    for i, names in enumerate(names_list):
        tl = _log_softmax(type_logits[i])
        fl = _log_softmax(np.where(np.arange(fn_logits.shape[-1])[None] < len(names), fn_logits[i], -1e9))
        nodes, score, weakest = [], 0.0, None
        for n in range(tl.shape[0]):
            t = int(tl[n].argmax())
            score += float(tl[n, t])
            if t == 0:
                break
            j = int(fl[n].argmax())
            score += float(fl[n, j])
            nodes.append(names[j])
            if len(names) > 1:
                srt = np.sort(fl[n])[::-1]
                margin = float(srt[0] - srt[1])
                if weakest is None or margin < weakest[0]:
                    weakest = (margin, len(nodes) - 1, int(np.argsort(fl[n])[::-1][1]))
        second = None
        if n_second and weakest is not None and weakest[2] < len(names):
            second = list(nodes)
            second[weakest[1]] = names[weakest[2]]
        out.append({"nodes": nodes, "score": score, "second": second})
    return out


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


# ---------------------------------------------------------------------------------------------------------------- torch
def build_net(enc_config, n_nodes=N_NODES, d=512, heads=8, layers=4, passes=3, vocab_full=50368):
    """The segment network (torch). Parameter names follow the typed Decomposer, so a compressed-decomposer checkpoint initialises it."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):                         # nn.RMSNorm's math (default eps), exportable to ONNX opset 17+
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(n))

        def forward(self, x):
            xf = x.float()
            y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + torch.finfo(torch.float32).eps)
            return (y * self.weight.float()).to(x.dtype)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.dk = heads, d // heads
            self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

        def forward(self, x, mem, bias=None, mask=None):
            B, N, _ = x.shape
            M = mem.shape[1]
            q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)
            k = self.k(mem).view(B, M, self.h, self.dk).transpose(1, 2)
            v = self.v(mem).view(B, M, self.h, self.dk).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
            if bias is not None:
                s = s + bias
            if mask is not None:
                s = s.masked_fill(~mask[:, None], -1e4)
            a = torch.softmax(s.float(), -1).to(v.dtype)
            return self.o((a @ v).transpose(1, 2).reshape(B, N, d))

    class FFN(nn.Module):
        def __init__(self, mult=2.67):
            super().__init__()
            h = int(d * mult)
            self.w1, self.w2, self.w3 = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class DecLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
            self.sa, self.ca, self.ff = Attn(), Attn(), FFN()
            self.plan_bias = nn.Linear(6, heads, bias=False)
            self.fn_bias = nn.Parameter(torch.zeros(heads))

        def forward(self, q, cells, cmask, rel, fnsel=None):
            sb = self.plan_bias(rel).permute(2, 0, 1)[None]
            q = q + self.sa(self.n1(q), self.n1(q), sb)
            cb = fnsel[:, None] * self.fn_bias[None, :, None, None] if fnsel is not None else None
            q = q + self.ca(self.n2(q), cells, cb, cmask[:, None, :].expand(-1, q.shape[1], -1))
            return q + self.ff(self.n3(q))

    class SegmentNet(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import ModernBertConfig, ModernBertModel
            if isinstance(enc_config, dict):
                c = dict(enc_config)
                c.pop("architectures", None)
                lt = c.get("layer_types")
                if isinstance(lt, list) and len(lt) != c.get("num_hidden_layers", len(lt)):
                    c["layer_types"] = lt[: c["num_hidden_layers"]]
                for k in ("pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id"):
                    if isinstance(c.get(k), int) and c[k] >= c.get("vocab_size", 1 << 30):
                        c[k] = 0                        # a cut vocabulary: ids are remapped before the encoder
                cfg = ModernBertConfig(**c)
            else:
                cfg = enc_config
            self.passes, self.n_nodes = passes, n_nodes
            self.register_buffer("remap", torch.arange(vocab_full, dtype=torch.long).clamp(max=cfg.vocab_size - 1))
            self.enc = ModernBertModel(cfg)
            self.proj = nn.Linear(cfg.hidden_size, d)
            self.kind_emb = nn.Embedding(len(CELL_KINDS), d)
            self.vt_emb = nn.Embedding(len(VTYPES), d)
            self.pos_emb = nn.Embedding(1024, d)
            self.depth_emb = nn.Embedding(256, d)
            self.ready_emb = nn.Embedding(3, d)
            nn.init.zeros_(self.ready_emb.weight)
            self.gnorm = RMSNorm(d)
            self.queries = nn.Parameter(torch.randn(n_nodes, d) * 0.02)
            self.type_fb = nn.Embedding(len(NODE_TYPES), d)
            self.dlayers = nn.ModuleList([DecLayer() for _ in range(layers)])
            self.dnorm = RMSNorm(d)
            self.h_type = nn.Linear(d, len(NODE_TYPES))
            self.p_fn = nn.Linear(d, d)
            rel = torch.zeros(n_nodes, n_nodes, 6)
            idx = torch.arange(n_nodes - 1)
            rel[idx, idx + 1, 0] = 1.0
            self.register_buffer("rel", rel, persistent=False)
            self.d = d

        def pool(self, ids, mask):
            """Cells' tokens [N, T] → mean-pooled encoder states [N, H] (each cell on its own)."""
            h = self.enc(input_ids=self.remap[ids], attention_mask=mask).last_hidden_state
            m = mask[..., None].to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1)

        def encode(self, ids, mask, kind, vt, depth, ready, dedup=True):
            B, C = kind.shape
            if dedup:                                   # identical cells (shared across segments) are encoded once
                u, inv = torch.unique(torch.cat([ids, mask], 1), dim=0, return_inverse=True)
                T = ids.shape[1]
                pooled = self.pool(u[:, :T], u[:, T:])[inv]
            else:
                pooled = self.pool(ids, mask)
            return self.cells(pooled.view(B, C, -1), kind, vt, depth, ready)

        def cells(self, pooled, kind, vt, depth, ready):
            C = kind.shape[1]
            x = self.proj(pooled) + self.kind_emb(kind) + self.vt_emb(vt) + self.depth_emb(depth) + self.ready_emb(ready)
            x = x + self.pos_emb(torch.arange(C, device=x.device).clamp(max=1023))[None]
            return self.gnorm(x)

        def forward(self, ids, mask, kind, vt, depth, ready, cmask, fn_idx, fn_mask, dedup=True):
            return self.decode(self.encode(ids, mask, kind, vt, depth, ready, dedup), cmask, fn_idx, fn_mask)

        def from_pooled(self, pooled, kind, vt, depth, ready, cmask, fn_idx, fn_mask):
            """The decoder on already pooled cells [B, C, H] (inference with a cell cache) → last pass (type, fn) logits."""
            return self.decode(self.cells(pooled, kind, vt, depth, ready), cmask, fn_idx, fn_mask)[-1]

        def decode(self, cells, cmask, fn_idx, fn_mask):
            B, C, D = cells.shape
            q = self.queries[None].expand(B, -1, -1)
            fn_c = torch.gather(cells, 1, fn_idx[..., None].expand(-1, -1, D))
            outs, fnsel = [], None
            for _ in range(self.passes):
                h = q
                for layer in self.dlayers:
                    h = layer(h, cells, cmask, self.rel, fnsel)
                h = self.dnorm(h)
                tl = self.h_type(h)
                fl = torch.einsum("bnd,bmd->bnm", self.p_fn(h), fn_c) / math.sqrt(D)
                fl = fl.masked_fill(~fn_mask[:, None], -1e4)
                outs.append((tl, fl))
                ty = tl.argmax(-1)
                cell = torch.gather(fn_idx, 1, fl.argmax(-1))
                fvec = torch.gather(cells, 1, cell[..., None].expand(-1, -1, D))
                q = self.queries[None] + self.type_fb(ty) + fvec
                fnsel = F.one_hot(cell, C).to(cells.dtype) * (ty[..., None] > 0).to(cells.dtype)
            return outs

    return SegmentNet()


def loss(outs, type_lab, fn_lab):
    """Cross-entropy over the slots, every pass (the last ×1, earlier ×0.5), as in the typed decomposer's training."""
    import torch.nn.functional as F
    tot = 0.0
    for p, (tl, fl) in enumerate(outs):
        w = 1.0 if p == len(outs) - 1 else 0.5
        tot = tot + w * F.cross_entropy(tl.float().reshape(-1, tl.shape[-1]), type_lab.reshape(-1))
        if (fn_lab != -100).any():
            tot = tot + w * F.cross_entropy(fl.float().reshape(-1, fl.shape[-1]), fn_lab.reshape(-1), ignore_index=-100)
    return tot


def init_from_l5(net, state):
    """Initialise from a compressed Decomposer state dict (TRUNC2_VOCAB): encoder, projections, embeddings, decoder, type and fn
    heads; the graph level and the other heads are dropped; queries keep the first N_NODES slots. → (loaded, skipped)."""
    import torch
    own = net.state_dict()
    load, skip = {}, []
    for k, v in state.items():
        if k == "queries":
            v = v[: own[k].shape[0]]
        if k in own and own[k].shape == v.shape:
            load[k] = v.to(own[k].dtype)
        else:
            skip.append(k)
    if "remap" in state:
        load["remap"] = state["remap"].to(torch.long)
    net.load_state_dict(load, strict=False)
    return sorted(load), skip


def arrays_to_torch(a, device="cpu"):
    import torch
    return {k: torch.from_numpy(v).to(device) for k, v in a.items()}


# ---------------------------------------------------------------------------------------------------------------- model
def cell_layout(segs):
    """Segments → (per segment cells: texts, kinds, vts, depths, ready, fn names) and the padded feature arrays (no tokens):
    kind, vt, depth, ready, cmask [B, C], fn_idx, fn_mask [B, F]."""
    cs = [seg_cells(s) for s in segs]
    B = len(segs)
    C = max(len(c[0]) for c in cs)
    F = max(1, max(len(c[5]) for c in cs))
    a = {k: np.zeros((B, C), dtype=np.int64) for k in ("kind", "vt", "depth", "ready")}
    a["cmask"] = np.zeros((B, C), dtype=bool)
    a["fn_idx"] = np.zeros((B, F), dtype=np.int64)
    a["fn_mask"] = np.zeros((B, F), dtype=bool)
    for i, (texts, kinds, vts, depth, ready, names) in enumerate(cs):
        n = len(texts)
        a["kind"][i, :n], a["vt"][i, :n], a["depth"][i, :n], a["ready"][i, :n] = kinds, vts, depth, ready
        a["cmask"][i, :n] = True
        fi = [j for j, k in enumerate(kinds) if k == 2]
        a["fn_idx"][i, :len(fi)] = fi
        a["fn_mask"][i, :len(fi)] = True
    return cs, a


class SegmentModel:
    """A loaded segment model: propose(segments) → [{"nodes", "score", "second"}]; fingerprint / info() for the trace.
    Cells are encoded once and cached by text (a catalog's parts recur in every plan), in batches of similar length."""

    def __init__(self, runner, tok, meta, path, backend, cache_size=50_000):
        self.runner, self.tok, self.meta, self.path, self.backend = runner, tok, meta, path, backend
        self.n_nodes = int(meta.get("n_nodes", N_NODES))
        self.cache, self.cache_size = {}, cache_size
        self._fp = None

    @classmethod
    def load(cls, path_or_id, backend="auto", threads=None, quantized=False):
        """backend "torch", "onnx" or "auto"; quantized=True (onnx): the int8 cell encoder (onnx/encoder_int8.onnx) —
        about 2× faster on CPU and 4× smaller, pooled states within ~0.1% (cosine) of fp32."""
        path = os.path.expanduser(str(path_or_id))
        from .loader import optional
        if backend not in ("auto", "torch", "onnx"):
            raise ValueError('backend must be "torch", "onnx" or "auto"')
        if not os.path.isdir(path):
            allow = {"onnx": ["*.json", "onnx/*"], "torch": ["*.json", "*.safetensors"]}.get(backend)
            hub = optional("huggingface_hub", "onnx", "SegmentModel.load of a Hugging Face id")
            path = hub.snapshot_download(path_or_id, allow_patterns=allow)
        mf = os.path.join(path, "solvi_strategist.json")
        if not os.path.isfile(mf):
            raise FileNotFoundError(f"{path} has no solvi_strategist.json: not a solvi strategist checkpoint")
        meta = json.load(open(mf))
        if not str(meta.get("format", "")).startswith(FORMAT):
            raise ValueError(f"unknown strategist format {meta.get('format')!r} (this solvi reads {FORMAT!r})")
        enc_file, dec_file = (os.path.join(path, "onnx", f) for f in ("encoder.onnx", "decoder.onnx"))
        if quantized:
            enc_file = os.path.join(path, "onnx", "encoder_int8.onnx")
        if backend == "auto":
            backend = _auto_backend(os.path.isfile(enc_file), "SegmentModel")
        tok = Tok(path, int(meta.get("cell_tok", CELL_TOK)))
        if backend == "onnx":
            runner = OnnxRunner(enc_file, dec_file, threads)
            files = [enc_file, dec_file]
        elif backend == "torch":
            runner = TorchRunner(path, meta, threads)
            files = [os.path.join(path, "model.safetensors")]
        else:
            raise ValueError('backend must be "torch", "onnx" or "auto"')
        m = cls(runner, tok, meta, path, backend)
        m._files = [os.path.join(path, f) for f in ("solvi_strategist.json", "config.json", "tokenizer.json")] + files
        m.model_id = str(path_or_id)
        return m

    @property
    def fingerprint(self):
        if self._fp is None:
            from .decide import _file_fingerprint
            self._fp = _file_fingerprint(self._files)
        return self._fp

    def info(self):
        return {"type": "ModelStrategist", "id": self.meta.get("name", os.path.basename(self.path.rstrip("/"))),
                "fp": self.fingerprint}

    def pooled(self, texts, bs=64):
        """Mean-pooled encoder states of cell texts (cached)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            ids = self.tok(todo)
            order = sorted(range(len(todo)), key=lambda j: len(ids[j]))
            for s in range(0, len(order), bs):
                ch = order[s:s + bs]
                T = max(len(ids[j]) for j in ch)
                x = np.full((len(ch), T), self.tok.pad, dtype=np.int64)
                m = np.zeros((len(ch), T), dtype=np.int64)
                for r, j in enumerate(ch):
                    x[r, :len(ids[j])] = ids[j]
                    m[r, :len(ids[j])] = 1
                v = self.runner.pool(x, m)
                for r, j in enumerate(ch):
                    self.cache[todo[j]] = v[r]
            if len(self.cache) > self.cache_size:
                self.cache = {t: self.cache[t] for t in texts if t in self.cache}
        return np.stack([self.cache[t] for t in texts])

    def propose(self, segs, bs=64):
        out = []
        for s in range(0, len(segs), bs):
            chunk = segs[s:s + bs]
            cs, a = cell_layout(chunk)
            B, C = a["kind"].shape
            texts = [t for c in cs for t in c[0]]
            vec = self.pooled(texts)
            P = np.zeros((B, C, vec.shape[1]), dtype=np.float32)
            k = 0
            for i, c in enumerate(cs):
                P[i, :len(c[0])] = vec[k:k + len(c[0])]
                k += len(c[0])
            tl, fl = self.runner.decode(P, a)
            out += decode(tl, fl, [c[5] for c in cs])
        return out


class TorchRunner:
    def __init__(self, path, meta, threads=None):
        from .loader import optional
        torch = optional("torch", "model", 'SegmentModel (backend="torch")')
        load_file = optional("safetensors.torch", "model", 'SegmentModel (backend="torch")').load_file
        if threads:
            torch.set_num_threads(int(threads))
        self.torch = torch
        cfg = json.load(open(os.path.join(path, "config.json")))
        cfg["attn_implementation"] = "sdpa"
        a = meta.get("arch", {})
        self.net = build_net(cfg, n_nodes=int(meta.get("n_nodes", N_NODES)), d=a.get("d", 512), heads=a.get("heads", 8),
                             layers=a.get("layers", 4), passes=a.get("passes", 3), vocab_full=a.get("vocab_full", 50368))
        sd = load_file(os.path.join(path, "model.safetensors"))
        self.net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
        self.net.eval()

    def pool(self, ids, mask):
        torch = self.torch
        with torch.inference_mode():
            return self.net.pool(torch.from_numpy(ids), torch.from_numpy(mask)).float().numpy()

    def decode(self, pooled, a):
        torch = self.torch
        t = {k: torch.from_numpy(a[k]) for k in DEC_NAMES[1:]}
        with torch.inference_mode():
            tl, fl = self.net.from_pooled(torch.from_numpy(pooled), *[t[k] for k in DEC_NAMES[1:]])
        return tl.float().numpy(), fl.float().numpy()


DEC_NAMES = ["pooled", "kind", "vt", "depth", "ready", "cmask", "fn_idx", "fn_mask"]


class OnnxRunner:
    def __init__(self, enc_file, dec_file, threads=None):
        from .loader import optional
        ort = optional("onnxruntime", "onnx", 'SegmentModel (backend="onnx")')
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.enc = ort.InferenceSession(enc_file, so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(dec_file, so, providers=["CPUExecutionProvider"])

    def pool(self, ids, mask):
        return self.enc.run(["pooled"], {"ids": ids, "mask": mask})[0]

    def decode(self, pooled, a):
        feed = {"pooled": pooled.astype(np.float32), **{k: a[k] for k in DEC_NAMES[1:]}}
        tl, fl = self.dec.run(["type_logits", "fn_logits"], feed)
        return tl, fl


# ---------------------------------------------------------------------------------------------------------------- export
def save(net, path, enc_config, tokenizer_dir, meta):
    """Write a checkpoint folder: solvi_strategist.json, config.json, tokenizer.json, model.safetensors (fp32)."""
    import shutil
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    save_file(sd, os.path.join(path, "model.safetensors"))
    json.dump(enc_config if isinstance(enc_config, dict) else enc_config.to_dict(), open(os.path.join(path, "config.json"), "w"),
              indent=1)
    shutil.copy(os.path.join(tokenizer_dir, "tokenizer.json"), os.path.join(path, "tokenizer.json"))
    json.dump({"format": FORMAT, **meta}, open(os.path.join(path, "solvi_strategist.json"), "w"), indent=1)


def export_onnx(path, opset=18):
    """Export a checkpoint folder's torch model to onnx/encoder.onnx (cell tokens → pooled states) and onnx/decoder.onnx
    (pooled cells + features → last-pass logits)."""
    import torch
    m = SegmentModel.load(path, backend="torch")
    net = m.runner.net

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, ids, mask):
            return self.net.pool(ids, mask)

    class Dec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, pooled, kind, vt, depth, ready, cmask, fn_idx, fn_mask):
            return self.net.from_pooled(pooled, kind, vt, depth, ready, cmask, fn_idx, fn_mask)
    seg = {"fact": "b", "type": "float", "question": "q", "available": [("a", "float", "given")],
           "candidates": [{"name": "b1", "kind": "fn", "params": [("a", "float")], "returns": "float", "doc": "x", "depth": 1},
                          {"name": "b2", "kind": "fn", "params": [("z", "float")], "returns": "float", "doc": "y", "depth": 1}]}
    a, _ = batch_arrays([seg, seg], m.tok)
    t = arrays_to_torch(a)
    os.makedirs(os.path.join(path, "onnx"), exist_ok=True)
    fe, fd = os.path.join(path, "onnx", "encoder.onnx"), os.path.join(path, "onnx", "decoder.onnx")
    with torch.no_grad():
        torch.onnx.export(Enc().eval(), (t["ids"], t["mask"]), fe, input_names=["ids", "mask"], output_names=["pooled"],
                          dynamic_axes={"ids": {0: "n", 1: "t"}, "mask": {0: "n", 1: "t"}, "pooled": {0: "n"}},
                          opset_version=opset, dynamo=False)
    with torch.no_grad():
        pooled = net.pool(t["ids"], t["mask"]).view(2, -1, net.proj.in_features)
    dyn = {"pooled": {0: "b", 1: "c"}, "kind": {0: "b", 1: "c"}, "vt": {0: "b", 1: "c"}, "depth": {0: "b", 1: "c"},
           "ready": {0: "b", 1: "c"}, "cmask": {0: "b", 1: "c"}, "fn_idx": {0: "b", 1: "f"}, "fn_mask": {0: "b", 1: "f"},
           "type_logits": {0: "b"}, "fn_logits": {0: "b", 2: "f"}}
    with torch.no_grad():
        torch.onnx.export(Dec().eval(), (pooled, *[t[k] for k in DEC_NAMES[1:]]), fd, input_names=DEC_NAMES,
                          output_names=["type_logits", "fn_logits"], dynamic_axes=dyn, opset_version=opset, dynamo=False)
    return fe, fd


def quantize_onnx(path):
    """Dynamic int8 quantization of onnx/encoder.onnx → onnx/encoder_int8.onnx (optional, smaller and faster on CPU)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    src = os.path.join(path, "onnx", "encoder.onnx")
    dst = os.path.join(path, "onnx", "encoder_int8.onnx")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    return dst
