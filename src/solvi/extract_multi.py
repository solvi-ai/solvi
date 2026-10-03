"""Single-pass @extract: ModernBERT reads the document once, with a pair of pointer heads (start / end) per field.
For solvi: extractor.field(name) returns a function doc → Quote; all fields of a document come from one pass (cached by text).

Moving into the knowledge memory in 1.0: this module will be folded into solvi's knowledge memory, and its API may
change then.

The extractor protocol it shares with solvi.extract_long.LongSpanExtractor: fit(items), predict(text, field),
field(name[, description]), save(path) / load(path), fingerprint(). items here are [(text, {field: (start, end) |
None})] (0.7's fit(docs, spans) and predict_doc(text) were removed in 0.9)."""
from __future__ import annotations

import hashlib
import math
import random
import time

import numpy as np

from . import _deprecate
from .core.catalog import Quote

CACHE = 5000                                     # documents whose predictions are kept (then the cache starts again)


class MultiSpanExtractor:
    def __init__(self, fields, model_name="answerdotai/ModernBERT-large", max_len=1024, device=None):
        from ._loader import optional
        torch = optional("torch", "model", "MultiSpanExtractor")
        tf = optional("transformers", "model", "MultiSpanExtractor")
        AutoModel, AutoTokenizer = tf.AutoModel, tf.AutoTokenizer
        self.torch = torch
        self.fields = list(fields)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 2 * len(self.fields)).to(self.device)
        self.max_len = max_len
        self._cache = {}
        self.temp = {f: 1.0 for f in self.fields}
        self.model_name = self.model_id = model_name
        self._fp_weights = None

    def _enc(self, texts):
        return self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_offsets_mapping=True, return_tensors="pt")

    def _logits(self, enc):
        h = self.enc(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device)).last_hidden_state
        return self.head(h)                                          # [B, L, 2F]

    @staticmethod
    def _tok_span(offs, span):
        """A character span → (first token, last token); None when the span starts past the encoded (truncated) text —
        the window cannot point at it, and (first token, last token) would be a wrong target."""
        s_c, e_c = span
        idx = [j for j, (a, b) in enumerate(offs) if b > a]
        if not idx or s_c >= offs[idx[-1]][1]:
            return None
        st = next((j for j in idx if offs[j][1] > s_c), idx[0])
        en = next((j for j in reversed(idx) if offs[j][0] < e_c), st)
        return st, max(st, en)

    def fit(self, items, *removed, epochs=4, lr=3e-5, bs=8, seed=0, log=print):
        """items: [(text, {field: (start, end) | None})] (0.7's fit(docs, spans) was removed in 0.9). A span past the
        encoded text (max_len tokens) is left out of training."""
        items = list(items)
        if removed or (items and isinstance(items[0], str)):
            raise TypeError("MultiSpanExtractor.fit(docs, spans) was renamed in 0.8 and removed in 0.9: use "
                            "fit([(text, spans), ...])")
        docs, spans = [t for t, _ in items], [dict(sp or {}) for _, sp in items]
        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        idx = list(range(len(docs)))
        opt = torch.optim.AdamW([{"params": list(self.enc.parameters()), "lr": lr}, {"params": list(self.head.parameters()), "lr": lr * 10}],
                                weight_decay=0.01)
        total = epochs * math.ceil(len(idx) / bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.0, 1 - s / total))
        params = list(self.enc.parameters()) + list(self.head.parameters())
        self.enc.train()
        t0 = time.time()
        for ep in range(epochs):
            random.shuffle(idx)
            tot = 0.0
            for b in range(0, len(idx), bs):
                ch = idx[b:b + bs]
                enc = self._enc([docs[i] for i in ch])
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    lg = self._logits(enc).float()
                mask = enc["attention_mask"].to(self.device).bool()
                loss, n = 0.0, 0
                for fi, f in enumerate(self.fields):
                    tgt_s, tgt_e, rows = [], [], []
                    for r, i in enumerate(ch):
                        sp = spans[i].get(f)
                        if sp is None:
                            continue
                        ts = self._tok_span(enc["offset_mapping"][r].tolist(), sp)
                        if ts is None:
                            continue
                        s, e = ts
                        tgt_s.append(s)
                        tgt_e.append(e)
                        rows.append(r)
                    if not rows:
                        continue
                    ls = lg[rows, :, 2 * fi].masked_fill(~mask[rows], -1e4)
                    le = lg[rows, :, 2 * fi + 1].masked_fill(~mask[rows], -1e4)
                    loss = loss + torch.nn.functional.cross_entropy(ls, torch.tensor(tgt_s, device=self.device)) + \
                        torch.nn.functional.cross_entropy(le, torch.tensor(tgt_e, device=self.device))
                    n += 2
                if n == 0:
                    continue
                loss = loss / n
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            log(f"[extract1] epoch {ep + 1}/{epochs}: loss {tot / max(1, math.ceil(len(idx) / bs)):.4f}, {time.time() - t0:.0f} s")
        self.enc.eval()
        self._cache = {}
        self._fp_weights = None
        return self

    predict_doc = _deprecate.removed_attr("predict_doc()", "predict(text[, field])", "MultiSpanExtractor")

    def predict(self, text, field=None, max_span=64):
        """→ {field: (start, end, confidence)} in a single pass, or one field's (start, end, confidence)."""
        if field is not None:
            return self.predict(text, max_span=max_span)[field]
        key = (hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest(), max_span)
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        with torch.no_grad():
            enc = self._enc([text])
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                lg = self._logits(enc).float()[0]
        offs = enc["offset_mapping"][0].tolist()
        valid = np.array([b > a for a, b in offs])
        out = {}
        for fi, f in enumerate(self.fields):
            zs = lg[:, 2 * fi].cpu().numpy()
            ze = lg[:, 2 * fi + 1].cpu().numpy()
            zs[~valid], ze[~valid] = -1e4, -1e4
            T = self.temp[f]
            ps = np.exp((zs - zs.max()) / T)
            ps /= ps.sum()
            pe = np.exp((ze - ze.max()) / T)
            pe /= pe.sum()
            sc = np.triu(np.outer(ps, pe)) - np.triu(np.outer(ps, pe), max_span)
            s, e = np.unravel_index(int(sc.argmax()), sc.shape)
            out[f] = (int(offs[s][0]), int(offs[e][1]), float(sc[s, e]))
        if len(self._cache) >= CACHE:
            self._cache = {}
        self._cache[key] = out
        return out

    def fingerprint(self):
        """A stable hash of this extractor: fields, settings, temperatures, sampled weights (see LongSpanExtractor)."""
        from .core.provenance import digest, torch_fingerprint
        if self._fp_weights is None:
            import os
            files = self.model_name if os.path.isdir(str(self.model_name)) else None
            self._fp_weights = torch_fingerprint([self.enc, self.head], files)
        return digest("MultiSpanExtractor", self.fields, self.max_len, self.temp, self._fp_weights)

    def field(self, name):
        def f(doc):
            s, e, c = self.predict(doc, name)
            return Quote(doc[s:e], s, e, confidence=c)
        f.__name__ = name
        f.__solvi_model__ = self                 # cat.extract records this model (and its fingerprint) in the trace
        f.__solvi_provenance__ = "quoted"
        return f

    def fit_temperature(self, docs, gold_ok, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)):
        """Per-field temperature fitted on held-out documents: minimizes log loss of confidence vs. whether the span matched gold.
        gold_ok(field, doc_index, (start, end)) → bool."""
        for f in self.fields:
            best = None
            for T in grid:
                self.temp[f] = T
                self._cache = {}
                nll = 0.0
                for i, d in enumerate(docs):
                    s, e, c = self.predict(d, f)
                    y = gold_ok(f, i, (s, e))
                    c = min(max(c, 1e-6), 1 - 1e-6)
                    nll -= math.log(c if y else 1 - c)
                if best is None or nll < best[0]:
                    best = (nll, T)
            self.temp[f] = best[1]
        self._cache = {}
        return dict(self.temp)

    def save(self, path):
        """Encoder weights (bf16 safetensors), tokenizer, the span heads and the settings into a directory (load reads
        it back) — as LongSpanExtractor.save."""
        import json
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        self.enc.to(self.torch.bfloat16).save_pretrained(path)
        self.enc.to(self.torch.float32)
        self._fp_weights = None                  # the weights in memory are now bf16-rounded
        self._cache = {}
        self.tok.save_pretrained(path)
        self.torch.save(self.head.state_dict(), f"{path}/span_heads.pt")
        with open(f"{path}/solvi_extract.json", "w") as fh:
            json.dump({"kind": "multi", "fields": self.fields, "max_len": self.max_len, "temp": self.temp,
                       "base_model": self.model_name}, fh, indent=1)

    @classmethod
    def load(cls, path, device=None):
        """From a directory written by save(), or a Hugging Face model id (downloaded once and cached)."""
        import json
        import os
        path_or_id = path
        if not os.path.isdir(path):
            from ._loader import optional
            path = optional("huggingface_hub", "model", "MultiSpanExtractor.load of a Hugging Face id").snapshot_download(path)
        with open(f"{path}/solvi_extract.json") as fh:
            cfg = json.load(fh)
        if cfg.get("kind") != "multi":
            raise ValueError(f"{path_or_id}: not a MultiSpanExtractor (solvi_extract.json has kind {cfg.get('kind')!r}; "
                             "a LongSpanExtractor loads with solvi.extract_long.LongSpanExtractor.load)")
        ex = cls(cfg["fields"], path, max_len=cfg["max_len"], device=device)
        ex.enc.to(ex.torch.float32)
        ex.head.load_state_dict(ex.torch.load(f"{path}/span_heads.pt", map_location=ex.device, weights_only=True))
        ex.temp = {f: float(cfg.get("temp", {}).get(f, 1.0)) for f in ex.fields}
        ex.model_id = path_or_id
        return ex


__all__ = ["MultiSpanExtractor"]
