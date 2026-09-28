"""Single-pass @extract: ModernBERT reads the document once, with a pair of pointer heads (start / end) per field.
For solvi: extractor.field(name) returns a function doc → Quote; all fields of a document come from one pass (cached by text)."""
from __future__ import annotations

import hashlib
import math
import random
import time

import numpy as np

from .core import Quote


class MultiSpanExtractor:
    def __init__(self, fields, model_name="answerdotai/ModernBERT-large", max_len=1024, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
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
        s_c, e_c = span
        idx = [j for j, (a, b) in enumerate(offs) if b > a]
        st = next((j for j in idx if offs[j][1] > s_c), idx[0])
        en = next((j for j in reversed(idx) if offs[j][0] < e_c), st)
        return st, max(st, en)

    def fit(self, docs, spans, epochs=4, lr=3e-5, bs=8, seed=0, log=print):
        """docs: [text]; spans: [{field: (start, end) | None}]."""
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
                        s, e = self._tok_span(enc["offset_mapping"][r].tolist(), sp)
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

    def predict_doc(self, text, max_span=64):
        """→ {field: (start, end, confidence)} in a single pass."""
        key = hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()
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
        if len(self._cache) > 5000:
            self._cache = {}
        self._cache[key] = out
        return out

    def fingerprint(self):
        """A stable hash of this extractor: fields, settings, temperatures, sampled weights (see LongSpanExtractor)."""
        from .provenance import digest, torch_fingerprint
        if self._fp_weights is None:
            import os
            files = self.model_name if os.path.isdir(str(self.model_name)) else None
            self._fp_weights = torch_fingerprint([self.enc, self.head], files)
        return digest("MultiSpanExtractor", self.fields, self.max_len, self.temp, self._fp_weights)

    def field(self, name):
        def f(doc):
            s, e, c = self.predict_doc(doc)[name]
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
                    s, e, c = self.predict_doc(d)[f]
                    y = gold_ok(f, i, (s, e))
                    c = min(max(c, 1e-6), 1 - 1e-6)
                    nll -= math.log(c if y else 1 - c)
                if best is None or nll < best[0]:
                    best = (nll, T)
            self.temp[f] = best[1]
        self._cache = {}
        return dict(self.temp)
