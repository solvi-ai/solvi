"""@extract for real-world, general-purpose documents: a field is defined by its DESCRIPTION, the document may be long
(windows), and the field may be absent ("no answer"). ModernBERT: input "field description [SEP] document window", two pointer
heads; "no answer" is position 0 (the special token). Prediction: the best span across all windows; an answer exists if its
score is above the field's threshold (tuned on held-out examples). Also works for fields unseen in training, from the
description alone ("new field")."""
from __future__ import annotations

import math
import random
import time

import numpy as np

from .core import Quote


class LongSpanExtractor:
    def __init__(self, model_name="answerdotai/ModernBERT-large", max_len=1024, stride=128, max_span=96, device=None):
        """stride: overlap between adjacent windows in tokens (as in transformers); max_span: maximum answer length in tokens."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 2).to(self.device)
        self.max_len, self.stride, self.max_span = max_len, stride, max_span
        self.thr = {}
        self.thr_default = 0.0                   # threshold for fields without their own examples (new field by description)
        self.model_name = model_name
        self.model_id = model_name               # recorded in the trace (load() sets the id or path it was loaded from)
        self._cache = {}
        self._tok_key, self._tok_val = None, None
        self._fp_weights = None

    def _windows(self, desc, text):
        """Windows "[CLS] description [SEP] text chunk [SEP]" overlapping by stride; split manually (the ModernBERT tokenizer's
        overflow returns only one extra window). → {input_ids, attention_mask, offset_mapping, ctx}."""
        key = hash(text)
        if self._tok_key != key:
            t = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            self._tok_key, self._tok_val = key, (t["input_ids"], t["offset_mapping"])
        ids, offs = self._tok_val
        d = self.tok(desc, add_special_tokens=False)["input_ids"][:64]
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        room = self.max_len - len(d) - 3
        step = max(1, room - self.stride)
        out = {"input_ids": [], "attention_mask": [], "offset_mapping": [], "ctx": []}
        for a in range(0, max(1, len(ids)), step):
            chunk = ids[a:a + room]
            seq = [cls] + d + [sep] + chunk + [sep]
            n = len(seq)
            out["input_ids"].append(seq + [pad] * (self.max_len - n))
            out["attention_mask"].append([1] * n + [0] * (self.max_len - n))
            out["offset_mapping"].append([(0, 0)] * (len(d) + 2) + list(offs[a:a + room]) + [(0, 0)] * (self.max_len - n + 1))
            out["ctx"].append(list(range(len(d) + 2, len(d) + 2 + len(chunk))))
            if a + room >= len(ids):
                break
        return out

    def _ctx(self, enc, i):
        return enc["ctx"][i]

    def make_examples(self, items, neg_per_item=3, seed=0):
        """items: [(text, description, (start, end) | None[, neg])] → training windows: every window with the answer + up to
        neg_per_item (or the item's own neg) windows without it."""
        rng = random.Random(seed)
        out = []
        for it in items:
            text, desc, span = it[:3]
            npi = it[3] if len(it) > 3 else neg_per_item
            enc = self._windows(desc, text)
            pos, neg = [], []
            for i in range(len(enc["input_ids"])):
                offs = enc["offset_mapping"][i]
                ctx = self._ctx(enc, i)
                if not ctx:
                    continue
                c0, c1 = offs[ctx[0]][0], offs[ctx[-1]][1]
                if span is not None and c0 <= span[0] < c1:            # answer starts in this window (a long answer is clipped at the window edge)
                    st = next((j for j in ctx if offs[j][1] > span[0]), ctx[0])
                    en = next((j for j in reversed(ctx) if offs[j][0] < span[1]), st)
                    pos.append((enc["input_ids"][i], enc["attention_mask"][i], st, min(max(st, en), st + self.max_span - 1)))
                else:
                    neg.append((enc["input_ids"][i], enc["attention_mask"][i], 0, 0))
            rng.shuffle(neg)
            out += pos + neg[:npi if pos or span is None else max(1, npi // 3)]
        return out

    def fit(self, items, epochs=3, lr=3e-5, bs=8, seed=0, neg_per_item=3, log=print):
        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        ex = self.make_examples(items, neg_per_item, seed)
        opt = torch.optim.AdamW([{"params": list(self.enc.parameters()), "lr": lr}, {"params": list(self.head.parameters()), "lr": lr * 10}],
                                weight_decay=0.01)
        total = epochs * math.ceil(len(ex) / bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.0, 1 - s / total))
        params = list(self.enc.parameters()) + list(self.head.parameters())
        self.enc.train()
        t0 = time.time()
        for ep in range(epochs):
            random.shuffle(ex)
            # batches of windows of similar length (sorted within chunks of 64 batches), in random order
            batches = []
            for k in range(0, len(ex), 64 * bs):
                part = sorted(ex[k:k + 64 * bs], key=lambda c: sum(c[1]))
                batches += [part[b:b + bs] for b in range(0, len(part), bs)]
            random.shuffle(batches)
            tot = 0.0
            for ch in batches:
                L = max(sum(c[1]) for c in ch)                  # trim the batch to its longest window
                ids = torch.tensor([c[0][:L] for c in ch], device=self.device)
                att = torch.tensor([c[1][:L] for c in ch], device=self.device)
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
                    lg = self.head(h).float()
                m = att.bool()
                ls, le = lg[..., 0].masked_fill(~m, -1e4), lg[..., 1].masked_fill(~m, -1e4)
                loss = (torch.nn.functional.cross_entropy(ls, torch.tensor([c[2] for c in ch], device=self.device)) +
                        torch.nn.functional.cross_entropy(le, torch.tensor([c[3] for c in ch], device=self.device))) / 2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            log(f"[extractL] epoch {ep + 1}/{epochs}: loss {tot / max(1, math.ceil(len(ex) / bs)):.4f}, windows {len(ex)}, {time.time() - t0:.0f} s")
        self.enc.eval()
        self._cache = {}
        self._fp_weights = None
        return self

    def predict(self, text, desc, bs=8):
        """→ (start, end, span score, "no answer" score) — the best span across all windows."""
        key = (hash(text), desc)
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        max_span = self.max_span
        enc = self._windows(desc, text)
        best, null = (0, 0, 0.0), 0.0                      # no window with text (an empty document): no span, score 0
        with torch.no_grad():
            n = len(enc["input_ids"])
            for b in range(0, n, bs):
                L = max(sum(a) for a in enc["attention_mask"][b:b + bs])
                ids = torch.tensor([x[:L] for x in enc["input_ids"][b:b + bs]], device=self.device)
                att = torch.tensor([x[:L] for x in enc["attention_mask"][b:b + bs]], device=self.device)
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    lg = self.head(self.enc(input_ids=ids, attention_mask=att).last_hidden_state).float()
                for r in range(lg.shape[0]):
                    i = b + r
                    ctx = self._ctx(enc, i)
                    if not ctx:
                        continue
                    offs = enc["offset_mapping"][i]
                    ps = torch.softmax(lg[r, :, 0].masked_fill(~att[r].bool(), -1e4), -1).cpu().numpy()
                    pe = torch.softmax(lg[r, :, 1].masked_fill(~att[r].bool(), -1e4), -1).cpu().numpy()
                    null = max(null, float(ps[0] * pe[0]))
                    c0, c1 = ctx[0], ctx[-1] + 1
                    sc = np.triu(np.outer(ps[c0:c1], pe[c0:c1])) - np.triu(np.outer(ps[c0:c1], pe[c0:c1]), max_span)
                    s, e = np.unravel_index(int(sc.argmax()), sc.shape)
                    if sc[s, e] > best[2]:
                        best = (offs[c0 + s][0], offs[c0 + e][1], float(sc[s, e]))
        st, en = int(best[0]), int(best[1])
        while st < en and text[st].isspace():              # BPE offsets include the leading space
            st += 1
        out = (st, en, best[2], null)
        self._cache[key] = out
        return out

    def tune_threshold(self, name, items, grid=(0.0003, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
        """Per-field "answer present" threshold from held-out examples: maximizes present/absent decision accuracy."""
        best = None
        for t in grid:
            acc = np.mean([(self.predict(x, d)[2] >= t) == (sp is not None) for x, d, sp in items])
            if best is None or acc > best[0]:
                best = (acc, t)
        self.thr[name] = best[1]
        return best

    def tune_default_threshold(self, items, grid=(0.0003, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)):
        """One threshold from held-out examples of many fields — used for fields that have no examples of their own."""
        best = None
        for t in grid:
            acc = np.mean([(self.predict(x, d)[2] >= t) == (sp is not None) for x, d, sp in items])
            if best is None or acc > best[0]:
                best = (acc, t)
        self.thr_default = best[1]
        return best

    def save(self, path):
        """Encoder weights (bf16 safetensors), tokenizer, span head and settings into a directory."""
        import json
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        self.enc.to(self.torch.bfloat16).save_pretrained(path)
        self.enc.to(self.torch.float32)
        self._fp_weights = None                  # the weights in memory are now bf16-rounded
        self.tok.save_pretrained(path)
        self.torch.save(self.head.state_dict(), f"{path}/span_head.pt")
        json.dump({"max_len": self.max_len, "stride": self.stride, "max_span": self.max_span, "thr_default": self.thr_default,
                   "thr": self.thr, "base_model": self.model_name}, open(f"{path}/solvi_extract.json", "w"), indent=1)

    @classmethod
    def load(cls, path, device=None):
        """From a directory written by save(), or a Hugging Face model id (downloaded once and cached)."""
        import json
        import os
        path_or_id = path
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        cfg = json.load(open(f"{path}/solvi_extract.json"))
        ex = cls(path, max_len=cfg["max_len"], stride=cfg["stride"], max_span=cfg["max_span"], device=device)
        ex.enc.to(ex.torch.float32)
        ex.head.load_state_dict(ex.torch.load(f"{path}/span_head.pt", map_location=ex.device, weights_only=True))
        ex.thr_default, ex.thr = cfg["thr_default"], cfg.get("thr", {})
        ex.model_id = path_or_id
        return ex

    def embed(self, text, bs=8):
        """Document embedding: the encoder's token states averaged over every window of the text (no field description)."""
        torch = self.torch
        key = ("__embed__", hash(text))
        if key in self._cache:
            return self._cache[key]
        enc = self._windows("", text)
        tot, cnt = None, 0
        with torch.no_grad():
            for b in range(0, len(enc["input_ids"]), bs):
                L = max(sum(a) for a in enc["attention_mask"][b:b + bs])
                ids = torch.tensor([x[:L] for x in enc["input_ids"][b:b + bs]], device=self.device)
                att = torch.tensor([x[:L] for x in enc["attention_mask"][b:b + bs]], device=self.device)
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state.float()
                m = att.unsqueeze(-1).float()
                s = (h * m).sum((0, 1))
                tot = s if tot is None else tot + s
                cnt += int(m.sum())
        v = (tot / max(1, cnt)).cpu().numpy()
        self._cache[key] = v
        return v

    def embedder(self, name="doc_embedding"):
        """A catalog part: doc → embedding vector, usable as a feature of System.fit_fast."""
        def f(doc):
            return self.embed(doc)
        f.__name__ = name
        f.__doc__ = "document embedding"
        f.__solvi_model__ = self
        f.__solvi_provenance__ = "learned"
        return f

    def fingerprint(self):
        """A stable hash of this extractor: settings, thresholds, the span head, sampled encoder weights and the weight files
        it was loaded from (recorded in the trace; replay compares it with the catalog's current model)."""
        from .provenance import digest, torch_fingerprint
        if self._fp_weights is None:
            import os
            files = self.model_name if os.path.isdir(str(self.model_name)) else None
            self._fp_weights = torch_fingerprint([self.enc, self.head], files)
        return digest("LongSpanExtractor", self.max_len, self.stride, self.max_span, self.thr_default, self.thr,
                      self._fp_weights)

    def field(self, name, desc):
        def f(doc):
            s, e, sc, _ = self.predict(doc, desc)
            sc = min(1.0, max(0.0, float(sc)))             # a probability: the trace never records one outside [0, 1]
            if sc < self.thr.get(name, self.thr_default):
                return Quote("", 0, 0, confidence=1 - sc)
            return Quote(doc[s:e], s, e, confidence=sc)
        f.__name__ = name
        f.__doc__ = desc
        f.__solvi_model__ = self                 # cat.extract records this model (and its fingerprint) in the trace
        f.__solvi_provenance__ = "quoted"
        return f
