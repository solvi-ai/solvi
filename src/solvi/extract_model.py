"""Model-backed @extract: ModernBERT + two pointer heads (span start and end) that "cut a field's value out of the text given
its description" (extractive QA). The answer is always a span of the text itself, so it comes with a quote and offsets by
construction. The field description is the @extract function's docstring (a field can be added without retraining if its
description resembles the trained ones)."""
from __future__ import annotations

import math
import random
import time

import numpy as np


class SpanExtractor:
    def __init__(self, model_name="answerdotai/ModernBERT-large", max_len=1024, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 2).to(self.device)
        self.max_len = max_len
        self.model_name = self.model_id = model_name
        self._fp_weights = None

    def _batch(self, items):
        """items: [(text, description, (start, end) | None)] → tensors and target positions in tokens."""
        enc = self.tok([d for _, d, _ in items], [t for t, _, _ in items], truncation="only_second", max_length=self.max_len,
                       padding=True, return_offsets_mapping=True, return_tensors="pt")
        starts, ends, ctx_masks = [], [], []
        for i, (_, _, span) in enumerate(items):
            seq = enc.sequence_ids(i)
            offs = enc["offset_mapping"][i].tolist()
            ctx = [j for j, s in enumerate(seq) if s == 1]
            m = [s == 1 for s in seq]
            ctx_masks.append(m)
            if span is None:
                starts.append(0)
                ends.append(0)
                continue
            s_c, e_c = span
            st = next((j for j in ctx if offs[j][0] <= s_c < offs[j][1] or offs[j][0] >= s_c), ctx[0])
            en = next((j for j in reversed(ctx) if offs[j][0] < e_c), st)
            starts.append(st)
            ends.append(max(en, st))
        return enc, starts, ends, ctx_masks

    def _logits(self, enc):
        out = self.enc(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device)).last_hidden_state
        lg = self.head(out)
        return lg[..., 0], lg[..., 1]

    def fit(self, examples, epochs=4, lr=3e-5, bs=8, seed=0, log=print):
        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        ex = [e for e in examples if e[2] is not None]
        params = list(self.enc.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW([{"params": list(self.enc.parameters()), "lr": lr}, {"params": list(self.head.parameters()), "lr": lr * 10}],
                                weight_decay=0.01)
        total = epochs * math.ceil(len(ex) / bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.0, 1 - s / total))
        self.enc.train()
        t0 = time.time()
        for ep in range(epochs):
            random.shuffle(ex)
            tot = 0.0
            for b in range(0, len(ex), bs):
                enc, st, en, m = self._batch(ex[b:b + bs])
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    ls, le = self._logits(enc)
                mask = torch.tensor(m, device=self.device)
                ls = ls.float().masked_fill(~mask, -1e4)
                le = le.float().masked_fill(~mask, -1e4)
                loss = (torch.nn.functional.cross_entropy(ls, torch.tensor(st, device=self.device)) +
                        torch.nn.functional.cross_entropy(le, torch.tensor(en, device=self.device))) / 2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            log(f"[extract] epoch {ep + 1}/{epochs}: loss {tot / max(1, math.ceil(len(ex) / bs)):.4f}, {time.time() - t0:.0f} s")
        self.enc.eval()
        self._fp_weights = None
        return self

    def fingerprint(self):
        """A stable hash of this extractor (settings and sampled weights). Register a part that uses it with
        `@cat.extract(model=extractor)` so the trace records it."""
        from .provenance import digest, torch_fingerprint
        if self._fp_weights is None:
            import os
            files = self.model_name if os.path.isdir(str(self.model_name)) else None
            self._fp_weights = torch_fingerprint([self.enc, self.head], files)
        return digest("SpanExtractor", self.max_len, self._fp_weights)

    def predict(self, items, bs=16, max_span=64):
        """items: [(text, description)] → [(start, end, confidence)] in text characters."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for b in range(0, len(items), bs):
                ch = [(t, d, None) for t, d in items[b:b + bs]]
                enc, _, _, m = self._batch(ch)
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    ls, le = self._logits(enc)
                mask = torch.tensor(m, device=self.device)
                ps = torch.softmax(ls.float().masked_fill(~mask, -1e4), -1).cpu().numpy()
                pe = torch.softmax(le.float().masked_fill(~mask, -1e4), -1).cpu().numpy()
                for i in range(len(ch)):
                    offs = enc["offset_mapping"][i].tolist()
                    L = len(offs)
                    sc = np.triu(np.outer(ps[i, :L], pe[i, :L])) - np.triu(np.outer(ps[i, :L], pe[i, :L]), max_span)
                    s, e = np.unravel_index(int(sc.argmax()), sc.shape)
                    out.append((int(offs[s][0]), int(offs[e][1]), float(sc[s, e])))
        return out
