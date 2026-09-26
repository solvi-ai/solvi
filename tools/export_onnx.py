"""Export a LongSpanExtractor (encoder + span head) to ONNX for browsers and CPUs: fp32 and fp16 (identical spans on our tests; int8 with INT8=1 loses accuracy), plus an
agreement check against the PyTorch model on your own texts.

Usage:
  uv run --extra model --with onnx --with onnxscript --with onnxruntime \
      python tools/export_onnx.py <model dir or HF id> <out dir> [texts.jsonl with {"text", "desc"} lines for the check]

Graph: inputs input_ids, attention_mask (int64, [batch, length]) -> logits (float, [batch, length, 2]): start and end scores.
Windowing, "no answer" (position 0) and span decoding stay in the caller (see LongSpanExtractor.predict)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def export(model, out):
    import torch
    from solvi.extract_long import LongSpanExtractor
    ex = LongSpanExtractor.load(model, device="cpu")

    class Span(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.head = ex.enc, ex.head

        def forward(self, input_ids, attention_mask):
            return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)

    os.makedirs(out, exist_ok=True)
    m = Span().eval()
    ids = torch.ones((1, 64), dtype=torch.long)
    fp32 = f"{out}/model.onnx"
    torch.onnx.export(m, (ids, ids), fp32, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "length"}, "attention_mask": {0: "batch", 1: "length"},
                                    "logits": {0: "batch", 1: "length"}}, opset_version=17, dynamo=False)
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), f"{out}/model_fp16.onnx")
    if os.environ.get("INT8"):                  # dynamic int8 loses a lot on this model (51% same spans) — off by default
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32, f"{out}/model_int8.onnx", weight_type=QuantType.QInt8)
    cfg = json.load(open(os.path.join(ex_dir(model), "solvi_extract.json")))
    json.dump(cfg, open(f"{out}/solvi_extract.json", "w"), indent=1)
    for f in ("model.onnx", "model_fp16.onnx", "model_int8.onnx"):
        if os.path.exists(f"{out}/{f}"):
            print(f, round(os.path.getsize(f"{out}/{f}") / 1e6), "MB")
    return ex


def ex_dir(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


def spans_with(ex, logits_fn, text, desc):
    """LongSpanExtractor.predict with a pluggable logits function → (start, end, score)."""
    enc = ex._windows(desc, text)
    best = (0, 0, -1.0)
    for i in range(len(enc["input_ids"])):
        n = sum(enc["attention_mask"][i])
        lg = logits_fn(np.array([enc["input_ids"][i][:n]], dtype=np.int64), np.array([enc["attention_mask"][i][:n]], dtype=np.int64))[0]
        ps, pe = np.exp(lg[:, 0] - lg[:, 0].max()), np.exp(lg[:, 1] - lg[:, 1].max())
        ps, pe = ps / ps.sum(), pe / pe.sum()
        ctx = enc["ctx"][i]
        c0, c1 = ctx[0], ctx[-1] + 1
        sc = np.triu(np.outer(ps[c0:c1], pe[c0:c1])) - np.triu(np.outer(ps[c0:c1], pe[c0:c1]), ex.max_span)
        s, e = np.unravel_index(int(sc.argmax()), sc.shape)
        if sc[s, e] > best[2]:
            offs = enc["offset_mapping"][i]
            best = (offs[c0 + s][0], offs[c0 + e][1], float(sc[s, e]))
    return best


def check(ex, out, items):
    import onnxruntime as ort
    import torch
    ref = []
    for t, d in items:
        def torch_logits(ids, att):
            with torch.no_grad():
                return ex.head(ex.enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(att)).last_hidden_state).numpy()
        ref.append(spans_with(ex, torch_logits, t, d))
    for f in [f for f in ("model.onnx", "model_fp16.onnx", "model_int8.onnx") if os.path.exists(f"{out}/{f}")]:
        sess = ort.InferenceSession(f"{out}/{f}", providers=["CPUExecutionProvider"])
        got = [spans_with(ex, lambda ids, att: sess.run(None, {"input_ids": ids, "attention_mask": att})[0], t, d) for t, d in items]
        same = np.mean([g[:2] == r[:2] for g, r in zip(got, ref)])
        dscore = np.mean([abs(g[2] - r[2]) for g, r in zip(got, ref)])
        print(f"{f}: same span as PyTorch {same:.1%}, mean |score diff| {dscore:.3f} on {len(items)} fields")


if __name__ == "__main__":
    ex = export(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        items = [(r["text"], r["desc"]) for r in map(json.loads, open(sys.argv[3]))]
        check(ex, sys.argv[2], items)
