"""Export a LongSpanExtractor (encoder + span head) to ONNX for browsers and CPUs: fp32 and fp16 (identical spans on our tests; int8 with INT8=1 loses accuracy), plus an
agreement check against the PyTorch model on your own texts.

Usage:
  uv run --extra model --with onnx --with onnxscript --with onnxruntime \
      python tools/export_onnx.py <model dir or HF id> <out dir> [texts.jsonl with {"text", "desc"} lines for the check]

Graph: inputs input_ids, attention_mask (int64, [batch, length]) -> logits (float, [batch, length, 2]): start and end scores.
Windowing, "no answer" (position 0) and span decoding stay in the caller (see LongSpanExtractor.predict).

<out dir> becomes a checkpoint for the ONNX backend: onnx/model.onnx (fp32), onnx/model_fp16.onnx, and the model's
solvi_extract.json, tokenizer.json, tokenizer_config.json and config.json beside them — load it without torch with
LongSpanExtractor.load(out_dir, backend="onnx"), or copy onnx/ into the model's own folder (or Hugging Face repo) so
one checkpoint serves both backends. The check compares the PyTorch model with that ONNX backend: the largest logit
difference and the share of identical spans."""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np


def export(model, out):
    import torch
    from solvi.core.extract import LongSpanExtractor
    ex = LongSpanExtractor.load(model, device="cpu")

    class Span(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.head = ex.enc, ex.head

        def forward(self, input_ids, attention_mask):
            return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)

    os.makedirs(f"{out}/onnx", exist_ok=True)
    m = Span().eval()
    ids = torch.ones((1, 64), dtype=torch.long)
    fp32 = f"{out}/onnx/model.onnx"
    torch.onnx.export(m, (ids, ids), fp32, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "length"}, "attention_mask": {0: "batch", 1: "length"},
                                    "logits": {0: "batch", 1: "length"}}, opset_version=17, dynamo=False)
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), f"{out}/onnx/model_fp16.onnx")
    if os.environ.get("INT8"):                  # dynamic int8 loses a lot on this model (51% same spans) — off by default
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32, f"{out}/onnx/model_int8.onnx", weight_type=QuantType.QInt8)
    src = ex_dir(model)
    if os.path.abspath(src) != os.path.abspath(out):
        for f in ("solvi_extract.json", "tokenizer.json", "tokenizer_config.json", "config.json"):
            if os.path.exists(f"{src}/{f}"):
                shutil.copyfile(f"{src}/{f}", f"{out}/{f}")
    for f in EXPORTS:
        if os.path.exists(f"{out}/onnx/{f}"):
            print(f, round(os.path.getsize(f"{out}/onnx/{f}") / 1e6), "MB")
    return ex


EXPORTS = ("model.onnx", "model_fp16.onnx", "model_int8.onnx")


def ex_dir(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


def check(ex, out, items):
    """The PyTorch model against each export through LongSpanExtractor's ONNX backend: the largest logit difference over
    every window, and the share of identical spans (and of the same found / not-found decision at the default threshold)."""
    import torch
    from solvi.core.extract import LongSpanExtractor
    ref = [ex.predict(t, d) for t, d in items]
    for f in [f for f in EXPORTS if os.path.exists(f"{out}/onnx/{f}")]:
        ox = LongSpanExtractor.load(out, backend="onnx", onnx_file=f"{out}/onnx/{f}")
        got = [ox.predict(t, d) for t, d in items]
        diff = 0.0
        for t, d in items:
            enc = ex._windows(d, t)
            for i in range(len(enc["input_ids"])):
                n = sum(enc["attention_mask"][i])
                ids, att = [enc["input_ids"][i][:n]], [enc["attention_mask"][i][:n]]
                with torch.no_grad():
                    a = ex.head(ex.enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(att)).last_hidden_state).numpy()
                b = ox.sess.run(None, {"input_ids": np.array(ids, np.int64), "attention_mask": np.array(att, np.int64)})[0]
                diff = max(diff, float(np.abs(a - b).max()))
        same = np.mean([g[:2] == r[:2] for g, r in zip(got, ref)])
        found = np.mean([(g[2] >= ex.thr_default) == (r[2] >= ex.thr_default) for g, r in zip(got, ref)])
        dscore = np.mean([abs(g[2] - r[2]) for g, r in zip(got, ref)])
        print(f"{f}: same span as PyTorch {same:.1%}, same found/not-found {found:.1%}, mean |score diff| {dscore:.4f}, "
              f"max |logit diff| {diff:.4f} on {len(items)} fields")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python tools/export_onnx.py <model dir or HF id> <out dir> [texts.jsonl]")
    ex = export(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        items = [(r["text"], r["desc"]) for r in map(json.loads, open(sys.argv[3]))]
        check(ex, sys.argv[2], items)
