---
title: solvi documents
emoji: 📑
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
license: apache-2.0
short_description: Cited, typed answers from documents in your browser
models:
  - solvi-ai/extract-base
  - solvi-ai/extract-receipts
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# solvi documents

Typed answers from documents with [solvi](https://github.com/solvi-ai/solvi), computed **entirely in your browser**:

- **Documents never leave the device.** A ModernBERT field extractor
  ([solvi-ai/extract-base](https://huggingface.co/solvi-ai/extract-base), ONNX fp16) runs in a Web Worker with
  onnxruntime-web (WebGPU, or WebAssembly on the CPU). The decision logic (solvi) runs in Python compiled to WebAssembly
  (Pyodide).
- **Fields are defined by a plain-English description.** No training, no labels. Add a field and it is extracted at once.
- **Every answer cites the exact span**: the field is highlighted in the document at its character offsets.
- **Rules and hard checks decide**, in plain Python you can edit on the page.
- **Built on solvi 1.0**: the page installs `solvi==1.0.0` from PyPI into its Python.
- **The system abstains instead of guessing**: missing field, failed parse or an answer outside the options gives no answer.
- **The trace re-verifies**: every run writes a hash-chained trace and replays it; "Tamper with a copy" shows the replay
  catching an edited fact.

## Use-case library

Receipt expense check, supplier invoice (IBAN checksum and due date as hard checks), services contract review, NDA key
terms, data processing agreement (breach notice within 72 hours), residential lease (deposit cap), job offer letter,
insurance claim vs policy, bill of lading, support e-mail triage, and a blank case for your own document. Every case has
synthetic sample documents and a "paste your own" box.

A use case is one file in `usecases/` (fields as `[name, description]`, documents, rule code). To add one, copy
`usecases/receipt.js`, change it and list it in `usecases/index.js`.

## How it works

1. **Tokenizer** (`tokenizer.js`): ModernBERT byte-level BPE reimplemented in JavaScript so every token has character
   offsets (Python code-point offsets for solvi quotes, UTF-16 offsets for highlighting). Identical ids and offsets to
   Hugging Face `tokenizers` on all sample documents and descriptions.
2. **Extractor** (`extractor.worker.js`): reproduces `solvi.core.extract.LongSpanExtractor.predict`: windows
   `[CLS] description [SEP] chunk [SEP]` (1024 tokens, stride 128), softmax of start/end logits per window, best span
   `start ≤ end < start + 256` by `p_start · p_end`, best over windows, present if the score reaches the threshold from
   `solvi_extract.json`. The model file (790 MB) is cached with the Cache API. WebGPU is used when the GPU supports
   16-bit float shaders (`shader-f16`); otherwise the model runs on the CPU (multi-threaded when the page is
   cross-origin isolated, which the headers above enable).
3. **Decisions** (`solvi_docs.py` in Pyodide, `solvi==1.0.0` from PyPI, pinned twice in `app.js`): each field becomes an `@cat.extract` part returning a
   `Quote(value, start, end, confidence)`; the use case's rule code adds `@cat.fn`, `@cat.check` and `@cat.rule` parts;
   `System(cat, QUESTIONS).ask({"doc": text, "today": ...})` answers, and the trace is replayed.

## Parity test

`tests/parity.html` runs the browser extractor on reference fields and compares spans with the PyTorch model
(`tests/parity_ref.json`, produced by `tests/make_ref.py`). URL options: `?set=quick|all|long`, `?backend=wasm`,
`?repo=<url prefix>`, `?onnx=<file>`.

## Honest note

`extract-base` is a general starting point (contracts, receipts, Wikipedia-style questions). Fields far from its training
can be missed or cited at the wrong place, which the highlight makes visible. For production, label 25–100 of your
documents and fine-tune (see the model card). Speed: on a laptop GPU (WebGPU) about 0.3 s per field on a receipt and
about 1 s per 1 000-token window; on the CPU (8 threads) about 2 s per receipt field and 6–9 s per window.

All sample documents are synthetic.
