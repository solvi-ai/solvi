# Changelog

## 0.2.0 — 2026-09-26

- `System.fit_fast`: a closed-form ridge answer head trained in milliseconds (ridge strength by exact leave-one-out accuracy,
  pairwise features when there are few), and `System.teach` now updates it instantly (rank-one update, ~0.2 ms).
- `LongSpanExtractor.embed` / `embedder()`: a document embedding usable as a `fit_fast` feature.
- A learned head abstains when its features could not be computed.
- `tools/export_onnx.py`: export an extractor to ONNX (fp32 / fp16 / int8) with an agreement check.
- Spaces now run entirely in the browser (Gradio-Lite + Pyodide): playground and arcade.
- Examples 09 (strategy at scale) and 10 (learning in milliseconds); benchmarks `strategist_scale.py`, `fast_head.py`.

## 0.1.1 — 2026-09-26

- Runs in the browser (Pyodide): parallel execution falls back to one-by-one where threads are unavailable.
- Example 07 uses the published receipts model's strong fields (date, total, cash, change) and checks the change.

## 0.1.0 — 2026-09-26

First public version.

- Catalog of `@extract`, `@fn`, `@check` (soft and hard) and `@rule` parts; contracts come from function signatures.
- Typed questions (yes/no, choice); the strategist plans only the parts the asked questions need, plus checkpoints.
- Answers with confidence, a quote or a formula, and a hash-chained trace that `replay` re-verifies.
- Early exit (hard checks first; a failed one skips what only the settled questions needed) and parallel execution of
  independent steps (`workers=`), with a scheduling-independent trace.
- Learning: answer heads from labeled examples (`fit`), readable rule lists (`learn_rule`), Platt calibration (`calibrate`).
- ModernBERT extractors: one-pass multi-field (`MultiSpanExtractor`), per-field QA (`SpanExtractor`), long documents by field
  description with "no answer" (`LongSpanExtractor`), save/load and Hugging Face loading.
