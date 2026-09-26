# Changelog

## 0.3.0 — 2026-09-27

- New answer types: `Answer.ordinal` (ordered levels; a learned head answers with the median) and `Answer.multi` (a subset;
  learned per option with fit / fit_fast, updated by teach). Options may carry descriptions (`{option: description}`).
- Constraints between answers (`@cat.constraint`) with joint decoding: contradictory learned answers are replaced by the most
  probable combination that satisfies every constraint; `Response.feasible` and `Response.violations` report the result.
- Example 11: a content guard with multi-label and ordinal answers tied by constraints.

## 0.2.2 — 2026-09-27

- Stronger `Trace.replay`: checks that the chain starts from the hash of the recorded input, flags inputs missing from the trace
  (a deleted step), re-runs steps recorded as failed (a faked error is caught), and with `replay(catalog, flow)` checks that every
  planned step was recorded or skipped at run time. Documented limit: a trace rebuilt honestly from a different input is
  consistent — compare `init_hash` with a receipt published elsewhere.
- Arcade: minesweeper, 20 questions, Mafia detective, hack the trace, bot arena.

## 0.2.1 — 2026-09-26

- A question's flow no longer depends on which other questions are asked (checks on computed facts are planned per question).
- A hard check with `then` governs only the questions listed there (and those naming it as a checkpoint); for others it is an
  ordinary failed check. Early exit follows the same rule. When several hard checks fail, the first declared in the catalog decides.
- Learned rule lists are deterministic (ties broken in sorted order).
- Gallery: twelve decision tasks with scenarios, runners and comparisons; synced into the browser playground (`tools/sync_gallery.py`).

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
