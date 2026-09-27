# Changelog

## 0.4.1 — 2026-09-27 — clearer audits

- The audit of a learned part now lists what the head reads (`reads …`) and which requested features it ignored and why
  (`ignored …`, e.g. a dict-valued fact); `fit_fast` warns when it drops an explicitly requested feature.
- A rule that returns `None` on purpose is reported as its own safeguard, `rule_abstained` ("rule abstained"), instead of
  "outside the options"; `System.stats` counts it separately.
- Gallery: 03 shows a surface-only head (constraints repair it, 8/10) next to one reading a computed risk score (10/10);
  the gallery audit summary no longer double-counts safeguard events.

## 0.4.0 — 2026-09-27 — grounded decisions

Fuzzy proposes, deterministic decides, everything is in the trace. Every fact now carries its provenance (given, computed,
quoted, decided, learned, proposed); model outputs are grounded or rejected; `Response.audit()` shows what an answer rests on;
decisions with a model (`solvi.decide`) plug into the same safeguards. The pretrained solvi-decide weights are not published
yet (training data licensing is being cleaned); `solvi.decide` loads any checkpoint in that format.

- Decisions with a model (`solvi.decide`): the solvi-decide cross-encoder ("[mode] task [opt] options … [SEP] text", one logit
  per option) as a catalog part.
  - `DecideModel.load(path_or_hf_id, device=None, backend="auto"|"torch"|"onnx")`, `fingerprint()`, `model_id`,
    `metadata()`; `score` / `decide` / `logits`, batched and cached. Any object with `logits(items)` can stand in.
  - `model.decision(name, task, text_fact=, options=, descriptions=, multi=)` → a function returning `Decision(value, probs)`:
    the value is one of the options by construction, provenance `decided`, the trace records the model with a fingerprint of
    the checkpoint and this decision's adaptation. `part.question(cat, ...)` makes it a question's answer. Catalog parts
    pick up a decision's options (`__solvi_options__`) as their closed set.
  - `adapt(unlabelled_texts)`: label-bias correction without labels (the mean logit per option is subtracted).
  - `fit(examples)`: few-shot shift per option + shared scale (L-BFGS) and a temperature on out-of-fold predictions;
    `teach(text, correct)` refits the shift at once (~1 ms). `System.teach` routes to it when a question's answer is a
    decision part (or a rule passing a decided fact on).
  - "other" / "none" among the options is an abstain threshold on the best real option's calibrated probability (fitted
    on labelled "other" examples, else from the checkpoint's metadata), not a label the model scores.
  - `save_adaptations` / `load_adaptations`.
  - New extra `solvi[onnx]` (onnxruntime, tokenizers, huggingface_hub): the decider without torch; `solvi[model]` runs it with
    torch.
- `solvi.calibration`: `reliability`, `ece`, `coverage_at`, `threshold_for`, `accuracy_at`, `summary`, `evaluate(system,
  question, examples)` — for any model or question.
- Example 13: support-email routing with a decision part, bias correction, 16 labelled examples, abstention, a constraint
  with a rule-based question, a hard check, the audit and teach (the real decider with `SOLVI_DECIDE_MODEL`, a stand-in
  otherwise).

- Online learning of the check order / producer choice is on only when a learned policy uses it (`learn=None` default); it refits inside `ask` and caused rare slow calls in long runs. Cost tracking stays on.

- Grounded decisions: fuzzy proposes, deterministic decides, everything is in the trace.
  - Provenance of every fact and answer: `given`, `computed`, `quoted`, `decided`, `learned`, `proposed` (`record.origin`,
    `result.provenance` / `.source`, `solvi.provenance`). Parts take `model=` and `provenance=` (`@cat.extract`, `@cat.fn`,
    `@cat.check`, `@cat.rule`); `@cat.fn(model=, options=)` returning `Decision(value, probs)` is a model decision.
    Extractor `field()` / `embedder()` functions carry their model. `computed_state` shows provenance and model.
  - Model identity in the trace: model-backed records store `{"type", "id", "fp"}`; extractors, `Head`, `FastHead`,
    `RuleList` have fingerprints. Answer-head decisions are trace records (`kind="head"`). `Trace.replay(catalog_or_system,
    trust_models=False)` re-runs a model when it is the same and deterministic, otherwise verifies the recorded output is
    grounded, and reports "model changed since this decision"; `rep["models"]` gives a verdict per model step.
  - Grounding: a model's quote must be literally `doc[start:end]` (strings up to whitespace, numbers as written;
    `exact=` per part); a quote outside its text is now rejected (the fact is missing) instead of being used with an error;
    `min_confidence` and `validate` apply to every part, not only alternative producers; `Question(min_confidence=)`
    abstains on low-confidence answers.
  - `Response.audit(question=None)`: what each answer rests on, the safeguards that fired, and the deterministic share of
    its support; `show` prints a compact version. `System.stats` / `safeguard_report()`: lifetime counts of model outputs,
    grounding rejections, answers outside options, low-confidence abstentions, hard-check decisions, constraint repairs,
    validator rejections and fallbacks.
  - Hashes: plain records hash exactly as before; model-backed and learned-rule records (and head records) are new.
  - Example 12: one catalog with and without models, a hallucination caught, a model changed since the decision.
- The learned strategist. `System.costs`: a moving average of each part's run time. `System(order="learned")` /
  `system.learn_order(examples)`: hard checks run one at a time, most expected saving first (P(fail) from a per-check online
  model × cost saved ÷ cost), with early exit; answers are identical to the default order (earlier-declared governing checks
  are always evaluated before a later one decides). `res.trace.explain_order()` and `show` explain the order.
- Several producers of one fact: `@cat.fn(provides="total", cost=, validate=)`, `@cat.extract(provides=..., min_confidence=)`,
  `@cat.features("total")`. A fallback chain in declaration order, or `System(producers="learned")` — a policy that orders
  producers per input by P(accepted), P(agrees with the reference producer) and cost, learning from every run (with shadow
  runs for exploration). Records carry `producer` and `tried`; replay recomputes with the producer that was used.

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
