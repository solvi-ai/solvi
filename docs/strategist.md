# The model strategist and name matching (experimental)

See also: [the guide](guide.md#how-the-strategist-plans-a-flow) for the default strategist, and
[decide_format.md](decide_format.md) for the decider checkpoint contract.

solvi's default strategist plans a flow by exact names: a parameter's name is the fact it reads, and for a fact with several
producers (`provides=`) it needs the inputs of all of them — a fallback chain in declaration order. Two things it cannot do:

- **plan around a dead end** — one producer of a fact whose inputs are never given makes the whole fact unreachable, and
  **choose between interchangeable producers** by what they cost;
- **wire parts whose parameter names match no fact** — parts written by different teams, each with its own naming.

`solvi.strategy.ModelStrategist` and `solvi.aliases` add both. The principle is the same everywhere: **a model proposes,
code verifies, answers come only from verified plans**. A model error costs cost or coverage (the question abstains, or
code's own plan is used); it never becomes a silent wrong wiring — as far as the checks and your examples can tell.

Status: experimental in 0.5.0. The code strategist (`ModelStrategist()`, `producers="equivalent"`) is ready and needs no
model. **The segment model's and the name matcher's weights are not published with 0.5.0**: `ModelStrategist.load` and
`NameMatcher.load` read a checkpoint in the format below that you trained yourself (the research repository has the recipe); without one,
examples/17 uses stand-ins with the same interfaces. The research behind it (the typed decomposer, name matching, the segment model) and what was measured are summarised
[below](#what-was-measured).

## Planning: `ModelStrategist`

```python
from solvi import System
from solvi.strategy import ModelStrategist

System(cat, questions, strategist=ModelStrategist())                                   # 1. code only
System(cat, questions, strategist=ModelStrategist(producers="equivalent"))             # 2. cheapest verified plan
System(cat, questions, strategist=ModelStrategist.load("path/to/strategist-checkpoint"))   # 3. + the model
```

1. **`producers="declared"` (default).** The deterministic strategist's plan, except that producers whose inputs cannot be
   computed are dropped (they no longer make the fact unreachable). The remaining producers keep their declaration order as
   a run-time fallback chain. Wherever the deterministic strategist answers, the answers are the same (checked on all 12
   gallery catalogs: 333 of 333 question × case). No model is ever asked: there is nothing to choose.
2. **`producers="equivalent"`.** You declare that the producers of a fact are interchangeable (any accepted output is the
   same fact). Then *points* are computed by code: an exact 0/1 program (scipy's HiGHS; branch and bound as a fallback)
   picks one producer per needed fact — the cheapest valid plan by declared `cost=` (a part without one counts 1).
   **Mandatory milestones**: every hard check that governs a question in the plan the deterministic strategist would build
   stays in every plan, so a shortcut that skips the fact such a check reads cannot drop the check. The other producers
   stay as run-time fallbacks when their inputs are computed anyway.

Facts that can be derived from each other (`net` from `gross` and `gross` from `net`, each also given directly) work in
both modes: the plan computes each fact from what is given, and a producer that would read its own fact back — through
any other fact — is not kept as a fallback, since the flow could not run it.
3. **With a model** (`ModelStrategist(model, producers="equivalent")`, or `ModelStrategist.load(path)`, which implies
   `"equivalent"`): where declared costs do not settle the choice (a fact with ≥ 2 usable producers, not all with a declared
   cost), the fact becomes a **segment**: code narrows the catalog to the fact's producers and, up to 3 levels back, the
   producers of their inputs that are not yet available; the model proposes 1–4 parts in order. All segments of a plan go
   through the model in one batch. Every proposal is **checked** (`check_segment`: each part is a candidate, the last one
   provides the fact, every input is available or made earlier in the segment, types fit, nothing idle); the runner-up
   proposal is tried once; an accepted segment fixes those choices and the 0/1 program completes the rest; the whole plan is
   **validated** again (`validate`: inputs bound, order acyclic, producer and reader types fit, every mandatory check
   kept). A rejected segment falls back to code's choice for that fact; a plan that fails validation falls back to
   `fallback="code"` (code's plan), `"deterministic"` (solvi.strategist.plan) or `"abstain"`.

### Costs from measurements

With no `cost=` declared, interchangeable producers tie and the first declared wins. `System(cat, questions,
producers="equivalent", costs="measured")` (the same as `strategist=ModelStrategist(producers="equivalent")` plus
`costs="measured"`) plans with the run times solvi measures anyway (`system.costs`, a moving average in ms per part):

- **Warm-up.** A producer counts its measured time once it has run `min_samples` times (default 3); before that its
  declared `cost=`, or 0 ms when it declares none — so each undeclared producer gets chosen, and measured, in turn.
- **Smoothing and adapting.** The moving average weighs the newest run by `alpha` (CostBook's 0.3 unless set). When the
  producer in use slows down, its average rises above the others' and the next plans switch.
- **Recheck.** A producer not run for `recheck` asks (default 50) counts 0 ms for one plan, so a source that got faster
  again is noticed (`recheck=None`: never).
- **Freeze.** `system.freeze_costs()` fixes the planner's costs at what was measured (a producer that never ran: its
  declared cost, else 1) — the choice stops changing, measuring goes on; `system.unfreeze_costs()` resumes.

```python
from solvi.learned import MeasuredCosts
system = System(cat, questions, producers="equivalent", costs=MeasuredCosts(min_samples=5, recheck=100, alpha=0.2))
```

Every plan record then carries `extra["costs"]`: `{"mode": "measured" | "frozen", "facts": {fact: {"chosen",
"producers": {name: {"ms", "from", "runs"}}, "why"}}}` — per fact with several usable producers, the cost of each and where
it came from (`measured`, `declared`, `warm-up`, `recheck`, `frozen: measured` ...), e.g. `"why": "cheapest plan: rate_table
measured 0.012 ms ×5; rate_live measured 301.448 ms ×3"`. The choice is made on the costs of the whole plan (a producer
that reads an expensive fact pays for it too). Since measured costs depend on timing, so does this record's hash — it
says what was known when the plan was made; the rest of the trace is as always.

`strategist.last` is the report of the last plan: the choice per fact, the mandatory checks, per segment what code chose,
what the model proposed, whether it was accepted (and why) or rejected (and why), the fallback if any, and the time.

### In the trace

A planned flow adds one hashed record at the end of the trace: kind `plan`, name `plan:strategy`, value = the chosen
producer per fact and the mandatory checks; `extra` = the strategist, and per segment `proposed`, `code`, `by`
(model / code), `accepted` / `rejected`. Provenance is `proposed` when the model chose any part (then `model` holds its
`{type, id, fp}` — the fingerprint of the checkpoint files), `computed` otherwise. `trace.replay(system)` re-verifies the
record against the catalog (every chosen producer exists, provides its fact, and its inputs are given or chosen); a
tampered record breaks the hash chain. Facts whose producer group was narrowed replay with the producers that ran.

## Name matching: `solvi.aliases`

```python
from solvi.aliases import NameMatcher, propose, accept, apply, unresolved

unresolved(cat, init_keys)                        # {name: [(reader part, type)]}: names that are neither given nor a fact
m = NameMatcher.load("path/to/strategist-checkpoint/matcher")
props = propose(cat, questions, init_keys, m, init_types={"net": float})   # [Proposal(aliases={name: fact}, score)]
got = accept(cat, questions, props, examples, probes=states)                # 5 labelled examples + unlabelled probes
got = accept(cat, questions, props, examples[:3], probes=states, oracle=ask_person, active=(3, 7))   # the active mode
if got.aliases is not None:
    cat2 = apply(cat, got.aliases)                # parts rewired to read the facts; cat2.aliases lists what was accepted
```

- **Propose.** The matcher (all-MiniLM-L6-v2 + a character CNN over identifiers, trained on a broad family of name
  perturbations) scores every unresolved name against the given facts and the facts parts produce (type-compatible
  only); a beam search over the names builds joint proposals (a part never reads one fact twice, no cycles; a name may stay
  unresolved — an input nobody gives).
- **Accept** (deterministic; the model does not take part). The first proposal whose answers match every labelled example
  is taken; then its *neighbours* — the other proposals, one alias replaced by another of its candidates, two aliases
  exchanged — are run on the unlabelled probes: if one of them also fits the examples but answers differently somewhere, the
  examples cannot tell the two wirings apart and nothing is accepted ("ambiguous: ask a person"). In the **active mode**,
  while such neighbours remain, solvi picks the probe on which most of them disagree with the proposal and asks
  `oracle(state)` for its right answers (a person labels that one case), up to `m` questions. Aliases that change no answer
  are dropped (the name stays unresolved).
- **Apply.** Accepted aliases are applied by rewiring: every part that read an aliased name now reads the fact itself
  (its docstring lists the aliases), so planning, checks, quotes into given texts, the trace and replay behave exactly as
  for a catalog written with one naming.

What acceptance can and cannot guarantee: a wiring is accepted only if it answers every example right and no nearby wiring
that also does so answers any probe differently. A wrong alias whose correct alternative is *not* among the neighbours, or
that differs from the right one only on inputs that no example or probe exercises, can still pass — give examples and probes
that cover the cases that matter.

## Installation and backends

```bash
pip install "solvi[model]"     # torch + transformers: ModelStrategist.load(..., backend="torch"), NameMatcher (torch)
pip install "solvi[onnx]"      # onnxruntime + tokenizers: backend="onnx", no torch (also in the browser via onnxruntime-web)
```

`backend="auto"` (default) uses ONNX when the checkpoint has it and onnxruntime is installed, else torch. The code-only
strategist and `accept` / `apply` need neither (scipy only).

## Checkpoint format (`solvi_strategist v1`)

```
solvi_strategist.json    format and architecture (below)
config.json              the cell encoder's transformers config (ModernBERT, 2 layers, vocabulary 8193)
tokenizer.json           ModernBERT's tokenizer (`tokenizers`)
model.safetensors        the whole network (fp32), incl. the `remap` table 50368 → 8193 token ids
onnx/encoder.onnx        cell tokens → pooled states:  ids, mask [N, T] int64 → pooled [N, 768]
onnx/decoder.onnx        pooled cells + features → logits:  pooled [B, C, 768] float32, kind, vt, depth, ready [B, C] int64,
                         cmask [B, C] bool, fn_idx [B, F] int64, fn_mask [B, F] bool → type_logits [B, 8, 9], fn_logits [B, 8, F]
matcher/                 the name matcher (`solvi_matcher v1`): solvi_matcher.json, config.json (BERT), tokenizer.json,
                         model.safetensors ("text.*" BertModel, "char.*" character CNN, "logw"), onnx/matcher.onnx
                         (input_ids, attention_mask, token_type_ids [B, T], chars [B, 40], has_char [B, 1] → emb [B, 640])
README.md                model card
```

```json
{"format": "solvi_strategist v1", "name": "strategist-base", "n_nodes": 8, "cell_tok": 48,
 "arch": {"d": 512, "heads": 8, "layers": 4, "passes": 3, "vocab_full": 50368, "encoder_layers": 2},
 "training": {...}}
```

**Input: a segment as cells.** Each cell is a short text, encoded on its own (≤ `cell_tok` tokens, mean-pooled):

| cell | text | kind | value type |
|---|---|---|---|
| goal | `produce <fact>: <type> \| for: <question text>` | 0 GOAL | goal |
| available fact (only those a candidate reads) | `<fact>: <type> \| given` / `\| computed` | 1 FACT | num, bool, str, entity, list… |
| candidate part | `<kind> <name> -> <type> \| <docstring> \| reads <p>: <type>, … \| provides <fact> \| cost <c>` | 2 FUNC | func |

Per cell also: `depth` (0 goal / facts; 1 a producer of the segment's fact; 2 a producer of an input of one; …) and
`ready` (0 not a part; 1 all its inputs are available; 2 not) — both computed by code. **Output**: 8 query slots; slot
*i* → a node type (0 EMPTY, 1 STEP; the other 7 of the decomposer's plan language are unused) and a pointer over the candidate cells;
the proposal is the slots up to the first EMPTY. `solvi.strategy_model.seg_cells`, `batch_arrays`, `decode` implement
exactly this; the training side (in the research repository) uses the same functions.

The fingerprint recorded in traces is a hash of `solvi_strategist.json`, `config.json`, `tokenizer.json` and the weights file
the backend loads (`model.safetensors` or the two ONNX files): a retrained or re-exported checkpoint has another one.

## What was measured

The full report is in the research repository; criteria were registered before training.
Synthetic long catalogs (chains of stages as real solvi catalogs, up to 128 steps; half the stages with an extra producer: a
dead end, a costly or a cheap shortcut, plus stages with a cheap and a costly producer; every producer of a fact gives the
same value), 100 tasks per length bucket on training themes, 60 on held-out themes with held-out docstring phrasings:

| planner | goal reached, 65–128 steps | cost vs optimum (own / held-out themes) | CPU per plan, 65–128 |
|---|---|---|---|
| deterministic strategist | 0% (one dead end suffices; 23% even at 1–16) | — | 0.5 ms |
| `ModelStrategist()` code, costs not declared | 100% | 1.48 / 1.48 | 9 ms |
| code with costs from a 10-word keyword list of the training docstrings | 100% | 1.07 / 1.47 | 9 ms |
| `ModelStrategist.load(...)` — the segment model | 100% | 1.07 / 1.44 | 420 ms cold (onnx, 4 threads); 51 ms for a catalog it has seen; 250 ms int8 |
| code with declared costs (the model is never asked) | 100% | **1.000** | 9 ms |

- **Validity is code's:** every planner that goes through the verifier reached the goal in every task, with 0 wrong answers;
  the model's first proposal passed the segment check in 98.8% of segments (the runner-up or code's choice covered the rest).
- **What the model learned is the cost hints in docstrings** — about as well as a 10-word keyword list; on phrasings it never
  saw it barely beats code with unit costs. Declare `cost=` and code alone is exact.
- **Name matching** on the 12 gallery catalogs renamed in held-out styles (5 renderings each): the matcher links 79% of the
  names right; with 5 labelled cases acceptance covers 10% of the catalogs, with 3 labels + up to 6 targeted questions 38%;
  every accepted wiring answered every case like the original, but 17–22% of them differ from the true wiring in a name the
  cases did not exercise. On long synthetic catalogs (1–64 steps) coverage is 0–75% and 4–8% of the answers of accepted
  wirings were wrong. Treat accepted aliases as a suggestion to review, not as proof.

Status for 0.5.0: `ModelStrategist()` (code only, the dead-end-aware plan) and `producers="equivalent"` with declared costs are
ready; the segment model and `solvi.aliases` ship as **experimental**.
