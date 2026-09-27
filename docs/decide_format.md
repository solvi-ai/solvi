# solvi-decide: checkpoint and input format

This is the contract between a decider checkpoint and solvi (`solvi.decide`): what the model reads, what it outputs, and the
fields of `solvi_decide.json` that say what a checkpoint can do. The training side builds its inputs with exactly these rules;
solvi reads any checkpoint that declares them. How to use a decider in a catalog: [guide](guide.md#types-questions-and-model-decisions); the strategist's own checkpoint
format is in [strategist.md](strategist.md).

## 1. Formats solvi reads

`solvi_decide.json["format"]` names the format; every capability field below may be declared over the format's defaults.

| `format` | Checkpoints | Defaults |
|---|---|---|
| `l14b_decider v1` | L14b – L14e (current releases) | modes `single`, `multi`; columns `single` 0, `multi` 1; text input; no act head; one question per pass |
| `l14f typed v1` | L14f | modes `single`, `multi`, `score`, `noul`; columns `single` / `score` / `noul` 0, `multi` 1, `act` 2; states `paths`, `tree`, `json`; act head in column 2; noul labels `true` / `false`; one question per pass unless `multi_question` is declared |
| `solvi_decide v2` | any other model | the `l14b_decider v1` defaults; everything else declared |
| `solvi_decide v2` with `"subformat": "l14g typed v2"` (also `format` `l14g typed v2` or `solvi_decide v3`) | L14g and later (the published decide-* previews) | the answer primitives, §9: the L14f defaults plus modes `rank`, `number`, `span`, "not stated" (column 3), a pointer (columns 4 / 5) |

`l14f typed v1.N`, `l14g typed v2.N` and `solvi_decide v2.N` / `v3.N` (a minor version) load the same way; any other format is refused
(`ValueError: unknown decider format`). A checkpoint without `format` is read as `l14b_decider v1`. An `l14b_decider v1`
checkpoint loads, scores and hashes exactly as in solvi 0.4 (same fingerprints, same trace hashes).

## 2. The checkpoint folder

```
config.json            the encoder's transformers config (ModernBERT)
tokenizer.json         the tokenizer (`tokenizers`); must contain the marker tokens
model.safetensors      encoder + head: head = Linear(h, h) → GELU → LayerNorm → Linear(h, C)  ("head.3.weight": [C, h])
onnx/model_fp16.onnx   optional; inputs input_ids, attention_mask [B, L] int64 → logits [B, L, C]
solvi_decide.json      format and capabilities (below)
```

## 3. `solvi_decide.json`

```json
{
  "format": "l14f typed v1",
  "base": "answerdotai/ModernBERT-base",
  "max_len": 512,
  "modes": ["single", "multi", "score", "noul"],
  "markers": {"option": "[unused0]", "single": "[unused1]", "multi": "[unused2]", "score": "[unused3]", "noul": "[unused4]"},
  "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "act": 2},
  "noul_labels": ["true", "false"],
  "state_serialization": ["paths", "tree", "json"],
  "multi_question": {"layout": "block", "max_questions": 6, "max_len": 1024, "window": 64},
  "temperature": {"choice": 1.08, "multi": 1.0, "score": 1.12, "noul": 0.95},
  "thresholds": {"other": 0.34, "multi": 0.44},
  "act": {
    "column": 2,
    "temperature": 1.0,
    "calibrator": {"features": ["confidence", "margin", "entropy", "act_logit", "n_options",
                                "kind=choice", "kind=multi", "kind=score", "kind=noul"],
                   "weights": [2.1, 1.3, -0.4, 0.8, -0.05, 0.0, -0.3, 0.1, 0.2], "bias": -1.9},
    "threshold": 0.5,
    "threshold_for_error": {"0.05": 0.83, "0.1": 0.66}
  }
}
```

| Field | Meaning | Default |
|---|---|---|
| `max_len` | tokens per sequence in the full layout (only the input is truncated) | 512 |
| `modes` | question kinds the model was trained on natively: `single` (one option), `multi` (every option that applies), `score` (ordered levels), `noul` (yes / no). A kind that is not native is asked as `single`: a score as a choice among its levels, a yes/no as a choice between "yes" and "no" | per format |
| `markers` | the mode marker of each kind and the option marker (tokens of `tokenizer.json`) | `[unused0]` option, `[unused1..4]` single, multi, score, noul |
| `columns` | which head output column holds each kind's option logits, and `act` the act logit | per format |
| `noul_labels` | the option labels of a native yes/no question, the "yes" one first | per format |
| `state_serialization` | the state serializations the model was trained on, preferred first; solvi uses the first of `paths`, `tree`, `json` it finds. `["text"]` (L14b–L14e): states are sent as `paths` lines, which the model reads as text. A v2 checkpoint that lists none of the three is refused | per format |
| `multi_question` | several questions per forward pass: `false`, `true`, a number (`max_questions`), or `{"layout", "max_questions", "max_len", "window"}` — layout `block` (§5), `max_questions` per pass (default 6), `max_len` of a block sequence (1024), `window` of the local attention layers (64, ±tokens) | `false` |
| `temperature` | a number (the single-choice temperature; `temperature_multi` then gives the multi-label one), or one per kind: `choice` (or `single`), `multi`, `score`, `noul` (missing kinds: the single-choice one) | L14b: 1.45; else 1.0 |
| `thresholds` | `other` (the "other" abstain threshold, §6), `multi` (a multi-label option applies at p ≥ it), `escalate_below` (a default confidence threshold for parts that set none). The top-level `other_threshold`, `multi_threshold`, `temperature_multi` of L14b–L14e are still read | 0.5, 0.5, none |
| `act` | the act / escalate head: `false` / absent — none; `{}` or `"act_head": true` — present with defaults. `column` (default `columns.act`), `temperature` (1.0), `calibrator` (optional, §6), `threshold` (0.5), `threshold_for_error` (`{"target error": threshold}`, used by `target_error=`) | per format |

Other keys (`base`, `training`, `licenses`, ...) are free: `model.metadata()` shows them. Every capability field, the
temperatures and thresholds are part of the checkpoint's fingerprint (a v2 or L14f checkpoint; `l14b_decider v1` hashes as
before).

## 4. One question: the sequence

A question's segment is its mode marker, the task, then an option marker before each option:

```
{mode} {task}{option} {label 1}{option} {label 2: description 2}...
[unused3] How urgent is this ticket?[unused0] low[unused0] medium[unused0] high[unused0] critical
```

An option with a description is `label: description`. Options by kind:

- `single`: the options in declaration order; an option named "other" / "none" (and variants) is left out when solvi treats
  it as an abstain threshold (§6), else it is an ordinary option;
- `multi`: the options in declaration order;
- `score`: the levels, lowest first (2–10);
- `noul`: two options, `noul_labels` (L14f: `true`, `false`), the "yes" one first, with the question's yes / no
  descriptions when it has them (e.g. `true: The sender expects a reply.`); asked as `single`, the labels are `yes`, `no`.

The **full layout** (one question per sequence, L14b–L14f): `[CLS] segment [SEP] input [SEP]`, full attention, only the
input truncated to `max_len`. The mode marker is at position 1.

## 5. Several questions: the block layout

With `multi_question.layout = "block"`: the input first, then one block per question:

```
[CLS] input [SEP] | segment 1 [SEP] | segment 2 [SEP] | ...
```

- the input is truncated to `max_len − Σ len(blocks) − 2` tokens (the questions are never cut; if fewer than 8 input tokens
  would remain, the questions go in smaller passes);
- position ids: the input `0 … Ls−1`; block j continues from `Ls` (`Ls, Ls+1, …`), so every block "stands right after the
  input";
- attention: the input sees only the input; block j sees the input and itself; blocks do not see each other; padding sees
  itself. The local (sliding) layers also require `|pos_i − pos_j| ≤ window`. In transformers this is
  `attention_mask={"full_attention": [B,1,L,L] bool, "sliding_attention": [B,1,L,L] bool}` with `position_ids`
  (sdpa attention); an ONNX export for this layout takes `input_ids`, `position_ids`, `full_attention_mask`,
  `sliding_attention_mask` ([B,1,L,L] bool) — without those inputs solvi scores one question per sequence;
- consequence: a question's logits do not depend on which other questions share its pass (verified: max |Δ| 9e-7 on the
  L14d weights), so solvi scores every question of a block checkpoint in the block layout — alone or together — and fit /
  teach / adapt see the same logits as the runtime. Which layout the temperatures and the act calibrator are fitted on must
  therefore be the block layout for a block checkpoint.

The mode marker of each block is its first token. (`"layout": "concat"`, all segments in the first segment of a full
sequence, exists for experiments only: there a question's answer depends on its neighbours.)

## 6. Outputs and what solvi does with them

The head gives `C` logits per position. solvi reads, per question: the column of its kind (`columns`) at each option marker
→ `z` [K]; and, with an act head, column `act.column` at the question's mode marker → the act logit. A native `noul` may
give one logit (the log-odds of "yes") instead of two.

Then, per question (task, options, descriptions, kind):

1. label-bias correction (`adapt`, optional): `z − bias`, bias = centered mean logit over unlabelled inputs;
2. scale and shift (`fit` / `teach`, optional; else scale = 1 / temperature of the kind): `s = (a·z + b) / τ`; b is free per
   option (choice, multi), `b = β₁·r + β₂·(r² − mean r²)` over the centered rank r ∈ [−1, 1] (score), one bias ±β/2 (noul);
3. probabilities: softmax over `s` (single, score, noul), a sigmoid per option (multi);
4. the value: argmax (single); options with p ≥ `thresholds.multi` (multi); the **median** level (score; `expected` — the
   probability-weighted level, in level units for numeric levels, else in rank units — and `median` are recorded); "yes" when
   p(yes) ≥ 0.5 (noul);
5. confidence: the top probability; multi-label: the least certain option's max(p, 1 − p); score: p(value);
6. "other" (single, when it is a threshold): chosen when the top real probability m < `thresholds.other`
   (or a threshold fitted on labelled "other" examples); p(other) = g / (1 + g), g = thr·(1 − m)/(1 − thr);
7. act probability: `sigmoid(act_logit / act.temperature)`, or with `act.calibrator`:
   `sigmoid(bias + Σ weights[i] · feature[i])` over these features of the calibrated decision:

   | feature | value |
   |---|---|
   | `confidence` | step 5 |
   | `margin` | top − second probability; multi-label: min over options of \|2p − 1\| |
   | `entropy` | Shannon entropy of the probabilities (nats); multi-label: the mean binary entropy |
   | `act_logit` | the raw act logit |
   | `n_options` | the number of scored options K |
   | `kind=choice`, `kind=multi`, `kind=score`, `kind=noul` | 1 for the question's kind, else 0 |

8. act or escalate: act probability < threshold (`act.threshold`, a part's `act_threshold`, or
   `act.threshold_for_error[target]` via `target_error=`) → "model escalated"; else confidence < `escalate_below` →
   "low confidence". Either rejects the decision (the answer abstains, a fallback producer may run).

## 7. The input: text or state

A text is sent as it is. A state (a dict, a list, a pydantic model, a dataclass) is first made JSON data, then serialized.
Reference implementations: `solvi.decide.state_text(obj, fmt)` and `serialize(state, fmt)` in
`exps_v2/experiments/l14f_format.py` (the L14f training code) — identical on JSON data (checked on 7 644 random states,
all three formats).

**To JSON data** (`solvi.decide.jsonable`; the training side starts from JSON already): a pydantic model → its
`model_dump()` (field order); a dataclass → its fields; `date` / `datetime` / `time` → `isoformat()`; an Enum → its value;
`Decimal`, `UUID` and other objects → `str()`; bytes → UTF-8 text; tuples → lists; sets → lists sorted by their JSON text;
numpy → Python numbers. Mapping keys keep their order.

**`paths`** (default): one line per leaf, `path: value`.

- path: keys joined by `.`; list items `[i]`; a key that does not match `[A-Za-z0-9_-]+` (a space, a dot, a bracket,
  non-ASCII, empty) is written `["key"]` (a JSON string, no leading dot); a leaf or empty container at the top has the path
  `.`;
- value: `null`, `true`, `false`; floats `repr(round(v, 6))` without trailing zeros and dot when `|v| < 1e15` (`25.5`, `2`,
  `0.333333`), else `repr(v)` (`1e+20`, `nan`); other values `str(v)` with new lines replaced by spaces (strings are not
  quoted); an empty dict `{}`, an empty list `[]`.

```
customer.name: Anna
customer.tier: enterprise
items[0].sku: A-17
items[0].qty: 2
["delivery note"]: leave at the door
tags: []
```

**`tree`**: YAML-like, two spaces per level; `key: value`, `key:` before a nested dict or list, list items `- value`
(a dict item: `- first key: value`, its other keys indented under it). **`json`**: `json.dumps(data, ensure_ascii=False,
separators=(", ", ": "))`.

A decision reading several facts serializes `{fact: value, ...}` in the order of its facts when any of them is a state;
several text facts are joined by new lines. A scalar fact (a number, a date) is sent as its value's text.

## 8. Checkpoints without a model: the scorer protocol

`DecideModel(scorer, meta)` accepts any object with:

- `logits(items)`: `items` are `solvi.decide.Item(task, options, descriptions, text, multi, kind)` (`item.mode` is the wire
  mode: `single`, `multi`, `score`, `noul`); return one array per item — `[K]` or `[K, C]` (the kind's column is taken) — or
  `{"logits": array, "act": logit}`;
- optionally `logits_pass(passes)`: `solvi.decide.Pass(text, items)` → per pass a list of the same per-question outputs
  (several questions in one forward pass; raise `ValueError` when they do not fit together);
- optionally `fingerprint()` (else the model is "unversioned") and `model_id`.

`meta` is the content of `solvi_decide.json`; a meta whose `format` is none of §1 (a stand-in) gets the `l14b_decider v1`
defaults and hashes as in solvi 0.4.

## 9. Answer primitives: `l14g typed v2` (proposed `solvi_decide v3`)

The contract for a decider that answers every answer primitive of solvi (see the
[guide](guide.md#answer-primitives-not-stated-evidence-spans-rankings-estimates)): "not stated", evidence quotes, spans,
rankings and numbers. It is the L14g training format — `exps_v2/experiments/l14g_format.py` writes it and is the reference
for the network side — and solvi 0.5 reads it. Where L14g left a choice open, solvi's choice is marked **(solvi)**.

A checkpoint is read with this contract when its `solvi_decide.json` has `"subformat": "l14g typed v2"` (L14g writes
`"format": "solvi_decide v2"` with it), or `"format": "l14g typed v2"` / `"solvi_decide v3"`. Only such checkpoints get the
new capability fields: `l14b_decider v1`, `l14f typed v1` and plain `solvi_decide v2` checkpoints parse, score and hash
exactly as before (same `model.caps`, temperatures and fingerprints).

### 9.1 `solvi_decide.json`

```json
{
  "format": "solvi_decide v2", "subformat": "l14g typed v2", "max_len": 512,
  "modes": ["single", "multi", "score", "noul", "rank", "number", "span"],
  "markers": {"option": "[unused0]", "single": "[unused1]", "multi": "[unused2]", "score": "[unused3]", "noul": "[unused4]",
              "rank": "[unused5]", "number": "[unused6]", "span": "[unused7]"},
  "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "rank": 0, "number": 0, "act": 2, "unknown": 3,
              "span_start": 4, "span_end": 5},
  "unknown": {"column": 3, "position": "mode", "label": "not stated",
              "joint": ["single", "score", "noul", "rank", "number"], "multi": "sigmoid", "span": "null_span"},
  "pointer": {"start": 4, "end": 5, "layouts": ["full"], "max_span_tokens": 40, "null": "mode",
              "evidence": {"threshold": 0.15, "max_spans": 3}},
  "number": {"interval": 0.8},
  "noul_labels": ["true", "false"], "state_serialization": ["paths", "tree", "json"],
  "multi_question": {"layout": "block", "max_questions": 6, "max_len": 1024, "window": 64, "pointer": false},
  "temperature": {"choice": 1.0, "multi": 1.0, "score": 1.0, "noul": 1.0, "rank": 1.0, "number": 1.0},
  "act": {"calibrator": {"features": ["confidence", "margin", "entropy", "act_logit", "n_options", "p_unknown",
                                      "kind=choice", "kind=multi", "kind=score", "kind=noul",
                                      "kind=rank", "kind=number", "kind=span"],
                         "weights": [2.0, 1.2, -0.4, 0.8, -0.05, -1.5, 0.0, -0.3, 0.1, 0.2, 0.0, 0.0, 0.0], "bias": -1.8},
          "threshold": 0.5}
}
```

| Field | Meaning | Default (`l14g typed v2`) |
|---|---|---|
| `modes` | adds `rank` (order the options), `number` (a number over bins), `span` (a piece of the input) | all seven |
| `markers` | `rank` `[unused5]`, `number` `[unused6]`, `span` `[unused7]` | as shown |
| `columns` | `rank`, `number`: the option logits (column 0); `unknown` 3; `span_start` 4, `span_end` 5 | as shown |
| `unknown` | the "not stated" logit u: `column` at the mode marker; `joint` — the kinds whose options compete with it in one softmax; `multi: "sigmoid"` — p = σ(u); `span: "null_span"` — the pointer's null span; `threshold` **(solvi)** — see 9.3 (0.5). `false` / absent: none | as shown |
| `pointer` | `start` / `end` columns over the input's tokens; `layouts` where it exists (`["full"]`: not in the block layout, where the input does not see the question); `max_span_tokens` (40); `null` (`"mode"`: the null span is scored at the mode marker); `evidence`: `threshold` (0.15), `max_spans` (3) | as shown |
| `number` | `interval`: the coverage L14g evaluates with (informational; each question sets its own, default 0.8) | 0.8 |
| `multi_question.pointer` | whether the pointer works in a shared pass (L14g: no) | false |
| `temperature` | adds `rank`, `number` (default: the `score` one), `span` (1.0; divides the pointer's start / end scores and the null span's before the softmax, for spans and evidence) | per kind |
| `act.calibrator.features` | may also use `p_unknown` (the decision's p("not stated"), 0 when not asked) and `kind=rank`, `kind=number`, `kind=span` | — |

`model.has_unknown`, `model.has_pointer` say what a loaded checkpoint can do. A decision that asks for what the checkpoint
cannot give raises when it is made (`model.decision(...)`): a span or evidence without a pointer, "not stated" without
`unknown`. `rank` and `number` work with older checkpoints too **(solvi)**: a rank is asked as a single choice (the order
is by probability), a number as a score over its bins (or a single choice).

### 9.2 Segments and layouts

- `rank`: `[unused5] task[opt] option 1[opt] option 2 …` — like `single`;
- `number`: `[unused6] task[opt] bin 1[opt] bin 2 …` — the bins lowest first, labelled by `bin_labels(edges, integer,
  unit)`: edges e0 < … < e_last give len(edges) + 1 bins (−∞, e0), [e0, e1), …, [e_last, ∞) labelled `less than e0`, then
  `a` (a one-wide integer bin), `a–(b−1)` (integer edges) or `a to b`, then `e_last or more`, the unit after a space
  (`solvi.core.bin_labels` is the same function);
- `span`: `[unused7] task` — no options.

The full layout carries every kind, the pointer and "not stated". The block layout (§5) carries every kind but `span`, and
no pointer: **(solvi)** a question that needs the pointer (a span, or a decision with `evidence=`) is scored alone, in the
full layout, even when the checkpoint has `multi_question`; the strategist does not put it in a shared pass.

### 9.3 Outputs and what solvi does with them

Per question, besides the option logits z (and act): u = column `unknown.column` at the mode marker; with the pointer,
`start_i`, `end_i` over the input's tokens and `start_m`, `end_m` at the mode marker.

**"Not stated".** For `joint` kinds, p = softmax([s₁ … s_K, u / T]) where s are the calibrated option scores (§6, steps 1–2)
and T the kind's temperature; p(not stated) = p_{K+1}. For `multi`, p(not stated) = σ(u). A question that does not allow
"not stated" drops u (softmax over the options only). The answer **(solvi)**:

| kind | "not stated" when | otherwise | confidence |
|---|---|---|---|
| single, noul | it is the most probable of the options and "not stated" | the most probable option | p(answer) (joint) |
| score, rank, number | p(not stated) ≥ `unknown.threshold` (0.5) | the median level / the order / the median bin of p(· \| stated) | see below, × p(stated) |
| multi | σ(u) ≥ threshold | the options at p ≥ `thresholds.multi` | min(least certain option's max(p, 1 − p), 1 − σ(u)) |
| span | p(null span) ≥ p(best span) | the best span | p(span) |

`decision.probs` has the options' joint probabilities and `solvi.Unknown` (written `"<not stated>"` in JSON and in the trace
hash) with p(not stated); the value of a "not stated" decision is `solvi.Unknown`.

**Pointer.** A span's score is S(i, j) = start_i + end_j over input tokens i ≤ j < i + `max_span_tokens`; the null span's is
N = start_m + end_m; p = softmax over {N / T} ∪ {S(i, j) / T}, T = `temperature.span` (L14g `span_dist` as its eval applies it). A span's text is the input from token i's start to
token j's end (tokenizer offsets), without surrounding whitespace — so it is literally in the input. A span question takes
the best span (without "not stated": p renormalized without the null span); evidence takes greedily up to `max_spans`
non-overlapping spans with p ≥ `evidence.threshold`, none when the null span is at least as probable as the best span
(L14g `evidence`). **(solvi)** Quotes are bound to the fact the decision read: a span or evidence needs a decision that reads
**one given text fact** (the pointer's offsets are into that text; a state serialization is not a text in the input), else a
span escalates and evidence is dropped. solvi re-checks every quote literally at its offsets (safeguard "grounding
rejected" if not) and replays it.

**rank.** p = softmax over the options (column 0, joint with u); the value is the options in decreasing p(· | stated),
the top k; confidence = the Plackett–Luce probability of that top k in that order × p(stated). **number.** p over the bins;
the value is the middle of the median bin (an open bin: its edge), the interval the bins from the (1 − c)/2 to the
(1 + c)/2 cumulative probability (L14g `number_summary`, c = the question's coverage), confidence = their mass × p(stated),
`decision.extra` = `{"interval", "coverage"}`. **span**: `Decision(Quote(text, start, end, fact), confidence=p)`, the answer
coerced to the question's type (`Span[float]`) when it is resolved — a failure is "type rejected".

**Act.** As §6 step 7–8, the calibrator's features computed on the decision above (`p_unknown` = p(not stated) in
`decision.probs`, 0 when the question does not allow it).

### 9.4 The scorer protocol (stand-ins, other backends)

A per-question output (§8) may also carry `"unknown": u` and, for an `Item` with `pointer=True`,
`"pointer": {"start": [T], "end": [T], "null": [start_m, end_m], "offsets": [(char start, char end)] per input token}` —
offsets into `item.text`. `Item.mode` is `rank`, `number` or `span` for the new kinds (`span` items have no options). The
ONNX / torch backends read these columns when the checkpoint declares them.

### 9.5 Adaptation

`adapt` / `fit` / `teach` work for `number` (as a score over the bins: a tilt and a spread; a label may be a number — its
bin) and, with "not stated", on the option logits only: examples labelled `Unknown` are left out and u is not adapted.
`rank` and `span` are not adapted from labels.
