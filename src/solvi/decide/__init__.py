"""Decisions with a model: a cross-encoder that answers typed questions about a text or a state ("decider").

Types declare questions, the model proposes, checks decide. A question's kind comes from its type (solvi.typed):

  choice  Literal[...] / an Enum          one option (softmax); "other" / "none" may be an abstain threshold
  multi   list[Literal[...]]              every option that applies (a sigmoid per option)
  score   Scale[...] (2–10 levels)        ordered levels (softmax; the value is the median, the expected level is recorded)
  noul    bool / Literal["yes", "no"]     yes or no

The decider (solvi-decide, ModernBERT) reads one sequence per question — or, when the checkpoint says it can, several
questions about the same input in one sequence:

    [mode] task[opt] option 1[opt] option 2 ... [mode] task 2[opt] ... [SEP] input

and gives one logit per option marker (and, with an act head, one "act" logit per question). The input is a text, or a
state (a dict, a list, a pydantic model) serialized by `state_text` — the one serialization the training side uses too
(see docs/decide_format.md, which also defines the checkpoint's capability fields).

In solvi a decider is a catalog part like any other: `model.decision(...)` returns a function that returns
`solvi.Decision(value, probs)`. The value is one of the declared options by construction, the part's provenance is `decided`
and the model's identity (weights, calibration and adaptation of this part) is in the trace, so the closed set,
`min_confidence`, constraints with joint decoding, hard checks, the audit and the stats apply unchanged. A decision the model
escalates (its act head, or a calibrated confidence below the part's `escalate_below`) is rejected like an unsure one: the
fact is missing, the answer abstains, and the audit and stats say "model escalated" / "low confidence".

On top of the raw logits, per question (task, options, kind):
  - label-bias correction without labels (`adapt`): the mean logit of each option over unlabelled inputs of the domain is
    subtracted before the softmax (the decider likes some labels regardless of the text; +7 points in research);
  - few-shot adaptation "S" (`fit`, `teach`): a shift and a shared scale fitted on k labelled examples (L-BFGS), with a
    temperature fitted on out-of-fold predictions, so confidences are calibrated; `teach` updates the shift at once. The
    shift is per option (choice, multi), a tilt / spread over the levels (score) or one yes−no bias (noul);
  - "other" / "none" as an abstain threshold: such an option is not scored by the model; it is chosen when the best real
    option's calibrated probability is below a threshold (fitted on labelled examples that include it, else the default);
  - `calibrate_for(examples, max_error=0.05)`: the escalation threshold for a target error rate.

Backends: "torch" (`solvi[model]`) or "onnx" (`solvi[onnx]`: onnxruntime + tokenizers, no torch), or any object with
`logits(items)` (and optionally `logits_pass(passes)`) — tests, other models: see DecideModel."""
from .kinds import ACT_FEATURES, ACT_FEATURES_V3, DEFAULT_T, FORMAT, KINDS, KINDS_V3, LEGACY_FORMAT, MANY, MARKERS, NULL_SOURCE, ONE, OPT, OTHER_NAMES, TYPED2_FORMAT, TYPED_FORMAT, V3_MARKERS, WIRE, _KIND, _OPTION_KINDS, _kind, _spec_names, _unused_options  # noqa: F401
from .state import SERIALIZATIONS, _KEY_OK, _is_text, _jkey, _paths, _pkey, _scalar, _single, _text, _tree, jsonable, state_text  # noqa: F401
from .wire import Item, Logits, Pass, _respan, _typed_span, decode_pointer, pass_prompt, pointer_evidence, prompt  # noqa: F401
from .capabilities import _DEFAULTS, _max_len_long, _multi_question, _pointer_caps, _unknown_caps, _v3, capabilities  # noqa: F401
from .backends import BlockUnsupported, LONG_CPU_TOKENS, LongInputWarning, OnnxScorer, SECTION_TOKENS, TorchScorer, _Encoder, _NetScorer, _batches, _file_fingerprint, _onnx_file, block_masks, need  # noqa: F401
from .adapt import Adaptation, _Spec, _basis, _fit_shift, _fit_temperature, _from_type, _given, _is_type, _sig, _softmax, _usable, lora_key  # noqa: F401
from .gate import Facts, GroupBy, SEPARATION_MIN, _group_guard, _group_info, _group_promise, _shown, _threshold, _vkey, act_features, confidence_source, decision_of, group_name, group_record, guard_promise, no_separation, one_source  # noqa: F401
from .part import DecisionPart, plan_batches  # noqa: F401
from .model import DecideModel, _json_default  # noqa: F401

__all__ = ["act_features", "Adaptation", "block_masks", "BlockUnsupported", "capabilities", "DecideModel", "decision_of", "DecisionPart", "decode_pointer", "Facts", "group_name", "group_record", "GroupBy", "guard_promise", "Item", "jsonable", "KINDS", "Logits", "LongInputWarning", "lora_key", "no_separation", "one_source", "OnnxScorer", "Pass", "pass_prompt", "plan_batches", "pointer_evidence", "prompt", "state_text", "TorchScorer"]
