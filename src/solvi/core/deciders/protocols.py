"""The extension points of the parts tier: what a model, a decider, an adapter and a learned answer head implement to
plug into solvi. Each is a `typing.Protocol` (runtime_checkable: `isinstance(x, Scorer)` checks the methods are there,
not their signatures); `solvi.testing.conformance` checks the behaviour.

    from solvi.core import Scorer, Decider, Adapter, Head

Nothing here has to be subclassed: an object with these methods is one. The built-ins conform — `OnnxScorer` /
`TorchScorer` (Scorer), a decision part and a `Cascade` / `Vote` / `Route` (Decider), the LoRA adapter of
`solvi.experimental.lora` (Adapter), `FastHead` and the multi-label head of `System.fit` (Head)."""
from __future__ import annotations

from typing import Any, ContextManager, Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    """A model's raw scores, under `DecideModel(scorer)`.

    You implement: `logits(items)` → one entry per `Item` (task, options, descriptions, mode, text): an array [K] or
    [K, C] (a column per mode: [:, 0] choose-one, [:, 1] multi-label) or {"logits": array, "act": logit}. Optional:
    `logits_pass(passes)` (several questions in one forward pass), `fingerprint()` (your weights' identity; without
    it the model is "unversioned" and a changed model cannot be detected), `model_id`.

    You get for free (through `DecideModel`): questions from types (choice, multi, score, yes/no, spans), label-bias
    correction (`adapt`), few-shot fitting and `teach`, calibration and `act_guard` / `calibrate_for` with a stated
    promise, the model's identity in every trace and replay that re-runs it.

    Stability: stable (no break in 1.x)."""

    def logits(self, items: list) -> list: ...


@runtime_checkable
class Decider(Protocol):
    """A model as a catalog part: called with the facts it reads, it returns a `solvi.Decision(value, probs)`.

    You implement: `__call__(**facts)` → Decision (the value one of `options`, or `solvi.Unknown` when "not stated"
    is allowed; `escalate=` a reason when unsure); `options` → the values it can take (None for a number or a span);
    `fingerprint()` → its identity (weights, calibration, adaptation: anything that changes its answers).

    You get for free (register it with `cat.fn(decider)` or return its Decision from `@cat.rule`): the closed set (an
    answer outside the options is refused), constraints with joint decoding, guarantees on its signal
    (`System.guarantee`), the low-confidence safeguard, its identity recorded with every answer, replay, the audit.

    Stability: stable. The built-in decision part and `Cascade` / `Vote` / `Route` are stable to use, provisional to
    subclass."""

    @property
    def options(self) -> list | None: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def fingerprint(self) -> str: ...


@runtime_checkable
class Adapter(Protocol):
    """A per-question adaptation of a model's weights (LoRA and the like), in a decision part's adapter slot.

    You implement: `kind` (a short name; a calibration file names it, `solvi.core.deciders.ADAPTERS` maps it to the
    module whose `load(part, path, strict, expect=)` reads its file); `fingerprint()` (its content's hash);
    `using(scorer, active=True)` → a context in which the scorer scores with it (active=False: with no adapter);
    `save(path)` → path; `load(path)` (a classmethod) → the adapter written there.

    You get for free: its fingerprint in the model's and the part's fingerprints (so a decision records which
    adapter made it, and replay refuses another one), `extra["lora"]`-style records in the trace, the file written
    next to a calibration file and loaded back with it (checked against the recorded hash).

    Stability: stable (the protocol); the LoRA adapter itself is experimental (`solvi.experimental.lora`)."""

    kind: str

    def fingerprint(self) -> str: ...

    def using(self, scorer: Any, active: bool = True) -> ContextManager: ...

    def save(self, path: Any) -> Any: ...

    @classmethod
    def load(cls, path: Any) -> Adapter: ...


@runtime_checkable
class Head(Protocol):
    """A learned answer head: the probability of each option of a question from the facts the System computed.

    You implement: `options` (the answers it scores), `features` (the facts it reads, set by fit), `fit(rows, answers,
    features)` → self (rows: {fact: value} dicts), `predict(row)` → {option: probability}, `contributions(row)` →
    {fact: its share of the answer} (the answer's `why`), `teach(row, answer)` → the milliseconds it took (one
    correction, absorbed at once), and `fingerprint()` (its parameters: it must change when fit or teach changes
    the answers).

    You get for free: `System.fit(question, examples, head=lambda options: YourHead(options))` builds the rows from
    the flow, selects the facts (select=True) and installs it (a multi-label question gets one per option); the
    System answers from it — abstaining when a feature could not be computed or the probabilities are not finite —
    with its fingerprint recorded with every answer (a replay with another head is a mismatch); `System.teach`
    updates it; guarantees on its confidence (`System.guarantee`).

    Stability: stable. The built-in `FastHead` is stable to use, provisional to subclass."""

    options: list
    features: list

    def fit(self, rows: list, answers: list, features: list) -> Any: ...

    def predict(self, row: dict) -> dict: ...

    def contributions(self, row: dict) -> dict: ...

    def teach(self, row: dict, answer: Any) -> float: ...

    def fingerprint(self) -> str: ...


__all__ = ["Adapter", "Decider", "Head", "Scorer"]
