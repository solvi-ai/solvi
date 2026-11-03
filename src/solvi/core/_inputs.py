"""What a question reads: the given facts its flow needs (`question_inputs`), their types, the pydantic model and JSON
schema of its input state (`input_model`, `input_schema` — what `solvi serve` publishes) and the questions as entry points
for a text (`entry_points`, behind System.entry_points and solvi.core.textin.TextIn). Moved out of solvi.serve in 0.9, which
re-exports them, so that reading a text (solvi.core.textin) does not import the HTTP server."""
from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, create_model

from .textin import EntryField, EntryPoint


def question_inputs(system, name):
    """The given facts a question's flow reads → {"properties": [fact], "required": [fact]}. Planned with every given fact
    of the catalog present; a fact is required when the question cannot be answered without it (the strategist leaves it
    unresolved), so the inputs of alternative producers are optional. Planned by the system's own strategist
    (System(strategist=)), as `ask` plans. Applicability declarations are always required initial facts,
    including facts no rule reads. Opaque callables need applicable_inputs (or explicit q.uses);
    their arbitrary Python reads cannot safely be inferred. Computed parts are not initial inputs."""
    from .plan.strategist import PlanError, given_facts
    cat, q = system.catalog, system.questions[name]
    contract = getattr(cat, 'contracts', {}).get(name)
    applicable_inputs = ()
    if contract is not None and contract.applicable is not None:
        applicable_inputs = contract.applicable_inputs
        if applicable_inputs is None:
            if q.uses is None:
                raise ValueError(f"question {name!r}: applicable callable has opaque reads; declare applicable_inputs "
                                 "(use () for no inputs), or explicitly set Question.uses")
            applicable_inputs = tuple(q.uses)
        computed = sorted(set(applicable_inputs) & set(cat.parts))
        if computed:
            raise ValueError(f"question {name!r}: applicable_inputs are initial given facts, not computed parts: "
                             f"{', '.join(computed)}")
    given = given_facts(cat, system.questions.values()) | set(applicable_inputs)
    if system._answer_refs():
        for candidate in getattr(cat, 'contracts', {}).values():
            given.update(candidate.applicable_inputs or ())
    try:
        flow = system._plan([q], given)
    except PlanError:
        flow = None
    used = {x for s in flow.steps for x in s.part.inputs if x in given} if flow is not None else set()
    if flow is not None and flow.answer_stages:
        for source in {n for stage in flow.answer_stages for n in stage} - {name}:
            source_contract = getattr(cat, 'contracts', {}).get(source)
            if source_contract is None or source_contract.applicable is None:
                continue
            declared = source_contract.applicable_inputs
            if declared is None:
                declared = system.questions[source].uses
                if declared is None:
                    raise ValueError(f'question {source!r}: applicable callable has opaque reads; declare applicable_inputs')
            computed = sorted(set(declared) & set(cat.parts))
            if computed:
                raise ValueError(f'question {source!r}: applicable_inputs are initial given facts, not computed parts: '
                                 + ', '.join(computed))
            applicable_inputs = tuple(set(applicable_inputs) | set(declared))
    used |= set(applicable_inputs)
    used |= set(q.uses or ()) & given
    head = system.heads.get(name)
    if head is not None:
        used |= set(head.features) & given
    # A guard may read a supplied fact which no catalog part reads. Internal
    # answer signals belong to the decider, never to the client's input state.
    guard = system.guards.get(name)
    signal = guard.signal if guard is not None else None
    external_signal = (isinstance(signal, str) and signal not in {"confidence", "probability", "act"}
                       and not signal.startswith("extra:") and signal not in cat.parts)
    if external_signal:
        used.add(signal)
    required = sorted(set(applicable_inputs) | ({signal} if external_signal else set()))
    for f in sorted(used - set(required)):
        try:
            if system._plan([q], given - {f}).unresolved.get(name):
                required.append(f)
        except PlanError:
            pass
    return {"properties": sorted(used), "required": required}


def _fact_type(system, fact):
    """The type of a given fact → (type or Any, description): System(input_model=...)'s field, else the one type its typed
    readers declare (Any when they disagree or nobody declares one)."""
    readers = system.catalog.readers.get(fact) or {}
    description = (getattr(system.catalog, "fact_descriptions", None) or {}).get(fact)
    m = system.input_model
    if m is not None and fact in m.model_fields:
        return m.model_fields[fact].annotation, m.model_fields[fact].description or description
    types = []
    for t in readers.values():
        if t not in types:
            types.append(t)
    return (types[0] if len(types) == 1 else Any), description


def _schema_ok(t):
    from pydantic import TypeAdapter
    try:
        TypeAdapter(t).json_schema()
        return True
    except Exception:  # noqa: BLE001 — a type with no JSON schema (an arbitrary class): documented as any value
        return False


def input_model(system, name):
    """A pydantic model of the input state a question reads (extra keys allowed) — for the schema; solvi validates."""
    from .types import type_name
    info = question_inputs(system, name)
    readers = system.catalog.readers
    fields = {}
    for i, f in enumerate(info["properties"]):
        t, desc = _fact_type(system, f)
        if t is not Any and not _schema_ok(t):
            desc, t = f"{type_name(t)} (no JSON schema)", Any
        who = sorted(readers.get(f) or ())
        desc = desc or (f"read by {', '.join(who)}" if who else None)
        default = ... if f in info["required"] else None
        fields[f"f{i}"] = (t, Field(default, alias=f, title=f, description=desc))
    return create_model(_camel(name) + "Input", __config__=ConfigDict(extra="allow", arbitrary_types_allowed=True),
                        __doc__=f"The input state of question {name!r}", **fields)


def input_schema(system, name, ref_template="#/$defs/{model}"):
    """The JSON schema of the input state a question reads."""
    return input_model(system, name).model_json_schema(ref_template=ref_template)


def _camel(name):
    s = "".join(w[:1].upper() + w[1:] for w in str(name).replace("-", "_").split("_") if w)
    return s if s.isidentifier() else "Question"


def entry_points(system, questions=None):
    """The questions of a system as entry points (see System.entry_points)."""
    names = questions
    out = []
    for q in system.questions.values():
        if names is not None and q.name not in names:
            continue
        info = question_inputs(system, q.name)
        fields = {}
        for f in info["properties"]:
            t, desc = _fact_type(system, f)
            fields[f] = EntryField(f, t, desc, f in info["required"])
        out.append(EntryPoint(q.name, q.text, fields, input_schema(system, q.name)))
    if names is not None:
        missing = [n for n in names if n not in system.questions]
        if missing:
            raise KeyError(f"no such question: {', '.join(missing)}")
    return out


__all__ = ["entry_points", "input_model", "input_schema", "question_inputs"]
