"""What a question reads: the given facts its flow needs (`question_inputs`), their types, the pydantic model and JSON
schema of its input state (`input_model`, `input_schema` — what `solvi serve` publishes) and the questions as entry points
for a text (`entry_points`, behind System.entry_points and solvi.core.textin.TextIn). Moved out of solvi.serve in 0.9, which
re-exports them, so that reading a text (solvi.core.textin) does not import the HTTP server."""
from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, create_model

from .core.textin import EntryField, EntryPoint


def question_inputs(system, name):
    """The given facts a question's flow reads → {"properties": [fact], "required": [fact]}. Planned with every given fact
    of the catalog present; a fact is required when the question cannot be answered without it (the strategist leaves it
    unresolved), so the inputs of alternative producers are optional. Planned by the system's own strategist
    (System(strategist=)), as `ask` plans."""
    from .strategist import PlanError, given_facts
    cat, q = system.catalog, system.questions[name]
    given = given_facts(cat, system.questions.values())
    try:
        flow = system._plan([q], given)
    except PlanError:
        return {"properties": [], "required": []}
    used = {x for s in flow.steps for x in s.part.inputs if x in given}
    used |= set(q.uses or ()) & given
    head = system.heads.get(name)
    if head is not None:
        used |= set(head.features) & given
    required = []
    for f in sorted(used):
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
    m = system.input_model
    if m is not None and fact in m.model_fields:
        return m.model_fields[fact].annotation, m.model_fields[fact].description
    types = []
    for t in readers.values():
        if t not in types:
            types.append(t)
    return (types[0] if len(types) == 1 else Any), None


def _schema_ok(t):
    from pydantic import TypeAdapter
    try:
        TypeAdapter(t).json_schema()
        return True
    except Exception:  # noqa: BLE001 — a type with no JSON schema (an arbitrary class): documented as any value
        return False


def input_model(system, name):
    """A pydantic model of the input state a question reads (extra keys allowed) — for the schema; solvi validates."""
    from .core.types import type_name
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
