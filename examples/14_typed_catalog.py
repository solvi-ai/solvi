"""Typed facts: a customs desk for parcels. The type hints of the catalog functions are the types of the facts; the request
is a pydantic model; the answer types come from the rules' return types (an Enum, a Literal, bool, a list of a Literal).

What it shows:
  1. a producer / consumer type mismatch is a catalog error when the function is registered, naming both functions;
  2. given facts and function outputs are validated / coerced at run time ("12.5" → 12.5, "2026-09-01" → a date);
  3. a value that fails its type is rejected like an ungrounded quote: the fact is missing, the next producer runs (fallback)
     or the answers that need it abstain; the audit and system.stats count it as "type rejected";
  4. a response is JSON (res.to_json()), loads back with the catalog restoring the typed values, and still replays;
     system.response_schema() is its JSON schema, with each answer as its closed set.

Run:  uv run python examples/14_typed_catalog.py"""

import re
from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel

from solvi import Catalog, Question, Response, System
from solvi.core.types import FactTypeError, type_name


class Item(BaseModel):
    description: str
    hs_code: str
    value_eur: float


class Parcel(BaseModel):
    """The request: one parcel with its label text and its manifest."""
    label: str
    items: list[Item]
    declared_weight_kg: Optional[float] = None
    destination: Literal["DE", "FR", "US", "CH"]
    shipped: date


class Duty(str, Enum):
    none = "none"
    standard = "standard"
    inspect = "inspect"


RESTRICTED = {"9303": "firearms", "3004": "medicine", "2402": "tobacco"}
cat = Catalog()


# the weight has two producers: the label text first, then the manifest. Each output must be a float.
@cat.fn(provides="weight_kg")
def weight_from_label(label: str) -> float:
    m = re.search(r"weight:\s*([^\n]+)", label, flags=re.I)
    return m.group(1).strip() if m else None              # "12.5" is coerced to 12.5; "12 kg" is not a number: rejected


@cat.fn(provides="weight_kg")
def weight_from_manifest(declared_weight_kg: Optional[float]) -> float:
    return declared_weight_kg


@cat.fn
def total_value(items: list[Item]) -> float:
    return round(sum(i.value_eur for i in items), 2)


@cat.fn
def restricted(items: list[Item]) -> list[str]:
    return sorted({RESTRICTED[i.hs_code[:4]] for i in items if i.hs_code[:4] in RESTRICTED})


@cat.check
def heavy(weight_kg: float) -> bool:
    return weight_kg > 30


@cat.rule("duty")
def duty(total_value: float, restricted: list[str], destination: str) -> Duty:
    if restricted:
        return Duty.inspect
    limit = 800 if destination == "US" else 150            # de minimis per destination
    return Duty.standard if total_value > limit else "none"   # a plain value is coerced to the Enum


@cat.rule("flags")
def flags(restricted: list[str], heavy: bool) -> list[Literal["restricted", "heavy", "late"]]:
    return (["restricted"] if restricted else []) + (["heavy"] if heavy else [])


@cat.rule("release")
def release(restricted: list[str], heavy: bool) -> bool:
    return not restricted and not heavy


QUESTIONS = [Question("duty", "Which customs duty applies?"),            # answer types from the rules' return types
             Question("flags", "What needs attention?"),
             Question("release", "Release without a manual check?")]

LABEL = "TO: J. Doe, Berlin\nWeight: 12.5\nRef: P-1182"
PARCEL = Parcel(label=LABEL, items=[Item(description="headphones", hs_code="851830", value_eur=189.0)],
                destination="DE", shipped=date(2026, 9, 1))


def line(res):
    return "  ".join(f"{q}={r.answer!r}" if r.status != "abstain" else f"{q}=— ({r.why[:60]})" for q, r in res.results.items())


if __name__ == "__main__":
    system = System(cat, QUESTIONS, input_model=Parcel)
    print("answer types from the rules:", {q: (s.answer.kind, s.answer.options) for q, s in system.questions.items()})
    print("fact types:", {f: type_name(t) for f, t in cat.types.items()})

    print("\n1. a mismatch is caught when the function is registered")
    try:
        @cat.fn
        def weight_band(weight_kg: str) -> str:
            return weight_kg
    except FactTypeError as e:
        print("   FactTypeError:", e)

    print("\n2. a typed request (a pydantic instance) and a plain dict validated against Parcel")
    res = system.ask(PARCEL)
    print("  ", line(res))
    res = system.ask({"label": LABEL, "items": [{"description": "cigars", "hs_code": "240210", "value_eur": "95"}],
                      "destination": "FR", "shipped": "2026-09-02"})
    print("  ", line(res), "| shipped =", repr(res.trace.init["shipped"]))

    print("\n3. rejected values: a label weight that is not a number, a manifest with a bad value")
    res = system.ask(PARCEL.model_copy(update={"label": "Weight: 12 kg", "declared_weight_kg": 41.0}))
    rec = next(r for r in res.trace.records if r.name == "weight_kg")
    print("  ", line(res))
    print(f"   weight_kg = {rec.value} by {rec.producer}; tried " + "; ".join(f"{n}: {w}" for n, w in rec.tried))
    res = system.ask({"label": LABEL, "items": [{"description": "lamp", "hs_code": "940510", "value_eur": "cheap"}],
                      "destination": "DE", "shipped": "2026-09-03"})
    print("  ", line(res))
    print("   " + res.audit("duty").safeguard_line() + ":", res.safeguards[0]["detail"].split(" is not ")[1])
    print("  ", system.safeguard_summary().splitlines()[2].strip(), "| lifetime:", system.stats["type_rejected"], "type rejected")

    print("\n4. JSON: the response, loaded back with the catalog, still replays")
    res = system.ask(PARCEL)
    text = res.to_json()
    back = Response.from_json(text, catalog=system)
    print(f"   {len(text)} characters; round trip identical: {back.to_json() == text}; "
          f"shipped restored as {type(back.trace.init['shipped']).__name__}; replay ok: {back.trace.replay(cat)['ok']}")
    schema = system.response_schema()
    duty_answer = schema["$defs"]["Result_duty"]["properties"]["answer"]["anyOf"][0]
    print("   response schema: duty answer", duty_answer)
