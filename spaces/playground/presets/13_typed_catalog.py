"""Typed facts (solvi 0.5): a customs desk. Type hints on the catalog functions are the types of the facts.

- Given facts are validated / coerced with pydantic before a typed function reads them: "12.5" → 12.5, "2026-09-01" → a date,
  the items → a list of `Item` models.
- `weight_kg` has two producers (a fallback chain): the label text first, then the manifest. The label says "12 kg", which
  is not a float: that output is REJECTED like an ungrounded quote ("type rejected" in the Audit), and the manifest's value
  is used instead.
- The answer types come from the rules' return types: `Literal[...]` is a closed set, `bool` is yes / no.
- A mismatch between a producer's return type and a reader's argument type is a catalog error when the function is
  registered (try: change `def total_value(...) -> float` to `-> str`).

Try: remove `declared_weight_kg` from init_state (nothing gives a valid weight: `heavy` cannot run), set a value_eur to
"abc", or change destination to "CH"."""
import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel

from solvi import Catalog, Question


class Item(BaseModel):
    description: str
    hs_code: str
    value_eur: float


RESTRICTED = {"9303": "firearms", "3004": "medicine", "2402": "tobacco"}
cat = Catalog()


@cat.fn(provides="weight_kg")
def weight_from_label(label: str) -> float:
    """the weight printed on the label (a string: must parse as a float)"""
    m = re.search(r"weight:\s*([^\n]+)", label, flags=re.I)
    return m.group(1).strip() if m else None


@cat.fn(provides="weight_kg")
def weight_from_manifest(declared_weight_kg: Optional[float]) -> float:
    """the weight declared in the manifest"""
    return declared_weight_kg


@cat.fn
def total_value(items: list[Item]) -> float:
    return round(sum(i.value_eur for i in items), 2)


@cat.fn
def restricted(items: list[Item]) -> list[str]:
    return sorted({RESTRICTED[i.hs_code[:4]] for i in items if i.hs_code[:4] in RESTRICTED})


@cat.fn
def days_in_transit(shipped: date, today: date) -> int:
    return (today - shipped).days


@cat.check
def heavy(weight_kg: float) -> bool:
    return weight_kg > 30


@cat.rule("duty")
def duty(total_value: float, restricted: list[str], destination: str) -> Literal["none", "standard", "inspect"]:
    if restricted:
        return "inspect"
    return "none" if total_value <= (1000 if destination == "CH" else 150) else "standard"


@cat.rule("courier")
def courier(heavy: bool, days_in_transit: int) -> bool:
    """a heavy parcel, or one that is late, goes by courier"""
    return heavy or days_in_transit > 10


QUESTIONS = [Question("duty", "Which customs duty applies?"), Question("courier", "Hand over to a courier?")]
