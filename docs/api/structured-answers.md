# Structured rule answers

`Answer.structured(type=dict)` accepts a rule's JSON object or array as a normal question answer, with its trace,
source and confidence. `Answer.from_type` also infers this kind for dictionary/list annotations, importable
Pydantic models, TypedDicts and dataclasses. Collections of Literal/Enum options retain their existing `multi` kind.
The schema must be serializable: custom classes must be importable at module scope. Local classes and unsupported
type expressions are refused at declaration. Class schemas are pinned in serialized types; loading a changed schema
fails. Loading an importable class uses the same trusted-code boundary as typed spans.

```python
from solvi import Answer, Catalog, Claim, Question, System

cat = Catalog()

@cat.rule("quote")
def quote(doc):
    return Claim({"price": 12, "currency": "EUR"}, extra={"fields": {
        "/price": {"why": "published tariff", "confidence": .9, "evidence": ["12"]},
        "/currency": {"why": "contract currency", "evidence": ["EUR"]},
    }})

system = System(cat, [Question("quote", "Quote?", Answer.structured())])
result = system.ask({"doc": "12 EUR"})["quote"]
assert result.answer == {"price": 12, "currency": "EUR"}
assert result.confidence == .9
assert result.extra["fields"]["/price"]["why"] == "published tariff"
```

Field explanations are optional, explicit producer declarations in `Claim.extra['fields']`. Each key is a JSON Pointer
(`/items/0/price`; `~1` escapes `/`, `~0` escapes `~`). Every supplied path must exist. Every supplied explanation needs
a nonempty `why`, finite confidence in [0,1] (default 1), and optional evidence strings or Quote-shaped dictionaries.
Evidence must occur literally in an initial source text; invalid metadata or evidence abstains with `type_rejected`.
No explanation is fabricated for omitted fields. These declarations do not prove causality or calibrate probabilities.
Overall confidence is at most every supplied field confidence and the confidence of the facts the rule reads.

Rule return annotations and the question schema are both enforced. Values normalize to JSON dictionaries/lists;
nonfinite numbers, non-string object keys and arbitrary objects are rejected. Tuples normalize to JSON arrays.
A numeric-key dictionary schema is refused, and validated nested numeric keys cannot silently merge JSON fields.
Normal type coercion may change values within the declared type (for example a numeric string into an integer).

Hard-check `then` objects are normalized against the question type. Regular answers use existing question contracts,
minimum-confidence refusal and guarantee machinery; structured values do not receive a new statistical guarantee.
No learned classification head is added for open objects/arrays. Use a rule to assemble the response.

JSON round trips retain schema and explicit field metadata. Cached structured answers, metadata, values and trace
records are copied so a caller cannot corrupt the next cached response. Class identities include their JSON schema
in System and Process fingerprints. This candidate has not been released.
