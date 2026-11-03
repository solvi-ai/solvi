# Fact-name aliases

`Catalog.alias(internal, source, type=None, description=None)` registers one ordinary fact computation reading
`source` and producing `internal`. Existing rules, checks and functions continue reading their original names.
It preserves the usual why, trace, fingerprint and replay rather than wrapping every consumer.

```python
from solvi import Catalog, Question, System

cat = Catalog()

@cat.rule("enough")
def enough(car_worth: float) -> bool:
    return car_worth >= 100

cat.alias("car_worth", "market_value", description="Vehicle value in EUR")
system = System(cat, [Question("enough", "Enough collateral?")])
result = system.ask({"market_value": 120})
assert result["enough"].answer == "yes"
assert system.entry_points()[0].fields["market_value"].description == "Vehicle value in EUR"
assert result.trace.replay(system, result.flow)["ok"]
```

The type is inferred only when existing consumers agree on one type. Otherwise pass `type=` explicitly (including
`typing.Any` when untyped values are intentional). Declare a type when registering an alias before its consumers.
Normal typed-fact coercion and rejection still apply. An explicit description documents the external/source input
in entry points and generated input schemas; conflicting descriptions for the same source are refused.

An alias cannot replace a declared part, rule or constraint, refer to itself or introduce a dependency cycle.
Names must be Python fact identifiers. The internal name becomes a computed fact: supply the external/source name
to `ask`. With `input_model=`, the model must expose external names; the adapter does not rewrite a model's fields.
An alias changes catalog identity, so calibrate under the new catalog before relying on a statistical guarantee.

This API adapts fact names. Question/answer renaming is a separate, still unimplemented request.

## Declared dependencies

`System.depends(question)` returns a new set of initial fact names read by the question's declared flow. It reuses the
input-schema planner, covering transitive computations, checks, heads, aliases and declared applicability inputs.
For the example above, `system.depends('enough') == {'market_value'}`. Opaque applicability predicates require an
explicit `applicable_inputs` declaration or `Question.uses`. The result describes declared dependencies under the
System's strategist, rather than causal relevance or proof that every input affects every output.
