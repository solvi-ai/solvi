# Decisions without audit records

`System.decide(state, questions=None)` evaluates a serial, unmodeled catalog without producing an audit record.
It runs the ordinary planning, typed facts, rules, hard checks, contracts, constraints and accepted answer-fact stages.
Use `result[question].answer` as with `ask`. Its separate `UnrecordedResponse` has `recorded=False` and
`replayable=False`; accessing a trace, replaying it or saving it as a decision record is refused.

Every call checks the configuration before validating inputs or running parts. Configured storage, question guards,
calibration, heads, models, learning, watches, caches, custom strategies, asynchronous parts, parallel workers and
experimental components require `System.ask`. This first version deliberately supports a narrow synchronous catalog.
Contract applicability must be a regular synchronous Python function. User functions retain their own side effects;
this method does not establish their purity or make retries safe.

```python
from solvi import Answer, Catalog, Question, System

cat = Catalog()

@cat.rule("total")
def total(price: int, count: int) -> int:
    return price * count

system = System(cat, [Question("total", "Total?", Answer.value())])
result = system.decide({"price": 12, "count": 3})
assert result["total"].answer == 36
assert result.recorded is False and result.replayable is False
```

Use this explicit entry point for application boundaries that do not require a decision journal. Ordinary `ask`
continues to produce its trace and update its production bookkeeping. No performance improvement is asserted here.
