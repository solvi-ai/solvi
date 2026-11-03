# Hypothetical rule evaluation

`System.probe(question, facts, override=False)` evaluates one real question against supplied fact values, returning
its normal Response with why, normalized answers and a trace. No duplicate catalog/System or copied business rule
is needed to test a boundary of an intermediate calculation.

```python
from solvi import Catalog, Question, System

cat = Catalog()

@cat.fn
def ratio(price: float, limit: float) -> float:
    return price / limit

@cat.rule("go")
def go(ratio: float, threshold: float) -> bool:
    return ratio >= threshold

system = System(cat, [Question("go", "Proceed?")])
result = system.probe("go", {"ratio": .9, "threshold": .75}, override=True)
assert result["go"].answer == "yes"
assert result["go"].extra["probe"]["hypothetical"]
assert result.trace.replay(system, result.flow)["ok"]
```

`override=True` permits a supplied fact to replace its computed producer for this evaluation. Checks and answer rules
cannot be overridden: supply their input facts instead. Ordinary `ask` retains its collision rejection. The System's
raw input model is bypassed, so intermediate facts can be supplied without the original raw inputs; consumers still
validate their declared argument types. Applicable contracts, hard checks, answer normalization and minimum confidence
are evaluated normally on the hypothetical facts.

Every result is marked hypothetical in its why and extra metadata. A reserved initial fact
`__solvi_hypothetical_probe__` binds this identity and the overridden names into the trace's initial hash. Ordinary
ask rejects that name. JSON round trips and replay preserve the marker. Probe replay rederives the diagnostic answer
without treating it as a production decision or applying a calibrated guarantee.

The diagnostic call does not store/cache a response, update System statistics, observe measured costs, train the
producer/order policies or apply calibrated guarantees. Evaluation is serial under `no_probes`, preventing identity
canaries from probing/counting/marking the diagnostic call in worker threads. Reported confidence is diagnostic;
it is not a new statistical guarantee on hypothetical inputs. The actual catalog functions still run, including any
I/O or application side effects those functions perform. The response does not attach the production System report
context. No production configuration or calibrated threshold changes.
