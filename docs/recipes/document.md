# The document recipe: `solvi.document`

One model read per field of a pydantic schema, each fact found in the text, the answer computed by your code, a
calibrated threshold on top. The library's 24th top-level name; built only from the low level
(`solvi.core`), so every piece under it is replaceable.

```python
import solvi
from solvi import Span
from solvi.models import llm
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    total: Span[float] = Field(description="The total amount due, exactly as written.")
    due: Span[str] = Field(description="The payment due date, exactly as written.")


recipe = solvi.document(Invoice, llm(URL, "openai/gpt-oss-120b"),
                        task="An invoice: the total and its due date.",
                        compute=lambda f, s: f.total,
                        storage="invoices.jsonl")
res = recipe.ask({"doc": invoice_text})
print(res["answer"].answer, res["answer"].confidence)
for f in res.facts:
    print(f.name, f.value, "at", f"{f.source}[{f.start}:{f.end}]:", f.quote)
```

One call: `res["answer"]` (answer, status "ok" / "abstain", confidence, `why`). Every read is grounded — a span
must be found in the text where the text writes it; a number is a number of its own, never a piece of a longer
one. What cannot be read or checked abstains with the reason; nothing is guessed.

## The schema is the prompt

The fields' names and `description=`s go to the model as the reading instructions. Types:

- `Span[float]` / `Span[int]` / `Span[Decimal]` — a number as the document writes it: thousands separators, a
  decimal comma, a scale word (`1,250 million` is 1250 in millions, never multiplied), a currency sign or code
  around it, parentheses as an accounting negative (`(1,250)` → −1250). The unit the text states (million,
  EUR, %) is kept in `facts.units`, not multiplied into the value.
- `Span[date]` — "21 July 2026", "July 21, 2026", "21.07.2026"; a numeric date that reads both ways
  (`03/04/2026`) refuses, day or month first is a guess.
- `Span[str]` — the exact text of the field.
- A `Literal[...]` choice — the model picks one option; its confidence is the stated one (or token
  log-probabilities with `logprobs=`).
- `Maybe[...]` — the field may be not stated; `solvi.Unknown` then, not an abstention.

One model call per field; a field's value must stand in the text at its offsets (`res.facts[i].quote ==
doc[start:end]`), else the read is rejected. A field whose quote left out the currency or scale takes it from
the text into its unit; a sign the text does not write there abstains.

## compute, checks, calibrate

`compute=(facts, state)` is the question's rule — plain code over the read fields (the answer is a number by
default; `answer=` for another type). `checks={"name": fn}` are hard checks over the same facts: a failed hard
check overrides any confidence. `calibrate(examples, max_risk=0.05)` sets a threshold on the answer from
labelled examples `[(input, correct answer)]`: below it the question abstains instead of answering alone.

```python
recipe.calibrate([({"doc": d}, gold) for d, gold in labelled], max_risk=0.05, workers=8)
```

- `workers > 1` asks the calibration examples in parallel threads (results in input order; one example whose
  ask breaks counts as escalated, calibration goes on). 300 examples of the fake-endpoint test calibrate in
  under 5 s.
- A heartbeat prints to stderr every 10 s once there are 50 or more examples; pass `progress=(done, total)`
  to receive each one instead.
- Warnings: fewer than 100 examples (the threshold rests on them), and more than 20 % of the examples
  escalating (the threshold then rests on few answers).
- `save_calibration(path)` writes it; `document(..., calibration=path)` loads it (refuses a file made for
  another catalog or model).

## What every answer carries

- `res.facts` — the quoted facts as data: name, value, the offsets and the quoted text, confidence.
- `why` — the rule's inputs and the reads behind them; an input text longer than 200 chars is named by its
  length and hash (`doc[12,345 chars, sha256:0123456789ab]`), never carried — the audit of a 12k-char
  document stays under 1 KB and still verifies against the text.
- `res.audit()` — what the answer rests on: given inputs, computed facts, quotes with offsets, model decisions
  with probabilities, checks, the rule, constraints.
- `storage="decisions.jsonl"` stores every ask (hash-chained); `recipe.verify()` checks the chain,
  `recipe.replay_all()` re-executes every stored decision and names any step that changed. Calibration asks are
  stored too: once the threshold is installed they replay against a guarded system, so a re-ask can differ —
  calibrate on a store-less system, or replay with filters (by id / time) that keep the live decisions.

## Knobs

`text="doc"` — the input key carrying the text. `units="same"` (default) refuses amounts whose written scales
or currencies disagree; `"lenient"` lets a bare `$` match any dollar and a field without a scale match any;
`"any"` turns the check off. `checks=` and `answer=` as in the low level. The model is anything with the
decider interface; a `logprobs="auto"` model is read with one stated confidence per read (a copy, with a
warning).

The runnable example: [examples/27_document_decision.py](https://github.com/solvi-ai/solvi/blob/main/examples/27_document_decision.py)
— a fake chat-completions endpoint, no network; the test suite runs it.
