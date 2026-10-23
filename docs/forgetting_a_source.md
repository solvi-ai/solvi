# Forgetting a source

A source has to go: a person asks for their data to be erased, a supplier's feed turns out to be wrong, a writer is
exposed. What it told must stop counting, every decision that rested on it must be looked at again, and the record of
all this must stay verifiable. In solvi that is one call on the knowledge store, plus an explicit erasure of personal
data in the decision records when a regulation requires it.

## One call: `KnowledgeStore.forget`

```python
import solvi
from solvi.core.store import JSONLStorage

decisions = JSONLStorage("decisions.jsonl")
system = solvi.System(catalog, questions, storage=decisions)      # its rules read the given fact "knowledge"
km = solvi.Knowledge()

km.tell(("drugX", "max_dose", "40mg"), source="spec", by="patient-42")
system.ask({"knowledge": km.snapshot(about="drugX")})            # a decision resting on it
km.tell(("drugX", "max_dose", "20mg"), source="spec", by="pharmacy")

out = km.store.forget("patient-42", decisions, system, by="dpo", why="erasure request")
out["retracted"]     # {item id: (status before, status after)}: what the source told, and what was derived from it
out["changed"]       # [{"id", "old", "new"}]: decisions whose answer changes — for a reviewer
out["same"]          # decisions whose snapshot changed but whose answer does not — recorded
```

`forget` does three things, each a public call you can also run on its own:

1. **Take back what the source told** — `km.store.retract_source("patient-42", by=, why=)`. Items only that source
   told are retracted with everything derived from them; items others told too stay on their support. Nothing is
   deleted: the journal keeps the entries, marked retracted, and the chain stays whole.
2. **Find and re-decide the decisions that rested on it** — `km.store.redecide(decisions, retracted, system,
   upto="now")`. A stored decision is listed when the snapshot it read changes once the retracted items are left out;
   it is re-run on the journal as it is now, so a correction told later (the 20mg above) gives the new answer instead
   of "unknown".
3. **Journal it** — one `"forget"` entry in the knowledge journal: the source, who asked and why, the retracted item
   ids and the ids of the decisions whose answer changed (and of those whose did not).

Without a decision store (`km.store.forget("patient-42")`) only steps 1 and 3 run.

## Erasing personal data in the decision records

`forget` keeps every decision record as it was: a decision's input is evidence of what was decided and why. When
the data itself must go, find where it is held and erase it explicitly — erasure is never a side effect:

```python
where = decisions.where_is("patient_id", "42")      # {"dependent": decisions resting on it, "stored": records holding it}
for d in where["dependent"]:
    decisions.redact(d["id"], by="dpo", note="erasure request")
for rid in where["stored"]:
    decisions.redact(rid, by="dpo", note="erasure request")
decisions.verify()["ok"]                             # still True: a redaction keeps the chain and is itself recorded
```

`redact` removes a record's content and keeps its place and hash, and appends a `"redaction"` record naming what was
erased; a redacted record no longer replays. `decisions.quarantine(fact, value)` lists the decisions whose answers rest
on a fact found to be wrong (the provenance of each answer, hard checks included) when the wrong value was a given
input rather than a knowledge item.

## What this does not do

- It does not decide for you which decisions to reverse: `changed` goes to a person; nothing is re-sent or undone.
- An item the source told that someone else also told stays: it is no longer the source's, but it is still known.
- A knowledge item told again after its retraction is a hypothesis waiting for a re-test (`pending_retest`) until it is
  confirmed; `store.pending()` lists them. On facts told with `valid=`, the same content told again by anyone on an
  overlapping validity waits too, and a copy accepted while another writer still held the content waits from the
  moment that writer is taken back.
