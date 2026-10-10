"""The determinism contract of a specialist's replay: a model proposer need not be deterministic (a remote LLM at
temperature 0 gives other outputs across identical runs), so the record keeps the proposer's output and replay
re-executes the code — parse, check, render — from it, never calling the proposer; only a record without a stored
output calls the proposer, and the replay says so."""
import itertools
import json
import random
import re
import warnings

import pytest
from pydantic import BaseModel

from solvi.core.specialist import BLOCKED, DROPPED, Checked, Issue, Specialist, replay_runs
from solvi.core.store import JSONLStorage
from solvi.testing.conformance import ConformanceError, check_specialist

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.charts import ChartSpecialist, LLMProposer

SOURCES = ["Sales by region: north 120 units, south 80 units, east 45 units.",
           "Staff by site: berlin 30 people, paris 12 people, rome 7 people."]


class Item(BaseModel):
    label: str
    value: int


class Items(BaseModel):
    items: list[Item]


class Numbers(Specialist):
    """A toy specialist: a value is kept when its label and its number are in the source; the order is the proposal's."""
    name = "numbers"
    version = "1"
    spec_type = Items

    def check(self, spec, source):
        kept = [x for x in spec.items if x.label in source and str(x.value) in re.findall(r"\d+", source)]
        issues = [Issue(DROPPED, "outside", f"{x.label} = {x.value}", f"items[{i}]")
                  for i, x in enumerate(spec.items) if x not in kept]
        if not kept:
            return Checked(None, issues + [Issue(BLOCKED, "nothing_verified", "nothing verified")])
        return Checked(Items(items=kept), issues, [f"{x.label} = {x.value}" for x in kept])

    def render(self, checked):
        return "\n".join(f"{x.label}: {x.value}" for x in checked.verified.items)


class Noisy:
    """A nondeterministic proposer: every call orders the pairs differently and writes its JSON with other spacing (a
    model at temperature 0 behind a remote API does the like)."""
    id = "noisy"

    def __init__(self, seed=0):
        self.calls, self.rng = 0, random.Random(seed)

    def __call__(self, source, question=None):
        self.calls += 1
        pairs = re.findall(r"(\w+) (\d+)", source)
        self.rng.shuffle(pairs)
        body = json.dumps({"items": [{"label": a, "value": int(b)} for a, b in pairs]}, indent=self.calls % 3 or None)
        return f"call {self.calls}:\n{body}" if self.calls % 2 else body


class LenientNumbers(Numbers):
    def parse(self, proposal):
        if isinstance(proposal, str):                                  # a model's text: the JSON object in it
            m = re.search(r"\{.*\}", proposal, re.S)
            if m is None:
                raise ValueError("no JSON object in the text")
            proposal = json.loads(m.group(0))
        return super().parse(proposal)


def tripwire(source, question=None):
    raise AssertionError("replay called the proposer")


def test_a_nondeterministic_proposer_replays_byte_identically_from_storage(tmp_path):
    store = JSONLStorage(str(tmp_path / "runs.jsonl"))
    noisy = Noisy()
    sp = LenientNumbers(noisy, storage=store)
    runs = [sp.run(s) for s in SOURCES for _ in range(3)]
    assert len({r.output for r in runs[:3]}) > 1                       # the proposer is not deterministic
    recs = [store.record(r.stored_id) for r in runs]
    assert all(isinstance(d["proposal_raw"], str) for d in recs)       # the model's own text is kept
    assert recs[0]["proposal_raw"] == runs[0].trace.step("propose")["raw"]
    calls = noisy.calls
    replayer = LenientNumbers(tripwire)
    assert replay_runs(store, replayer, SOURCES) == []
    for r in runs:
        rp = replayer.replay(r.stored_id, r.source, storage=store)
        assert rp.ok and rp.from_record and rp.output_sha256 == store.record(r.stored_id)["output_sha256"]
        again = replayer.replay(r.to_dict(), r.source)                 # the run's own record: the same
        assert again.ok and again.from_record
    assert noisy.calls == calls                                        # the proposer was never called by a replay
    assert store.verify()["ok"]


def test_the_recorded_output_is_parsed_again_and_a_changed_parser_is_caught(tmp_path):
    sp = LenientNumbers(Noisy(), storage=str(tmp_path / "runs.jsonl"))
    run = sp.run(SOURCES[0])

    class Stricter(LenientNumbers):
        def parse(self, proposal):
            spec = super().parse(proposal)
            return Items(items=spec.items[:1])                          # reads less of the same output now

    rp = Stricter(tripwire).replay(run.stored_id, SOURCES[0], storage=sp.storage)
    assert not rp.ok and any("re-parsing the recorded model output" in p for p in rp.problems)
    # a blocked run with an invalid output: it must still be invalid when parsed again
    bad = sp.run(SOURCES[0], proposal="no json here")
    assert bad.output is None and LenientNumbers(tripwire).replay(bad.stored_id, SOURCES[0], storage=sp.storage).ok

    class Accepting(LenientNumbers):
        def parse(self, proposal):
            return Items(items=[]) if proposal == "no json here" else super().parse(proposal)

    rb = Accepting(tripwire).replay(bad.stored_id, SOURCES[0], storage=sp.storage)
    assert not rb.ok and any("parses now" in p for p in rb.problems)


def test_a_record_without_a_stored_output_calls_the_proposer_and_says_so(tmp_path):
    sp = LenientNumbers(Noisy(), storage=str(tmp_path / "runs.jsonl"))
    run = sp.run(SOURCES[0])
    assert run.ok
    legacy = dict(sp.storage.record(run.stored_id))                    # a record that kept no model output
    for k in ("hash", "proposal_raw"):
        legacy.pop(k, None)
    legacy["proposal"] = None
    noisy = Noisy(seed=5)
    rp = LenientNumbers(noisy).replay(legacy, SOURCES[0])
    assert not rp.from_record and noisy.calls == 1 and "called the proposer" in rp.note
    without = LenientNumbers().replay(legacy, SOURCES[0])
    assert not without.ok and not without.from_record and "no proposer" in " ".join(without.problems)


def test_check_specialist_checks_replay_from_record():
    out = check_specialist(LenientNumbers(Noisy()), SOURCES, deterministic=False)
    assert out["replayed_from_record"] == len(SOURCES)
    with pytest.raises(ConformanceError, match="two runs"):
        check_specialist(LenientNumbers(Noisy()), SOURCES)              # two runs differ: not deterministic

    class Recalls(LenientNumbers):
        """Replays by calling the proposer again — what the contract forbids."""
        def replay(self, record, source=None, storage=None, *, proposer=None):
            self.proposer(source, None)
            return super().replay(record, source, storage, proposer=proposer)

    with pytest.raises(ConformanceError, match="replays from its record"):
        check_specialist(Recalls(Noisy()), SOURCES, deterministic=False)


PRESS = "Revenue by region: Europe accounted for 42%, North America for 35% and Asia-Pacific for 23%."


def test_the_chart_llm_proposer_keeps_the_model_text_and_replays_without_the_model(tmp_path):
    counter = itertools.count()

    class _Resp:
        def __init__(self, b):
            self.b = b

        def read(self):
            return self.b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):                                     # another order and spacing on every call
        n = next(counter)
        pts = [{"label": "Europe", "value": 42, "quote": {"text": "Europe accounted for 42%"}},
               {"label": "North America", "value": 35, "quote": {"text": "North America for 35%"}}]
        reply = {"kind": "bar", "title": "Revenue by region", "unit": "%",
                 "series": [{"points": pts if n % 2 else pts[::-1]}]}
        text = "```json\n" + json.dumps(reply, indent=n % 3 or None) + "\n```"
        return _Resp(json.dumps({"choices": [{"message": {"content": text}}]}).encode())

    store = JSONLStorage(str(tmp_path / "runs.jsonl"))
    sp = ChartSpecialist(LLMProposer("http://127.0.0.1:9/v1", "tiny", opener=opener), storage=store)
    a, b = sp.run(PRESS), sp.run(PRESS)
    assert a.output != b.output and a.ok and b.ok                     # the model answers differently each time
    rec = store.record(a.stored_id)
    assert rec["proposal_raw"].startswith("```json")                  # the model's text, as it came
    seen = next(counter)
    replayer = ChartSpecialist(tripwire)
    assert replay_runs(store, replayer, [PRESS]) == []
    assert all(replayer.replay(r.stored_id, PRESS, storage=store).ok for r in (a, b))
    assert next(counter) == seen + 1                                   # no request was made by the replays
