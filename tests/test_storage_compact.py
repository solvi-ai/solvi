"""Compact records (TraceStorage(record="compact" | "sample:N")) and rotation of a JSON-lines store: what a compact
record keeps, that the chain still verifies, what replay checks of it and what it says it cannot check, and a rotated
store whose files are linked head to head."""
import io
import json
import os
import threading
import urllib.error  # noqa: F401 — the fake server below mirrors tests/test_generate.py

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.diff import diff
from solvi.refine import Fail
from solvi.storage import CompactRecord, JSONLStorage, SQLiteStorage, record_mode


def _game(storage=None, threshold=0.9):
    """A frequent-decision loop in miniature: a screen, notes, the last failed moves; a proposal, two hard checks with
    reasons, a rule."""
    cat = Catalog()

    @cat.fn
    def exits(screen: str) -> list:
        return [d for d, ch in zip(("north", "south", "west", "east"), screen) if ch == "."]

    @cat.fn
    def danger(screen: str) -> float:
        return round(screen.count("E") / 10, 3)

    @cat.fn
    def proposal(exits: list) -> str:
        return exits[0] if exits else "wait"

    @cat.check(hard=True, then={"move": "wait"})
    def not_repeat_failed(proposal: str, failed: list) -> bool:
        return Fail(f"{proposal} failed in the last steps") if proposal in failed else True

    @cat.rule("move")
    def move(proposal: str, danger: float) -> str:
        return "wait" if danger > threshold else proposal

    q = Question("move", "Which way?", Answer.choice(["north", "south", "west", "east", "wait"]),
                 requires=["not_repeat_failed"])
    return System(cat, [q], storage=storage)


STATES = [{"screen": s, "failed": f, "notes": "the key is in the cave north of the lake " * 5}
          for s, f in [("..#.EE", []), ("#..#E", ["south"]), ("##..", []), ("....EEEEEEEEEE", ["north"]), ("####", [])]]


def _fill(store, system=None):
    s = system or _game(store)
    return s, [s.ask(st) for st in STATES]


def test_record_modes_are_checked():
    assert record_mode("full") == ("full", None) and record_mode("compact") == ("compact", None)
    assert record_mode("sample:10") == ("sample", 10)
    for bad in ("sample:0", "sample:x", "lean", None):
        with pytest.raises(ValueError, match="sample:N"):
            record_mode(bad)


def test_a_compact_record_keeps_the_input_answers_checks_and_hashes_and_verifies(tmp_path):
    full, comp = JSONLStorage(tmp_path / "f.jsonl"), JSONLStorage(tmp_path / "c.jsonl", record="compact")
    _fill(full)
    s, rs = _fill(comp)
    a, b = full.record(full.query()[0].id), comp.record(rs[0].stored_id)
    assert "response" in a and "record" not in a                     # a full record is as before 1.0
    assert "response" not in b and b["record"] == "compact"
    c = b["compact"]
    assert c["trace"]["init"] == STATES[0] and c["results"]["move"]["answer"] == b["answers"]["move"][0]
    assert set(c["checks"]) == {"not_repeat_failed"} and b["records"] == a["records"]   # the same step hashes
    reasons = comp.record(rs[1].stored_id)["compact"]["checks"]["not_repeat_failed"]
    assert reasons == [False, ["south failed in the last steps"]]
    assert os.path.getsize(tmp_path / "c.jsonl") * 2 < os.path.getsize(tmp_path / "f.jsonl")
    assert comp.verify()["ok"] and len(comp) == len(STATES)
    st = comp.query()[0]
    assert st.compact and st.input == STATES[0] and st.answers == {"move": rs[0]["move"].answer}
    with pytest.raises(CompactRecord, match="rederive"):
        comp.get(rs[0].stored_id)


def test_replay_reruns_a_compact_decision_and_checks_every_step_hash(tmp_path):
    store = JSONLStorage(tmp_path / "c.jsonl", record="compact")
    s, rs = _fill(store)
    assert store.replay_all(s) == []
    back = store.rederive(rs[3].stored_id, s)
    assert [(r.step, r.name, r.hash) for r in back.trace.records] == [(r.step, r.name, r.hash) for r in rs[3].trace.records]
    assert back["move"].answer == rs[3]["move"].answer and back.trace.replay(s, back.flow)["ok"]
    changed = _game(threshold=0.1)                                    # a rule changed since: the old value is not kept
    bad = store.replay_all(changed)
    assert bad and all(b["record"] == "compact" for b in bad)
    m = bad[0]["mismatches"][0]
    assert m.kind == "recompute" and "keeps the hash of the old step" in m[2]


def test_an_edited_compact_record_does_not_verify(tmp_path):
    p = tmp_path / "c.jsonl"
    store = JSONLStorage(p, record="compact")
    _fill(store)
    lines = p.read_text().splitlines()
    d = json.loads(lines[2])
    d["compact"]["trace"]["init"]["screen"] = "...."                   # the input the decision was made on
    lines[2] = json.dumps(d)
    p.write_text("\n".join(lines) + "\n")
    v = JSONLStorage(p).verify()
    assert not v["ok"] and any(seq == 2 and "edited" in why for seq, _, why in v["problems"])


def test_a_compact_record_whose_answer_was_swapped_is_named_by_verify(tmp_path):
    p = tmp_path / "c.jsonl"
    store = JSONLStorage(p, record="compact")
    _fill(store)
    lines = p.read_text().splitlines()
    d = json.loads(lines[0])
    d["compact"]["results"]["move"]["answer"] = "east"
    lines[0] = json.dumps(d)
    p.write_text("\n".join(lines) + "\n")
    probs = JSONLStorage(p).verify()["problems"]
    assert any("compact answers differ" in why for _, _, why in probs)


def test_sample_keeps_one_decision_in_n_in_full(tmp_path):
    store = JSONLStorage(tmp_path / "s.jsonl", record="sample:2")
    s, rs = _fill(store)
    kinds = [store.record(r.stored_id).get("record", "full") for r in rs]
    assert kinds == ["full", "compact", "full", "compact", "full"]
    assert store.verify()["ok"] and store.replay_all(s) == []
    assert store.get(rs[0].stored_id)["move"].answer == rs[0]["move"].answer


def test_sqlite_takes_record_too(tmp_path):
    with SQLiteStorage(tmp_path / "c.db", record="compact") as store:
        s, rs = _fill(store)
        assert store.verify()["ok"] and store.replay_all(s) == [] and store.query()[0].compact
        assert store.query(answer=rs[0]["move"].answer)


def test_quarantine_where_is_diff_and_the_reports_read_compact_records(tmp_path):
    store, full = JSONLStorage(tmp_path / "c.jsonl", record="compact"), JSONLStorage(tmp_path / "f.jsonl")
    s, rs = _fill(store)
    _fill(full)
    hit = store.quarantine("failed")                                    # re-derived: the same as on full records
    want = full.quarantine("failed")
    assert hit and [(h["seq"], h["questions"]) for h in hit] == [(h["seq"], h["questions"]) for h in want]
    blind = JSONLStorage(tmp_path / "c.jsonl")                          # a store that does not know its System
    with pytest.warns(UserWarning, match="could not be re-derived"):
        assert blind.quarantine("failed") == []
    wi = store.where_is("screen", "####")
    assert wi["stored"] == [rs[4].stored_id] and wi["dependent"] == []
    rep = diff(store, _game(threshold=0.1))
    assert rep.total == len(STATES) and rep.changed
    ch = rep.changed[0]
    assert ch["record"] == "compact" and ch["questions"]["move"]["causes"] == []
    assert "only the answers are compared" in ch["questions"]["move"]["first_step"]["why"]
    sysrep = s.report()
    assert sysrep.period["decisions"] == len(STATES)
    assert sum(sysrep.system1["move"]["answered_by"].values()) + sysrep.system1["move"]["abstained"] == len(STATES)
    assert "move" in s.storage.report()


def test_a_redacted_compact_record_still_verifies(tmp_path):
    store = JSONLStorage(tmp_path / "c.jsonl", record="compact")
    _, rs = _fill(store)
    store.redact(rs[1].stored_id, by="dpo", note="erasure request")
    assert store.verify()["ok"] and "compact" not in store.record(rs[1].stored_id)


# --- a model step: a generator's output is kept whole and given back on replay instead of calling the model
class FakeServer:
    def __init__(self, reply):
        self.reply, self.n = reply, 0

    def __call__(self, req, timeout=None):
        self.n += 1
        body = json.loads(req.data.decode())
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [{"message": {"role": "assistant",
                                      "content": self.reply}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode())


def _planner(storage, server, rerun=False):
    from solvi.generate import generator
    g = generator("http://127.0.0.1:9/v1", "m-1", opener=server, sleep=lambda s: None)
    cat = Catalog()
    cat.fn(g.part("plan", lambda goal: f"plan for {goal}", replay="rerun" if rerun else "trust"))

    @cat.check(hard=True, then={"ok": "no"})
    def short(plan) -> bool:
        return len(plan) < 40

    @cat.rule("ok")
    def ok(short) -> bool:
        return True
    return System(cat, [Question("ok", "?", Answer.yes_no(), requires=["short"])], storage=storage)


def test_a_compact_record_with_a_generator_replays_from_the_kept_output_without_calling_the_model(tmp_path):
    srv = FakeServer("go north")
    store = JSONLStorage(tmp_path / "g.jsonl", record="compact")
    s = _planner(store, srv)
    res = s.ask({"goal": "the cave"})
    kept = store.record(res.stored_id)["compact"]["trace"]["records"]
    assert [r["name"] for r in kept] == ["plan"] and kept[0]["extra"]["generated"]["usage"]["input_tokens"] == 11
    n = srv.n
    assert store.replay_all(s) == [] and srv.n == n                    # not called again
    assert store.rederive(res.stored_id, s).values["plan"] == "go north"
    from solvi.storage import rederive
    d = store.record(res.stored_id)                                     # the kept output edited
    d["compact"]["trace"]["records"][0]["value"] = "go south"
    back, why = rederive(d, s)
    assert back is None and why[0].kind == "integrity" and why[0][1] == "plan"
    d = store.record(res.stored_id)                                     # the model step's record not kept: no verdict
    d["compact"]["trace"]["records"] = []                               # past it, and the replay says so
    back, why = rederive(d, s)
    assert back is None and why[0].kind == "not_kept" and "not checked past this step" in why[0][2]
    from solvi.runtime import mismatch_summary
    assert mismatch_summary(why)["summary"].startswith("not verified")


def test_trust_models_on_a_rerunnable_model_takes_the_kept_output(tmp_path):
    srv = FakeServer("go north")
    store = JSONLStorage(tmp_path / "g.jsonl", record="compact")
    s = _planner(store, srv, rerun=True)
    s.ask({"goal": "the cave"})
    n = srv.n
    assert store.replay_all(s, trust_models=True) == [] and srv.n == n
    assert store.replay_all(s) == [] and srv.n == n + 1                 # re-run: the model is asked again, same output


# --- rotation
def test_rotation_links_each_file_to_the_head_of_the_one_before(tmp_path):
    p = tmp_path / "d.jsonl"
    store = JSONLStorage(p, record="compact", rotate_records=3)
    s = _game(store)
    for _ in range(3):
        for st in STATES:
            s.ask(st)
    segs = store.segments()
    assert [os.path.basename(x) for x in segs] == [f"d.{i:06d}.jsonl" for i in range(1, len(segs) + 1)] and len(segs) >= 4
    first = json.loads(p.read_text().splitlines()[0])
    assert first["kind"] == "segment" and first["segment"]["previous"] == os.path.basename(segs[-1])
    v = store.verify_segments()
    assert v["ok"] and v["count"] == 3 * len(STATES) + len(segs) and store.verify()["ok"]
    assert store.replay_all(s) == [] and JSONLStorage(segs[1]).verify()["ok"]
    os.remove(segs[1])                                                  # a segment removed: a link breaks
    v = store.verify_segments()
    assert not v["ok"] and any("the link names" in why for _, _, _, why in v["problems"])


def test_rotation_by_size_and_bad_options(tmp_path):
    store = JSONLStorage(tmp_path / "d.jsonl", rotate_bytes=4000)
    s = _game(store)
    for st in STATES * 2:
        s.ask(st)
    assert store.segments() and store.verify_segments()["ok"]
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="rotate_bytes"):
            JSONLStorage(tmp_path / "x.jsonl", rotate_bytes=bad)


def test_two_writers_of_a_rotated_store_keep_one_chain(tmp_path):
    p = tmp_path / "d.jsonl"

    def write(k):
        s = _game(JSONLStorage(p, record="compact", rotate_records=7))
        for i in range(20):
            s.ask(dict(STATES[i % len(STATES)], notes=f"writer {k} step {i}"))
    ts = [threading.Thread(target=write, args=(k,)) for k in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    v = JSONLStorage(p).verify_segments()
    assert v["ok"], v["problems"][:3]
    segs = len(JSONLStorage(p).segments())
    assert v["count"] == 40 + segs
