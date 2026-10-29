"""solvi.experimental.process, stage 2: replay of a process from its records — across the segments of a run, with a
person's answer as a leaf recorded and not recomputed, the first divergence named by node and step
(docs/experimental.md#process).

Admission (PROCESS_DESIGN §F, S2): corrupting any one of a node's record, an edge choice, a passed fact or a "human"
record (its answer, reviewer or parent) makes replay name exactly that node — 100 of 100 seeded random tampers; a clean
replay is ok on every run; a pending run replays to {"ok": True, "pending": ticket}."""
import json
import random
import shutil
import warnings

import pytest

from solvi.core.runtime import vhash
from solvi.core.store import GENESIS, JSONLStorage, record_hash

from test_experimental_process_run import claim_process, claim_systems, clock, store

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import (REPLAY_STEPS, CallbackChannel, MemoryQueue, Process, ProcessRunError)


# --- editing a store as an attacker would

def _lines(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def _write(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for d in recs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def edit(path, rid, fn):
    """Change one record in place (its hash is not recomputed)."""
    recs = _lines(path)
    for d in recs:
        if d.get("id") == rid:
            fn(d)
            break
    else:
        raise KeyError(rid)
    _write(path, recs)


def reseal(path):
    """Recompute every hash, id and link of a store and its head, as one who rewrites the whole file would."""
    recs, prev, ids = _lines(path), GENESIS, {}
    for d in recs:
        d["prev"] = prev
        old = d["id"]
        d["hash"] = record_hash(d)
        d["id"] = d["hash"][:16]
        ids[old], prev = d["id"], d["hash"]
    _write(path, recs)
    with open(str(path) + ".head", "w") as fh:
        json.dump({"count": len(recs), "hash": prev}, fh)
    return ids


# --- a world of runs: plain runs, runs paused for a person and resumed, answered at once, pending, cancelled

def review(ps, storage, channel, name="claim"):
    p = Process(name, storage=storage, human=channel)
    p.node("triage", ps["triage"], reads=["item"], writes={"kind": "answer"})
    p.node("review", kind="human", question="fraud_risk", reads=["item", "amount", "kind", "provenance"],
           writes={"fraud_risk": "answer"})
    p.node("payout", ps["payout"], reads=["amount", "kind", "fraud_risk"], writes={"payout": "answer"}, end=True)
    p.edge("triage", "review").edge("review", "payout")
    return p


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("replay")
    clk = clock()
    ps = claim_systems(tmp, "w", clk)
    plain = claim_process(ps, storage=store(tmp / "plain.jsonl", clk))
    q = MemoryQueue()
    queued = review(ps, store(tmp / "queued.jsonl", clk), q)
    at_once = review(ps, store(tmp / "at_once.jsonl", clk),
                     CallbackChannel(lambda req: "high" if req.facts["amount"] > 1000 else "low", reviewer="desk"),
                     name="claim_desk")
    runs = {"plain": [], "queued": [], "at_once": [], "pending": [], "cancelled": []}
    rng = random.Random(7)
    items = ["car", "bumper", "roof", "door", "windscreen"]
    for i in range(6):
        c = {"item": rng.choice(items), "amount": float(rng.randint(10, 6000))}
        runs["plain"].append(plain.run(c, run=f"plain{i}").run)
    for i in range(6):
        c = {"item": rng.choice(["car", "roof", "door"]), "amount": float(rng.randint(10, 6000))}
        r = queued.run(c, run=f"queued{i}")
        q.answer(r.ticket, rng.choice(["low", "high"]), reviewer=rng.choice(["anna", "boris"]))
        assert queued.resume(r.run).status == "decided"
        runs["queued"].append(r.run)
        runs["at_once"].append(at_once.run(c, run=f"desk{i}").run)
    for i in range(2):
        runs["pending"].append(queued.run({"item": "car", "amount": 100.0 + i}, run=f"pending{i}").run)
    r = queued.run({"item": "roof", "amount": 900.0}, run="cancelled0")
    queued.cancel(r.run, "case closed")
    runs["cancelled"].append(r.run)
    files = sorted(p for p in tmp.iterdir() if p.suffix in (".jsonl", ".head"))
    snap = tmp / "snapshot"
    snap.mkdir()
    for f in files:
        shutil.copy(f, snap / f.name)
    return {"tmp": tmp, "snap": snap, "files": files, "ps": ps, "procs": {"plain": plain, "queued": queued,
                                                                          "at_once": at_once}, "runs": runs}


def restore(w):
    for f in w["files"]:
        shutil.copy(w["snap"] / f.name, f)


def proc_of(w, kind):
    return w["procs"]["queued" if kind in ("pending", "cancelled") else kind]


# --- clean replay

def test_every_clean_run_replays(world):
    restore(world)
    for kind, runs in world["runs"].items():
        p = proc_of(world, kind)
        for rid in runs:
            rep = p.replay(rid)
            assert rep["ok"], (kind, rid, rep["at"], rep["reason"])
            assert rep["at"] is None and rep["run"] == rid
            assert all(st["checked"] for st in rep["steps"])
    for p in world["procs"].values():
        out = p.replay_all()
        assert out["ok"] and not out["failed"] and all(v["ok"] for v in out["verify"].values())


def test_a_resumed_run_replays_across_its_segments_with_the_person_as_a_leaf(world):
    restore(world)
    p = world["procs"]["queued"]
    rep = p.replay(world["runs"]["queued"][0])
    assert rep["ok"] and len(rep["segments"]) == 2 and rep["status"] == "decided"
    assert [(s["segment"], s["node"], s["kind"]) for s in rep["steps"]] == [(0, "triage", "decision"),
                                                                           (1, "review", "human"),
                                                                           (1, "payout", "decision")]
    hum = rep["steps"][1]
    assert hum["human"] == "recorded, not recomputed" and "replay" not in hum["checked"] and "human" in hum["checked"]
    assert "replay" in rep["steps"][0]["checked"] and "facts" in rep["steps"][2]["checked"]
    once = world["procs"]["at_once"].replay(world["runs"]["at_once"][0])
    assert once["ok"] and len(once["segments"]) == 1 and once["steps"][1]["human"] == "recorded, not recomputed"


def test_a_pending_run_replays_up_to_its_question(world):
    restore(world)
    p = world["procs"]["queued"]
    for rid in world["runs"]["pending"]:
        rep = p.replay(rid)
        assert rep["ok"] is True and rep["pending"] and rep["status"] == "pending"
        assert rep["pending"] in {x["ticket"] for x in p.pending()}
    out = p.replay_all()
    assert sorted(out["pending"]) == sorted(world["runs"]["pending"])
    rep = p.replay(world["runs"]["cancelled"][0])
    assert rep["ok"] and rep["cancelled"] and rep["status"] == "cancelled"


def test_replay_needs_the_records():
    p = Process("x").node("a", kind="human", question="q", writes={"q": "answer"}, end=True)
    with pytest.raises(ProcessRunError, match="storage"):
        p.replay("r0")


# --- S2 admission: one tamper, exactly that node

def _segments(p, rid):
    return [s for s in p.records() if s.run == rid]


def _steps(p, rid):
    """(segment record, position, step) for every recorded step of a run."""
    return [(s, i, st) for s in _segments(p, rid) for i, st in enumerate(s.data["path"])]


def _store_path(w, node):
    return w["ps"][node].storage.path


def _other(v, choices):
    return next(x for x in choices if x != v)


def tamper(w, rng):
    """One random tamper of one recorded item → (process, run, the node that has to be named, what was done)."""
    kind = rng.choice(["plain", "queued", "at_once"])
    p = proc_of(w, kind)
    rid = rng.choice(w["runs"][kind])
    pst = p._open().path
    steps = _steps(p, rid)
    what = rng.choice(["node", "edge", "fact", "human"])
    if what == "human" and kind == "plain":
        what = "node"
    if what == "node":
        seg, i, st = rng.choice([x for x in steps if x[2]["kind"] != "human"])
        path = _store_path(w, st["node"])
        field = rng.choice(["answer", "by", "parent", "init", "s1answer"])

        def fn(d):
            if field == "answer":
                d["answer"] = "forged"
            elif field == "by":
                d["by"] = "s2"
            elif field == "parent":
                d["run"]["parent"] = "0" * 16
            elif field == "init":
                init = d["s1"]["trace"]["init"]
                k = rng.choice(sorted(init))
                init[k] = init[k] + 1 if isinstance(init[k], (int, float)) else str(init[k]) + "x"
            else:
                d["s1"]["results"] = {k: dict(v, confidence=0.123) for k, v in d["s1"]["results"].items()}
        edit(path, st["stored_id"], fn)
        if rng.random() < 0.5:
            reseal(path)                              # the node's store rewritten whole: ids change
        return p, rid, st["node"], f"node record {st['node']}.{field}"
    if what == "edge":
        seg, i, st = rng.choice([x for x in steps if x[2]["edge"] is not None])

        def fn(d):
            e = d["path"][i]["edge"]
            if rng.random() < 0.5:
                e["to"] = _other(e["to"], ["payout", "glass", "fraud", "review", "triage"])
            else:
                e["when"] = {"status": "abstain"} if e["when"] != {"status": "abstain"} else None
        edit(pst, seg.id, fn)
        return p, rid, st["node"], f"edge of {st['node']}"
    if what == "fact":
        cands = [x for x in steps if x[1] > 0 or x[0].data.get("continues")]
        seg, i, st = rng.choice(cands)
        kept = st["kind"] == "human" and seg.data.get("continues")
        mode = rng.choice(["index", "record"] if st["node"] in ("fraud", "payout") else
                          ["index", "kept"] if kept else ["index"])
        if mode == "kept":                            # the pending segment kept another value for the person
            where = rng.choice(["asking", "resume"])

            def fn(d):
                f = d[where]["facts"]
                f["kind"] = _other(f["kind"], ["auto", "property", "glass"])
            edit(pst, seg.data["continues"], fn)
        elif mode == "index":
            edit(pst, seg.id, lambda d: d["path"][i].__setitem__("facts_in_hash", vhash(rng.random())))
        else:                                         # the receiving node's record holds another value of the fact
            def fn(d):
                init = d["s1"]["trace"]["init"]
                init["kind"] = _other(init["kind"], ["auto", "property", "glass"])
            edit(_store_path(w, st["node"]), st["stored_id"], fn)
        return p, rid, st["node"], f"fact passed to {st['node']} ({mode})"
    seg, i, st = rng.choice([x for x in steps if x[2]["kind"] == "human"])
    field = rng.choice(["value", "reviewer", "parent"])

    def fn(d):
        if field == "value":
            d["value"] = _other(d["value"], ["low", "high"])
        elif field == "reviewer":
            d["reviewer"] = d["reviewer"] + "_forged"
        else:
            d["run"]["parent"] = "f" * 16
    edit(pst, st["stored_id"], fn)
    if rng.random() < 0.3:
        reseal(pst)
    return p, rid, st["node"], f"human {field}"


def test_s2_admission_one_tamper_names_exactly_that_node(world):
    rng = random.Random(2026_10_23)
    hits, misses = 0, []
    for k in range(100):
        restore(world)
        p, rid, node, what = tamper(world, rng)
        rep = p.replay(rid)
        assert rep["at"] is None or rep["at"]["step"] in REPLAY_STEPS
        if not rep["ok"] and rep["at"]["node"] == node:
            hits += 1
        else:
            misses.append((k, what, rid, rep["at"], rep["reason"]))
    restore(world)
    assert hits == 100, misses


# --- the steps beyond the hashes: a forger who rewrites the hashes too

def test_an_edge_rewritten_with_its_hashes_is_caught_by_the_predicate(world):
    restore(world)
    p = world["procs"]["plain"]
    rid = world["runs"]["plain"][-1]                  # the last record of the store: a reseal changes only it
    seg = _segments(p, rid)[0]
    path = p._open().path
    to = seg.data["path"][0]["edge"]["to"]
    edit(path, seg.id, lambda d: d["path"][0]["edge"].__setitem__("to", "glass" if to != "glass" else "fraud"))
    reseal(path)
    rep = p.replay(rid)
    assert not rep["ok"] and rep["at"]["node"] == "triage" and rep["at"]["step"] == "edge"
    assert rep["kind"] == "integrity"
    restore(world)


def test_a_forged_answer_with_its_hashes_rewritten_is_caught_by_the_node_replay(world):
    restore(world)
    p = world["procs"]["queued"]
    rid = world["runs"]["cancelled"][0]               # its triage decision is the last of the triage store
    seg, i, st = _steps(p, rid)[0]
    spath = _store_path(world, "triage")
    assert _lines(spath)[-1]["id"] == st["stored_id"]
    new = {}

    def forge(d):
        new["kind"] = "auto" if d["answer"] != "auto" else "property"
        d["answer"] = new["kind"]
    edit(spath, st["stored_id"], forge)
    ids = reseal(spath)                               # only the last record changes

    def fix(d):                                       # the index made to agree with the forged record
        e = d["path"][i]
        e["stored_id"] = ids[st["stored_id"]]
        e["answers_hash"] = e["facts_out_hash"] = vhash({"kind": new["kind"]})
    edit(p._open().path, seg.id, fix)
    reseal(p._open().path)
    rep = p.replay(rid)
    assert not rep["ok"] and rep["at"]["node"] == "triage" and rep["at"]["step"] == "replay"
    assert rep["kind"] == "recompute" and "answer" in rep["reason"]
    restore(world)


def test_a_human_answer_moved_to_another_parent_with_its_hashes_rewritten(world):
    restore(world)
    p = world["procs"]["queued"]
    pst = p._open().path
    rid = world["runs"]["queued"][0]
    seg, i, st = next(x for x in _steps(p, rid) if x[2]["kind"] == "human")
    edit(pst, st["stored_id"], lambda d: d["run"].__setitem__("parent", "f" * 16))
    ids = reseal(pst)
    recs = _lines(pst)
    for d in recs:                                    # every reference made to agree with the rewritten ids
        for e in d.get("path") or []:
            e["stored_id"] = ids.get(e["stored_id"], e["stored_id"])
        if d.get("continues"):
            d["continues"] = ids.get(d["continues"], d["continues"])
        if isinstance(d.get("run"), dict) and d["run"].get("parent") in ids:
            d["run"]["parent"] = ids[d["run"]["parent"]]
    _write(pst, recs)
    reseal(pst)
    rep = p.replay(rid)
    assert not rep["ok"] and rep["at"]["node"] == "review" and rep["at"]["step"] == "human"
    restore(world)


def test_a_process_changed_since_the_run_is_named(world, tmp_path):
    restore(world)
    ps = world["ps"]
    p = world["procs"]["plain"]
    q = claim_process(ps, storage=JSONLStorage(p._open().path))
    q.edge("glass", "payout", when={"status": "abstain"})
    rep = q.replay(world["runs"]["plain"][0])
    assert not rep["ok"] and rep["at"]["step"] == "spec" and rep["kind"] == "recompute"


def test_a_system_and_a_guard_replay_and_a_tampered_guard_record_is_named(tmp_path):
    from solvi import Answer, Catalog, Question, System
    from solvi.solutions.guard import Guard
    cat = Catalog()

    @cat.rule("size")
    def size(amount: float) -> str:
        return "big" if amount > 100 else "small"
    sysm = System(cat, [Question("size", "Size?", Answer.choice(["small", "big"]))],
                  storage=store(tmp_path / "sys.jsonl"))
    g = Guard(storage=store(tmp_path / "guard.jsonl"), fact_names=["size"])

    @g.tool
    def pay(amount: float) -> str:
        """Pay."""
        return f"paid {amount}"

    @g.policy("pay", on_fail="escalate")
    def small_only(size: str) -> bool:
        return size == "small"
    p = Process("pay", storage=store(tmp_path / "process.jsonl"), on_human="abstain")
    p.node("size", sysm, reads=["amount"], writes={"size": "answer"})
    p.node("guard", g, reads=["call", "size"], writes={"outcome": "answer", "paid": "answer:result"}, end=True)
    p.edge("size", "guard")
    runs = [p.run({"amount": a, "call": {"name": "pay", "arguments": {"amount": a}}}, run=f"g{i}")
            for i, a in enumerate([30.0, 500.0])]
    for r in runs:
        rep = p.replay(r.run)
        assert rep["ok"], rep["reason"]
        assert [s["kind"] for s in rep["steps"]] == ["system", "guard"]
    assert any("tool's result" in u for u in p.replay(runs[0].run)["unchecked"])      # kept as a hash only
    assert p.replay_all()["ok"]
    edit(g.storage.path, runs[1].path[1]["stored_id"], lambda d: d["meta"]["guard"].__setitem__("outcome", "allow"))
    rep = p.replay(runs[1].run)
    assert not rep["ok"] and rep["at"]["node"] == "guard" and rep["at"]["step"] == "record"
    assert not p.replay_all()["ok"]
