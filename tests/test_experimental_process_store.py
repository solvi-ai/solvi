"""solvi.experimental.process and its store: the fixes of the 2026-10-30 external handoff that touch how the
process reads its records (S267/S276 — the question-to-a-person cycle rescanned the whole store five times per
run; S277 — replay scanned it once per run), a replay under a swapped node naming the node (S269), the resumed
run's path stitched over its segments (S268), the spec hash no longer depending on the nodes' store paths
(S271), merge="all" refused at spec time (S272) and a fully open end_to_end threshold warned about (S274).

Small synthetic processes and systems, no LLM, no torch; the perf pins are generous absolute budgets in the
style of test_knowledge_store_revisions.py (the broken baseline: ~87 ms per run per MB of store, replay_all
53 ms per run at 1.3 MB)."""
import json
import time
import warnings

import pytest

import solvi
from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import MemoryQueue, Process, ProcessSpecError, _edge_list, spec_hash


def _build(tmp_path, name, examples, max_risk=0.05):
    """A solvi.build result on a small separable choice question, with its own store."""
    return solvi.build(Question(name, f"{name}?", Answer.choice(sorted({y for _, y in examples}))), examples,
                       catalog=Catalog(), max_risk=max_risk, novel=False,
                       storage=tmp_path / f"{name}.jsonl")


def _system(tmp_path, name, fn, qname, options=("a", "b")):
    cat = Catalog()
    cat.rule(qname)(fn)
    return System(cat, [Question(qname, "?", Answer.choice(list(options)))],
                  storage=tmp_path / f"{name}.jsonl")


# --- S267/S276: the question cycle reads the store's tail, not the store

def test_the_question_cycle_stays_flat_as_the_store_grows(tmp_path):
    """Every run of a human-in-the-loop process used to make five full-store scans (run, answer, resume); with the
    store's parsed-lines cache and the process's index over it, a cycle reads only what was appended since. Four
    equal batches as the store grows 4x: the per-run time stays under a generous budget (the broken tree needed
    ~20 ms/run at 0.3 MB growing to ~110 ms/run at 1.3 MB) and does not track the store's size."""
    cat = Catalog()

    @cat.rule("q")
    def q(x: int) -> str:
        return "a" if x % 2 else "b"

    s = System(cat, [Question("q", "?", Answer.choice(["a", "b"]))], storage=tmp_path / "node.jsonl")
    queue = MemoryQueue()
    p = Process("p", storage=tmp_path / "proc.jsonl", human=queue)
    p.node("n", s, reads=["x"], writes={"q": "answer"})
    p.node("h", kind="human", question="q", end=True)
    p.edge("n", "h")
    per_run, done = [], 0
    for k in (60, 60, 60, 60):
        t0 = time.perf_counter()
        for i in range(done, done + k):
            r = p.run({"x": i})
            assert r.status == "pending", r.status
            queue.answer(r.ticket, "a", reviewer="r")
            assert p.resume(r.run).status == "decided"
        per_run.append((time.perf_counter() - t0) / k * 1000)
        done += k
    assert all(ms < 12.0 for ms in per_run), per_run            # ~87 ms/run/MB before, ~5 ms/run at 1.3 MB now
    assert per_run[-1] < 2.5 * per_run[0] + 4.0, per_run        # flat within a generous margin (before: ~5x)
    st = p._open()
    assert p._index()["seq"] == len(st._cache) - 1              # the index ends where the store does
    assert p._index()["seq"] == p._index()["seq"]               # a second refresh consumes nothing new


def test_the_store_cache_sees_a_file_rewritten_in_place(tmp_path):
    """The parsed-lines cache is validated against the file's bytes: an edit made behind the store's back — even of
    the same length, in the same clock tick — is what iter() returns (verify() catches it; nothing serves stale
    records). iter(after=seq) serves the same freshness to an incremental reader."""
    st = JSONLStorage(tmp_path / "s.jsonl")
    for i in range(3):
        st._append({"v": 2, "kind": "ask", "x": i})
    assert [s.data["x"] for s in st.iter(None)] == [0, 1, 2]
    assert [s.data["x"] for s in st.iter(None, after=1)] == [2]
    recs = [json.loads(l) for l in open(st.path) if l.strip()]
    recs[0]["x"] = 9                                       # same length, written in the same clock tick
    with open(st.path, "w") as fh:
        for d in recs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    assert [s.data["x"] for s in st.iter(None)] == [9, 1, 2]
    assert [s.data["x"] for s in st.iter(None, after=0)] == [1, 2]
    assert not st.verify()["ok"]                            # and the chain still reports the edit


# --- S277: replay rides the index, not one scan per run

def test_replay_all_walks_the_records_it_already_has(tmp_path):
    """p.replay(run) scanned the whole process store once per run (replay_all: once per run per run). With the
    index, replaying every run of a grown store reads no line of it again (the cache is current); every run still
    replays and every chain verifies."""
    cat = Catalog()

    @cat.rule("q1")
    def q1(x: int) -> str:
        return "a" if x % 2 else "b"

    @cat.rule("q2")
    def q2(q1: str) -> str:
        return "c" if q1 == "a" else "d"

    s = System(cat, [Question("q1", "?", Answer.choice(["a", "b"]))], storage=tmp_path / "m1.jsonl")
    t = System(cat, [Question("q2", "?", Answer.choice(["c", "d"]))], storage=tmp_path / "m2.jsonl")
    p = Process("decided", storage=tmp_path / "proc.jsonl")
    p.node("m1", s, reads=["x"], writes={"q1": "answer"})
    p.node("m2", t, reads=["q1"], writes={"r": "answer"}, end=True)
    p.edge("m1", "m2")
    for i in range(80):
        assert p.run({"x": i}).status == "decided"
    st, reads = p._open(), []
    real = st._read_tail

    def counting(fh):
        reads.append(1)
        return real(fh)
    st._read_tail = counting
    t0 = time.perf_counter()
    out = p.replay_all()
    dt = time.perf_counter() - t0
    assert out["ok"] and out["runs"] == 80 and not out["failed"]
    assert reads == []                                        # the store was already read: nothing parsed again
    assert dt < 6.0                                           # ~4.7 ms/run now, ~50 ms/run before (400 runs)


# --- S269: a run under a swapped node fails at the node, not at the spec

def test_replay_under_a_swapped_node_names_the_node(tmp_path):
    """strict=False loads a process with another system at a node and carries the actual fingerprint, so the spec
    hash differed and every replayed run died at the blanket "spec" step — the divergence was never located. Now
    replay walks by the recorded spec: the run through the swapped node fails at that node's own step; the run
    that bypassed it replays; a strict load still refuses."""
    sa = _system(tmp_path, "a", lambda x: "take" if x else "skip", "qa", options=("take", "skip"))
    sb = _system(tmp_path, "b", lambda qa: "mid:" + qa, "qb", options=("mid:take", "mid:skip", "ALT:take"))
    sb_alt = _system(tmp_path, "balt", lambda qa: "ALT:" + qa, "qb", options=("mid:take", "mid:skip", "ALT:take"))
    sc = _system(tmp_path, "c", lambda qb: "end", "qc", options=("end",))
    store = tmp_path / "proc.jsonl"
    p = Process("swap", storage=store, human=MemoryQueue())
    p.node("a", sa, reads=["x"], writes={"qa": "answer"})
    p.node("b", sb, reads=["qa"], writes={"qb": "answer"})
    p.node("c", sc, reads=["qb"], writes={"out": "answer"}, end=True)
    p.edge("a", "b", when={"answer": "qa", "eq": "take"})
    p.edge("a", "c", when={"answer": "qa", "in": ["skip"]})
    p.edge("b", "c")
    through = p.run({"x": 1}).run
    bypass = p.run({"x": 0}).run
    assert p.replay(through)["ok"] and p.replay(bypass)["ok"]
    spec = p.spec()
    with pytest.raises(ProcessSpecError, match="not the recorded one"):
        Process.from_spec(spec, systems={"a": sa, "b": sb_alt, "c": sc}, storage=store, strict=True)
    p2 = Process.from_spec(spec, systems={"a": sa, "b": sb_alt, "c": sc}, storage=store, strict=False)
    assert list(p2.mismatched) == ["b"]
    rep = p2.replay(through)                                  # the run through b: b's own step, not the spec's
    assert not rep["ok"] and rep["at"]["node"] == "b"
    assert rep["at"]["step"] in ("record", "replay", "facts", "edge")
    assert p2.replay(bypass)["ok"]                            # the run that never reached b still replays


# --- S268: the resumed run's path is the whole run

def test_a_resumed_run_carries_its_whole_path(tmp_path):
    """A resumed run's ProcessResult.path (and so explain() and monitor(reference=)) used to cover the last
    segment only — the nodes before the question and their edges were gone, and a watch set on such runs was set
    on a biased sample. The path is now the run's: the earlier segments' steps first, as the live watch's edge
    union always was."""
    cat = Catalog()

    @cat.rule("kind")
    def kind(x: int) -> str:
        return "auto"

    @cat.rule("verdict")
    def verdict(zz: str) -> str:
        return "ok"                                  # never runs: zz is never given -> the question abstains

    @cat.rule("final")
    def final(verdict: str) -> str:
        return "done" if verdict == "fine" else "stop"

    s1 = System(cat, [Question("kind", "?", Answer.choice(["auto"]))], storage=tmp_path / "n1.jsonl")
    s2 = System(cat, [Question("verdict", "?", Answer.choice(["fine", "bad"]))], storage=tmp_path / "n2.jsonl")
    s3 = System(cat, [Question("final", "?", Answer.choice(["done", "stop"]))], storage=tmp_path / "n3.jsonl")
    queue = MemoryQueue()
    p = Process("seg", storage=tmp_path / "proc.jsonl", human=queue)
    p.node("n1", s1, reads=["x"], writes={"kind": "answer"})
    p.node("n2", s2, reads=["kind", "zz"], writes={"verdict": "answer"}, question="verdict")
    p.node("n3", s3, reads=["verdict"], writes={"out": "answer"}, question="final", end=True)
    p.edge("n1", "n2")
    p.edge("n2", "n3", when={"status": "ok"})
    r0 = p.run({"x": 1})
    assert r0.status == "pending" and [st["node"] for st in r0.path] == ["n1", "n2"]
    queue.answer(r0.ticket, "fine", reviewer="anna")
    r1 = p.resume(r0.run)
    assert r1.status == "decided"
    assert [st["node"] for st in r1.path] == ["n1", "n2", "n2", "n3"]       # n2's abstention, then its answer
    assert [st["kind"] for st in r1.path] == ["system", "system", "human", "system"]
    assert _edge_list(p.edges, r1) == [0, 1]                                # both edges, as the live watch sees
    text = p.explain(r1)
    assert "1. n1 (system)" in text and "2. n2 (system)" in text            # explain walks the whole run
    assert "3. n2 (human)" in text and "4. n3 (system)" in text
    assert p.replay(r1.run)["ok"]                                           # and replay is unchanged by it


# --- S271: the spec hash does not depend on the nodes' store paths

def test_the_spec_hash_ignores_where_the_nodes_keep_their_records(tmp_path):
    """The same system with its store under another directory hashed differently (only nodes.n.store differed), so
    a process was not portable: rewriting the path in a recorded spec broke its hash and a strict load refused.
    The store paths are now outside the hashed body: same system, same hash; a moved spec loads strictly.
    Behaviour change over 1.4.0: hashes of specs whose nodes carry a store differ from the recorded ones."""
    def rule(x: int) -> str:
        return "a" if x % 2 else "b"

    def make(d):
        cat = Catalog()
        cat.rule("q")(rule)
        return System(cat, [Question("q", "?", Answer.choice(["a", "b"]))], storage=d / "n.jsonl")

    s1, s2 = make(tmp_path / "one"), make(tmp_path / "two")
    p1 = Process("loc").node("n", s1, reads=["x"], writes={"q": "answer"}, end=True)
    p2 = Process("loc").node("n", s2, reads=["x"], writes={"q": "answer"}, end=True)
    h1, h2 = p1.spec(), p2.spec()
    assert h1["hash"] == h2["hash"]                                          # where the records live is not what it is
    moved = json.loads(json.dumps(h1))
    moved["nodes"]["n"]["store"] = str(tmp_path / "two" / "n.jsonl")
    assert moved["hash"] == spec_hash(moved)                                 # rewriting the path keeps the hash valid
    Process.from_spec(moved, systems={"n": s2}, strict=True)                 # and a strict load takes it


# --- S272: merge="all" is refused where it is written

def test_merge_all_of_several_inputs_is_refused_at_spec_time(tmp_path):
    """merge="all" on a multi-input node validated in spec() and raised NotImplementedError at run time — a spec
    that could be written but never run. spec() now refuses it, naming the node and saying why (a recorded spec of
    an earlier release still loads; see test_a_merge_of_all_inputs_and_an_unbound_node_are_refused)."""
    from test_experimental_process_run import Echo
    p = Process("diamond")
    for n in ("a", "b", "c", "d"):
        p.node(n, Echo(), kind="system", writes={n: "answer"}, end=n == "d")
    p.edge("a", "b").edge("a", "c").edge("b", "d").edge("c", "d").merge("d", "all")
    with pytest.raises(ProcessSpecError, match="node 'd' merges 'all' of several inputs.*not\\s+implemented"):
        p.spec()
    quiet = Process("chain")                                                 # one input: "all" is the same as "any"
    quiet.node("a", Echo(), kind="system", writes={"a": "answer"})
    quiet.node("b", Echo(), kind="system", reads=["a"], writes={"b": "answer"}, end=True)
    quiet.edge("a", "b").merge("b", "all")
    assert quiet.spec()["merge"] == {"b": "all"} and quiet.run({"x": 1}).status == "decided"


# --- S274: a threshold that holds no node back says so

def test_a_fully_open_end_to_end_threshold_warns_naming_the_nodes(tmp_path):
    """per_node_fixed's last grid step opens the last node to its smallest calibration signal; on clean labels
    every step certifies and the chosen threshold holds no node back — the end_to_end promise then adds nothing
    over the nodes answering alone, and nothing said so. calibrate() warns, naming the nodes (here over a fixed
    family of three steps — closed, held, fully open — over nodes that always answer alone at 0.9)."""
    from solvi.experimental.process import NodeResult, NodeRunner, ThresholdFamily

    class Confident(NodeRunner):
        kind = "system"

        def __init__(self):
            super().__init__(None, {})

        def run(self, facts):
            return NodeResult("a", {"q": "a"}, "s1", "ok", 0.9)

        def fingerprint(self):
            return "confident/1"

    class ThreeSteps(ThresholdFamily):
        procedure = "fixed"

        def grid(self, spec, signals):
            ss = {n: sorted(v) for n, v in signals.items()}
            return [{n: (max(v) + 1.0 if v else None) for n, v in ss.items()},        # closed
                    {n: ((min(v) + max(v)) / 2 + 0.05 if v else None) for n, v in ss.items()},   # held
                    {n: (min(v) if v else None) for n, v in ss.items()}]              # fully open

        def automatic(self, node, signal, t):
            return signal is not None and t.get(node) is not None and float(signal) >= float(t[node])

        def fingerprint(self):
            return "three-steps/1"

    p = Process("e2e")
    p.node("n1", Confident(), kind="system", reads=["x"], writes={"q": "answer"})
    p.node("n2", Confident(), kind="system", reads=["q"], writes={"r": "answer"}, end=True)
    p.edge("n1", "n2")
    right = [({"x": i}, "a") for i in range(120)]
    p.promise("end_to_end", risk=0.10, delta=0.05, family=ThreeSteps(), examples=right)
    with pytest.warns(UserWarning, match="holds no node back \\(n1, n2 open to their smallest calibration"):
        th = p.calibrate()
    assert th["t"] == {"n1": 0.9, "n2": 0.9} and th["automatic"] == th["n"]    # everything goes through

    wrong = [({"x": i}, "a") for i in range(120)]             # labels wrong often enough that the fully open step
    for i in range(0, 120, 5):                                # cannot certify: the chosen one holds nodes back
        wrong[i] = (wrong[i][0], "b")
    p.promise("end_to_end", risk=0.10, delta=0.05, family=ThreeSteps(), examples=wrong)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                        # a threshold that holds nodes back: no warning
        th2 = p.calibrate()
    assert th2["t"] is not None and th2["automatic"] < th2["n"]
