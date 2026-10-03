"""The Pokémon world-map showcase (spaces/pokemon, examples/23): System 1 + System 2 over a recorded world. Offline and
small: the bundled recording replays, a live run takes the same decisions, the numbers hold, and the Space ships only
plain data."""
import gzip
import io
import json
import re
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1] / "spaces" / "pokemon"
sys.path.insert(0, str(SPACE))

from pokeworld import record as rec  # noqa: E402
from pokeworld.agent import Memory, dispatcher, play, system1  # noqa: E402
from pokeworld.world import World  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return World.load()


def test_the_world_shows_exits_without_destinations_and_gates_them_by_the_story(world):
    assert len(world.places) > 150 and len(world.story) == 15
    assert world.on_offer("PALLET_TOWN", 0) == sorted(world.on_offer("PALLET_TOWN", 0))
    fired = set()
    st = world.take("PALLET_TOWN", "Leave north", 0, fired)
    assert st.to == "OAKS_LAB" and st.event["done"] == "meet_oak"          # Oak walks you to his lab, once
    assert world.take("PALLET_TOWN", "Leave north", 3, fired).to == "ROUTE_1"
    assert world.take("VIRIDIAN_CITY", "Leave west", 3, fired).blocked       # Route 22: later in the story
    with pytest.raises(ValueError, match="not on offer"):
        world.take("PALLET_TOWN", "Leave east", 0, fired)


def test_the_bundled_recording_replays_without_the_game():
    r = rec.replay()
    assert r["ok"], r["mismatches"][:3]
    assert r["decisions"] == r["replayed"] == 245
    assert all(m["verify"] for m in r["maps"].values())


def test_a_live_run_takes_the_recorded_decisions(world, tmp_path):
    s = rec.record(tmp_path, world)
    for name in rec.NAMES:
        assert rec.same_as_recorded(rec._moves(tmp_path / f"{name}.moves.jsonl"), run=name) is None
    assert rec.replay(tmp_path)["ok"]
    assert s["objectives_done"] == {"run1": 15, "run2": 15}


def test_the_second_run_needs_far_fewer_slow_decisions():
    nums = rec.numbers()
    r1, r2 = nums["runs"]["run1"], nums["runs"]["run2"]
    assert (r1["decisions"], r1["s1"], r1["s2"]) == (183, 36, 147)
    assert (r2["decisions"], r2["s1"], r2["s2"]) == (62, 60, 2)
    assert r2["decisions"] == sum(o["shortest"] for o in nums["objectives"])     # the fewest moves, every goal
    assert r2["surprises"] == 2 and r2["human"] == 0
    summary = json.loads((rec.RUNS / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"] == nums["runs"]


def test_system_1_stands_back_when_surprised_and_system_2_takes_over():
    m2 = rec._moves(rec.RUNS / "run2.moves.jsonl")
    for i, m in enumerate(m2):
        if m["surprise"]:
            nxt = m2[i + 1]
            assert nxt["by"] == "s2" and nxt["why"].startswith("surprised:")
    d = dispatcher(storage=rec.open_store(rec.RUNS, "run2")).stored()
    slow = [x for x in d if x.by == "s2"]
    assert len(slow) == 2 and all(x.action == "think" and x.s2.mode == "search" for x in slow)


def test_consolidation_compiles_routes_from_confirmed_claims_only():
    mem = Memory()
    mem.map.see("A", "Leave north", step="t#1")
    mem.map.arrive("A", "Leave north", "B", step="t#1")
    mem.map.told("B", "Door at (1,1)", "C", quote="a sign says so")      # a claim nobody walked
    mem.goals = {"B", "C"}
    rep = mem.consolidate("t#2", "test")
    assert mem.routes["B"] == {"A": ["Leave north", "B", 1]} and mem.routes["C"] == {}
    assert rep["added"] == 1 and mem.s1_route("A", ["B"])["exit"] == "Leave north"


def test_an_llm_hint_that_fails_falls_back_to_no_hint(world):
    """System 2 with an LLM (off by default): an invalid reply is no hint, never a guess — same decisions as without."""
    from solvi.core.deciders.llm import llm

    def broken(req, timeout=None):
        ch = {"index": 0, "message": {"role": "assistant", "content": "not json"}, "finish_reason": "stop"}
        return io.BytesIO(json.dumps({"model": "fake", "choices": [ch],
                                      "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode())
    m = llm("http://fake.invalid/v1", "fake", opener=broken, sleep=lambda s: None, retries=0)
    story = world.story[:8]
    with_llm, _, _ = play(world, Memory(), "run1", llm=m, story=story)
    without, _, _ = play(world, Memory(), "run1", story=story)
    assert [(x.here, x.exit, x.by) for x in with_llm] == [(x.here, x.exit, x.by) for x in without]


def test_the_system_report_reads_the_run_from_its_store():
    text = str(system1().report(store=rec.open_store(rec.RUNS, "run2")))
    assert "62 dispatched decision(s)" in text and "System 1 60" in text


def test_the_space_ships_plain_data_only_and_every_file_it_mounts():
    files = [p for p in SPACE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert sum(p.stat().st_size for p in files) < 5_000_000
    data = [p for p in files if "data" in p.relative_to(SPACE).parts]
    for p in data:
        assert p.suffix in (".json", ".jsonl", ".gz"), p
        raw = gzip.open(p).read() if p.suffix == ".gz" else p.read_bytes()
        raw.decode("utf-8")                                   # text only: no ROM bytes, no images
        for line in raw.splitlines()[:3]:
            json.loads(line) if p.suffix != ".json" else None
    html = (SPACE / "index.html").read_text(encoding="utf-8")
    mounted = set(re.findall(r'<gradio-file name="([^"]+)"', html))
    shipped = {p.relative_to(SPACE).as_posix() for p in files
               if p.suffix in (".py", ".json", ".jsonl", ".gz") and p.parts[-2] != "tools"}
    assert shipped - {"tools/extract_world.py"} <= mounted, shipped - mounted
