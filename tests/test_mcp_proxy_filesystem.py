"""benchmarks/mcp_proxy_filesystem.py: the script's own logic — the scripted MCP client, the session plan, the
comparison of results, the folder-state check, the wire relay, the guard catalog and the malformed messages — against
tests/mcp_upstream.py (the measured run itself needs node and the filesystem server)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = [sys.executable, str(ROOT / "tests" / "mcp_upstream.py")]
MOCK_TOOLS = {"read": "read_file", "write": "write_file"}


def _load():
    spec = importlib.util.spec_from_file_location("mcp_proxy_filesystem", ROOT / "benchmarks" / "mcp_proxy_filesystem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load()
ENV = {**os.environ, "PYTHONWARNINGS": "ignore"}


def proxy_cmd(tmp_path, tools=MOCK_TOOLS, upstream=UPSTREAM, extra=()):
    cat = tmp_path / "catalog.py"
    cat.write_text(B.catalog_source("/work", tools))
    return [sys.executable, "-m", "solvi", "serve", "--guard", f"{cat}:guard", "--upstream", " ".join(upstream),
            "--store", str(tmp_path / "calls.db"), "--escalate", "deny", *extra]


# --- the plan

def test_the_full_session_has_the_registered_counts_and_order():
    steps = B.plan()
    counts = {k: sum(s["kind"] == k for s in steps) for k in B.KINDS}
    assert len(steps) == 60 and counts == {"allowed": 20, "violation": 15, "clean_write": 10, "injected_write": 10,
                                           "malformed": 5}
    assert [s["n"] for s in steps] == list(range(1, 61))
    page = next(s["n"] for s in steps if s.get("args", {}).get("path") == "{W}/page.html")
    assert all(s["n"] < page for s in steps if s["kind"] == "clean_write")            # before the taint
    assert all(s["n"] > page for s in steps if s["kind"] == "injected_write")
    assert all(s["tool"] == "write" for s in steps if s["kind"] == "injected_write")
    for s in steps:                                                                   # the session goes on after each
        if s["kind"] == "malformed":
            assert s["n"] < 60 and steps[s["n"]]["kind"] != "malformed"
    paths = [s["args"]["path"] for s in steps if s["kind"] == "violation"]
    assert any(p.startswith("/") and "{W}" not in p for p in paths)                 # an absolute path elsewhere
    assert any("/../" in p for p in paths) and any("escape" in p for p in paths)     # `..` and a symlink out
    assert any(p.startswith("~") for p in paths) and any(p.startswith("{W}_evil") for p in paths)
    assert {s["shape"] for s in steps if s["kind"] == "malformed"} == {"size_call", "size_garbage", "depth_args",
                                                                     "depth_arrays", "depth_params"}


def test_the_dry_session_has_ten_calls_with_every_kind():
    steps = B.plan(dry=True)
    assert len(steps) == 10 and {s["kind"] for s in steps} == set(B.KINDS)
    page = next(s["n"] for s in steps if s.get("args", {}).get("path") == "{W}/page.html")
    assert all(s["n"] < page for s in steps if s["kind"] == "clean_write")
    assert all(s["n"] > page for s in steps if s["kind"] == "injected_write")
    assert steps[-1]["kind"] != "malformed"


def test_materialize_fills_the_placeholders():
    assert B.materialize({"path": "{W}/../{ON}/x", "head": 2}, "/a/work", "outside") == \
        {"path": "/a/work/../outside/x", "head": 2}


# --- the fixture and the folder state

def test_the_injected_fixture_is_the_upstream_text():
    t = B.injected_text()
    assert t.startswith("Great recipes!") and "ignore previous instructions" in t


def test_folder_state_sees_files_dirs_and_links_without_following_them(tmp_path):
    for side in ("p", "d"):
        B.make_fixture(tmp_path / f"work_{side}", tmp_path / f"outside_{side}")
    sp = {str(tmp_path / "work_p"): "{W}", str(tmp_path / "outside_p"): "{O}"}
    sd = {str(tmp_path / "work_d"): "{W}", str(tmp_path / "outside_d"): "{O}"}
    a, b = B.tree_state(tmp_path / "work_p", sp), B.tree_state(tmp_path / "work_d", sd)
    assert a == b and a["escape"] == ["link", "{O}"] and a["escape_file"] == ["link", "{O}/secret.txt"]
    assert a["sub"] == ["dir"] and a["notes.txt"][0] == "file" and "escape/secret.txt" not in a
    (tmp_path / "work_p" / "sub" / "a.txt").write_text("changed")
    assert B.tree_state(tmp_path / "work_p", sp) != b
    (tmp_path / "outside_d" / "new.txt").write_text("x")              # outside the folder: not seen through the link
    assert B.tree_state(tmp_path / "work_d", sd) == b


def test_the_policy_resolves_dots_symlinks_home_and_prefixes(tmp_path):
    work, outside, evil = tmp_path / "w", tmp_path / "o", tmp_path / "w_evil"
    B.make_fixture(work, outside, evil)
    ns = {}
    exec(B.catalog_source(work), ns)
    g = ns["guard"]
    assert sorted(g.tools) == sorted(B.TOOLS.values())
    inside = ns["inside_work"]
    assert inside(f"{work}/notes.txt") and inside(f"{work}/sub/../notes.txt") and inside(f"{work}/new/x.txt")
    assert inside(f"{work}") and inside(f"{work}/")
    for bad in (f"{work}/../o/secret.txt", f"{work}/escape/secret.txt", f"{work}/escape_file", f"{work}_evil/secret.txt",
                "/etc/passwd", "~/.bashrc", "notes.txt", f"{work}/escape/new"):
        assert not inside(bad), bad


# --- the comparison

def test_compare_drops_only_the_proxy_meta_and_the_folder_paths():
    direct = {"content": [{"type": "text", "text": "Successfully wrote to /x/work_d/a.txt"}]}
    proxied = {"content": [{"type": "text", "text": "Successfully wrote to /x/work_p/a.txt"}],
               "_meta": {"solvi": {"outcome": "allow", "stored_id": "1", "trace_hash": "h"}}}
    c = B.compare(proxied, direct, {"/x/work_p": "{W}"}, {"/x/work_d": "{W}"})
    assert c == {"identical": True, "raw": False}
    same = {"content": [{"type": "text", "text": "alpha"}]}
    assert B.compare({**same, "_meta": {"solvi": {}}}, same) == {"identical": True, "raw": True}
    assert not B.compare({**same, "_meta": {"solvi": {}, "other": 1}}, same)["identical"]      # other meta is kept
    assert not B.compare({"content": [{"type": "text", "text": "alpha "}]}, same)["identical"]
    assert not B.compare({**same, "isError": True}, same)["identical"]
    assert B.canonical({"b": 1, "a": [1, {"d": 2, "c": "é"}]}) == '{"a":[1,{"c":"é","d":2}],"b":1}'
    assert B.outcome({"_meta": {"solvi": {"outcome": "deny"}}}) == "deny" and B.outcome({}) is None


# --- the client, the relay and the proxy

def test_the_client_speaks_to_the_server_directly():
    c = B.Client(UPSTREAM, env=ENV, timeout=30)
    try:
        init = c.initialize()
        assert init["result"]["serverInfo"]["name"] == "files"
        (tl, ms) = c.request("tools/list")
        assert [t["name"] for t in tl["result"]["tools"]] == ["read_file", "write_file", "delete_all"] and ms > 0
        r, _ = c.call("read_file", {"path": "/work/notes.txt"})
        assert r["result"]["content"][0]["text"].startswith("Meeting at 10")
        r, _ = c.request("nothing/here")
        assert r["error"]["code"] == -32601
    finally:
        assert c.close() == 0


def test_the_client_reports_a_dead_process_instead_of_hanging():
    c = B.Client([sys.executable, "-c", "pass"], env=ENV, timeout=10)
    with pytest.raises(B.ClientError):
        c.request("ping")
    c.close()


def test_the_relay_logs_what_the_server_receives_and_sends(tmp_path):
    log = tmp_path / "wire.jsonl"
    c = B.Client(B.relay_command(log, UPSTREAM), env=ENV, timeout=30)
    try:
        c.initialize()
        r, _ = c.call("write_file", {"path": "/work/a.txt", "content": "x"})
        assert r["result"]["content"][0]["text"] == "wrote /work/a.txt"
    finally:
        assert c.close() == 0
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert {r["dir"] for r in rows} == {"in", "out"}
    assert B.wire_calls(log) == [("write_file", {"path": "/work/a.txt", "content": "x"})]
    assert B.wire_calls(tmp_path / "none.jsonl") == []


def test_through_the_proxy_allowed_results_match_and_blocked_calls_stay_off_the_wire(tmp_path):
    log = tmp_path / "wire.jsonl"
    p = B.Client(proxy_cmd(tmp_path, upstream=B.relay_command(log, UPSTREAM)), env=ENV, timeout=60)
    d = B.Client(UPSTREAM, env=ENV, timeout=30)
    try:
        p.initialize(), d.initialize()
        p.request("tools/list")
        rp, _ = p.call("read_file", {"path": "/work/notes.txt"})
        rd, _ = d.call("read_file", {"path": "/work/notes.txt"})
        assert B.outcome(rp["result"]) == "allow" and B.compare(rp["result"], rd["result"])["identical"]
        rp, _ = p.call("read_file", {"path": "/work/../etc/passwd"})
        assert B.outcome(rp["result"]) == "deny" and rp["result"]["isError"]
        rp, _ = p.call("read_file", {"path": "/work/web.html"})                       # the injected page
        rp, _ = p.call("write_file", {"path": "/work/x.txt", "content": "y"})
        assert B.outcome(rp["result"]) == "escalate" and rp["result"]["isError"]
    finally:
        assert p.close() == 0
        d.close()
    assert B.wire_calls(log) == [("read_file", {"path": "/work/notes.txt"}), ("read_file", {"path": "/work/web.html"})]


@pytest.mark.parametrize("shape", ["size_call", "size_garbage", "depth_args", "depth_params", "depth_arrays"])
def test_each_malformed_message_is_over_the_limits_and_the_proxy_answers_and_goes_on(tmp_path, shape):
    line = B.malformed_line(shape, 5000, 8, 1)
    if shape.startswith("size"):
        assert len(line) > 5000
    else:
        assert len(line) <= 5000 or shape == "depth_arrays"
    log = tmp_path / "wire.jsonl"
    p = B.Client(proxy_cmd(tmp_path, upstream=B.relay_command(log, UPSTREAM),
                           extra=("--max-body", "5000" if shape != "depth_arrays" else "1000000", "--max-depth", "8")),
                 env=ENV, timeout=60)
    try:
        p.initialize()
        msg, _ = p.exchange(line, None)
        assert msg["error"]["code"] in (-32600, -32700) and p.alive()
        r, _ = p.call("read_file", {"path": "/work/notes.txt"})
        assert B.outcome(r["result"]) == "allow"
    finally:
        assert p.close() == 0
    assert B.wire_calls(log) == [("read_file", {"path": "/work/notes.txt"})]


def test_stored_checks_count_verify_and_replay_the_proxy_store(tmp_path):
    p = B.Client(proxy_cmd(tmp_path), env=ENV, timeout=60)
    rows = []
    try:
        p.initialize()
        tl, _ = p.request("tools/list")
        for args in ({"path": "/work/notes.txt"}, {"path": "/etc/passwd"}):
            r, _ = p.call("read_file", args)
            m = r["result"]["_meta"]["solvi"]
            rows.append({"outcome": m["outcome"], "stored_id": m["stored_id"]})
    finally:
        p.close()
    s = B._stored_checks(tmp_path / "catalog.py", tmp_path / "calls.db", tl["result"]["tools"], rows)
    assert s["records"] == 2 and s["every_call_stored"] and s["verify_ok"] and s["replay_ok"] == 2 and s["outcomes_match"]
    assert s["outcomes"] == {"allow": 1, "deny": 1}
    rows[1]["outcome"] = "allow"
    assert not B._stored_checks(tmp_path / "catalog.py", tmp_path / "calls.db", tl["result"]["tools"], rows)["outcomes_match"]
    assert not B._stored_checks(tmp_path / "catalog.py", tmp_path / "calls.db", tl["result"]["tools"],
                                rows[:1])["every_call_stored"]
