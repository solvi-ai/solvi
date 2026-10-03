"""The MCP proxy: `solvi serve --guard catalog.py:guard --upstream CMD` sits between an agent and an MCP server and
checks every tools/call with the guard (tests/mcp_upstream.py is the server)."""
import io
import json
import subprocess
import sys
from pathlib import Path

from solvi.experimental.mcp import run_proxy

UPSTREAM = f"{sys.executable} {Path(__file__).parent / 'mcp_upstream.py'}"
CATALOG = '''
from solvi.solutions.guard import Guard
guard = Guard()
guard.declare("read_file")
guard.declare("write_file", ground=["path"], injections="any")

@guard.policy(["read_file", "write_file"])
def inside_work(path: str) -> bool:
    """Only files under /work."""
    return path.startswith("/work/")

@guard.policy("write_file", on_fail="escalate")
def editors_write(role: str) -> bool:
    """Only editors write without asking."""
    return role == "editor"
'''


def rpc(id_, method, params=None):
    return json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})


def call(id_, name, **args):
    return rpc(id_, "tools/call", {"name": name, "arguments": args})


def guard(tmp_path):
    ns = {}
    exec(CATALOG, ns)
    g = ns["guard"]
    from solvi.core.store import open_storage
    g.storage = open_storage(tmp_path / "calls.jsonl")
    return g


def run(g, lines, upstream=UPSTREAM, **kw):
    out = io.StringIO()
    run_proxy(g, upstream, stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out, **kw)
    return {m["id"]: m for m in map(json.loads, out.getvalue().splitlines()) if "id" in m}, out.getvalue()


def test_proxy_lists_declared_tools_and_guards_each_call(tmp_path):
    g = guard(tmp_path)
    init = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    out, _ = run(g, [init, json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}), rpc(2, "tools/list"),
                     call(3, "read_file", path="/work/notes.txt"),
                     call(4, "read_file", path="/etc/passwd"),
                     call(5, "delete_all"),
                     call(6, "write_file", path="/work/budget.xlsx", content="x"),
                     call(7, "write_file", path="/work/other.txt", content="x"),
                     call(8, "read_file", path="/work/web.html"),
                     call(9, "write_file", path="/work/budget.xlsx", content="x"),
                     rpc(10, "ping"), rpc(11, "resources/list")], facts={"role": "editor"})
    assert out[1]["result"]["serverInfo"]["name"] == "solvi-guard/files"
    assert [t["name"] for t in out[2]["result"]["tools"]] == ["read_file", "write_file"]       # delete_all hidden
    ok = out[3]["result"]
    assert ok["content"][0]["text"].startswith("Meeting at 10") and ok["_meta"]["solvi"]["outcome"] == "allow"
    assert out[4]["result"]["isError"] and "inside_work: Only files under /work." in out[4]["result"]["content"][0]["text"]
    assert out[5]["result"]["isError"] and "unknown tool 'delete_all'" in out[5]["result"]["content"][0]["text"]
    assert out[6]["result"]["content"][0]["text"] == "wrote /work/budget.xlsx"    # the path was in a tool output
    assert "not in the conversation: path='/work/other.txt'" in out[7]["result"]["content"][0]["text"]
    assert out[8]["result"]["_meta"]["solvi"]["outcome"] == "allow"
    esc = out[9]["result"]                                   # after a tool output with instructions: writes escalate
    assert esc["isError"] and esc["structuredContent"]["solvi"]["outcome"] == "escalate"
    assert out[10]["result"] == {} and out[11]["error"]["code"] == -32601
    stored = list(g.storage.iter())
    assert [s.meta["guard"]["outcome"] for s in stored] == ["allow", "deny", "deny", "allow", "deny", "allow", "escalate"]
    assert stored[0].meta["guard"]["executed"] and g.storage.verify()["ok"]
    assert all(g.replay(s.id)["ok"] for s in stored)


def test_escalation_asks_the_user_by_elicitation(tmp_path):
    g = guard(tmp_path)
    init = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {"elicitation": {}}})
    yes = json.dumps({"jsonrpc": "2.0", "id": "solvi-elicit-1", "result": {"action": "accept", "content": {"approve": True}}})
    no = json.dumps({"jsonrpc": "2.0", "id": "solvi-elicit-2", "result": {"action": "decline"}})
    out, raw = run(g, [init, call(2, "read_file", path="/work/notes.txt"),
                       call(3, "write_file", path="/work/budget.xlsx", content="v2"), yes,
                       call(4, "write_file", path="/work/budget.xlsx", content="v3"), no], facts={"role": "viewer"})
    asks = [m for m in map(json.loads, raw.splitlines()) if m.get("method") == "elicitation/create"]
    assert len(asks) == 2 and "editors_write: Only editors write without asking." in asks[0]["params"]["message"]
    assert out[3]["result"]["content"][0]["text"] == "wrote /work/budget.xlsx"
    assert out[4]["result"]["isError"] and "rejected by mcp elicitation" in out[4]["result"]["content"][0]["text"]
    assert [c["answer"] for c in g.storage.corrections()] == ["allow", "deny"]
    (tmp_path / "again").mkdir()
    out, _ = run(guard(tmp_path / "again"), [init, call(2, "read_file", path="/work/notes.txt"),
                 call(3, "write_file", path="/work/budget.xlsx", content="v2")], facts={"role": "viewer"},
                 escalate="deny")
    assert out[3]["result"]["isError"] and "escalated to a person" in out[3]["result"]["content"][0]["text"]


def test_solvi_serve_guard_command(tmp_path):
    cat = tmp_path / "catalog.py"
    cat.write_text(CATALOG)
    lines = [rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}), rpc(2, "tools/list"),
             call(3, "read_file", path="/work/notes.txt"), call(4, "read_file", path="/etc/shadow")]
    p = subprocess.run([sys.executable, "-m", "solvi", "serve", "--guard", f"{cat}:guard", "--upstream", UPSTREAM,
                        "--store", str(tmp_path / "calls.db")], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    out = {m["id"]: m for m in map(json.loads, p.stdout.splitlines())}
    assert len(out[2]["result"]["tools"]) == 2 and not out[3]["result"].get("isError") and out[4]["result"]["isError"]
    from solvi.core.store import open_storage
    assert len(open_storage(tmp_path / "calls.db")) == 2
    bad = subprocess.run([sys.executable, "-m", "solvi", "serve", "--guard", f"{cat}:guard"], capture_output=True,
                         text=True, timeout=60)
    assert bad.returncode == 2 and "--upstream" in bad.stderr


def test_proxy_escalates_the_repeat_of_a_once_tool_and_not_a_call_that_failed(tmp_path):
    from solvi.solutions.guard import Guard
    g = Guard()
    g.declare("write_file", once=True)
    g.declare("read_file", once=True)
    out, _ = run(g, [rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}), rpc(2, "tools/list"),
                     call(3, "write_file", path="/work/a.txt", content="x"),
                     call(4, "write_file", path="/work/a.txt", content="x"),      # the same call again: a person
                     call(5, "write_file", path="/work/a.txt", content="y"),      # other arguments: another call
                     call(6, "read_file", path="/work/none.txt"),                 # the server answers with an error
                     call(7, "read_file", path="/work/none.txt")], escalate="deny")   # ... so it may be tried again
    assert [out[i]["result"]["_meta"]["solvi"]["outcome"] for i in (3, 4, 5, 6, 7)] == \
        ["allow", "escalate", "allow", "allow", "allow"]
    assert "already made" in out[4]["result"]["content"][0]["text"]


def test_proxy_reports_a_dead_upstream_as_an_upstream_error():
    from solvi.solutions.guard import Guard
    g = Guard()
    g.declare("read_file")
    dead = f"{sys.executable} -c pass"                       # a server that exits at once
    out, _ = run(g, [rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}), rpc(2, "tools/list"),
                     call(3, "read_file", path="/work/notes.txt")], upstream=dead)
    for i in (1, 2, 3):
        assert out[i]["error"]["message"].startswith("upstream: the upstream MCP server closed"), out[i]
