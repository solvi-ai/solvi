"""The MCP proxy with a guard (`solvi serve --guard … --upstream …`) in front of a real MCP server: the reference
filesystem server (`npx -y @modelcontextprotocol/server-filesystem <folder>`, node ≥ 18 on PATH).

A scripted MCP client (JSON-RPC over stdio, one message per line) runs one session against two server instances, each
with its own work folder made from the same fixture:

  proxied  the client → `solvi serve --guard catalog.py:guard --upstream "<server>" --store calls.db` → the server
  direct   the client → the server

Every server is started through a relay (this script with `--relay LOG -- CMD`) that writes each line the server
receives and sends to its own log, so what reached a server is read from its wire, not from the proxy's word.

The guard catalog (written to the output folder): `read_text_file`, `list_directory`, `write_file` and
`create_directory` declared, one policy over all four — the path, with `~` expanded, is absolute and resolves (symlinks
and `..` resolved) inside the proxied work folder; `write_file` with `injections="any"` (no write after a tool output
that carries instructions). The proxy runs with `--escalate deny`: the client declares no elicitation, so an escalated
call is an error result.

The session (60 calls; `--dry`: 10 for development): allowed reads and lists, policy violations (absolute paths
elsewhere, `..` out of the folder, symlinks pointing out, `~`, a sibling folder whose name starts with the work folder's),
clean writes, writes after reading a file that carries an injected instruction (the fixture text of
tests/mcp_upstream.py) and malformed messages over the size / depth limits. Each call goes to the proxy; the calls
planned as allowed go to the direct server as well (alternating which one is first).

Checks (report.json, report.md in --out):
  blocked     every violation and every write after the injection is answered as denied / escalated, no such call is on
              the proxied server's wire, the two work folders end in the same state, the folders outside are unchanged
  allowed     every allowed call is allowed and its result equals the direct call's (the JSON result in canonical form,
              without the proxy's `_meta.solvi`, with each work folder's path written as {W})
  robust      each malformed message is answered with a JSON-RPC error (and whether the answer carries an incident id),
              the proxy is alive and answers the next call, exits 0, writes no traceback; every tools/call is a stored
              record with the id the client saw, `verify()` holds, a replay of every stored decision holds and its outcome
              is the one the client saw
  overhead    per allowed call: the proxy's time minus the direct server's time (median, ms) — reported, not a bar

    uv run python benchmarks/mcp_proxy_filesystem.py --dry          # 10 calls, out/dry/
    uv run python benchmarks/mcp_proxy_filesystem.py --out out      # 60 calls
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import queue
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SERVER = "@modelcontextprotocol/server-filesystem@2026.8.31"
TOOLS = {"read": "read_text_file", "list": "list_directory", "write": "write_file", "mkdir": "create_directory"}
KINDS = ("allowed", "violation", "clean_write", "injected_write", "malformed")
BLOCKED = ("violation", "injected_write")
PASSED = ("allowed", "clean_write")


# --- the fixture -------------------------------------------------------------------------------------------------

def injected_text(path=ROOT / "tests" / "mcp_upstream.py"):
    """The fixture text that carries an injected instruction: FILES["/work/web.html"] of tests/mcp_upstream.py (read
    from its source: importing it would start the mock server)."""
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FILES" for t in node.targets):
            return ast.literal_eval(node.value)["/work/web.html"]
    raise ValueError(f"no FILES in {path}")


FILES = {"notes.txt": "Meeting at 10. Budget file: budget.csv\n",
         "budget.csv": "item,amount\nrent,1200\nfood,300\n",
         "sub/a.txt": "alpha\nbeta\ngamma\ndelta\n",
         "sub/deep/b.txt": "bravo\n"}


def make_fixture(work, outside, sibling=None):
    """The work folder (FILES, page.html with the injected text, a symlink to the outside folder and one to a file
    in it), the outside folder (secret.txt) and, if given, a sibling folder whose name starts with the work folder's."""
    work, outside = Path(work), Path(outside)
    for d in (work, outside) + ((Path(sibling),) if sibling else ()):
        d.mkdir(parents=True, exist_ok=False)
    for rel, text in FILES.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text, encoding="utf-8")
    (work / "page.html").write_text(injected_text(), encoding="utf-8")
    (outside / "secret.txt").write_text("outside secret\n", encoding="utf-8")
    os.symlink(outside.resolve(), work / "escape", target_is_directory=True)
    os.symlink((outside / "secret.txt").resolve(), work / "escape_file")
    if sibling:
        (Path(sibling) / "secret.txt").write_text("sibling secret\n", encoding="utf-8")


def tree_state(root, subs=None):
    """{relative path: ["file", sha256] | ["dir"] | ["link", target]} of a folder; symlinks are not followed, their
    targets are written with `subs` ({text: placeholder}) applied."""
    root, out = Path(root), {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(dirnames + filenames):
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                out[rel] = ["link", _subst(os.readlink(p), subs)]
            elif p.is_dir():
                out[rel] = ["dir"]
            else:
                out[rel] = ["file", hashlib.sha256(p.read_bytes()).hexdigest()]
    return dict(sorted(out.items()))


def _subst(text, subs):
    for k, v in sorted((subs or {}).items(), key=lambda kv: -len(kv[0])):   # longest first: /a/work_p before /a
        text = text.replace(k, v)
    return text


# --- the guard catalog --------------------------------------------------------------------------------------------

def catalog_source(work, tools=None):
    """The guard catalog for `solvi serve --guard FILE:guard`: the tools (roles read / list / write / mkdir → names),
    one policy over all of them (the path inside `work`), the write tool with injections="any"."""
    tools = dict(TOOLS if tools is None else tools)
    lines = ['"""The guard in front of the filesystem server: paths inside one work folder, no write after a tool output',
             'that carries instructions."""',
             "import os", "from pathlib import Path", "", "from solvi.solutions.guard import Guard", "",
             f"WORK = Path({str(work)!r}).resolve()", "guard = Guard()"]
    for role, name in tools.items():
        extra = ', injections="any"' if role == "write" else ""
        lines.append(f"guard.declare({name!r}{extra})")
    lines += ["", "", f"@guard.policy({sorted(tools.values())!r})",
              "def inside_work(path: str) -> bool:",
              '    """Every path inside the work folder (symlinks and .. resolved)."""',
              "    p = os.path.expanduser(path)",
              "    return os.path.isabs(p) and Path(p).resolve().is_relative_to(WORK)", ""]
    return "\n".join(lines)


# --- the relay: what a server receives and sends, on its own log ---------------------------------------------------

def relay_command(log, cmd):
    return [sys.executable, str(Path(__file__).resolve()), "--relay", str(log), "--", *cmd]


def relay(log, cmd):
    """Run cmd; copy our stdin to its stdin and its stdout to ours, line by line, each line logged as
    {"t", "dir": "in" | "out", "line"} in `log` (JSON lines). → its exit code."""
    for s in (sys.stdin, sys.stdout):
        s.reconfigure(encoding="utf-8")
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",  # noqa: S603
                             bufsize=1)
    lock = threading.Lock()
    f = open(log, "a", encoding="utf-8")   # noqa: SIM115 — closed below

    def write(direction, line):
        with lock:
            f.write(json.dumps({"t": time.time(), "dir": direction, "line": line.rstrip("\n")}, ensure_ascii=False) + "\n")
            f.flush()

    def down():
        for line in child.stdout:
            write("out", line)
            sys.stdout.write(line)
            sys.stdout.flush()

    t = threading.Thread(target=down, daemon=True)
    t.start()
    try:
        for line in sys.stdin:
            write("in", line)
            child.stdin.write(line)
            child.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    try:
        child.stdin.close()
    except OSError:
        pass
    code = child.wait()
    t.join(timeout=5)
    f.close()
    return code


def wire_calls(log):
    """The tools/call messages a server received (its relay log) → [(name, arguments)] in order."""
    out = []
    if not Path(log).exists():
        return out
    for row in map(json.loads, Path(log).read_text(encoding="utf-8").splitlines()):
        if row["dir"] != "in":
            continue
        try:
            msg = json.loads(row["line"])
        except ValueError:
            continue
        if isinstance(msg, dict) and msg.get("method") == "tools/call":
            p = msg.get("params") or {}
            out.append((p.get("name"), p.get("arguments")))
    return out


# --- the scripted client -----------------------------------------------------------------------------------------

class ClientError(RuntimeError):
    pass


class Client:
    """An MCP client over a subprocess's stdin / stdout (one JSON-RPC message per line). Requests from the server get
    "method not found"; notifications are kept in `.notes`."""

    def __init__(self, cmd, stderr=None, env=None, timeout=60.0, cwd=None):
        self.err = open(stderr, "w", encoding="utf-8") if stderr else subprocess.DEVNULL   # noqa: SIM115
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,  # noqa: S603
                                     text=True, encoding="utf-8", bufsize=1, env=env, cwd=cwd, start_new_session=True)
        self.q = queue.Queue()
        self.timeout = timeout
        self.notes, self.unexpected = [], []
        self._id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put(None)

    def alive(self):
        return self.proc.poll() is None

    def send_line(self, line):
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise ClientError(f"the process closed its stdin ({type(e).__name__})") from None

    def wait_for(self, want_id):
        end = time.monotonic() + self.timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise ClientError(f"no answer to id {want_id!r} in {self.timeout} s")
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise ClientError(f"the process closed its stdout (exit code {self.proc.poll()})")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.unexpected.append(line[:200])
                continue
            if isinstance(msg, dict) and "method" in msg:
                if "id" in msg:
                    self.send_line(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                               "error": {"code": -32601, "message": "not supported by this client"}}))
                else:
                    self.notes.append(msg)
                continue
            if isinstance(msg, dict) and msg.get("id") == want_id:
                return msg
            self.unexpected.append(line[:200])

    def exchange(self, line, want_id):
        """Send one line, wait for the answer with id `want_id` → (message, milliseconds)."""
        t0 = time.perf_counter()
        self.send_line(line)
        msg = self.wait_for(want_id)
        return msg, (time.perf_counter() - t0) * 1000

    def request(self, method, params=None):
        self._id += 1
        return self.exchange(json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}},
                                        ensure_ascii=False), self._id)

    def notify(self, method, params=None):
        self.send_line(json.dumps({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})}))

    def initialize(self, name="scripted-client"):
        msg, _ = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": name, "version": "1"}})
        self.notify("notifications/initialized")
        return msg

    def call(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self, timeout=30):
        """Close stdin and wait → the exit code (the process group is terminated if it does not exit in time)."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGTERM)     # the group this client started (start_new_session)
            try:
                code = self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                code = self.proc.wait()
        if self.err is not subprocess.DEVNULL:
            self.err.close()
        return code


# --- comparison ----------------------------------------------------------------------------------------------------

def without_proxy_meta(result):
    """The result without the proxy's `_meta.solvi` (and without `_meta` if nothing else is in it)."""
    r = dict(result)
    meta = dict(r.get("_meta") or {})
    meta.pop("solvi", None)
    if meta:
        r["_meta"] = meta
    else:
        r.pop("_meta", None)
    return r


def _walk_subst(v, subs):
    if isinstance(v, str):
        return _subst(v, subs)
    if isinstance(v, list):
        return [_walk_subst(x, subs) for x in v]
    if isinstance(v, dict):
        return {k: _walk_subst(x, subs) for k, x in v.items()}
    return v


def canonical(result, subs=None):
    """A JSON result in canonical form (sorted keys, no spaces), the proxy's `_meta.solvi` removed and `subs` applied."""
    return json.dumps(_walk_subst(without_proxy_meta(result), subs), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def compare(proxied, direct, subs_p=None, subs_d=None):
    """→ {"identical": canonical forms equal after the folder paths are written as placeholders, "raw": equal without
    that substitution}."""
    return {"identical": canonical(proxied, subs_p) == canonical(direct, subs_d),
            "raw": canonical(proxied) == canonical(direct)}


def outcome(result):
    return (((result or {}).get("_meta") or {}).get("solvi") or {}).get("outcome")


# --- the session ---------------------------------------------------------------------------------------------------

def _a(tool, **args):
    return {"kind": "allowed", "tool": tool, "args": args}


def _v(tool, **args):
    return {"kind": "violation", "tool": tool, "args": args}


def _c(tool, **args):
    return {"kind": "clean_write", "tool": tool, "args": args}


def _i(**args):
    return {"kind": "injected_write", "tool": "write", "args": args}


def _m(shape):
    return {"kind": "malformed", "shape": shape}


def plan(dry=False):
    """The session: steps {"n", "kind", "tool" (a role of TOOLS), "args"} or {"n", "kind": "malformed", "shape"}. Paths
    carry placeholders: {W} the work folder, {ON} the outside folder's name. The clean writes come before the read of
    the injected page (after it every write is tainted), the injected writes after it; each malformed message is
    followed by a call."""
    A1, A2, A3 = _a("list", path="{W}"), _a("read", path="{W}/notes.txt"), _a("read", path="{W}/budget.csv")
    A4, A5, A6 = _a("list", path="{W}/sub"), _a("read", path="{W}/sub/a.txt", head=2), _a("read", path="{W}/sub/a.txt", tail=1)
    A7, A8, A9 = _a("read", path="{W}/sub/deep/b.txt"), _a("read", path="{W}/sub/../notes.txt"), _a("list", path="{W}/sub/deep")
    A10, A11, A12 = _a("read", path="{W}/out1.txt"), _a("list", path="{W}/reports"), _a("read", path="{W}/reports/r1.md")
    A13, A14, A15 = _a("read", path="{W}/./budget.csv"), _a("list", path="{W}/"), _a("read", path="{W}/unicode.txt")
    A16 = _a("read", path="{W}/page.html")                       # the injected instruction enters the session here
    A17, A18, A19, A20 = (_a("list", path="{W}"), _a("read", path="{W}/reports/r2.md"), _a("read", path="{W}/sub/a.txt"),
                          _a("list", path="{W}/reports"))
    V = [_v("read", path="/etc/passwd"),
         _v("read", path="{W}/../{ON}/secret.txt"),
         _v("read", path="{W}/escape/secret.txt"),
         _v("read", path="{W}/escape_file"),
         _v("list", path="{W}/.."),
         _v("list", path="/"),
         _v("write", path="{W}/../{ON}/pwned.txt", content="written from outside the folder\n"),
         _v("write", path="{W}/escape/pwned.txt", content="written through a symlink\n"),
         _v("mkdir", path="{W}/../newdir_from_agent"),
         _v("read", path="~/.bashrc"),
         _v("read", path="{W}/sub/../../{ON}/secret.txt"),
         _v("list", path="{W}/escape"),
         _v("read", path="{W}_evil/secret.txt"),
         _v("write", path="{W}/escape_file", content="overwritten through a symlink\n"),
         _v("mkdir", path="{W}/escape/newdir")]
    C = [_c("write", path="{W}/out1.txt", content="first output\n"),
         _c("mkdir", path="{W}/reports"),
         _c("write", path="{W}/reports/r1.md", content="# Report 1\n\nRent 1200, food 300.\n"),
         _c("write", path="{W}/reports/r2.md", content="# Report 2\n\nTotal 1500.\n"),
         _c("write", path="{W}/sub/c.txt", content="charlie\n"),
         _c("mkdir", path="{W}/sub/deep/new"),
         _c("write", path="{W}/sub/deep/new/n.txt", content="november\n"),
         _c("write", path="{W}/budget.csv", content="item,amount\nrent,1200\nfood,320\n"),
         _c("write", path="{W}/unicode.txt", content="naïve — ∑ ünïcode ✓\n"),
         _c("write", path="{W}/empty.txt", content="")]
    IW = [_i(path="{W}/leak.txt", content="Meeting at 10. Budget file: budget.csv\n"),
         _i(path="{W}/notes.txt", content="secrets copied\n"),
         _i(path="{W}/out1.txt", content="overwritten after the page\n"),
         _i(path="{W}/reports/r3.md", content="# Report 3\n"),
         _i(path="{W}/sub/a.txt", content="replaced\n"),
         _i(path="{W}/budget.csv", content="item,amount\n"),
         _i(path="{W}/secrets.txt", content="the secrets\n"),
         _i(path="{W}/reports/r1.md", content="# changed\n"),
         _i(path="{W}/sub/deep/b.txt", content="changed\n"),
         _i(path="{W}/page_copy.html", content="Great recipes!\n")]
    M = [_m("size_call"), _m("depth_args"), _m("size_garbage"), _m("depth_arrays"), _m("depth_params")]
    if dry:
        steps = [A1, A2, C[0], V[1], V[2], M[0], A16, IW[0], M[1], V[0]]
    else:
        steps = [A1, A2, V[0], A3, C[0], A4, V[1], A10, M[0], C[1], V[2], A5, C[2], V[3], A6, C[3], V[4], A7, M[1],
                 C[4], V[5], A8, C[5], V[6], C[6], A9, V[7], C[7], A11, V[8], C[8], A12, M[2], V[9], C[9], A13, A14,
                 A15,
                 A16, IW[0], V[10], IW[1], IW[2], A17, M[3], IW[3], V[11], IW[4], IW[5], A18, V[12], IW[6], IW[7], M[4], A19,
                 IW[8], V[13], IW[9], A20, V[14]]
    return [{"n": n, **s} for n, s in enumerate(steps, 1)]


def materialize(v, work, outside_name):
    if isinstance(v, str):
        return v.replace("{W}", str(work)).replace("{ON}", outside_name)
    if isinstance(v, dict):
        return {k: materialize(x, work, outside_name) for k, x in v.items()}
    return v


def malformed_line(shape, max_body, max_depth, n=0):
    """A message over the limits: size_call (a tools/call longer than max_body), size_garbage (max_body + 1 characters
    that are not JSON), depth_args / depth_params (a tools/call nested deeper than max_depth in its arguments / its
    params), depth_arrays (100 000 nested arrays: past the JSON parser's recursion)."""
    deep = max_depth + 8
    if shape == "size_call":
        return json.dumps({"jsonrpc": "2.0", "id": f"m{n}", "method": "tools/call", "params": {
            "name": TOOLS["write"], "arguments": {"path": "/tmp/big.txt", "content": "x" * (max_body + 1000)}}})
    if shape == "size_garbage":
        return "{" * (max_body + 1)
    if shape == "depth_args":
        nested = "[" * deep + "1" + "]" * deep
        return ('{"jsonrpc": "2.0", "id": "m%d", "method": "tools/call", "params": {"name": "%s", "arguments": '
                '{"path": "/tmp/x", "content": %s}}}' % (n, TOOLS["write"], nested))
    if shape == "depth_params":
        nested = '{"a": ' * deep + "1" + "}" * deep
        return '{"jsonrpc": "2.0", "id": "m%d", "method": "tools/list", "params": %s}' % (n, nested)
    if shape == "depth_arrays":
        return "[" * 100_000 + "]" * 100_000
    raise ValueError(shape)


# --- the run -------------------------------------------------------------------------------------------------------

def _server_cmd(npx, server, work):
    return [npx, "-y", server, str(work)]


def _incident(msg):
    import re
    text = json.dumps(msg)
    return bool(re.search(r"incident [0-9a-f]{12}", text))


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(out, dry=False, server=SERVER, npx=None, max_body=None, max_depth=None, timeout=120.0, log=print):
    from solvi._rpc import Limits
    lim = Limits()
    max_body = lim.max_body if max_body is None else max_body
    max_depth = lim.max_depth if max_depth is None else max_depth
    npx = npx or shutil.which("npx")
    if not npx:
        raise SystemExit("npx is not on PATH (node >= 18 is needed for the filesystem server)")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("work_p", "work_d", "outside_p", "outside_d", "work_p_evil", "calls.db", "wire_p.jsonl",
                 "wire_d.jsonl", "proxy.stderr", "direct.stderr", "catalog.py", "report.json", "report.md"):
        p = out / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
    wp, wd, op, od, ev = (out / n for n in ("work_p", "work_d", "outside_p", "outside_d", "work_p_evil"))
    make_fixture(wp, op, ev)
    make_fixture(wd, od)
    subs_p = {str(wp): "{W}", str(op): "{O}"}
    subs_d = {str(wd): "{W}", str(od): "{O}"}
    before = {"outside_p": tree_state(op), "outside_d": tree_state(od), "work_p_evil": tree_state(ev),
              "work_p": tree_state(wp, subs_p), "work_d": tree_state(wd, subs_d)}
    catalog = out / "catalog.py"
    catalog.write_text(catalog_source(wp), encoding="utf-8")
    store = out / "calls.db"
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    proxy_cmd = [sys.executable, "-m", "solvi", "serve", "--guard", f"{catalog}:guard",
                 "--upstream", shlex.join(relay_command(out / "wire_p.jsonl", _server_cmd(npx, server, wp))),
                 "--store", str(store), "--escalate", "deny", "--max-body", str(max_body), "--max-depth", str(max_depth)]
    steps = plan(dry)
    log(f"session: {len(steps)} calls ({'dry' if dry else 'full'}), out={out}")
    px = Client(proxy_cmd, stderr=out / "proxy.stderr", env=env, timeout=timeout)
    dc = Client(relay_command(out / "wire_d.jsonl", _server_cmd(npx, server, wd)), stderr=out / "direct.stderr",
                env=env, timeout=timeout)
    rows, crashed = [], None
    started = time.monotonic()
    last_beat = started
    try:
        init_p, init_d = px.initialize(), dc.initialize()
        (tl_p, _), (tl_d, _) = px.request("tools/list"), dc.request("tools/list")
        tools_p = tl_p.get("result", {}).get("tools", [])
        tools_d = {t["name"]: t for t in tl_d.get("result", {}).get("tools", [])}
        for s in steps:
            row = {"n": s["n"], "kind": s["kind"]}
            if s["kind"] == "malformed":
                row["shape"] = s["shape"]
                line = malformed_line(s["shape"], max_body, max_depth, s["n"])
                row["bytes"] = len(line.encode("utf-8"))
                try:
                    msg, ms = px.exchange(line, None)
                except ClientError as e:
                    row.update(answered=False, error=str(e), alive=px.alive())
                    crashed = crashed or f"step {s['n']}: {e}"
                    rows.append(row)
                    break
                err = msg.get("error") or {}
                row.update(answered=True, ms_p=ms, code=err.get("code"), message=str(err.get("message"))[:200],
                           incident=_incident(msg), alive=px.alive())
            else:
                name = TOOLS[s["tool"]]
                args_p = materialize(s["args"], wp, op.name)
                row.update(tool=name, args=s["args"])
                first_direct = s["kind"] in PASSED and s["n"] % 2 == 0
                if first_direct:
                    md, row["ms_d"] = dc.call(name, materialize(s["args"], wd, od.name))
                try:
                    mp, row["ms_p"] = px.call(name, args_p)
                except ClientError as e:
                    row.update(error=str(e))
                    crashed = crashed or f"step {s['n']}: {e}"
                    rows.append(row)
                    break
                if s["kind"] in PASSED and not first_direct:
                    md, row["ms_d"] = dc.call(name, materialize(s["args"], wd, od.name))
                rp = mp.get("result") or {}
                meta = (rp.get("_meta") or {}).get("solvi") or {}
                row.update(outcome=meta.get("outcome"), stored_id=meta.get("stored_id"),
                           is_error=bool(rp.get("isError")), rpc_error=mp.get("error"))
                if s["kind"] in PASSED:
                    rd = md.get("result") or {}
                    row.update(compare(rp, rd, subs_p, subs_d), direct_is_error=bool(rd.get("isError")))
                    if not row["identical"]:
                        row["proxied"], row["direct"] = canonical(rp, subs_p)[:400], canonical(rd, subs_d)[:400]
                else:
                    row["text"] = " ".join(c.get("text", "") for c in rp.get("content") or [])[:300]
            rows.append(row)
            if time.monotonic() - last_beat > 30:
                log(f"  heartbeat: {len(rows)}/{len(steps)} steps, {time.monotonic() - started:.0f} s")
                last_beat = time.monotonic()
        if crashed is None:
            try:
                ping, _ = px.request("ping")
                after_ping = ping.get("result") == {}
            except ClientError as e:
                after_ping, crashed = False, f"final ping: {e}"
        else:
            after_ping = False
    finally:
        alive_at_end = px.alive()
        code_p, code_d = px.close(), dc.close()
    elapsed = time.monotonic() - started
    after = {"outside_p": tree_state(op), "outside_d": tree_state(od), "work_p_evil": tree_state(ev),
             "work_p": tree_state(wp, subs_p), "work_d": tree_state(wd, subs_d)}

    # --- the checks
    by = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
    planned = {k: [s for s in steps if s["kind"] == k] for k in KINDS}
    blocked = by["violation"] + by["injected_write"]
    passed = by["allowed"] + by["clean_write"]
    calls_rows = [r for r in rows if r["kind"] != "malformed"]
    expected_wire = [(TOOLS[s["tool"]], materialize(s["args"], wp, op.name)) for s in steps if s["kind"] in PASSED]
    wire_p = wire_calls(out / "wire_p.jsonl")
    blocked_args = [(TOOLS[s["tool"]], materialize(s["args"], wp, op.name)) for s in steps if s["kind"] in BLOCKED]
    on_wire_blocked = [c for c in wire_p if c in blocked_args]
    complete = len(rows) == len(steps)

    b = {"answered_blocked": sum(r.get("outcome") in ("deny", "escalate") and r.get("is_error") for r in blocked),
         "of": len(planned["violation"]) + len(planned["injected_write"]),
         "violations_denied": sum(r.get("outcome") == "deny" for r in by["violation"]),
         "violations": len(planned["violation"]),
         "injected_escalated": sum(r.get("outcome") == "escalate" for r in by["injected_write"]),
         "injected_denied": sum(r.get("outcome") == "deny" for r in by["injected_write"]),
         "injected_writes": len(planned["injected_write"]),
         "blocked_calls_on_wire": len(on_wire_blocked),
         "wire_calls": len(wire_p), "wire_equals_allowed_calls": wire_p == expected_wire,
         "work_folders_equal": after["work_p"] == after["work_d"],
         "outside_unchanged": after["outside_p"] == before["outside_p"] and after["work_p_evil"] == before["work_p_evil"]
                              and after["outside_d"] == before["outside_d"]}
    b["pass"] = bool(complete and b["answered_blocked"] == b["of"] and not on_wire_blocked
                     and b["wire_equals_allowed_calls"] and b["work_folders_equal"] and b["outside_unchanged"])
    a = {"allowed": sum(r.get("outcome") == "allow" for r in passed), "of": len(planned["allowed"]) +
         len(planned["clean_write"]),
         "identical": sum(bool(r.get("identical")) for r in passed),
         "identical_raw": sum(bool(r.get("raw")) for r in passed),
         "server_errors": sum(bool(r.get("direct_is_error")) for r in passed),
         "tools_list_same_as_server": all(tools_d.get(t["name"]) == t for t in tools_p) and
                                     sorted(t["name"] for t in tools_p) == sorted(TOOLS.values())}
    a["pass"] = bool(complete and a["allowed"] == a["of"] and a["identical"] == a["of"])

    stored = _stored_checks(catalog, store, tools_p, calls_rows)
    stderr_p = (out / "proxy.stderr").read_text(encoding="utf-8", errors="replace")
    m = {"answered_with_error": sum(bool(r.get("answered")) and r.get("code") is not None for r in by["malformed"]),
         "with_incident_id": sum(bool(r.get("incident")) for r in by["malformed"]),
         "alive_after_each": all(r.get("alive") for r in by["malformed"]),
         "of": len(planned["malformed"]), "codes": [r.get("code") for r in by["malformed"]],
         "malformed_on_wire": sum(1 for _, args in wire_p if args and ("big.txt" in str(args.get("path")) or
                                                                     args.get("path") == "/tmp/x")),
         "final_ping": after_ping, "proxy_alive_at_end": alive_at_end, "proxy_exit_code": code_p,
         "direct_exit_code": code_d, "tracebacks_in_proxy_log": stderr_p.count("Traceback"), "crash": crashed}
    r3 = {"malformed": m, "store": stored}
    r3["pass_without_incident_id"] = bool(
        complete and m["answered_with_error"] == m["of"] and m["alive_after_each"] and after_ping
        and code_p == 0 and not m["tracebacks_in_proxy_log"] and not crashed and not m["malformed_on_wire"]
        and stored["every_call_stored"] and stored["verify_ok"] and stored["replay_ok"] == stored["records"]
        and stored["outcomes_match"])
    r3["pass"] = bool(r3["pass_without_incident_id"] and m["with_incident_id"] == m["of"])

    over = [r["ms_p"] - r["ms_d"] for r in passed if "ms_p" in r and "ms_d" in r]
    ov = {"calls": len(over), "median_ms": statistics.median(over) if over else None,
          "p10_ms": _pct(over, 0.1), "p90_ms": _pct(over, 0.9),
          "median_proxied_ms": statistics.median([r["ms_p"] for r in passed if "ms_p" in r]) if passed else None,
          "median_direct_ms": statistics.median([r["ms_d"] for r in passed if "ms_d" in r]) if passed else None,
          "median_blocked_ms": statistics.median([r["ms_p"] for r in blocked if "ms_p" in r]) if blocked else None,
          "by_tool": {t: statistics.median([r["ms_p"] - r["ms_d"] for r in passed if r.get("tool") == t and "ms_d" in r])
                      for t in sorted({r.get("tool") for r in passed if "ms_d" in r})}}
    report = {
        "session": "dry" if dry else "full", "calls": len(steps), "steps_run": len(rows),
        "counts": {k: len(planned[k]) for k in KINDS}, "seconds": round(elapsed, 2),
        "environment": _environment(npx, server, init_p, init_d),
        "limits": {"max_body": max_body, "max_depth": max_depth},
        "proxy_command": [str(x) for x in proxy_cmd],
        "blocked": b, "allowed": a, "robust": r3, "overhead": ov,
        "folders": {"before": before, "after": after},
        "rows": rows,
        "client": {"unexpected_p": px.unexpected[:20], "unexpected_d": dc.unexpected[:20]},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_md(report), encoding="utf-8")
    return report


def _stored_checks(catalog, store, tools_p, calls_rows):
    """Every tools/call a stored record with the id the client saw; verify(); a replay of each stored decision (the
    catalog loaded afresh, each tool given the server's schema as the proxy did) and its outcome against the client's."""
    import runpy
    import warnings
    from solvi.core.store import open_storage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = runpy.run_path(str(catalog))["guard"]
    for t in tools_p:
        if t["name"] in g.tools:
            g.adopt(t["name"], t.get("inputSchema") or {"type": "object"}, t.get("description") or "")
    if not Path(store).exists():
        return {"records": 0, "every_call_stored": False, "verify_ok": False, "replay_ok": 0, "outcomes_match": False}
    g.storage = open_storage(store)
    recs = list(g.storage.iter())
    ids = [r.id for r in recs]
    seen = [r.get("stored_id") for r in calls_rows]
    ver = g.storage.verify()
    replays = [g.replay(i) for i in ids]
    stored_outcomes = {r.id: ((r.meta or {}).get("guard") or {}).get("outcome") for r in recs}
    return {"records": len(recs), "calls": len(calls_rows),
            "every_call_stored": len(recs) == len(calls_rows) and None not in seen and sorted(map(str, seen)) == sorted(map(str, ids)),
            "verify_ok": bool(ver.get("ok")), "verify": {k: v for k, v in ver.items() if k != "records"},
            "replay_ok": sum(bool(v.get("ok")) for v in replays),
            "replay_mismatches": [{"id": i, "mismatches": v.get("mismatches")} for i, v in zip(ids, replays)
                                  if not v.get("ok")][:10],
            "outcomes_match": all(stored_outcomes.get(r.get("stored_id")) == r.get("outcome") for r in calls_rows),
            "outcomes": {o: list(stored_outcomes.values()).count(o) for o in sorted(set(stored_outcomes.values()), key=str)}}


def _environment(npx, server, init_p, init_d):
    def ver(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()  # noqa: S603
        except Exception:  # noqa: BLE001
            return None
    import solvi
    node = shutil.which("node", path=os.path.dirname(npx)) or shutil.which("node")
    return {"python": platform.python_version(), "platform": platform.platform(), "solvi": solvi.__version__,
            "node": ver([node, "--version"]) if node else None, "npx": ver([npx, "--version"]), "server": server,
            "server_info_direct": (init_d.get("result") or {}).get("serverInfo"),
            "server_info_proxied": (init_p.get("result") or {}).get("serverInfo"),
            "protocol": (init_d.get("result") or {}).get("protocolVersion")}


def render_md(r):
    b, a, s, o = r["blocked"], r["allowed"], r["robust"], r["overhead"]
    m, st = s["malformed"], s["store"]

    def f(x):
        return "—" if x is None else f"{x:.2f}"
    lines = [f"# MCP proxy with a guard against the filesystem server — {r['session']} session", "",
             f"{r['calls']} calls ({', '.join(f'{k} {v}' for k, v in r['counts'].items())}), {r['steps_run']} run, "
             f"{r['seconds']} s. Server `{r['environment']['server']}`, node {r['environment']['node']}, solvi "
             f"{r['environment']['solvi']}, limits {r['limits']}.", "",
             "| check | result | pass |", "|---|---|---|",
             f"| blocked calls never reach the server | {b['answered_blocked']}/{b['of']} answered as blocked "
             f"(violations denied {b['violations_denied']}/{b['violations']}, injected writes escalated "
             f"{b['injected_escalated']}/{b['injected_writes']}); blocked calls on the wire {b['blocked_calls_on_wire']}; "
             f"wire = allowed calls: {b['wire_equals_allowed_calls']}; folders equal: {b['work_folders_equal']}; "
             f"outside unchanged: {b['outside_unchanged']} | {b['pass']} |",
             f"| allowed calls identical to the direct call | allowed {a['allowed']}/{a['of']}, identical "
             f"{a['identical']}/{a['of']} (without the path placeholder: {a['identical_raw']}); tools/list as the "
             f"server's: {a['tools_list_same_as_server']} | {a['pass']} |",
             f"| malformed messages, store, replay | answered with an error {m['answered_with_error']}/{m['of']} (codes "
             f"{m['codes']}), with an incident id {m['with_incident_id']}/{m['of']}, alive after each: "
             f"{m['alive_after_each']}, final ping: {m['final_ping']}, exit {m['proxy_exit_code']}, tracebacks "
             f"{m['tracebacks_in_proxy_log']}; records {st['records']}/{st['calls']} (every call stored: "
             f"{st['every_call_stored']}), verify {st['verify_ok']}, replay {st['replay_ok']}/{st['records']}, "
             f"outcomes match {st['outcomes_match']} | {s['pass']} (without the incident id: "
             f"{s['pass_without_incident_id']}) |",
             f"| overhead per allowed call (report) | median {f(o['median_ms'])} ms (p10 {f(o['p10_ms'])}, p90 "
             f"{f(o['p90_ms'])}; proxied {f(o['median_proxied_ms'])}, direct {f(o['median_direct_ms'])}, blocked "
             f"{f(o['median_blocked_ms'])}) | — |", "", "## Calls", "", "| n | kind | tool | args / shape | outcome | ms |",
             "|---|---|---|---|---|---|"]
    for row in r["rows"]:
        what = row.get("shape") or json.dumps(row.get("args"), ensure_ascii=False)[:80]
        res = row.get("outcome") or (f"error {row.get('code')}" if row["kind"] == "malformed" else "—")
        if row["kind"] in PASSED:
            res += " =" if row.get("identical") else " ≠"
        lines.append(f"| {row['n']} | {row['kind']} | {row.get('tool', '')} | `{what.replace('|', '/')}` | {res} | "
                     f"{f(row.get('ms_p'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--relay"]:
        if len(argv) < 4 or argv[2] != "--":
            raise SystemExit("--relay LOG -- CMD ...")
        return relay(argv[1], argv[3:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="the 10-call development session (output in OUT/dry)")
    ap.add_argument("--out", default="out", help="the output folder (default: out)")
    ap.add_argument("--server", default=SERVER, help=f"the npm package of the server (default: {SERVER})")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one answer")
    a = ap.parse_args(argv)
    out = Path(a.out) / "dry" if a.dry else Path(a.out)
    r = run(out, dry=a.dry, server=a.server, timeout=a.timeout)
    print(render_md(r).split("## Calls")[0])
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
