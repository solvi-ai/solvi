"""An MCP proxy with a solvi Guard: it sits between an agent (the MCP client) and an MCP server, and every tools/call
passes the guard before it reaches the server.

    solvi serve --guard catalog.py:guard --upstream "npx -y @modelcontextprotocol/server-filesystem /work" \\
                [--store calls.db] [--facts '{"role": "viewer"}'] [--escalate elicit|deny] \\
                [--context-messages 50] [--context-chars 100000]

`catalog.py` declares which of the server's tools the agent may call and the policies over them — without functions
(the server runs them) and usually without schemas (the proxy takes each tool's inputSchema from the server):

    from pathlib import Path

    guard = Guard(storage="calls.db")
    guard.declare("read_text_file")
    guard.declare("write_file", injections="any")          # no writes after a tool output that carries instructions

    @guard.policy(["read_text_file", "write_file"])
    def inside_work(path: str) -> bool:         # resolved: "/work/../etc/passwd" and symlinks out of /work fail
        return Path(path).resolve().is_relative_to(Path("/work").resolve())

The proxy speaks MCP over stdio (JSON-RPC, one message per line) to the client and to the server it starts:

  initialize   initializes the server, answers with the tools capability (and the server's name in ours)
  tools/list   the server's tools that the guard declares (others are hidden; each declared tool without a schema
               adopts the server's inputSchema — one that cannot be read gives that tool a permissive schema, a warning in
               the log, and every call of it escalates; a tool whose arguments collide with the guard's facts is hidden)
  tools/call   the guard checks the call — allow: forwarded to the server with the arguments as the guard validated
               them (coerced to the schema's types: "2" for an integer is sent as 2, "no" for a boolean as false; the
               arguments the client did not send are not added), and its result's text is kept as a tool
               output in the proxy's session, so later calls are checked against it (grounding, instruction-like text);
               deny: an error result with the reasons; escalate: with --escalate elicit (default) and a client that
               declares the elicitation capability, the user is asked (elicitation/create: approve yes / no) and the
               answer is recorded as a person's resolution; otherwise an error result saying it waits for a person
  ping         answered

Every decision goes to the guard's store (or --store) with the call's outcome; `_meta.solvi` on each result carries the
outcome, the stored id and the trace hash. The proxy does not see the user's messages: an argument declared with ground=
is found only in the tool outputs of this session (and denied otherwise). The session keeps the last `context_messages`
tool outputs, at most `context_chars` characters in all (Session's max_messages / max_chars): each decision's trace
records the context it was checked against, so the cap bounds what every stored decision holds — an output that has
left the window no longer grounds values or taints calls."""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys

from .guard import Guard, _text

CONTEXT_MESSAGES = 50            # the tool outputs the proxy's session keeps for checking
CONTEXT_CHARS = 100_000          # ... and their characters in all
log = logging.getLogger("solvi.agents")

PROTOCOL = "2025-06-18"


class UpstreamError(RuntimeError):
    pass


class Upstream:
    """An MCP server started as a subprocess, spoken to over its stdin / stdout (one JSON-RPC message per line)."""

    def __init__(self, command, env=None):
        args = shlex.split(command) if isinstance(command, str) else list(command)
        # the command line the operator gave (--upstream), never request data
        self.proc = subprocess.Popen(  # noqa: S603
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env)
        self._id = 0
        self.info = None

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        """Send a request and wait for its response → the result (UpstreamError on an error response or a closed
        server). Requests from the server meanwhile get "method not found"; notifications are ignored."""
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "method" in msg:
                if "id" in msg:
                    self._send({"jsonrpc": "2.0", "id": msg["id"],
                                "error": {"code": -32601, "message": "not supported by the solvi proxy"}})
                continue
            if msg.get("id") == rid:
                if "error" in msg:
                    raise UpstreamError(msg["error"].get("message", str(msg["error"])))
                return msg.get("result") or {}
        raise UpstreamError(f"the upstream MCP server closed (exit code {self.proc.poll()})")

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def initialize(self, version=PROTOCOL):
        if self.info is None:
            from .. import __version__
            self.info = self.request("initialize", {"protocolVersion": version, "capabilities": {},
                                                    "clientInfo": {"name": "solvi-guard", "version": __version__}})
            self.notify("notifications/initialized")
        return self.info

    def tools(self):
        out, cursor = [], None
        while True:
            r = self.request("tools/list", {"cursor": cursor} if cursor else {})
            out += r.get("tools") or []
            cursor = r.get("nextCursor")
            if not cursor:
                return out

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()


class Proxy:
    """The proxy's state: the guard, the upstream server, the session (tool outputs seen so far, the facts)."""

    def __init__(self, guard, upstream, facts=None, escalate="elicit", context_messages=CONTEXT_MESSAGES,
                 context_chars=CONTEXT_CHARS):
        if not isinstance(guard, Guard):
            raise TypeError("--guard names a solvi.agents.Guard")
        if escalate not in ("elicit", "deny"):
            raise ValueError('escalate: "elicit" | "deny"')
        self.guard = guard
        self.upstream = upstream if isinstance(upstream, Upstream) else Upstream(upstream)
        self.session = guard.session(facts=facts, max_messages=context_messages, max_chars=context_chars)
        self.escalate = escalate
        self.listed = None
        self.client_caps = {}
        self._asked = 0

    def tools(self):
        """The server's tools that the guard declares; each declared tool without a schema adopts the server's."""
        g, out = self.guard, []
        for t in self.upstream.tools():
            name = t.get("name")
            if name in g.tools:
                try:
                    g.adopt(name, t.get("inputSchema") or {"type": "object"}, t.get("description") or "")
                except ValueError as e:              # its arguments collide with the guard's facts: not offered
                    log.warning("solvi proxy: tool %s is hidden: %s", name, e)
                    continue
                out.append(t)
        self.listed = {t["name"] for t in out}
        return out

    def call(self, params, ask=None):
        """tools/call → the result for the client. ask(message) → the user's answer (True / False / None) or None."""
        if self.listed is None:
            self.tools()
        name, args = params.get("name"), params.get("arguments") or {}
        g = self.guard
        if name in g.tools and name not in self.listed:          # declared, but hidden (see tools)
            return {"content": [{"type": "text", "text": f"{name} is not available through this proxy"}],
                    "isError": True}
        d = g.check({"name": name, "arguments": args}, self.session.context, self.session.facts, store=False)
        if d.outcome == "escalate" and ask is not None:
            answer = ask(d)
            if answer is not None:
                g._save(d)
                d = g.resolve(d, approve=bool(answer), reviewer="mcp elicitation", execute=False)
        self.session.decisions.append(d)
        if d.outcome != "allow":
            g._save(d)
            return {"content": [{"type": "text", "text": d.message()}], "structuredContent": {"solvi": d.to_dict()},
                    "isError": True, "_meta": {"solvi": _meta(d)}}
        d.executed = True
        try:
            result = self.upstream.request("tools/call", {"name": name, "arguments": forwarded(g.tools[name], args)})
        except UpstreamError as e:
            d.error = str(e)
            result = {"content": [{"type": "text", "text": f"{name} failed: {e}"}], "isError": True}
        else:
            if result.get("isError"):
                d.error = _text(result.get("content")) or "error"
            d.result = result.get("structuredContent", _text(result.get("content")))
        self.session.add("tool", f"{name}: {_text(result.get('content'))}")
        if d.stored_id is None:
            g._save(d)
        result = dict(result)
        result["_meta"] = {**(result.get("_meta") or {}), "solvi": _meta(d)}
        return result


def forwarded(tool, args):
    """The arguments sent to the server for an allowed call: the ones the client sent, as the guard validated them
    (coerced to the schema's types — what the checks read is what the server gets), as JSON."""
    return tool.model.model_validate(args).model_dump(mode="json", exclude_unset=True, by_alias=True)


def _meta(d):
    return {"outcome": d.outcome, "stored_id": d.stored_id, "trace_hash": d.trace_hash}


def run_proxy(guard, upstream, facts=None, escalate="elicit", stdin=None, stdout=None, limits=None,
              context_messages=CONTEXT_MESSAGES, context_chars=CONTEXT_CHARS):
    """The proxy over stdio (see the module docs). upstream: a command line (or an Upstream). context_messages /
    context_chars: the session's context kept for checking (None: unbounded). limits: a
    solvi.serve.Limits — a client message is at most max_body characters and max_depth levels of JSON; a failure of the
    proxy itself is logged (logger solvi.serve) and answered with an incident id, never the exception's text."""
    from .. import __version__
    from ..schema import dumps
    from ..serve import Limits, RequestError, _readline, internal_error, parse_json
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    lim = limits or Limits()
    px = Proxy(guard, upstream, facts, escalate, context_messages, context_chars)

    def lines():
        while True:
            line, too_long = _readline(stdin, lim.max_body)
            if too_long:
                error(None, -32600, f"invalid request: a message is at most {lim.max_body} characters")
                continue
            if not line:
                return
            yield line

    def send(msg):
        stdout.write(dumps(msg, ensure_ascii=False, default=repr) + "\n")
        stdout.flush()

    def error(id_, code, msg):
        send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": msg}})

    def ask(d):
        """Escalation to the user through the client (elicitation/create) → True / False, or None (no answer)."""
        if px.escalate != "elicit" or "elicitation" not in px.client_caps:
            return None
        px._asked += 1
        rid = f"solvi-elicit-{px._asked}"
        send({"jsonrpc": "2.0", "id": rid, "method": "elicitation/create", "params": {
            "message": f"The agent wants to call {d.tool} with {json.dumps(d.to_dict()['arguments'], ensure_ascii=False)}. "
                       "It needs your approval: " + "; ".join(d.reasons),
            "requestedSchema": {"type": "object", "properties": {"approve": {"type": "boolean", "title": "Approve"}},
                                "required": ["approve"]}}})
        for line in lines():
            try:
                msg = parse_json(line, lim)
            except RequestError:
                continue
            if isinstance(msg, dict) and msg.get("id") == rid and "method" not in msg:
                r = msg.get("result") or {}
                return bool(r.get("action") == "accept" and (r.get("content") or {}).get("approve"))
            if isinstance(msg, dict) and "method" in msg and "id" in msg:
                error(msg["id"], -32603, "the solvi proxy is waiting for the user's answer to an escalated call")
        return None

    try:
        for line in lines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = parse_json(line, lim)
            except RequestError as e:
                bad_json = str(e) == "the request is not JSON"
                error(None, -32700 if bad_json else -32600, "parse error" if bad_json else f"invalid request: {e}")
                continue
            if not isinstance(msg, dict) or "method" not in msg:
                continue
            method, params, id_ = msg["method"], msg.get("params") or {}, msg.get("id")
            if "id" not in msg:
                continue
            try:
                if method == "initialize":
                    px.client_caps = params.get("capabilities") or {}
                    info = px.upstream.initialize(params.get("protocolVersion") or PROTOCOL)
                    up = (info.get("serverInfo") or {}).get("name", "upstream")
                    send({"jsonrpc": "2.0", "id": id_, "result": {
                        "protocolVersion": info.get("protocolVersion") or PROTOCOL,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": f"solvi-guard/{up}", "version": __version__},
                        "instructions": "Every tool call is checked by a solvi guard before it reaches the server: a denied "
                                        "or escalated call returns an error result with the reasons."}})
                elif method == "ping":
                    send({"jsonrpc": "2.0", "id": id_, "result": {}})
                elif method == "tools/list":
                    send({"jsonrpc": "2.0", "id": id_, "result": {"tools": px.tools()}})
                elif method == "tools/call":
                    send({"jsonrpc": "2.0", "id": id_, "result": px.call(params, ask)})
                else:
                    error(id_, -32601, f"method not found: {str(method)[:64]}")
            except UpstreamError as e:
                error(id_, -32603, f"upstream: {e}")
            except Exception:  # noqa: BLE001 — the proxy keeps serving; the details stay in the server log
                error(id_, -32603, internal_error(f"proxy {method}"))
    finally:
        px.upstream.close()
    return px
