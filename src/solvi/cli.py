"""The `solvi` command (also `python -m solvi ...`).

    solvi test PATH...        decision regression tests from cases.json files (solvi.testing)
    solvi honesty SET.json    honesty numbers of a labelled set, gated against a baseline (solvi.honesty)
    solvi verify | replay | diff over a TraceStorage:

    solvi verify decisions.db [--anchor COUNT:HASH] [--signature SIG.json] [--sign SIG.json [--alg syndrome|octonion]]
    solvi replay decisions.db --system myapp.decisions:system
    solvi diff   decisions.db --system myapp.decisions_v2:build_system [--question Q] [--since ISO] [--limit N] [--json]
    solvi serve  myapp.decisions:system [--store decisions.db] [--decider ID] [--port 8000] [--mcp]   (solvi.serve)
    solvi serve  --guard catalog.py:guard --upstream "MCP SERVER COMMAND" [--store calls.db]         (solvi.agents.mcp)
    solvi check  myapp.decisions:system [--strict] [--json]                                          (solvi.check)
    solvi report decisions.db [--since ISO] [--until ISO] [--question Q] [--id ID] [--html out.html] [--md out.md] [--json]
                                                                                                     (solvi.report)
    solvi init [DIR] [--template support|refunds|minimal] [--with-model] [--force]                  (solvi.scaffold)
    solvi ask myapp.decisions:system (STATE.json | - | --state '{...}' | --text "...") [--question Q] [--decider MODEL]
              [--audit] [--report md|html] [--lang ru] [--store decisions.db] [--json]
    solvi calibrate myapp.decisions:system PART labels.csv --risk 0.1 [--groups a,b] [--method crc|ltt] [--out F]
                                                                                                     (solvi.calibfile)
    solvi models [list | pull ID | check MODEL --examples labels.jsonl --task Q]                       (solvi.models)
    solvi hook [install | uninstall | pre-edit --rules rules.toml | pick-skill --skills-dir DIR]        (solvi.hooks)

--system names a System: "package.module:attribute" or "path/to/file.py:attribute", where the attribute is a System or a
function without arguments that returns one (`solvi serve` and `solvi check` take it as their first argument; check also
takes a Catalog). `solvi report` prints a Markdown report of the stored decisions (or of one with --id), or writes it
as a self-contained HTML page (--html) / a Markdown file (--md); --system is optional there (typed values, the replay of
one decision).
`solvi ask` asks once (a state, or a text through ask_text) and prints the answers, the audit or a report; a module-level
`prepare(state)` next to the System runs first, as in `solvi test`.
Exit status: 0 — verified / everything replays / nothing changes / no catalog errors / report written / scaffold written /
every asked question answered / something answered alone after calibrating; 1 — problems / mismatches / changes / catalog
errors / a file exists / a question abstained / the calibration escalates everything / a model that does not load; 2 —
usage errors."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys


def _fail(msg):
    print(f"solvi: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_object(spec):
    """"module:attr" or "file.py:attr" → the attribute (called when it is a function: a System factory)."""
    return load_module(spec)[1]


def load_module(spec):
    """"module:attr" or "file.py:attr" → (the module, the attribute — called when it is a function: a System factory).
    For the command: a name that cannot be read ends it with status 2 (library code uses solvi.loader, which raises
    LoadError)."""
    from .loader import LoadError, load_module as load
    try:
        return load(spec)
    except LoadError as e:
        _fail(str(e))


def load_system(spec):
    """"module:attr" or "file.py:attr" → the System (calling attr when it is a function)."""
    obj = load_object(spec)
    if not hasattr(obj, "ask") or not hasattr(obj, "catalog"):
        _fail(f"--system {spec}: not a solvi System")
    return obj


STORE_KINDS = (".db / .sqlite / .sqlite3 (SQLite), .duckdb (DuckDB), a postgresql:// URL (PostgreSQL), else a "
               "JSON-lines file")
STORE_HELP = "a TraceStorage: " + STORE_KINDS


def _store(path, system=None):
    from .storage import open_storage
    if not path.startswith(("postgresql://", "postgres://")) and not os.path.exists(path):
        _fail(f"no such store: {path}")
    return open_storage(path, system)


def _filters(a):
    f = {}
    for k in ("question", "status", "safeguard", "model", "since", "until"):
        v = getattr(a, k, None)
        if v is not None:
            f[k] = v
    _check_when(a)
    return f


def _check_when(a):
    """--since / --until that cannot be read as a time are usage errors."""
    from .storage import _when
    for k in ("since", "until"):
        v = getattr(a, k, None)
        if v is not None:
            try:
                _when(v)
            except (ValueError, TypeError) as e:
                _fail(f"--{k} {v}: not a time ({e}); write an ISO date or time (2026-09-01, 2026-09-01T12:00)")


def _dump(obj):
    from .schema import dumps
    print(dumps(obj, ensure_ascii=False, indent=2, default=repr))


def cmd_verify(a):
    anchor = None
    if a.anchor:
        n, _, h = a.anchor.partition(":")
        if not n.isdigit() or not h:
            _fail(f"--anchor {a.anchor}: expected COUNT:HASH (a head() kept elsewhere)")
        anchor = {"count": int(n), "hash": h}
    store = _store(a.store)
    sig = None
    if a.signature:
        from .signature import load
        try:
            sig = load(a.signature)
        except (OSError, ValueError) as e:
            _fail(f"cannot read the signature {a.signature}: {e}")
    try:
        v = store.verify(anchor, signature=sig)
    except ValueError as e:
        _fail(str(e))
    if a.sign and v["ok"]:                            # never sign a store that does not verify
        with open(a.sign, "w") as fh:
            json.dump(store.signature(a.alg), fh)
    if a.json:
        _dump(v)
    else:
        print(f"{a.store}: {v['count']} record(s)" + (f", {v['legacy']} legacy line(s)" if v["legacy"] else "")
              + f"; head {v['head']['count']}:{v['head']['hash']}")
        print("chain verified" if v["ok"] else f"{len(v['problems'])} problem(s):")
        for seq, rid, why in v["problems"]:
            print(f"  {'' if seq is None else f'#{seq} '}{rid or ''}: {why}".replace(" :", ":"))
        if sig is not None and v["signature"]["ok"]:
            print(f"signature verified ({v['signature']['signed']} signed record(s))")
        elif sig is not None and v["signature"]["digest"]:
            print(f"  original content hash of #{v['signature']['index']}: {v['signature']['digest']}")
        if a.sign:
            print(f"signature written to {a.sign}" if v["ok"] else "signature not written: the store does not verify")
    return 0 if v["ok"] else 1


def _checked_filters(a, system, store):
    """The query filters of replay / diff, checked: a --question the system does not have or a --status that is none
    is a usage error (a mistyped filter would match nothing and pass forever) → (filters, how many decisions match)."""
    f = _filters(a)
    q = f.get("question")
    if q is not None and q not in system.questions:
        _fail(f"--question {q}: no such question (the system has: {', '.join(system.questions)})")
    if f.get("status") is not None and f["status"] not in ("ok", "forced", "abstain"):
        _fail(f"--status {f['status']}: one of ok, forced, abstain")
    n = len(store.query(**f))
    if n == 0:
        print(f"solvi: no stored decision matches in {a.store}: nothing was checked", file=sys.stderr)
    return f, n


def cmd_replay(a):
    system = load_system(a.system)
    store = _store(a.store, system)
    filters, n = _checked_filters(a, system, store)
    bad = store.replay_all(system, trust_models=a.trust_models, **filters)
    if a.json:
        _dump(bad)
    else:
        print(f"every stored trace replays ({n})" if not bad else f"{len(bad)} of {n} stored trace(s) do not replay:")
        for b in bad:
            print(f"- {b['id']} (#{b['seq']}): {b['summary']}" + (f" — {b['note']}" if b.get("note") else ""))
            for m in b["mismatches"][:5]:
                step, name, why = m
                print(f"    step {step} {name} [{m.kind}]: {why}")
    return 0 if not bad else 1


def cmd_diff(a):
    from .diff import diff
    system = load_system(a.system)
    if a.limit is not None and a.limit < 1:
        _fail(f"--limit {a.limit}: at least 1")
    store = _store(a.store, system)
    filters, _ = _checked_filters(a, system, store)
    rep = diff(store, system, confidence=None if a.confidence < 0 else a.confidence, limit=a.limit, **filters)
    if a.json:
        _dump(rep.to_dict())
    else:
        print(rep)
    return 0 if rep.ok else 1


def cmd_report(a):
    from .report import decision, period, render
    system = load_system(a.system) if a.system else None
    store = _store(a.store, system)
    if a.id:
        try:
            res = store.get(a.id, system)
        except KeyError as e:
            _fail(str(e.args[0]))
        data = decision(res, system=system, replay="trusted" if system is not None else False)
    else:
        _check_when(a)
        f = {k: getattr(a, k) for k in ("status", "safeguard", "model") if getattr(a, k, None) is not None}
        data = period(store, a.since, a.until, a.question, a.examples, system, **f)
    if a.json:
        _dump(data)
    if a.html:
        with open(a.html, "w", encoding="utf-8") as fh:
            fh.write(render(data, "html"))
        print(f"wrote {a.html}", file=sys.stderr)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(render(data, "md"))
        print(f"wrote {a.md}", file=sys.stderr)
    if not (a.json or a.html or a.md):
        print(render(data, "md"), end="")
    return 0


def _read_json(src, what):
    """A JSON value from a file path, or from stdin when src is "-"."""
    try:
        text = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()
    except OSError as e:
        _fail(f"cannot read {what} {src}: {e.strerror}")
    try:
        return json.loads(text)
    except ValueError as e:
        _fail(f"{what} {'from stdin' if src == '-' else src}: not JSON: {e}")


def _answers(res):
    from .storage import plain
    out = {}
    for q, r in res.results.items():
        out[q] = {"answer": plain(r.answer), "status": r.status, "confidence": r.confidence, "why": r.why}
        if getattr(r, "guard", None):
            out[q]["guard"] = r.guard
    return out


def cmd_ask(a):
    """`solvi ask` → 0: every asked question answered; 1: at least one abstained (a person should look); 2: usage."""
    from . import i18n
    mod, system = load_module(a.system)
    if not hasattr(system, "ask") or not hasattr(system, "catalog"):
        _fail(f"ask {a.system}: not a solvi System")
    given = [x for x in (a.state_file, a.state, a.text) if x is not None]
    if len(given) != 1:
        _fail("ask: give one input — a STATE.json file ('-': stdin), --state '{...}' or --text '...'")
    if a.lang:
        try:
            i18n.check(a.lang)
        except ValueError as e:
            _fail(str(e))
    names = None
    if a.question:
        names = [q for x in a.question for q in x.split(",") if q]
        bad = [q for q in names if q not in system.questions]
        if bad:
            _fail(f"ask: no question {bad[0]!r} (the system asks: {', '.join(system.questions)})")
    if a.text is not None:
        if names is not None and len(names) != 1:
            _fail("ask --text: at most one --question (it skips routing)")
        text = sys.stdin.read() if a.text == "-" else a.text
        decider = None
        if a.decider:
            from .models import ModelError, load as load_model
            try:
                decider = load_model(a.decider, a.backend, api_key=a.api_key)
            except ModelError as e:
                _fail(str(e))
        from .llm import LLMError
        tin = None
        if a.today:                                    # as `solvi serve` reads a text: year-less and relative dates
            import datetime as dt
            from .textin import TextIn
            try:
                today = dt.date.today() if a.today == "today" else dt.date.fromisoformat(a.today)
            except ValueError:
                _fail(f"ask --today {a.today}: an ISO date (2026-09-28) or the word today")
            tin = TextIn(system, decider, today=today)
        try:
            res = system.ask_text(text, decider, textin=tin, question=names[0] if names else None)
        except (KeyError, ValueError) as e:
            _fail(f"ask --text: {e.args[0] if e.args else e}")
        except LLMError as e:                          # a wrong key, model or URL: said plainly, no traceback
            _fail(f"ask --decider {a.decider}: {e}")
    else:
        if a.decider:
            _fail("ask: --decider routes a --text; a state is asked as it is")
        if a.today:
            _fail("ask: --today is the date a --text is read on; a state is asked as it is")
        if a.state is not None:
            try:
                state = json.loads(a.state)
            except ValueError as e:
                _fail(f"--state: not JSON: {e}")
        else:
            state = _read_json(a.state_file, "state")
        if not isinstance(state, dict):
            _fail("ask: the state must be a JSON object")
        prep = getattr(mod, "prepare", None)
        if callable(prep):                             # the module's prepare(state), as `solvi test` runs it
            state = prep(state)
        res = system.ask(state, names)
    stored = None
    if a.store:
        from .storage import open_storage
        stored = open_storage(a.store, system).save(res)
    lang = a.lang or getattr(system, "lang", None)
    if a.json:
        out = {"answers": _answers(res), "safeguards": res.safeguards or [], "ms": res.ms}
        if a.text is not None and getattr(res, "textin", None) is not None:
            out["textin"] = res.textin.to_dict()
        if a.audit:
            out["audit"] = res.audit().to_dict()
        if stored:
            out["stored_id"] = stored
        _dump(out)
    elif a.report:
        print(res.report(a.report), end="")
    else:
        from .show import show
        show(res, flow=False, state=False, audit=False, lang=lang)
        tin = getattr(res, "textin", None)
        if tin is not None and tin.missing:
            print(tin.clarify())
        if a.audit:
            print(res.audit(lang=lang))
    if stored:
        print(f"stored {stored} in {a.store}", file=sys.stderr)
    return 0 if all(r.status != "abstain" for r in res.results.values()) else 1


def ask_parser(sub):
    s = sub.add_parser("ask", help="ask a System once: a state (JSON) or a text; print the answers, the audit or a report")
    s.add_argument("system", help="module:attr or file.py:attr — a System or a function returning one")
    s.add_argument("state_file", nargs="?", metavar="STATE.json", help="the input state; '-' reads it from stdin")
    s.add_argument("--state", help="the input state as inline JSON")
    s.add_argument("--text", help="a free text: the question it asks and its fields are read from it (ask_text); "
                                  "'-' reads stdin")
    s.add_argument("--question", action="append", help="ask only these questions (repeat, or comma-separated); with "
                                                       "--text: the question, without routing")
    s.add_argument("--today", metavar="DATE", help="with --text: the date it is read on — an ISO date, or the word today "
                                                   "(as solvi serve does); without it a date with no year, a two-digit "
                                                   "year or \"yesterday\" is not read")
    s.add_argument("--decider", help="with --text: the model that picks the question (a folder, a cached Hugging Face id, "
                                     "systemone:URL#model, llm:URL#model or module:attr; see solvi models)")
    s.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="the decider's backend")
    s.add_argument("--api-key", help="for a systemone: / llm: decider (default $SOLVI_SYSTEMONE_API_KEY / "
                   "$SOLVI_LLM_API_KEY; prefer the variable: arguments are visible to other local users)")
    s.add_argument("--audit", action="store_true", help="also print what each answer rests on (res.audit())")
    s.add_argument("--report", choices=["md", "html"], help="print the decision's report instead (res.report)")
    s.add_argument("--lang", help="the language of the answers and the audit: en (default) or ru")
    s.add_argument("--store", metavar="PATH", help="save the response to this TraceStorage: " + STORE_KINDS)
    s.add_argument("--json", action="store_true", help="print the answers (and --audit) as JSON")
    return s


COMMANDS = {"test": ("solvi.testing", "decision regression tests from cases.json files"),
            "honesty": ("solvi.honesty", "honesty numbers of a labelled set, gated against a baseline"),
            "hook": ("solvi.hooks", "a coding agent's hooks: check edits against rules, pick a skill, install them")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "-V"):
        from . import __version__
        print(f"solvi {__version__}")
        return 0
    if argv and argv[0] in COMMANDS:                  # commands with their own option parsers
        return importlib.import_module(COMMANDS[argv[0]][0]).main(argv[1:])
    p = argparse.ArgumentParser(prog="solvi", description="solvi: init a project; ask, check, serve a system; test, "
                                                          "honesty; calibrate a model decision; models; verify, replay, "
                                                          "diff and report stored decisions",
                                epilog="also: " + "; ".join(f"solvi {k} — {w}" for k, (_, w) in COMMANDS.items()))
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, system=True, filters=True):
        sp.add_argument("store", help=STORE_HELP)
        if system:
            sp.add_argument("--system", required=True, help="module:attr or file.py:attr — a System or a function returning one")
        if filters:
            for k in ("question", "status", "safeguard", "model", "since", "until"):
                sp.add_argument(f"--{k}", help=f"only stored decisions with this {k} (see TraceStorage.query)")
        sp.add_argument("--json", action="store_true", help="print JSON")
    v = sub.add_parser("verify", help="check the hash chain across stored decisions")
    common(v, system=False, filters=False)
    v.add_argument("--anchor", help="COUNT:HASH — a head() kept elsewhere")
    v.add_argument("--signature", help="SIG.json (a file or the JSON) — a signature() kept elsewhere: names the one changed "
                                       "record and its original content hash (solvi.signature)")
    v.add_argument("--sign", metavar="OUT.json", help="write the store's current signature() to this file")
    v.add_argument("--alg", choices=["syndrome", "octonion"], default="syndrome",
                   help="the code --sign writes (default syndrome; --signature reads it from the file)")
    r = sub.add_parser("replay", help="re-compute every stored trace against a system")
    common(r)
    r.add_argument("--trust-models", action="store_true", help="do not re-run models; verify their recorded outputs")
    d = sub.add_parser("diff", help="re-run stored decisions with another system and list what changes")
    common(d)
    d.add_argument("--limit", type=int, help="at most this many stored decisions")
    d.add_argument("--confidence", type=float, default=0.01,
                   help="report a confidence change above this (default 0.01; negative: ignore confidence)")
    rp = sub.add_parser("report", help="a human-readable report of stored decisions (a period, or one with --id)")
    rp.add_argument("store", help=STORE_HELP)
    rp.add_argument("--system", help="module:attr or file.py:attr — optional: restores typed values; replays one decision")
    for k in ("question", "status", "safeguard", "model", "since", "until"):
        rp.add_argument(f"--{k}", help=f"only stored decisions with this {k} (see TraceStorage.query)")
    rp.add_argument("--id", help="the report of this one stored decision")
    rp.add_argument("--examples", type=int, default=3, help="example ids per answer, escalation and safeguard (default 3)")
    rp.add_argument("--html", metavar="OUT.html", help="write a self-contained HTML page")
    rp.add_argument("--md", metavar="OUT.md", help="write Markdown to a file")
    rp.add_argument("--json", action="store_true", help="print the report's data as JSON")
    from .serve import add_parser as serve_parser, cmd_serve
    serve_parser(sub)
    from .check import add_parser as check_parser, cmd_check
    check_parser(sub)
    ask_parser(sub)
    from .calibfile import add_parser as calibrate_parser, cmd_calibrate
    calibrate_parser(sub)
    from .models import add_parser as models_parser, cmd_models
    models_parser(sub)
    from .scaffold import add_parser as init_parser, cmd_init
    init_parser(sub)
    try:
        a = p.parse_args(argv)
    except SystemExit as e:                            # --help: 0; usage errors: 2 — returned, not raised
        return e.code if isinstance(e.code, int) else 2
    try:
        return {"verify": cmd_verify, "replay": cmd_replay, "diff": cmd_diff, "serve": cmd_serve,
                "check": cmd_check, "report": cmd_report, "ask": cmd_ask, "calibrate": cmd_calibrate,
                "models": cmd_models, "init": cmd_init}[a.cmd](a)
    except OSError as e:                               # a path that cannot be read or written: usage, not a finding
        print(f"solvi: {e.filename}: {e.strerror}" if e.filename and e.strerror else f"solvi: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
