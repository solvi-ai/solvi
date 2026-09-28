"""The `solvi` command (also `python -m solvi ...`).

    solvi test PATH...        decision regression tests from cases.json files (solvi.testing)
    solvi honesty SET.json    honesty numbers of a labelled set, gated against a baseline (solvi.honesty)
    solvi verify | replay | diff over a TraceStorage:

    solvi verify decisions.db [--anchor COUNT:HASH]
    solvi replay decisions.db --system myapp.decisions:system
    solvi diff   decisions.db --system myapp.decisions_v2:build_system [--question Q] [--since ISO] [--limit N] [--json]

--system names a System: "package.module:attribute" or "path/to/file.py:attribute", where the attribute is a System or a
function without arguments that returns one. Exit status: 0 — verified / everything replays / nothing changes; 1 —
problems / mismatches / changes; 2 — usage errors."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys


def _fail(msg):
    print(f"solvi: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_system(spec):
    """"module:attr" or "file.py:attr" → the System (calling attr when it is a function)."""
    mod_name, _, attr = spec.rpartition(":")
    if not mod_name or not attr:
        _fail(f"--system: expected module:attribute or file.py:attribute, got {spec!r}")
    if mod_name.endswith(".py") or os.sep in mod_name:
        path = os.path.abspath(mod_name)
        sys.path.insert(0, os.path.dirname(path))
        s = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(s)
        sys.modules[s.name] = mod
        s.loader.exec_module(mod)
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if callable(obj) and not hasattr(obj, "ask"):
        obj = obj()
    if not hasattr(obj, "ask") or not hasattr(obj, "catalog"):
        _fail(f"--system {spec}: not a solvi System")
    return obj


def _store(path, system=None):
    from .storage import open_storage
    if not os.path.exists(path):
        _fail(f"no such store: {path}")
    return open_storage(path, system)


def _filters(a):
    f = {}
    for k in ("question", "status", "safeguard", "model", "since", "until"):
        v = getattr(a, k, None)
        if v is not None:
            f[k] = v
    return f


def _dump(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=repr))


def cmd_verify(a):
    anchor = None
    if a.anchor:
        n, _, h = a.anchor.partition(":")
        anchor = {"count": int(n), "hash": h}
    v = _store(a.store).verify(anchor)
    if a.json:
        _dump(v)
    else:
        print(f"{a.store}: {v['count']} record(s)" + (f", {v['legacy']} legacy line(s)" if v["legacy"] else "")
              + f"; head {v['head']['count']}:{v['head']['hash']}")
        print("chain verified" if v["ok"] else f"{len(v['problems'])} problem(s):")
        for seq, rid, why in v["problems"]:
            print(f"  #{seq} {rid or ''}: {why}")
    return 0 if v["ok"] else 1


def cmd_replay(a):
    system = load_system(a.system)
    bad = _store(a.store, system).replay_all(system, trust_models=a.trust_models, **_filters(a))
    if a.json:
        _dump(bad)
    else:
        print("every stored trace replays" if not bad else f"{len(bad)} stored trace(s) do not replay:")
        for b in bad:
            print(f"- {b['id']} (#{b['seq']})")
            for step, name, why in b["mismatches"][:5]:
                print(f"    step {step} {name}: {why}")
    return 0 if not bad else 1


def cmd_diff(a):
    from .diff import diff
    system = load_system(a.system)
    rep = diff(_store(a.store, system), system, confidence=None if a.confidence < 0 else a.confidence, limit=a.limit,
               **_filters(a))
    if a.json:
        _dump(rep.to_dict())
    else:
        print(rep)
    return 0 if rep.ok else 1


COMMANDS = {"test": ("solvi.testing", "decision regression tests from cases.json files"),
            "honesty": ("solvi.honesty", "honesty numbers of a labelled set, gated against a baseline")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "-V"):
        from . import __version__
        print(f"solvi {__version__}")
        return 0
    if argv and argv[0] in COMMANDS:                  # commands with their own option parsers
        return importlib.import_module(COMMANDS[argv[0]][0]).main(argv[1:])
    p = argparse.ArgumentParser(prog="solvi", description="solvi: test, honesty; verify, replay and diff stored decisions",
                                epilog="also: " + "; ".join(f"solvi {k} — {w}" for k, (_, w) in COMMANDS.items()))
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, system=True, filters=True):
        sp.add_argument("store", help="a TraceStorage: .db / .sqlite (SQLite) or a JSON-lines file")
        if system:
            sp.add_argument("--system", required=True, help="module:attr or file.py:attr — a System or a function returning one")
        if filters:
            for k in ("question", "status", "safeguard", "model", "since", "until"):
                sp.add_argument(f"--{k}", help=f"only stored decisions with this {k} (see TraceStorage.query)")
        sp.add_argument("--json", action="store_true", help="print JSON")
    v = sub.add_parser("verify", help="check the hash chain across stored decisions")
    common(v, system=False, filters=False)
    v.add_argument("--anchor", help="COUNT:HASH — a head() kept elsewhere")
    r = sub.add_parser("replay", help="re-compute every stored trace against a system")
    common(r)
    r.add_argument("--trust-models", action="store_true", help="do not re-run models; verify their recorded outputs")
    d = sub.add_parser("diff", help="re-run stored decisions with another system and list what changes")
    common(d)
    d.add_argument("--limit", type=int, help="at most this many stored decisions")
    d.add_argument("--confidence", type=float, default=0.01,
                   help="report a confidence change above this (default 0.01; negative: ignore confidence)")
    try:
        a = p.parse_args(argv)
    except SystemExit as e:                            # --help: 0; usage errors: 2 — returned, not raised
        return e.code if isinstance(e.code, int) else 2
    return {"verify": cmd_verify, "replay": cmd_replay, "diff": cmd_diff}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
