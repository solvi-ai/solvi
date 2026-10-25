"""The adapter loop's checkpoint (experimental; the LoRA learning loop, stage 4): `AdapterLoop.save(dir)` and
`AdapterLoop.load(dir, learner, ...)`.

    info = loop.save("loop.ckpt")                       # → {"path", "manifest", "n", "files"}
    loop = AdapterLoop.load("loop.ckpt", LoraLearner(part, "loop/"), system=part, secret=key)

What is saved: the loop's whole state at the moment of the save — the decision count, the served version, the
candidate and the rollback history; every version (state, fingerprint, frozen_at, the keys it trained on, its paired
samples, its `StreamPromise` through `dump()`) and the files of every version that still has a model (through its
`Served.save`; version 0 too when it has a model) plus those of discarded and evicted versions found in the loop's
`versions_dir`; the δ budget's index; the training budget's counters (start times, attempts); the policy as it is now
and as it was built (pickled; the replay starts from the latter); the `RiskDetector` through `dump()`, the score monitor
(pickled, now and as built); the adaptive rate's window; the alarms, probation, the held-out window; pause, close and
error; the label store `labels.jsonl` — every accepted label (content key, input, label, source, decision, split; an
input or a label that does not round-trip through JSON is pickled in its line); and the event record `records.jsonl`
(when the loop keeps its records in memory; with a `TraceStorage`, the store's chain head and the count of this loop's
records instead).

Layout: `dir/CURRENT` names the checkpoint folder `dir/ckpt-<n>-<id>/`, which holds `manifest.json` (the format, the
question, the configuration, the learner's class, the model's weights fingerprint, the store's head and every other
file's SHA-256), `manifest.sig` (HMAC-SHA256 of the manifest under the loop's `secret`, else its SHA-256), `state.json`,
`labels.jsonl`, `records.jsonl`, `policy.pkl`, `policy0.pkl`, `monitor.pkl`, `monitor0.pkl` and `versions/<v>.*`.
A save writes a new folder under a temporary name, syncs it, renames it, then replaces `CURRENT` (`os.replace`): a
crash at any point leaves `CURRENT` naming a complete checkpoint, the previous one until the switch. Older folders are
removed after the switch, under the same locks as the save (so a second save of the same loop waits for the first one's
cleanup), and the folder `CURRENT` names is never removed. One loop saves into a directory: two loops sharing one are
not serialised against each other. A save holds the part's version lock and the loop's lock for all its writes — the
state, the labels, the pickles and every version's files (a LoRA version's adapter), each fsynced — so the part's
decisions and the loop's controls wait for the whole save; the cost is unmeasured. `load` also takes the checkpoint
folder itself.

What load refuses (ValueError): a manifest whose signature does not match (tampered, or another secret — the secret is
never written), a file whose hash is not the manifest's or that is missing, a file in the folder that the manifest does
not list, a checkpoint saved with a secret loaded without one, a learner of another class, a part whose
model has other weights, a version file whose fingerprint is not the recorded one, an `initial=` with another
fingerprint, a store that does not contain the checkpoint's chain head or holds records of this loop after it (those
decisions are not in the saved state), a store given to a checkpoint saved without one or the other way round, a
System missing when the loop served on one, and a learner whose label count after the train labels are fed again is
not the saved one.

The learner is not saved: `load` takes it as built (fresh) and feeds it the saved train labels again, in order, through
`update` — enough for a batch learner (a LoRA learner trains from the base checkpoint every time) and exact for a
deterministic online learner. The key function, the `served` function, the clock and the secret are not saved either:
give them again. A training in flight refuses the save unless `save(cancel=True)`, which cancels it first (a "cancel"
event; it stays counted against the budget; nothing restarts it — the policy may start another on a later decision).
A save and a load write no record: a loaded loop continues the saved records, so it and the loop it was saved from give
the same records on the same continuation, and `replay()` on it rebuilds every decision, before and after the load.

Trust: the pickles (policy, monitor, a `HeadVersion`'s model, a label that is not JSON) are run only after the
manifest's signature and every file's hash are verified, and a folder holding a file the manifest does not list is
refused; the version files are read from the verified bytes only (written to a private temporary folder the loaders
read, so a file swapped in the checkpoint after its hash was checked is never read), and the ones copied into
`versions_dir` are written from them too. With `secret=` the signature
is an HMAC, so a file changed or added by someone without the secret is refused. Without a secret it is a plain SHA-256
that anyone can recompute: a check against accidental damage, with no tamper protection against whoever can write the
folder, and `load` warns (`CheckpointTrustWarning`). Whether the checkpoint was keyed is recorded inside the signed
manifest (`keyed`), so with the secret a flip of it is caught by the HMAC, and a keyed checkpoint is refused without
the secret; nothing written in the folder can stop its writer from forging an unkeyed checkpoint, which is why the
unkeyed path warns. Load checkpoints you wrote. Unmeasured: tests on fake learners and a tiny random checkpoint
only."""
from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import inspect
import json
import math
import os
import pickle
import secrets
import shutil
import tempfile
import warnings
from collections import deque
from pathlib import Path

from ._eprocess import DeltaBudget
from .certificate import StreamPromise
from .detector import RiskDetector
from .policy import LabelRow
from .serving import HeadVersion

FORMAT = "solvi adapter loop checkpoint v1"
# An additive loop's checkpoint (AdapterLoop(additive=True)) has its own format: a library without additive coverage
# refuses it with this string in its message; a replacement loop's checkpoint stays v1, byte for byte as before.
FORMAT_ADDITIVE = ("solvi adapter loop checkpoint v2 (additive coverage: load it with a solvi whose AdapterLoop takes "
                   "additive=True)")
_FORMATS = (FORMAT, FORMAT_ADDITIVE)
CURRENT = "CURRENT"
_META = ("manifest.json", "manifest.sig")


class CheckpointTrustWarning(UserWarning):
    """A checkpoint loaded without a secret: its integrity check detects damage, not tampering."""


# ───────────────────────────────────────────────────────────── small helpers
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dumps(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sig(secret, data: bytes) -> dict:
    if secret:
        return {"kind": "hmac-sha256", "sig": hmac.new(secret, data, hashlib.sha256).hexdigest()}
    return {"kind": "sha256", "sig": _sha(data)}


def _cls_path(cls) -> str | None:
    return None if cls is None else f"{cls.__module__}:{cls.__qualname__}"


def _cls_of(path):
    if path is None:
        return None
    mod, _, qual = path.partition(":")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _float(x):
    return None if x is None else float(x)


def _fsync(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _round_trips(v) -> bool:
    try:
        enc = json.dumps(v, allow_nan=False)
    except (TypeError, ValueError):
        return False
    back = json.loads(enc)
    return _same(back, v)


def _same(a, b) -> bool:
    """Equal and of the same JSON types all the way down (a tuple is not a list here, 1 is not True)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _label_line(r: LabelRow, split: str) -> dict:
    d = {"key": r.key, "source": r.source, "n": r.t, "split": split}
    if _round_trips(r.x) and _round_trips(r.y):
        d["x"], d["y"] = r.x, r.y
    else:
        d["pickle"] = base64.b64encode(pickle.dumps((r.x, r.y))).decode()
    return d


def _label_row(d) -> tuple:
    if "pickle" in d:
        x, y = pickle.loads(base64.b64decode(d["pickle"]))  # noqa: S301 — hash-checked against the signed manifest
    else:
        x, y = d["x"], d["y"]
    return LabelRow(d["key"], x, y, d["source"], d["n"]), d["split"]


def _items_of(storage, question) -> int:
    n = 0
    for s in storage.iter(kind=None):
        d = s.data
        if d.get("kind") == "loop" and d.get("question") == question:
            n += 1
        elif d.get("kind") == "stream" and (d.get("stream") or {}).get("question") == question:
            n += 1
    return n


def _weights(part):
    m = getattr(part, "model", None)
    f = getattr(m, "weights_fingerprint", None)
    return str(f()) if callable(f) else None


# ───────────────────────────────────────────────────────────── save
def _state(loop) -> dict:
    vs = []
    for v, ver in sorted(loop._vs.items()):
        p = ver["promise"]
        vs.append({"version": v, "state": ver["state"], "frozen_at": ver["frozen_at"], "seen": ver["seen"],
                   "n_train": ver["n_train"], "keys": sorted(ver["keys"]), "fingerprint": ver["fingerprint"],
                   "samples": [[_float(sk), _float(ss), bool(wk), bool(ws), bool(seen)]
                               for sk, ss, wk, ws, seen in ver["samples"]],
                   "fed": list(ver["fed"]), "discarded_at": ver["discarded_at"], "stale": ver.get("stale"),
                   "served_cls": _cls_path(ver.get("served_cls")), "has_model": ver["model"] is not None,
                   "promise": None if p is None else p.dump(),
                   "cmp": None if ver.get("cmp") is None else ver["cmp"].dump(),
                   "fixed": (None if ver.get("fixed") is None or not math.isfinite(ver["fixed"])
                             else float(ver["fixed"]))})
    rows_keys = {r.key for r in loop._rows}
    return {"n": loop.n, "served": loop.served, "candidate": loop.candidate, "left": loop.left,
            "paused": loop.paused, "closed": loop.closed, "error": loop.error, "attempt": loop._attempt,
            "trainings": list(loop._trainings), "history": list(loop._history), "served_since": loop._served_since,
            "alarms": [dict(a) for a in loop._alarms], "holdout": [[bool(a), bool(w)] for a, w in loop._holdout],
            "watch": None if loop._watch is None else {**loop._watch, "lw": [float(x) for x in loop._watch["lw"]]},
            "split_n": dict(loop._split_n),
            "refused": dict(loop._refused),
            "split_extra": {k: sp for k, sp in loop._split_of.items() if k not in rows_keys},
            "rate": {"alone": [bool(a) for a in loop.rate._alone], "count": loop.rate._count},
            "budget_v": loop.budget.v, "ev_seen": {str(v): k for v, k in loop._ev_seen.items()},
            "seen": loop.live.seen(), "seen0": loop._seen0,
            "detector": None if loop.detector is None else loop.detector.dump(),
            "versions": vs, **({} if not loop.additive else {"additive": _additive_state(loop)})}


def _additive_state(loop) -> dict:
    """An additive loop's chain: the layers (versions, L0 first), the routing thresholds Λ by version, the frozen and
    the restarted-and-relearning layers, each layer's RiskDetector (L0's is the loop's `detector`) and probation left
    (L0's is the loop's `left`), the last train refusal and each layer's zone counts. A candidate in trial is the
    loop's `candidate` (its promise, samples and no comparison certificate are in its version row)."""
    return {"layers": list(loop._layers),
            "route": {str(v): float(lam) for v, lam in sorted(loop._route.items())},
            "frozen": sorted(loop._frozen), "relearn": sorted(loop._relearn),
            "detectors": {str(v): None if d is None else d.dump() for v, d in sorted(loop._dets.items())},
            "left_of": {str(v): int(k) for v, k in sorted(loop._left_of.items())},
            "refused_why": loop._refused_why,
            "zones": {str(v): dict(z) for v, z in sorted(loop._zstat.items())}}


def _save_versions(loop, out: Path) -> dict:
    """Every version's files into out (versions/): a live version through its Served.save, version 0's model too (a
    model without save wrapped in a HeadVersion), the files of a version without a model copied from versions_dir. →
    {version: how version 0 / the version is rebuilt}."""
    out.mkdir()
    kinds = {}
    for v, ver in sorted(loop._vs.items()):
        if v == 0:
            m = ver["model"]
            if m is None:
                kinds["0"] = {"kind": "none"}
            elif callable(getattr(m, "save", None)) and callable(getattr(type(m), "load", None)):
                m.save(out, 0)
                kinds["0"] = {"kind": "served", "cls": _cls_path(type(m))}
            else:
                HeadVersion(m, loop.question, loop.feature).save(out, 0)
                kinds["0"] = {"kind": "head"}
            continue
        srv = ver.get("served_as")
        if ver["model"] is not None and srv is not None:
            srv.save(out, v)
        elif loop.versions_dir is not None:
            for f in sorted(loop.versions_dir.glob(f"{v}.*")):
                if f.is_file():
                    shutil.copy2(f, out / f.name)
    return kinds


def save_loop(loop, dir, *, cancel=False) -> dict:
    """See the module docstring. → {"path": the checkpoint folder, "manifest": its SHA-256, "n", "files"}."""
    root = Path(dir)
    with loop._locked():
        if loop._run is not None:
            if not cancel:
                raise RuntimeError(f"a training is in flight (job {loop._run['job'].id}): let it be applied "
                                   "(wait_training(), then one decision), or save(cancel=True) to cancel it first")
            loop.cancel_training(why="checkpoint")
        root.mkdir(parents=True, exist_ok=True)
        name = f"ckpt-{loop.n:09d}-{secrets.token_hex(4)}"
        tmp = root / f".tmp-{name}"
        tmp.mkdir()
        try:
            files = {}

            def put(rel, data: bytes):
                _write(tmp / rel, data)
                files[rel] = _sha(data)
            state = _state(loop)
            put("state.json", _dumps(state))
            lines = []
            for r in loop._rows:
                lines.append(json.dumps(_label_line(r, loop._split_of[r.key]), sort_keys=True))
            put("labels.jsonl", ("\n".join(lines) + ("\n" if lines else "")).encode())
            if loop.storage is None:
                put("records.jsonl", "".join(json.dumps([k, e], sort_keys=True) + "\n" for k, e in loop.log).encode())
                store = None
            else:
                store = {"head": loop.storage.head(), "items": _items_of(loop.storage, loop.question)}
            put("policy.pkl", pickle.dumps(loop.policy))
            put("policy0.pkl", loop._policy0)
            put("monitor.pkl", pickle.dumps(loop.monitor))
            put("monitor0.pkl", pickle.dumps(loop._monitor0))
            v0 = _save_versions(loop, tmp / "versions")
            for f in sorted((tmp / "versions").iterdir()):
                b = f.read_bytes()
                files[f"versions/{f.name}"] = _sha(b)
                _fsync(f)
            kw = loop._kw
            manifest = {
                "format": FORMAT_ADDITIVE if loop.additive else FORMAT, "question": loop.question, "n": loop.n, "served": loop.served,
                "candidate": loop.candidate, "delta_spent": loop.budget.spent, "learner": _cls_path(type(loop.live)),
                "weights": None if loop.part is None else _weights(loop.part),
                "config": {k: kw[k] for k in ("max_risk", "max_error", "delta", "probation", "source", "seed",
                                              "sampling", "promise_options", "feature", "wait", "mode",
                                              "warn_refused", "trace_candidate")
                           + (("additive", "max_layers") if loop.additive else ())},
                "detector_alone": loop._detector_alone, "batch": loop._batch,
                "detector": None if loop.detector is None else loop.detector.config(),
                "system": loop.system is not None, "version0": v0["0"], "keyed": loop._secret is not None,
                "store": store, "files": files}
            mbytes = _dumps(manifest)
            _write(tmp / "manifest.json", mbytes)
            _write(tmp / "manifest.sig", _dumps(_sig(loop._secret, mbytes)))
            _fsync(tmp / "versions")
            _fsync(tmp)
            final = root / name
            os.rename(tmp, final)
            _fsync(root)
            _switch(root, name)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        _cleanup(root, name)                             # under the lock: no other save of this loop runs meanwhile
    return {"path": root / name, "manifest": _sha(mbytes), "n": manifest["n"], "files": sorted(files)}


def _current(root: Path):
    try:
        return (root / CURRENT).read_text().strip()
    except OSError:
        return None


def _cleanup(root: Path, name: str):
    """Remove older checkpoints, leftovers of crashed saves and a stale `.CURRENT.tmp`; never `name` (this save's) nor
    the folder `CURRENT` names (read again before every removal)."""
    for old in sorted(root.iterdir()):
        if old.name == name or old.name == _current(root):
            continue
        if old.is_dir() and (old.name.startswith("ckpt-") or old.name.startswith(".tmp-ckpt-")):
            shutil.rmtree(old, ignore_errors=True)
        elif old.name == f".{CURRENT}.tmp" and old.is_file():
            old.unlink(missing_ok=True)


def _switch(root: Path, name: str):
    """CURRENT ← name, atomically (a crash leaves the old or the new name, never a partial file)."""
    tmp = root / f".{CURRENT}.tmp"
    _write(tmp, (name + "\n").encode())
    os.replace(tmp, root / CURRENT)
    _fsync(root)


# ───────────────────────────────────────────────────────────── load
def _folder(dir) -> Path:
    d = Path(dir)
    if (d / "manifest.json").exists():
        return d
    cur = d / CURRENT
    if not cur.exists():
        raise FileNotFoundError(f"{d}: no checkpoint here (neither {CURRENT} nor manifest.json)")
    name = cur.read_text().strip()
    if "/" in name or name.startswith("."):
        raise ValueError(f"{cur}: not a checkpoint folder name ({name!r})")
    return d / name


def _verified(folder: Path, secret) -> tuple:
    """→ (manifest, {rel: bytes}) once the signature and every file's hash match."""
    mbytes = (folder / "manifest.json").read_bytes()
    if not (folder / "manifest.sig").is_file():
        raise ValueError(f"{folder}: manifest.sig is missing")
    sig = json.loads((folder / "manifest.sig").read_bytes())
    m = json.loads(mbytes)
    fmt = m.get("format")
    if fmt not in _FORMATS:
        if isinstance(fmt, str) and fmt.startswith("solvi adapter loop checkpoint"):
            raise ValueError(f"{folder}: an adapter loop checkpoint of a format this solvi does not read ({fmt!r}): "
                             "it was written by a newer solvi — load it with that version")
        raise ValueError(f"{folder}: not an adapter loop checkpoint ({fmt!r})")
    if (fmt == FORMAT_ADDITIVE) != bool((m.get("config") or {}).get("additive")):
        raise ValueError(f"{folder}: the format ({fmt!r}) and the configuration's additive "
                         f"({(m.get('config') or {}).get('additive')!r}) disagree")
    # `keyed` is read before the signature check only to pick the refusal; with the secret, a flipped `keyed` fails the
    # HMAC below; without it nothing can be authenticated (see the module docstring).
    keyed = bool(m.get("keyed")) or sig.get("kind") == "hmac-sha256"
    if keyed and not secret:
        raise ValueError("this checkpoint was saved with a secret: give secret= to load it (without the secret it "
                         "cannot be verified, so it is refused)")
    if not keyed and secret:
        raise ValueError("this checkpoint was saved by a loop without a secret; the coin would differ with one")
    want = _sig(secret, mbytes)
    if sig.get("kind") != want["kind"] or not hmac.compare_digest(str(sig.get("sig")), want["sig"]):
        raise ValueError("the manifest's signature does not match: the manifest was changed, or this is another "
                         "secret's checkpoint")
    files = m["files"]
    for rel in files:
        parts = rel.split("/")
        if rel.startswith("/") or "\\" in rel or any(q in ("", ".", "..") for q in parts) or rel in _META:
            raise ValueError(f"checkpoint file name {rel!r} is not allowed")
    extra = sorted(rel for p in folder.rglob("*") if not p.is_dir()
                   for rel in [p.relative_to(folder).as_posix()] if rel not in files and rel not in _META)
    if extra:
        raise ValueError(f"checkpoint folder holds files the manifest does not list (unlisted, refused): {extra[:5]}")
    data = {}
    for rel, h in files.items():
        p = folder / rel
        if not p.is_file():
            raise ValueError(f"checkpoint file {rel} is missing")
        b = p.read_bytes()
        if _sha(b) != h:
            raise ValueError(f"checkpoint file {rel}: SHA-256 {_sha(b)[:12]}, the manifest has {h[:12]} (tampered)")
        data[rel] = b
    return m, data


def _load_version(cls, folder, v, fp, *, part, question, feature, groups):
    params = inspect.signature(cls.load).parameters
    kw = {}
    for k, val in (("part", part), ("question", question), ("feature", feature), ("groups", groups)):
        if k in params and val is not None:
            kw[k] = val
    try:
        return cls.load(folder, v, expect_fp=fp, **kw)
    except FileNotFoundError as e:
        raise ValueError(f"version {v}: {Path(e.filename).name} is missing from the checkpoint") from e


def load_loop(cls, dir, learner, *, system=None, storage=None, versions_dir=None, part=None, initial=None, key=None,
              served=None, clock=None, secret=None, groups=None, additive=None):
    """See the module docstring. learner: the learner as built (fresh: the saved train labels are fed to it again);
    system: the System (or part) the loop served on, when it had one — the served version is installed on it;
    storage: the TraceStorage that holds the loop's records, when it had one; versions_dir: where the loop saves
    new versions (the checkpoint's version files are copied there when missing, so replay(rescore=True) finds
    them); part: the part whose version lock the loop takes (None: the learner's or version 0's); initial:
    version 0's model given again instead of read from its file (its fingerprint must be the recorded one); key,
    served, clock: as the constructor's; secret: the loop's secret (required when it had one); groups: the
    grouping function of version 0's thresholds per group, when it was calibrated with one; additive: what the caller
    expects (None: whatever the checkpoint is) — a replacement checkpoint (v1) loaded with additive=True, or an additive
    one (v2) with additive=False, is refused: a loop never changes its protocol at a load (a replacement loop's chain
    cannot be rebuilt into layers; start an additive loop from its served version as version 0 instead)."""
    if isinstance(secret, str):
        secret = secret.encode()
    secret = bytes(secret) if secret else None
    if secret is None:
        warnings.warn("AdapterLoop.load without secret=: the checkpoint's integrity check is a plain SHA-256 that "
                      "anyone can recompute — it detects damage, not tampering, and the policy, monitor, version and "
                      "label pickles in it are run; load only checkpoints you wrote", CheckpointTrustWarning,
                      stacklevel=3)
    folder = _folder(dir)
    m, data = _verified(folder, secret)
    saved_additive = m["format"] == FORMAT_ADDITIVE
    if additive is not None and bool(additive) != saved_additive:
        raise ValueError(
            "this checkpoint is of an additive loop (format v2): it loads as one, not with additive=False"
            if saved_additive else
            "this checkpoint is of a replacement loop (format v1): it loads as one, not with additive=True — a loop "
            "never changes its protocol at a load; start an additive loop with its served version as initial=")
    with tempfile.TemporaryDirectory(prefix="solvi-ckpt-") as tmp:   # private (0700): the verified bytes only
        vdir = _private_versions(data, Path(tmp))
        return _load_verified(cls, folder, m, data, vdir, learner, system=system, storage=storage,
                              versions_dir=versions_dir, part=part, initial=initial, key=key, served=served,
                              clock=clock, secret=secret, groups=groups)


def _private_versions(data, tmp: Path) -> Path:
    """The verified bytes of the checkpoint's version files, written to a private folder the version loaders read:
    a file swapped in the checkpoint folder after `_verified` checked it is never read."""
    vdir = tmp / "versions"
    vdir.mkdir()
    for rel, b in data.items():
        if rel.startswith("versions/"):
            fname = rel[len("versions/"):]
            if "/" in fname:
                continue
            _write(vdir / fname, b)
    return vdir


def _load_verified(cls, folder, m, data, vdir, learner, *, system, storage, versions_dir, part, initial, key, served,
                   clock, secret, groups):
    st = json.loads(data["state.json"])
    if _cls_path(type(learner)) != m["learner"]:
        raise ValueError(f"the checkpoint's learner is {m['learner']}, not {_cls_path(type(learner))}")
    if learner.seen() != st["seen0"]:
        raise ValueError(f"the learner counts {learner.seen()} labels, the saved loop's counted {st['seen0']} when it "
                         "was built: give the learner as built (fresh)")
    if m["system"] and system is None:
        raise ValueError("the loop served on a System (system=): give it again, the served version is installed on it")
    if (m["store"] is None) != (storage is None):
        raise ValueError("the loop kept its records in a TraceStorage: give storage=" if m["store"] is not None else
                         "the loop kept its records in memory (records.jsonl): no storage= for this checkpoint")
    if storage is not None:
        head = m["store"]["head"]
        chk = storage.verify(anchor=head, deep=False)
        if not chk.get("ok"):
            raise ValueError(f"the store does not hold the checkpoint's chain head ({head}): "
                             f"{chk.get('problems', [])[:3]}")
        got = _items_of(storage, m["question"])
        if got != m["store"]["items"]:
            raise ValueError(f"the store holds {got} records of this loop, the checkpoint {m['store']['items']}: "
                             "records written after the save are not in its state")
    cfg = m["config"]
    vrows = {r["version"]: r for r in st["versions"]}
    # version 0
    v0 = m["version0"]
    fp0 = vrows[0]["fingerprint"]
    if initial is not None:
        if initial.fingerprint() != fp0:
            raise ValueError(f"initial= has fingerprint {initial.fingerprint()[:12]}, version 0 was {str(fp0)[:12]}")
    elif v0["kind"] == "served":
        p0 = part or getattr(learner, "part", None)
        initial = _load_version(_cls_of(v0["cls"]), vdir, 0, fp0, part=p0, question=m["question"],
                                feature=cfg["feature"], groups=groups)
    elif v0["kind"] == "head":
        initial = HeadVersion.load(vdir, 0, expect_fp=fp0).frozen
    the_part = part or getattr(learner, "part", None) or getattr(initial, "part", None)
    if m["weights"] is not None:
        if the_part is None:
            raise ValueError("the loop had a part (its model's weights fingerprint is recorded): give part=")
        if _weights(the_part) != m["weights"]:
            raise ValueError(f"the part's model has weights {str(_weights(the_part))[:12]}, the checkpoint was saved "
                             f"with {m['weights'][:12]}")
    policy = pickle.loads(data["policy.pkl"])  # noqa: S301 — hash-checked against the signed manifest
    monitor = pickle.loads(data["monitor.pkl"])  # noqa: S301
    det = None if m["detector"] is None else RiskDetector(**m["detector"])
    loop = cls(learner, question=m["question"], policy=policy, initial=initial, detector=det,
               monitor=monitor, secret=secret, storage=storage, versions_dir=versions_dir, system=system, key=key,
               served=served, clock=clock, part=the_part, _detector_alone=m["detector_alone"], _batch=m["batch"],
               **cfg)
    loop._policy0 = data["policy0.pkl"]
    loop._monitor0 = pickle.loads(data["monitor0.pkl"])  # noqa: S301
    _restore(loop, m, st, data, vdir, groups)
    if versions_dir is not None:                         # the version files a replay reads: the verified bytes only
        for rel, b in sorted(data.items()):
            if not rel.startswith("versions/"):
                continue
            fname = rel[len("versions/"):]
            dst = loop.versions_dir / fname
            if "/" not in fname and not fname.startswith("0.") and not dst.exists():   # version 0 is never saved there
                _write(dst, b)
    if system is not None:                               # serve the saved served version
        srv = loop._base if loop.served == 0 else loop._vs[loop.served].get("served_as")
        if srv is not None:
            srv.install(system)
    return loop


def _watch_restored(w) -> dict:
    """The watch's counts and its e-process's log wealths ("lw"). A checkpoint saved before the watch had an e-process
    restarts it at wealth 1: still valid, it only delays a rollback."""
    from .policy import watch_eprocess
    out = {k: int(v) for k, v in w.items() if k != "lw"}
    out["lw"] = [float(x) for x in w["lw"]] if w.get("lw") is not None else watch_eprocess().vector()
    return out


def _restore_additive(loop, st):
    a = st.get("additive")
    if a is None:
        raise ValueError("an additive checkpoint without its chain state (state.json has no \"additive\")")
    loop._layers = [int(v) for v in a["layers"]]
    loop._route = {int(v): float(lam) for v, lam in a["route"].items()}
    loop._frozen, loop._relearn = {int(v) for v in a["frozen"]}, {int(v) for v in a["relearn"]}
    dets = {}
    for v, d in a["detectors"].items():
        if d is None:
            dets[int(v)] = None
            continue
        if loop.detector is None:
            raise ValueError(f"layer version {v} has a detector dump and the loop has no detector")
        dets[int(v)] = RiskDetector(**loop.detector.config()).restore(d)
    loop._dets = dets
    loop._left_of = {int(v): int(k) for v, k in a["left_of"].items()}
    loop._refused_why = a["refused_why"]
    loop._zstat = {int(v): {k: int(x) for k, x in z.items()} for v, z in a["zones"].items()}
    if loop._layers[0] != 0 or set(loop._zstat) != set(loop._layers) or not set(loop._route) <= set(loop._layers):
        raise ValueError(f"the additive chain state is inconsistent: layers {loop._layers}, routing thresholds of "
                         f"{sorted(loop._route)}, zone counts of {sorted(loop._zstat)}")


def _restore(loop, m, st, data, vdir, groups):
    loop.n, loop.served, loop.candidate, loop.left = st["n"], st["served"], st["candidate"], st["left"]
    loop.paused, loop.closed, loop.error = st["paused"], st["closed"], st["error"]
    loop._attempt, loop._trainings = st["attempt"], list(st["trainings"])
    loop._history, loop._served_since = list(st["history"]), st["served_since"]
    loop._alarms = [dict(a) for a in st["alarms"]]
    loop._holdout = deque(((bool(a), bool(w)) for a, w in st["holdout"]), maxlen=loop._holdout.maxlen)
    loop._watch = None if st.get("watch") is None else _watch_restored(st["watch"])
    r = st.get("refused") or {}
    loop._refused = {"n": int(r.get("n", 0)), "run": int(r.get("run", 0)), "warned": bool(r.get("warned", False))}
    loop.rate._alone = deque((bool(a) for a in st["rate"]["alone"]), maxlen=loop.rate.window)
    loop.rate._count = int(st["rate"]["count"])
    loop.budget.v = int(st["budget_v"])
    loop._ev_seen = {int(v): k for v, k in st["ev_seen"].items()}
    if loop.detector is not None:
        loop.detector.restore(st["detector"])
    if loop.additive:
        _restore_additive(loop, st)
    # versions (vdir: the private copy of the verified bytes)
    v0_model = loop._vs[0]["model"]
    loop._vs = {}
    for r in st["versions"]:
        v = r["version"]
        promise = None
        if r["promise"] is not None:
            promise = StreamPromise(**loop._pkw, budget=DeltaBudget(loop.budget.delta, loop.budget.schedule))
            promise.restore(r["promise"])
            promise.budget = loop.budget
        scls = _cls_of(r["served_cls"])
        model, srv = None, None
        if v == 0:
            model = v0_model
        elif r["has_model"]:
            if scls is None:
                raise ValueError(f"version {v}: no Served class recorded to load it with")
            srv = _load_version(scls, vdir, v, r["fingerprint"], part=loop.part, question=loop.question,
                                feature=loop.feature, groups=groups)
            model = srv.frozen if isinstance(srv, HeadVersion) else srv
        row = loop._row(v, r["state"], model, promise, r["frozen_at"], r["seen"], r["n_train"], frozenset(r["keys"]),
                        r["fingerprint"])
        row["samples"] = [tuple(s) for s in r["samples"]]
        row["fed"] = list(r["fed"])
        row["discarded_at"] = r["discarded_at"]
        row["served_cls"] = scls
        row["fixed"] = r.get("fixed")                     # label mode's threshold of the version
        if r.get("cmp") is not None:                      # the served version's comparison certificate (1.4.0)
            cmp = StreamPromise(**loop._pkw, budget=DeltaBudget(loop.budget.delta, loop.budget.schedule))
            cmp.restore(r["cmp"])
            from .loop import _Mirror
            cmp.budget = _Mirror(promise)                 # the candidate's index: it spends nothing from the loop's
            row["cmp"] = cmp
        if r["stale"] is not None:
            row["stale"] = r["stale"]
        if v != 0:
            row["served_as"] = srv
        loop._vs[v] = row
    # labels: the store, the splits, and the learner fed again
    rows, split_of = [], {}
    for line in data["labels.jsonl"].decode().splitlines():
        if line:
            row, sp = _label_row(json.loads(line))
            rows.append(row)
            split_of.setdefault(row.key, sp)
    split_of.update(st["split_extra"])
    loop._rows, loop._split_of = rows, split_of
    n_split = {"train": 0, "calib": 0, "holdout": 0}
    for r in rows:
        n_split[split_of[r.key]] += 1
    if n_split != st["split_n"]:
        raise ValueError(f"the label store has {n_split} labels by split, the state {st['split_n']}")
    loop._split_n = dict(st["split_n"])
    loop._labelled = {r.key for r in rows}
    loop._split_keys = {"train": 0, "calib": 0, "holdout": 0}
    for k in loop._labelled:
        loop._split_keys[split_of[k]] += 1
    want = int(st["seen"])
    fed = set()
    for r in rows:
        if loop.live.seen() >= want:
            break
        if split_of[r.key] == "train":
            kw = {k: v for k, v in (("key", r.key), ("source", r.source)) if k in loop._upd_kw}
            loop.live.update(r.x, r.y, **kw)
            fed.add(r.key)
    if loop.live.seen() != want:
        raise ValueError(f"the learner counts {loop.live.seen()} labels after the saved train labels were fed again, "
                         f"the checkpoint {want}: give the learner as built (fresh)")
    loop._fed_keys = fed
    # the records
    if loop.storage is None:
        loop.log = [tuple(json.loads(line)) for line in data["records.jsonl"].decode().splitlines() if line]


__all__ = ["CURRENT", "FORMAT", "FORMAT_ADDITIVE", "CheckpointTrustWarning", "load_loop", "save_loop"]
