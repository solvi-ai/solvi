"""Verifiable specialists: a model proposes, code checks against the source, code renders, everything is in the trace.

One contract for every specialist (charts first; slides, tables, speech later):

  1. propose — a model (small and fast, an LLM, or a rule-based extractor) writes a typed intermediate spec (a pydantic
     model), never the result itself;
  2. check   — deterministic code verifies the spec against the source and the specialist's rules; what does not verify
     is dropped or changed, each with a reason (an Issue) — it is marked, never invented or repaired by a guess;
  3. render  — deterministic code builds the result from the verified spec only (same verified spec → same bytes);
  4. trace   — every step goes into a hash chain: the source's hash, the proposal, the check, the output's hash.
     `Specialist.replay` re-checks the recorded proposal and re-renders it: identical issues and identical bytes, or
     it says what differs.

    from solvi.charts import ChartSpecialist, RuleProposer
    run = ChartSpecialist(RuleProposer()).run(text, "revenue by region")
    run.output          # the SVG, or None when nothing verified
    print(run.report()) # what was kept, dropped, changed and why
    ChartSpecialist().replay(run.to_dict(), text).ok     # True: same checks, same bytes

A subclass defines `spec_type` (the pydantic model of the proposal), `check(spec, source) -> Checked` and
`render(checked) -> str | None`; the proposer is any callable `(source, question) -> spec | dict` (an `id` attribute,
when it has one, names it in the trace). A proposer is not replayed — its output is recorded; the check and the render
are, from that record."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

# what happened to a part of the proposal
DROPPED = "dropped"       # not verified: left out of the result
CHANGED = "changed"       # verified, but the proposal could not be kept as it was (e.g. a pie drawn as bars)
WARNING = "warning"       # kept, with a remark the reader should see
BLOCKED = "blocked"       # nothing can be produced (no valid proposal, nothing verified)
SEVERITIES = (DROPPED, CHANGED, WARNING, BLOCKED)


@dataclass(frozen=True)
class Issue:
    """One finding of the check: where in the spec (`path`, e.g. "series[0].points[2]"), what (`code`, stable,
    machine-readable), how bad (`severity`) and why (`message`, for people)."""
    severity: str
    code: str
    message: str
    path: str = ""

    def __str__(self):
        return f"{self.severity}: {self.path + ' — ' if self.path else ''}{self.message} [{self.code}]"


@dataclass
class Checked:
    """The check's result: the verified spec (only what passed; None when nothing did) and every issue."""
    verified: Any
    issues: list = field(default_factory=list)
    kept: list = field(default_factory=list)       # human-readable lines of what was verified
    meta: dict = field(default_factory=dict)       # counts and the like, for the renderer and the report

    def of(self, severity):
        return [i for i in self.issues if i.severity == severity]


def canonical(obj):
    """Canonical JSON bytes (sorted keys, no spaces, UTF-8): the input of every hash in the trace."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _plain(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)                                  # Decimal and the like: their exact text


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


class Trace:
    """A hash chain of steps: entry i stores its step, its data, the previous entry's hash and its own hash =
    sha256(prev + canonical(step, data)). Editing any entry breaks every hash after it."""

    GENESIS = "0" * 64

    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def add(self, step, data):
        prev = self.entries[-1]["hash"] if self.entries else self.GENESIS
        data = _plain(data)
        h = sha256(prev.encode() + canonical({"step": step, "data": data}))
        self.entries.append({"step": step, "data": data, "prev": prev, "hash": h})
        return h

    @property
    def head(self):
        return self.entries[-1]["hash"] if self.entries else self.GENESIS

    def step(self, name):
        for e in self.entries:
            if e["step"] == name:
                return e["data"]
        return None

    def verify(self):
        """→ a list of problems ([] when the chain is intact)."""
        out, prev = [], self.GENESIS
        for i, e in enumerate(self.entries):
            if e.get("prev") != prev:
                out.append(f"entry {i} ({e.get('step')}): its prev is not the hash of entry {i - 1}")
            h = sha256(prev.encode() + canonical({"step": e.get("step"), "data": e.get("data")}))
            if e.get("hash") != h:
                out.append(f"entry {i} ({e.get('step')}): its hash does not match its content")
            prev = e.get("hash") or ""
        return out


@dataclass
class Run:
    """One run of a specialist: the proposal, the check, the output (None when nothing verified) and the trace."""
    specialist: str
    version: str
    source: str
    question: str | None
    proposal: Any
    checked: Checked
    output: str | None
    trace: Trace

    @property
    def ok(self):
        """True when there is an output (with or without dropped parts: see `issues`)."""
        return self.output is not None

    @property
    def issues(self):
        return list(self.checked.issues)

    @property
    def dropped(self):
        return self.checked.of(DROPPED)

    def report(self):
        """What was kept, dropped, changed and why — plain text, one line per item."""
        lines = [f"{self.specialist} {self.version}: " + ("rendered" if self.ok else "NOT rendered")
                 + f" · trace {self.trace.head[:12]}"]
        lines += [f"  kept: {k}" for k in self.checked.kept]
        for sev in (BLOCKED, DROPPED, CHANGED, WARNING):
            lines += [f"  {sev}: {(i.path + ' — ') if i.path else ''}{i.message}" for i in self.checked.of(sev)]
        return "\n".join(lines)

    def to_dict(self, with_source=False):
        """The record to store: everything replay needs except the source (pass with_source=True to include it)."""
        d = {"specialist": self.specialist, "version": self.version, "question": self.question,
             "proposal": _plain(self.proposal), "issues": [dataclasses.asdict(i) for i in self.checked.issues],
             "output_sha256": sha256(self.output.encode()) if self.output is not None else None,
             "trace": self.trace.entries}
        if with_source:
            d["source"] = self.source
        return d


@dataclass
class Replay:
    ok: bool
    problems: list

    def __bool__(self):
        return self.ok


class Specialist:
    """The contract (see the module docs). Subclasses set `name`, `version`, `spec_type` and implement `check` and
    `render`; `proposer` is the default proposer for `run`."""

    name = "specialist"
    version = "1"
    spec_type: type[BaseModel] = BaseModel

    def __init__(self, proposer=None):
        self.proposer = proposer

    # --- to implement
    def check(self, spec, source) -> Checked:            # pragma: no cover - abstract
        raise NotImplementedError

    def render(self, checked: Checked) -> str | None:    # pragma: no cover - abstract
        raise NotImplementedError

    # --- the pipeline
    def parse(self, proposal):
        """A proposer's output (a spec_type, a dict or a JSON string) → spec_type; ValidationError / ValueError."""
        if isinstance(proposal, self.spec_type):
            return proposal
        if isinstance(proposal, (str, bytes)):
            return self.spec_type.model_validate_json(proposal)
        return self.spec_type.model_validate(proposal)

    def run(self, source, question=None, *, proposer=None, proposal=None):
        """source (a text) → Run. proposer overrides the default; proposal= skips proposing (a recorded or hand-written
        spec)."""
        proposer = proposer or self.proposer
        trace = Trace()
        trace.add("input", {"specialist": self.name, "version": self.version, "source_sha256": sha256(source.encode()),
                            "source_chars": len(source), "question": question})
        raw, spec, pid = proposal, None, "given"
        if proposal is None:
            if proposer is None:
                raise ValueError(f"{self.name}: no proposer and no proposal")
            pid = getattr(proposer, "id", None) or getattr(proposer, "__name__", None) or type(proposer).__name__
            try:
                raw = proposer(source, question)
            except Exception as e:  # noqa: BLE001 — a failing proposer is an issue in the trace, not a crash
                raw = None
                trace.add("propose", {"proposer": pid, "error": f"{type(e).__name__}: {e}"})
                return self._blocked(source, question, None, trace, "proposer_failed",
                                     f"the proposer failed: {type(e).__name__}: {e}")
        try:
            spec = self.parse(raw)
        except (ValidationError, ValueError, TypeError) as e:
            trace.add("propose", {"proposer": pid, "raw": _plain(raw), "error": str(e)[:2000]})
            return self._blocked(source, question, raw, trace, "invalid_proposal",
                                 f"the proposal is not a valid {self.spec_type.__name__}: {str(e).splitlines()[0]}")
        trace.add("propose", {"proposer": pid, "spec": spec})
        return self._finish(source, question, spec, trace)

    def _finish(self, source, question, spec, trace):
        checked = self.check(spec, source)
        trace.add("check", {"issues": checked.issues, "verified": checked.verified, "kept": checked.kept,
                            "meta": checked.meta})
        out = self.render(checked) if checked.verified is not None else None
        trace.add("render", {"renderer": f"{self.name} {self.version}",
                             "output_sha256": sha256(out.encode()) if out is not None else None,
                             "output_bytes": len(out.encode()) if out is not None else 0})
        return Run(self.name, self.version, source, question, spec, checked, out, trace)

    def _blocked(self, source, question, raw, trace, code, message):
        checked = Checked(None, [Issue(BLOCKED, code, message)])
        trace.add("check", {"issues": checked.issues, "verified": None, "kept": []})
        trace.add("render", {"renderer": f"{self.name} {self.version}", "output_sha256": None, "output_bytes": 0})
        return Run(self.name, self.version, source, question, raw, checked, None, trace)

    def replay(self, record, source=None):
        """Re-check the recorded proposal against the source and re-render it: the trace chain intact, the same source,
        the same issues — in the trace and in the record's own `issues` — identical output bytes. A run that was blocked
        before any spec (a failed proposer, an invalid proposal) has nothing to re-check: its chain and its record's
        copies are verified. record: a Run or its to_dict(); source: the text (or in the record)."""
        if isinstance(record, Run):
            source = record.source if source is None else source
            record = record.to_dict()
        source = record.get("source") if source is None else source
        if source is None:
            return Replay(False, ["no source: pass the text the run read"])
        trace = Trace(record.get("trace"))
        problems = trace.verify()
        inp = trace.step("input") or {}
        if inp.get("source_sha256") != sha256(source.encode()):
            problems.append("the source is not the one the run read (its sha256 differs)")
        if inp.get("specialist") != self.name:
            problems.append(f"recorded by {inp.get('specialist')!r}, replayed by {self.name!r}")
        if inp.get("version") != self.version:
            problems.append(f"recorded by version {inp.get('version')!r}, replayed by {self.version!r}")
        chk = trace.step("check") or {}
        if canonical(record.get("issues") or []) != canonical(chk.get("issues") or []):   # the record's own copies of
            problems.append("the record's issues are not the ones in the trace")           # what the trace holds
        if record.get("output_sha256") != (trace.step("render") or {}).get("output_sha256"):
            problems.append("the record's output hash is not the one in the trace")
        prop = trace.step("propose") or {}
        if "spec" not in prop:
            return Replay(not problems, problems + (["the run had no valid proposal"] if "error" not in prop else []))
        if canonical(prop["spec"]) != canonical(record.get("proposal")):
            problems.append("the record's proposal is not the one in the trace")
        again = self._finish(source, record.get("question"), self.parse(prop["spec"]), Trace())
        if canonical(again.checked.issues) != canonical(chk.get("issues")):
            problems.append("the check gives other issues now")
        if canonical(again.checked.verified) != canonical(chk.get("verified")):
            problems.append("the check verifies another spec now")
        sha = sha256(again.output.encode()) if again.output is not None else None
        if sha != (trace.step("render") or {}).get("output_sha256") or sha != record.get("output_sha256"):
            problems.append("the re-rendered output differs from the recorded one")
        return Replay(not problems, problems)
