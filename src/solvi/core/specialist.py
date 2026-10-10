"""Verifiable specialists: a model proposes, code checks against the source, code renders, the store keeps the run
(stable since 1.1; until then solvi.experimental.specialist, which re-exports every name for one release).

One contract for every specialist (charts first; slides, tables, speech later):

  1. propose — a model (small and fast, an LLM, or a rule-based extractor) writes a typed intermediate spec (a pydantic
     model), never the result itself;
  2. check   — deterministic code verifies the spec against the source and the specialist's rules; what does not verify
     is dropped or changed, each with a reason (an Issue) — it is marked, never invented or repaired by a guess;
  3. render  — deterministic code builds the result from the verified spec only (same verified spec → same bytes);
  4. trace   — every step goes into the run's own hash chain (`Run.trace`): the source's hash, the proposal, the
     check, the output's hash; with a TraceStorage (`Specialist(proposer, storage=...)`) the run is also one record of
     kind "specialist" in the store's chain — the chain the decisions use, so `store.verify()` covers it.
     `Specialist.replay` re-checks the recorded proposal and re-renders it: identical issues and identical bytes, or
     it says what differs.

The determinism contract of replay: replay re-executes code and re-renders from recorded model outputs; it does not
re-call a model. A model proposer need not be deterministic (a remote LLM at temperature 0 returns other outputs across
identical runs), so the record keeps what the proposer returned — the parsed spec, and, when the proposer returned
text (a model's reply), that text as it came — and replay parses that output again (it must give the
recorded spec), checks and renders it, and compares the bytes. Only a record that holds no output at all (a record
written without one) makes replay call the proposer — `Replay.from_record` is then False and `Replay.note` says so, and
the bytes are equal only if the proposer happens to repeat itself. Two runs of the same source give the same bytes
only with a deterministic proposer; `solvi.testing.conformance.check_specialist(..., deterministic=False)` checks the
replay alone.

    from solvi.experimental.charts import ChartSpecialist, RuleProposer
    sp = ChartSpecialist(RuleProposer(), storage="runs.jsonl")
    run = sp.run(text, "revenue by region")
    run.output          # the SVG, or None when nothing verified
    print(run.report()) # what was kept, dropped, changed and why
    sp.replay(run.stored_id, text).ok                    # True: same checks, same bytes, from the stored record
    ChartSpecialist().replay(run.to_dict(), text).ok     # the same from the run's own record (no storage needed)
    replay_runs(store, ChartSpecialist(), texts)         # every stored run of that specialist → [] when all replay

The stored record ("specialist" kind): specialist, version, question, source_sha256 and source_chars (the source itself
is not stored unless `store_source=True`: pass it to replay), proposer, proposal (the spec; for a blocked run the raw proposal and
proposal_error), proposal_raw (the proposer's output when it was text — a model's reply, as it came), issues, kept,
verified, meta, renderer, output_sha256, output_bytes, trace_head (the head of the run's own chain) and — when the
specialist reports them (`fingerprints()`) — fingerprints. `Run.to_dict()` keeps its shape;
`Run.stored_id` names the record. The in-memory Trace stays the run's own summary: the record holds the same data,
flattened, under the store's hash.

A subclass defines `spec_type` (the pydantic model of the proposal), `check(spec, source) -> Checked` and
`render(checked) -> str | None`; the proposer is any callable `(source, question) -> spec | dict` (an `id` attribute,
when it has one, names it in the trace). A proposer is not replayed — its output is recorded; the parse, the check and
the render are, from that record."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

KIND = "specialist"

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
    sha256(prev + canonical(step, data)). Editing any entry breaks every hash after it.

    Not solvi.core.runtime.Trace: a run's own chain, kept with the run (`to_dict` / `replay`). With a storage the run is
    also a record of kind "specialist" in the store's chain (see the module docs), which `verify()` covers."""

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
    stored_id: str | None = None                     # the record of kind "specialist" when the specialist has a storage

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
        raw = (self.trace.step("propose") or {})
        if "spec" in raw and "raw" in raw:
            d["proposal_raw"] = raw["raw"]
        if with_source:
            d["source"] = self.source
        return d


@dataclass
class Replay:
    """The result of replaying a specialist's recorded run: `ok` (true in a boolean context) and the problems found;
    `from_record` — False when the record held no proposer output and the proposer was called (`note` says so);
    `output_sha256` — the hash of the re-rendered output (None when nothing was rendered or nothing re-ran)."""
    ok: bool
    problems: list
    from_record: bool = True
    note: str = ""
    output_sha256: str | None = None

    def __bool__(self):
        return self.ok


class Specialist:
    """The contract (see the module docs). Subclasses set `name`, `version`, `spec_type` and implement `check` and
    `render`; `proposer` is the default proposer for `run`; `storage` (a TraceStorage or a path) keeps every run as a
    record of kind "specialist" (see the module docs). A subclass may report fingerprints of what made a run (a model,
    a renderer) by `fingerprints()`. store_source=True keeps the source text in each stored record ("source"), so
    `replay(stored_id)` needs no text passed — the record then grows by the source's size, and the store holds the text
    (mind what it may contain); the default keeps only its hash and length."""

    name = "specialist"
    version = "1"
    spec_type: type[BaseModel] = BaseModel

    def __init__(self, proposer=None, storage=None, *, store_source=False):
        self.proposer, self.store_source = proposer, bool(store_source)
        if storage is not None:
            from .store import open_storage
            storage = open_storage(storage)
        self.storage = storage

    def fingerprints(self) -> dict | None:
        """What made a run, as fingerprints the stored record keeps (e.g. {"proposer": ..., "model": ...}); None: none
        reported."""
        return None

    # --- the contract (solvi.core.contract): called by applicability, never an echo
    def applicable(self, source, question) -> bool:
        """Does this specialist apply to the source and the question? Checked before the proposer is called: a False
        (or an exception) blocks the run with the issue "not_applicable", the proposer is not called. Override it to
        state the precondition — a specialist is called because its task applies, not because the input has its
        form. Default: always."""
        return True

    def _applies(self, source, question):
        try:
            ok = self.applicable(source, question)
        except Exception as e:  # noqa: BLE001 — a precondition that cannot be evaluated is not met
            return False, f"{type(e).__name__}: {e}"
        return ok is True, None

    def _output_contract(self, out, source):
        """The output's contract: a text (or None), and not the source itself → an Issue, or None."""
        if out is None:
            return None
        if not isinstance(out, str):
            return Issue(BLOCKED, "wrong_type", f"the renderer returned {type(out).__name__}, not a text")
        from .contract import canon
        if canon(out) is not None and canon(out) == canon(source):
            return Issue(BLOCKED, "echo", "the output is the source itself: nothing was read from it")
        return None

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
        spec). With a storage the run is stored (Run.stored_id)."""
        run = self._run(source, question, proposer=proposer, proposal=proposal)
        if self.storage is not None:
            run.stored_id = str(self.storage._append(json.loads(json.dumps(self._record(run), default=str)))["id"])
        return run

    def _record(self, run):
        """The body of a run's record of kind "specialist" (from the run's own chain)."""
        from .store import FORMAT
        t = run.trace
        inp, prop, chk, ren = (t.step(n) or {} for n in ("input", "propose", "check", "render"))
        body = {"v": FORMAT, "kind": KIND, "specialist": inp.get("specialist"), "version": inp.get("version"),
                "question": inp.get("question"), "source_sha256": inp.get("source_sha256"),
                "source_chars": inp.get("source_chars"), "proposer": prop.get("proposer"),
                "proposal": prop["spec"] if "spec" in prop else prop.get("raw"), "proposal_error": prop.get("error"),
                "issues": chk.get("issues") or [], "kept": chk.get("kept") or [], "verified": chk.get("verified"),
                "meta": chk.get("meta") or {}, "renderer": ren.get("renderer"),
                "output_sha256": ren.get("output_sha256"), "output_bytes": ren.get("output_bytes"),
                "trace_head": t.head}
        if "spec" in prop and "raw" in prop:
            body["proposal_raw"] = prop["raw"]
        if self.store_source:
            body["source"] = run.source
        fps = self.fingerprints()
        if fps is not None:
            body["fingerprints"] = _plain(fps)
        return body

    def _run(self, source, question=None, *, proposer=None, proposal=None):
        proposer = proposer or self.proposer
        trace = Trace()
        trace.add("input", {"specialist": self.name, "version": self.version, "source_sha256": sha256(source.encode()),
                            "source_chars": len(source), "question": question})
        raw, spec, pid = proposal, None, "given"
        ok, err = self._applies(source, question)
        if not ok:                                    # checked before the proposer: it is not called
            if proposal is None and proposer is not None:
                pid = getattr(proposer, "id", None) or getattr(proposer, "__name__", None) or type(proposer).__name__
            why = f"{self.name} does not apply to this source" + (f" (its precondition failed: {err})" if err else "")
            trace.add("propose", {"proposer": pid, "error": f"not applicable: {why}", "not_applicable": True})
            return self._blocked(source, question, None, trace, "not_applicable", why)
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
        entry = {"proposer": pid, "spec": spec}
        if isinstance(raw, (str, bytes)):              # the proposer's text (a model's reply): replay parses it again
            entry["raw"] = raw.decode() if isinstance(raw, bytes) else raw
        trace.add("propose", entry)
        return self._finish(source, question, spec, trace)

    def _finish(self, source, question, spec, trace):
        checked = self.check(spec, source)
        out = self.render(checked) if checked.verified is not None else None
        bad = self._output_contract(out, source)
        if bad is not None:                           # the output breaks the contract: withheld, with the reason
            checked = Checked(checked.verified, list(checked.issues) + [bad], checked.kept, checked.meta)
            out = None
        trace.add("check", {"issues": checked.issues, "verified": checked.verified, "kept": checked.kept,
                            "meta": checked.meta})
        trace.add("render", {"renderer": f"{self.name} {self.version}",
                             "output_sha256": sha256(out.encode()) if out is not None else None,
                             "output_bytes": len(out.encode()) if out is not None else 0})
        return Run(self.name, self.version, source, question, spec, checked, out, trace)

    def _blocked(self, source, question, raw, trace, code, message):
        checked = Checked(None, [Issue(BLOCKED, code, message)])
        trace.add("check", {"issues": checked.issues, "verified": None, "kept": []})
        trace.add("render", {"renderer": f"{self.name} {self.version}", "output_sha256": None, "output_bytes": 0})
        return Run(self.name, self.version, source, question, raw, checked, None, trace)

    def replay(self, record, source=None, storage=None, *, proposer=None):
        """Re-parse the recorded proposer output, re-check it against the source and re-render it: the same source, the
        same spec, the same issues, identical output bytes — without calling the proposer (see the determinism contract
        in the module docs). record: a Run, its to_dict() (the run's own chain is verified, and the record's copies
        against it), a stored record of kind "specialist" (its hash is checked) or a stored id (read from `storage`,
        default the specialist's own); source: the text (or in the record). A run that was blocked before any spec (a
        failed proposer, an invalid proposal) has nothing to re-check: its record is verified, and a recorded invalid
        output must still be invalid. Only a record with no proposer output at all calls `proposer` (default the
        specialist's own): Replay.from_record is then False and Replay.note says so."""
        if isinstance(record, str):
            storage = storage if storage is not None else self.storage
            if storage is None:
                raise ValueError("replay by a stored id: the specialist has no storage — pass storage=")
            if not hasattr(storage, "record"):
                from .store import open_storage
                storage = open_storage(storage)
            record = storage.record(record)
        if isinstance(record, dict) and record.get("kind") == KIND:
            return self._replay_stored(record, source, proposer)
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
        prop = trace.step("propose")
        rsha = (trace.step("render") or {}).get("output_sha256")
        if prop is None:                              # no proposer output recorded: the proposer is called, and it says so
            return self._repropose(source, record.get("question"), rsha, record.get("output_sha256"), problems, proposer)
        if prop.get("not_applicable") and self._applies(source, record.get("question"))[0]:
            problems.append("recorded as not applicable; the specialist applies to this source now")
        if "spec" not in prop:
            if "error" in prop:
                self._still_invalid(prop.get("raw"), problems)
                return Replay(not problems, problems)
            return Replay(False, problems + ["the run had no valid proposal"])
        if canonical(prop["spec"]) != canonical(record.get("proposal")):
            problems.append("the record's proposal is not the one in the trace")
        if "raw" in prop and canonical(prop["raw"]) != canonical(record.get("proposal_raw", prop["raw"])):
            problems.append("the record's model output is not the one in the trace")
        spec = self._reparse(prop["spec"], prop.get("raw"), "raw" in prop, problems)
        if spec is None:
            return Replay(False, problems)
        again = self._finish(source, record.get("question"), spec, Trace())
        return self._compare(again, chk, rsha, record.get("output_sha256"), problems)

    def _reparse(self, spec, raw, has_raw, problems):
        """The spec to re-check: the recorded raw output parsed again (it must give the recorded spec), else the recorded
        spec. None (with a problem) when the recorded output does not parse now."""
        if not has_raw:
            try:
                return self.parse(spec)
            except (ValidationError, ValueError, TypeError) as e:
                problems.append(f"the recorded proposal is not a valid {self.spec_type.__name__}: {e}"[:300])
                return None
        try:
            again = self.parse(raw)
        except (ValidationError, ValueError, TypeError, AttributeError) as e:
            problems.append(f"the recorded model output does not parse now: {e}"[:300])
            return None
        if canonical(again) != canonical(spec):
            problems.append("re-parsing the recorded model output gives another spec now")
        return again

    def _still_invalid(self, raw, problems):
        """A run blocked on an invalid proposal: its recorded output must not parse now either."""
        if raw is None:
            return
        try:
            self.parse(raw)
        except Exception:  # noqa: BLE001 — still invalid, as recorded
            return
        problems.append("the recorded model output parses now (it was an invalid proposal when recorded)")

    def _repropose(self, source, question, sha_a, sha_b, problems, proposer):
        """A record with no proposer output: call the proposer (the replay is then not from the record)."""
        proposer = proposer or self.proposer
        if proposer is None:
            return Replay(False, problems + ["the record holds no proposer output and no proposer was given to call"],
                          from_record=False, note="no proposer output recorded; no proposer to call")
        again = self._run(source, question, proposer=proposer)
        sha = sha256(again.output.encode()) if again.output is not None else None
        if sha != sha_a or sha != sha_b:
            problems.append("the record holds no proposer output; the proposer was called and its output differs")
        return Replay(not problems, problems, from_record=False, output_sha256=sha,
                      note="the record holds no proposer output: replay called the proposer (not a replay from the record)")

    def _compare(self, again, chk, sha_a, sha_b, problems):
        if canonical(again.checked.issues) != canonical(chk.get("issues")):
            problems.append("the check gives other issues now")
        if canonical(again.checked.verified) != canonical(chk.get("verified")):
            problems.append("the check verifies another spec now")
        sha = sha256(again.output.encode()) if again.output is not None else None
        if sha != sha_a or sha != sha_b:
            problems.append("the re-rendered output differs from the recorded one")
        return Replay(not problems, problems, output_sha256=sha)

    def _replay_stored(self, d, source, proposer=None):
        """Replay a stored record of kind "specialist" (see replay)."""
        from .store import record_hash
        problems = []
        if "hash" in d and d.get("hash") != record_hash(d):
            problems.append("the stored record was edited after it was stored (its hash does not match)")
        source = d.get("source") if source is None else source
        if source is None:
            return Replay(False, problems + ["no source: pass the text the run read (the store keeps its hash only; "
                                             "Specialist(..., store_source=True) keeps the text)"])
        if d.get("source_sha256") != sha256(source.encode()):       # nothing to re-check against another text
            return Replay(False, problems + ["the source is not the one the run read (its sha256 differs)"])
        if d.get("specialist") != self.name:
            problems.append(f"recorded by {d.get('specialist')!r}, replayed by {self.name!r}")
        if d.get("version") != self.version:
            problems.append(f"recorded by version {d.get('version')!r}, replayed by {self.version!r}")
        if d.get("proposal_error") is not None:
            if str(d["proposal_error"]).startswith("not applicable") and self._applies(source, d.get("question"))[0]:
                problems.append("recorded as not applicable; the specialist applies to this source now")
            if "proposal_error" in d and d.get("proposal") is not None:
                self._still_invalid(d["proposal"], problems)
            return Replay(not problems, problems)
        if d.get("proposal") is None:                 # no proposer output recorded: the proposer is called, and it says so
            return self._repropose(source, d.get("question"), d.get("output_sha256"), d.get("output_sha256"), problems,
                                   proposer)
        spec = self._reparse(d["proposal"], d.get("proposal_raw"), "proposal_raw" in d, problems)
        if spec is None:
            return Replay(False, problems)
        again = self._finish(source, d.get("question"), spec, Trace())
        return self._compare(again, d, d.get("output_sha256"), d.get("output_sha256"), problems)


def replay_runs(storage, specialist, sources) -> list:
    """Replay every stored run of `specialist` (its records of kind "specialist" with its name) → the ones that do not
    replay: [{"id", "seq", "problems"}] ([] — all replay). sources: the texts the runs read — a list of texts or {sha256:
    text}; a record that kept its source (store_source=True) uses it when the texts do not hold it; a run whose source
    is not among them is listed ("no source given ...")."""
    if not hasattr(storage, "iter"):
        from .store import open_storage
        storage = open_storage(storage)
    texts = dict(sources) if isinstance(sources, dict) else {sha256(t.encode()): t for t in sources}
    bad = []
    for s in storage.iter(KIND):
        d = s.data
        if d.get("specialist") != specialist.name:
            continue
        src = texts.get(str(d.get("source_sha256")), d.get("source"))
        if src is None:
            bad.append({"id": s.id, "seq": s.seq, "problems": [f"no source given with sha256 {d.get('source_sha256')}"]})
            continue
        r = specialist.replay(d, src)
        if not r.ok:
            bad.append({"id": s.id, "seq": s.seq, "problems": r.problems})
    return bad


# ───────────────────────────────────────────────────────────── two readers: alignment of two hypotheses
@dataclass(frozen=True)
class Reading:
    """One reader's reading of a span of the source: its value, where it is (`span`: (start, end) for an interval —
    seconds, characters — or (x0, y0, x1, y1) for a box), the reader's confidence (None: not given) and what it read
    before any normalising (`raw`)."""
    value: Any
    span: tuple
    confidence: float | None = None
    raw: Any = None


@dataclass(frozen=True)
class Agreed:
    """A value both readers gave on overlapping spans: the value (a's), both readings — with their spans and
    confidences, for the provenance record ("quoted by two readers") — and the overlap."""
    value: Any
    a: Reading
    b: Reading
    overlap: float


@dataclass
class Alignment:
    """align's result: `agreed` (the readings a check may accept), `only_a` / `only_b` (readings without a counterpart
    at min_overlap or above), `conflicts` [(a, b, overlap)] (overlapping spans, different values: material for a
    conflict or a review)."""
    agreed: list = field(default_factory=list)
    only_a: list = field(default_factory=list)
    only_b: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)


def interval_overlap(a, b) -> float:
    """Two intervals (start, end) → the length they share over the shorter one's length, in [0, 1] (0 for an empty
    span)."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    shorter = min(a[1] - a[0], b[1] - b[0])
    if inter <= 0 or shorter <= 0:
        return 0.0
    return min(1.0, inter / shorter)


def box_iou(a, b) -> float:
    """Two boxes (x0, y0, x1, y1) → intersection over union, in [0, 1] (touching boxes: 0)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def align(a, b, overlap=interval_overlap, min_overlap=0.5, same=None, min_a=0.0, min_b=0.0) -> Alignment:
    """Two readers' readings of one source (two speech models' words with their times, two OCR engines' words with
    their boxes, two extractors' quotes with their character offsets) → Alignment.

    a, b: Readings. Readings whose confidence is below min_a / min_b are dropped before matching (None passes; a
    dropped reading is in no list). Then pairs are matched one-to-one, greedily by descending `overlap(span_a, span_b)`
    (interval_overlap, box_iou or your own, in [0, 1]), ties by (a's order, b's order); a pair below `min_overlap` is
    not matched, and a reading matched once is not reused. A matched pair is `agreed` when `same(value_a, value_b)`
    (default: equality — a normaliser goes here), else a conflict. agreed, conflicts and only_a come in a's order,
    only_b in b's.

    Nothing here decides acceptance: a specialist's check takes `agreed` as what it may accept and `conflicts` as its
    conflict / review material; the thresholds are the task's (calibrated), not this function's."""
    same = same or (lambda x, y: x == y)
    ka = [r for r in a if r.confidence is None or r.confidence >= min_a]
    kb = [r for r in b if r.confidence is None or r.confidence >= min_b]
    pairs = []
    for i, ra in enumerate(ka):
        for j, rb in enumerate(kb):
            o = float(overlap(ra.span, rb.span))
            if o >= min_overlap and o > 0:
                pairs.append((-o, i, j))
    pairs.sort()
    used_a, used_b, matched = set(), set(), {}
    for o, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched[i] = (j, -o)
    out = Alignment()
    for i, ra in enumerate(ka):
        if i not in matched:
            out.only_a.append(ra)
            continue
        j, o = matched[i]
        rb = kb[j]
        if same(ra.value, rb.value):
            out.agreed.append(Agreed(ra.value, ra, rb, o))
        else:
            out.conflicts.append((ra, rb, o))
    out.only_b = [rb for j, rb in enumerate(kb) if j not in used_b]
    return out


__all__ = ["Agreed", "align", "Alignment", "BLOCKED", "box_iou", "canonical", "CHANGED", "Checked", "DROPPED",
           "interval_overlap", "Issue", "KIND", "Reading", "Replay", "replay_runs", "Run", "SEVERITIES", "sha256",
           "Specialist", "Trace", "WARNING"]
