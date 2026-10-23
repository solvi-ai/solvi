"""Versions of a document and their impact (stable since 1.1; until then in solvi.experimental.compile): a specification
as numbered clauses, a catalog that implements it as a version citing those clauses, a folder of versions with their
parents and the version in force, and the decisions a new version moves — with the clauses why.

    from solvi.core.versions import CatalogVersion, Spec, Versions, decision_diff

    spec = Spec(policy_text)                       # clauses c1, c2, ... (paragraphs, list items, table rows)
    spec2 = spec.revise(new_text)                  # unchanged clauses keep their ids; spec2.changes
    v1 = CatalogVersion(spec, questions, module_source, parts)    # parts: {name: {"kind", "clauses", ...}}
    versions = Versions("policy_versions")         # v1/, v2/ ... each module.py + compiled.json + version.json
    versions.add(v1, "first"); versions.add(v2, "after the change")     # each with its parent: the version in force
    print(decision_diff(v1, v2, store=store))      # which stored decisions change, and the clauses of their causes
    versions.replay_all(store)                     # every stored decision against the version that made it
    versions.rollback(1)                           # the version in force goes back (every version is kept)

`decision_diff` also takes plain Systems (hand-written catalogs whose parts cite their clauses with `clause=` /
`cite=`), and `only=` names the decisions a change may move: `extraneous` lists the others it moves, `exact` says
there were none and every decision re-ran. A version's module is pure functions, checked before it is loaded
(`_sandbox`: standard-library imports only, no files, reflection or dunders) — the compiler that writes such modules
with an LLM stays experimental (solvi.experimental.compile, whose Compiled is a CatalogVersion and which re-exports
every name here for one release)."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..catalog import Catalog, Question
from . import _sandbox


def _extra():
    """The trusted names a version's module may use without an import: Fail (a check that says why)."""
    from ..slow.refine import Fail
    return {"Fail": Fail}


class Rejected(RuntimeError):
    """A compilation that was not accepted was asked for its catalog. `.reason` says why it was not accepted."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"the compilation was not accepted: {reason}")


# ───────────────────────────────────────────────────────────── the specification as clauses
_CL_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CL_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_CL_SEP = re.compile(r"^:?-{2,}:?$")


def split_clauses(text: str) -> list[tuple[str, str]]:
    """A text → [(section, clause text)]: each paragraph, list item and table row is a clause (a table row is written
    as `header: cell; ...`); headings name the section and are not clauses themselves."""
    out, buf, path, header = [], [], [], None

    def flush():
        if buf:
            out.append((" / ".join(t for _, t in path), " ".join(buf).strip()))
            buf.clear()
    for line in text.splitlines():
        s = line.strip()
        h = _CL_HEADING.match(s)
        if h:
            flush()
            header = None
            level = len(h.group(1))
            path[:] = [p for p in path if p[0] < level] + [(level, h.group(2))]
            continue
        if s.startswith("|"):
            flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(_CL_SEP.match(c) for c in cells if c):
                continue
            if header is None:
                header = cells
                continue
            out.append((" / ".join(t for _, t in path), "; ".join(f"{a}: {b}" for a, b in zip(header, cells) if b)))
            continue
        header = None
        if not s:
            flush()
        elif _CL_ITEM.match(line):
            flush()
            buf.append(_CL_ITEM.match(line).group(1).strip())
        else:
            buf.append(s)
    flush()
    return [(sec, t) for sec, t in out if t]


@dataclass
class Clause:
    id: str
    text: str
    section: str = ""


class Spec:
    """A specification as numbered clauses (`c1`, `c2`, ...). `Spec(text)` splits a text (Markdown-like: paragraphs, list
    items and table rows are clauses, headings are sections); `Spec(clauses={id: text})` takes them as given.
    `spec.revise(new_text)` aligns a changed text with this one: unchanged clauses keep their ids, and `changes` says
    which clauses changed, were added or removed."""

    def __init__(self, text: str | None = None, *, clauses=None, name: str = "specification"):
        self.name = name
        self.changes: dict | None = None
        if clauses is not None:
            items = clauses.items() if isinstance(clauses, dict) else [(f"c{i}", t) for i, t in enumerate(clauses, 1)]
            self.clauses = {str(k): Clause(str(k), str(v)) for k, v in items}
            self.text = "\n\n".join(c.text for c in self.clauses.values()) if text is None else text
        elif text is not None:
            self.text = text
            self.clauses = {f"c{i}": Clause(f"c{i}", t, sec) for i, (sec, t) in enumerate(split_clauses(text), 1)}
        else:
            raise ValueError("Spec needs a text or clauses")
        if not self.clauses:
            raise ValueError("the specification has no clauses")

    @property
    def hash(self) -> str:
        """sha256 of the clauses (ids, sections, texts)."""
        blob = json.dumps([[c.id, c.section, c.text] for c in self.clauses.values()], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def render(self, marks: dict | None = None) -> str:
        """The clauses as the writer reads them: `[c3] (section) text`, with `marks` ({id: "changed"}) in front."""
        out, sec = [], None
        for c in self.clauses.values():
            if c.section != sec:
                sec = c.section
                if sec:
                    out.append(f"\n### {sec}")
            m = f"[{marks[c.id]}] " if marks and c.id in marks else ""
            out.append(f"[{c.id}] {m}{c.text}")
        return "\n".join(out).strip()

    def revise(self, text: str | None = None, *, clauses=None) -> "Spec":
        """A new version of this specification, aligned with it: a clause whose text (and section) did not change keeps its
        id; a replaced clause keeps the id of the one it replaces (`changed`); new clauses get new ids (`added`); the
        others are `removed`. `new.changes` = {"unchanged", "changed", "added", "removed"} (lists of ids)."""
        if clauses is not None:
            items = list(clauses.values()) if isinstance(clauses, dict) else list(clauses)
            new = [("", str(t)) for t in items]
        else:
            new = split_clauses(text)
        old = list(self.clauses.values())
        sm = difflib.SequenceMatcher(a=[(c.section, c.text) for c in old], b=new, autojunk=False)
        n = max(int(i[1:]) for i in self.clauses if re.fullmatch(r"c\d+", i)) if any(
            re.fullmatch(r"c\d+", i) for i in self.clauses) else 0
        ids = [None] * len(new)
        ch = {"unchanged": [], "changed": [], "added": [], "removed": []}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    ids[j1 + k] = old[i1 + k].id
                    ch["unchanged"].append(old[i1 + k].id)
                continue
            pairs = min(i2 - i1, j2 - j1) if tag == "replace" else 0
            for k in range(pairs):
                ids[j1 + k] = old[i1 + k].id
                ch["changed"].append(old[i1 + k].id)
            for k in range(j1 + pairs, j2):
                n += 1
                ids[k] = f"c{n}"
                ch["added"].append(ids[k])
            ch["removed"] += [old[k].id for k in range(i1 + pairs, i2)]
        s = Spec.__new__(Spec)
        s.name, s.text = self.name, text if text is not None else "\n\n".join(t for _, t in new)
        s.clauses = {i: Clause(i, t, sec) for i, (sec, t) in zip(ids, new)}
        s.changes = ch
        s.parent = self.hash
        return s

    def to_dict(self):
        return {"name": self.name, "hash": self.hash, "clauses": [[c.id, c.section, c.text] for c in self.clauses.values()],
                "changes": self.changes}

    @classmethod
    def from_dict(cls, d):
        s = cls.__new__(cls)
        s.name, s.changes = d.get("name", "specification"), d.get("changes")
        s.clauses = {i: Clause(i, t, sec) for i, sec, t in d["clauses"]}
        s.text = "\n\n".join(c.text for c in s.clauses.values())
        return s


# ───────────────────────────────────────────────────────────── a catalog version
def build_catalog(ns: dict, parts: dict, questions) -> tuple[Catalog, list]:
    """The catalog of a loaded module (in PARTS order) and the questions, each requiring the hard checks that name it."""
    cat = Catalog()
    for name, p in parts.items():
        f = ns[name]
        if p["kind"] == "fn":
            cat.fn(f)
        elif p["kind"] == "check":
            cat.check(f, hard=bool(p.get("hard")), then=dict(p.get("then") or {}) or None)
        else:
            cat.rule(p["question"])(f)
    qs = []
    for q in questions:
        need = [n for n, p in parts.items() if p["kind"] == "check" and p.get("hard") and q.name in (p.get("then") or {})]
        qs.append(Question(q.name, q.text, q.answer, requires=list(dict.fromkeys(list(q.requires) + need)),
                           uses=q.uses, min_confidence=q.min_confidence, require_evidence=q.require_evidence))
    return cat, qs


def _norm(a):
    if isinstance(a, bool):
        return "yes" if a else "no"
    return None if a is None else str(a)


@dataclass
class CatalogVersion:
    """A version of a catalog that implements a document: the module's source (pure functions — a part's name is the
    fact it sets, its argument names what it reads), `parts` ({name: {"kind": "fn" | "check" | "rule", "clauses": [...],
    for a hard check "hard" and "then", for a rule "question"}}), the questions, and the specification whose clauses the
    parts cite. `not_normative` lists clauses no part implements, with why; `accepted` / `reason` say whether the version
    may be loaded (a draft that was not accepted is refused); `record` and `changes` are free for whoever made it (the
    compiler keeps its prompts, tests and rounds there). `catalog()` / `system()` load the module in this process —
    checked first: standard-library imports only, no files, reflection or dunders (solvi.core.versions._sandbox) — and
    refuse an unaccepted version unless `allow_unaccepted=True`. solvi.experimental.compile.Compiled is a
    CatalogVersion."""
    spec: Spec
    questions: list
    source: str
    parts: dict
    not_normative: dict = field(default_factory=dict)
    accepted: bool = True
    reason: str = ""
    record: dict = field(default_factory=dict)
    changes: dict | None = None
    _ns: dict | None = field(default=None, repr=False)

    def clauses_of(self, part: str) -> list:
        """The clauses a part implements (a rule may be named `answer:<question>`)."""
        if part.startswith("answer:"):
            q = part.split(":", 1)[1]
            part = next((n for n, p in self.parts.items() if p.get("kind") == "rule" and p.get("question") == q), part)
        return list((self.parts.get(part) or {}).get("clauses") or [])

    def part_of(self, step: str) -> str | None:
        """The PARTS name of a trace step (`answer:<question>` → its rule's function)."""
        if step.startswith("answer:"):
            q = step.split(":", 1)[1]
            return next((n for n, p in self.parts.items() if p.get("kind") == "rule" and p.get("question") == q), None)
        return step if step in self.parts else None

    def catalog(self, allow_unaccepted: bool = False) -> tuple[Catalog, list]:
        """→ (Catalog, questions): the module loaded in this process (sandbox.load), each hard check required by the
        questions it names. Raises Rejected for a compilation that was not accepted."""
        if not self.accepted and not allow_unaccepted:
            raise Rejected(self.reason)
        if self._ns is None:
            self._ns = _sandbox.load(self.source, _extra())
        return build_catalog(self._ns, self.parts, self.questions)

    def system(self, allow_unaccepted: bool = False, **kw):
        """An ordinary System over the compiled catalog (kw: System's own — storage, ...)."""
        from ..system import System
        cat, qs = self.catalog(allow_unaccepted)
        return System(cat, qs, **kw)

    @property
    def fingerprint(self) -> str:
        """The catalog's fingerprint (what a stored decision records as the catalog that decided)."""
        return self.system(allow_unaccepted=True).fingerprint()["catalog"]

    def to_dict(self) -> dict:
        return {"spec": self.spec.to_dict(), "questions": [q.model_dump() for q in self.questions], "source": self.source,
                "parts": self.parts, "not_normative": self.not_normative, "accepted": self.accepted,
                "reason": self.reason, "record": self.record, "changes": self.changes}

    @classmethod
    def from_dict(cls, d) -> "CatalogVersion":
        return cls(Spec.from_dict(d["spec"]), [Question.model_validate(q) for q in d["questions"]], d["source"],
                   d["parts"], d.get("not_normative") or {}, d.get("accepted", True), d.get("reason", ""),
                   d.get("record") or {}, d.get("changes"))

    def save(self, path) -> Path:
        """Write `module.py` (the code, as accepted) and `compiled.json` (everything) into a folder."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "module.py").write_text(self.source)
        (p / "compiled.json").write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False, default=str))
        return p

    @classmethod
    def load(cls, path) -> "CatalogVersion":
        d = json.loads((Path(path) / "compiled.json").read_text())
        c = cls.from_dict(d)
        if (Path(path) / "module.py").read_text() != c.source:
            raise ValueError(f"{path}/module.py differs from the source recorded in compiled.json")
        return c


# ───────────────────────────────────────────────────────────── what a change does to decisions
@dataclass
class DecisionDiff:
    """The decisions a new compilation changes: `changed` = [{"id", "question", "old", "new", "causes": [{"step",
    "part", "clauses", "why"}]}] (with `inputs=`, also "input": the input's id); `report` is solvi.core.store.diff's own
    report. With `only=`: `extraneous`, the ids of the decisions whose answer moves (`moved`) that are not in it, and
    `exact`, True when there are none and every decision re-ran (`report.errors` empty)."""
    changed: list
    report: Any
    total: int
    extraneous: list = field(default_factory=list)
    exact: bool = False

    @property
    def moved(self) -> list:
        """The changes whose answer moves (the others keep their answer and are decided another way: by another check,
        or forced where a rule answered)."""
        return [x for x in self.changed if _norm(x["old"]) != _norm(x["new"])]

    def __str__(self):
        from collections import Counter
        moved = self.moved
        c = Counter((x["question"], _norm(x["old"]), _norm(x["new"])) for x in moved)
        lines = [f"{len(moved)} of {self.total} decisions change their answer"]
        lines += [f"  {q}: {o} → {n}: {k}" for (q, o, n), k in c.most_common()]
        cl = Counter(tuple(sorted({c for cause in x["causes"] for c in cause["clauses"]})) for x in moved)
        lines += ["  because of clauses " + ", ".join(k) + f": {v}" for k, v in cl.most_common()]
        same = len(self.changed) - len(moved)
        if same:
            lines.append(f"{same} more keep their answer and are decided another way (another check, or forced)")
        return "\n".join(lines)


def _plain_answer(a):
    return a.get("answer") if isinstance(a, dict) else a


def _status(a):
    return a.get("status") if isinstance(a, dict) else None


def _decides(x):
    """A CatalogVersion or a System → its System (a version's, built for this diff)."""
    return x.system() if isinstance(x, CatalogVersion) else x


def decision_diff(old, new, *, store=None, inputs=None, only=None, **filters) -> DecisionDiff:
    """Which decisions `new` would change: the stored ones (`store`, decided by `old`'s catalog — filters as for
    solvi.core.store.diff.diff) or `inputs` (each decided by `old` first, in a temporary store). old / new: a CatalogVersion
    (a Compiled too), or a plain System (a hand-written catalog). Each change's causes are the steps solvi.core.store.diff names, with the
    clauses their parts implement — a version's parts (in the new compilation, else the old one), for a plain catalog
    the clause ids its parts cite (`clause=` on the decorators) — and "cites", the parts' citations [{"clause",
    "quote"}] when they have them.

    only: the decisions the change is allowed to move — stored decision ids, or with `inputs=` the inputs' ids: an
    input's "id" key when it has one, else its position in `inputs` (0, 1, ...); each change then carries "input".
    `extraneous` lists the moved decisions outside it (by those ids), `exact` is True when there are none and every
    decision re-ran (a decision that could not be re-run makes it False)."""
    from ..store import JSONLStorage
    from ..store.diff import diff, part_cites
    tmp = None
    input_ids: dict = {}
    if store is None:
        if inputs is None:
            raise ValueError("decision_diff needs a store of decisions or inputs")
        tmp = tempfile.mkdtemp(prefix="solvi_ddiff_")
        store = JSONLStorage(str(Path(tmp) / "decisions.jsonl"))
        if isinstance(old, CatalogVersion):
            s_old = old.system(storage=store)
            for i, x in enumerate(inputs):
                input_ids[s_old.ask(dict(x)).stored_id] = x["id"] if isinstance(x, dict) and "id" in x else i
        else:
            store.catalog = old
            for i, x in enumerate(inputs):
                input_ids[store.save(old.ask(dict(x), store=False))] = x["id"] if isinstance(x, dict) and "id" in x else i
    try:
        s_new = _decides(new)
        rep = diff(store, s_new, **filters)
        s_old = _decides(old) if not isinstance(old, CatalogVersion) else None
        out = []
        for ch in rep.changed:
            for q, d in ch["questions"].items():
                if not d.get("changed"):
                    continue
                causes = []
                for c in d.get("causes") or []:
                    cites = c.get("cites")
                    if cites is None and s_old is not None:
                        cites = part_cites(s_old.catalog, c["name"])
                    if isinstance(new, CatalogVersion) or isinstance(old, CatalogVersion):
                        part = (new.part_of(c["name"]) if isinstance(new, CatalogVersion) else None) or \
                            (old.part_of(c["name"]) if isinstance(old, CatalogVersion) else None)
                        src = new if isinstance(new, CatalogVersion) and new.part_of(c["name"]) else old
                        clauses = src.clauses_of(part) if part and isinstance(src, CatalogVersion) else []
                    else:
                        part = c["name"]
                        clauses = list(dict.fromkeys(x["clause"] for x in cites or () if x["clause"] is not None))
                    cause = {"step": c["name"], "part": part, "clauses": clauses, "why": c.get("why")}
                    if cites:
                        cause["cites"] = cites
                    causes.append(cause)
                row = {"id": ch["id"], "seq": ch.get("seq"), "question": q, "old": _plain_answer(d["old"]),
                       "new": _plain_answer(d["new"]), "status": [_status(d["old"]), _status(d["new"])],
                       "causes": causes}
                if input_ids:
                    row["input"] = input_ids.get(ch["id"])
                out.append(row)
        total = len(list(store.iter()))
        dd = DecisionDiff(out, rep, total)
        if only is not None:
            allowed = set(only)
            dd.extraneous = [i for i in dict.fromkeys(x.get("input", x["id"]) for x in dd.moved) if i not in allowed]
            dd.exact = not dd.extraneous and not rep.errors
        return dd
    finally:
        if tmp is not None:
            store.close()
            shutil.rmtree(tmp, ignore_errors=True)


# ───────────────────────────────────────────────────────────── versions
class Versions:
    """Catalog versions as numbered folders (`v1/`, `v2/`, ... each `module.py` + `compiled.json` + `version.json`).
    Each version records its parent — the version in force when it was added — and the folder keeps which version is in
    force (`current`: the last one added, unless rolled back; `in_force.json`). `get()` / `system()` give the version in
    force, `get(n)` / `system(n)` any one; `replay_all(store)` replays every stored decision with the version whose
    catalog fingerprint it recorded — an old decision is checked against the rules it was made with."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.numbers())

    def numbers(self) -> list[int]:
        return sorted(int(p.name[1:]) for p in self.path.glob("v*") if p.name[1:].isdigit() and (p / "compiled.json").exists())

    def _state(self) -> dict:
        p = self.path / "in_force.json"
        return json.loads(p.read_text()) if p.exists() else {"version": None, "history": []}

    def _write_state(self, st):
        (self.path / "in_force.json").write_text(json.dumps(st, indent=1))

    @property
    def current(self) -> int | None:
        """The version in force (None: no version yet)."""
        v = self._state()["version"]
        if v is None and self.numbers():                # a folder written before 1.1: the latest is in force
            return self.numbers()[-1]
        return v

    def add(self, version: CatalogVersion, label: str | None = None, *, parent: int | None = None) -> int:
        """Store an accepted version as the next number, in force from now on → its number. parent: the version it
        changes (default: the one in force)."""
        if not version.accepted:
            raise Rejected(version.reason)
        parent = self.current if parent is None else parent
        n = (self.numbers() or [0])[-1] + 1
        p = version.save(self.path / f"v{n}")
        (p / "version.json").write_text(json.dumps({"version": n, "label": label, "fingerprint": version.fingerprint,
                                                    "spec_hash": version.spec.hash, "parent": parent,
                                                    "time": time.time()}, indent=1))
        st = self._state()
        st["version"] = n
        self._write_state(st)
        return n

    def get(self, n: int | None = None) -> CatalogVersion:
        """Version n (the one in force when None)."""
        n = n if n is not None else self.current
        if n is None:
            raise KeyError("no version yet")
        return CatalogVersion.load(self.path / f"v{n}")

    def info(self, n: int) -> dict:
        """{"version", "label", "fingerprint", "spec_hash", "parent", "time"} of version n."""
        d = json.loads((self.path / f"v{n}" / "version.json").read_text())
        d.setdefault("parent", None)
        return d

    def system(self, n: int | None = None, **kw):
        """The System of version n (the one in force when None); kw: System's own (storage, ...)."""
        return self.get(n).system(**kw)

    def find(self, fingerprint: str) -> int | None:
        """The version whose catalog has this fingerprint."""
        for n in self.numbers():
            if self.info(n)["fingerprint"] == fingerprint:
                return n
        return None

    def rollback(self, n: int, *, by=None, why=None) -> int:
        """Put version n back in force (every version stays; a later add takes n as its parent); recorded in
        `history()` → n."""
        if n not in self.numbers():
            raise KeyError(f"no version {n} in {self.path}")
        st = self._state()
        st["history"] = list(st.get("history") or []) + [{"to": n, "from": self.current, "by": None if by is None
                                                          else str(by), "why": why, "time": time.time()}]
        st["version"] = n
        self._write_state(st)
        return n

    def history(self) -> list:
        """The rollbacks, in order: [{"to", "from", "by", "why", "time"}]."""
        return list(self._state().get("history") or [])

    def replay_all(self, store, **filters) -> list:
        """Replay every stored decision against the version that made it (by the recorded catalog fingerprint) → the
        ones that do not replay (as TraceStorage.replay_all), plus decisions whose catalog is no version here."""
        bad, known = [], set()
        for n in self.numbers():
            fp = self.info(n)["fingerprint"]
            known.add(fp)
            bad += [{**b, "version": n} for b in store.replay_all(self.system(n), catalog_fp=fp, **filters)]
        for s in (store.query(**filters) if filters else store.iter()):
            if s.data.get("catalog") not in known:
                bad.append({"id": s.id, "seq": s.seq, "time": s.time, "version": None, "mismatches": [],
                            "summary": "decided by a catalog that is no version here"})
        return bad


def serve_head(system, question, head) -> dict:
    """Install `head` as the answer head of `question` — the hook through which a learned version is served (the
    closed learning loop, solvi.experimental.stream.versions, installs each promoted or rolled-back frozen version
    here). The head is `system.heads[question]`, so the System's fingerprint ("models": "answer:<question>") changes
    with it and every trace records which model answered, as for any fitted head. head: the Head protocol (options,
    features, predict, contributions, fingerprint); a question a rule answers is refused (the rule would answer and
    the head would never be asked) → {"question", "from": the previous head's fingerprint or None, "to": the new one}.
    Experimental (1.4): the loop that calls it is unmeasured."""
    from ..provenance import fingerprint
    if question not in system.questions:
        raise KeyError(f"no question {question!r} in the System")
    if question in system.catalog.rules:
        raise ValueError(f"a rule answers {question!r}: a served head would never be asked")
    lacks = [a for a in ("options", "features", "predict", "contributions", "fingerprint") if not hasattr(head, a)]
    if lacks:
        raise TypeError(f"a served head has the Head protocol: missing {lacks}")
    old = system.heads.get(question)
    system.heads[question] = head
    return {"question": question, "from": None if old is None else fingerprint(old), "to": fingerprint(head)}


__all__ = ["CatalogVersion", "Clause", "DecisionDiff", "Rejected", "Spec", "Versions", "build_catalog", "decision_diff",
           "serve_head", "split_clauses"]
