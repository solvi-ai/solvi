"""Compile a specification into catalog parts: an LLM writes the rules, hard checks and computed facts a policy, a
regulation or a constraint description states; solvi accepts them only after checks that need no labels.

    from solvi.compile import Inputs, Spec, compile_spec, recompile, Versions
    from solvi.generate import generator

    spec = Spec(policy_text)                                  # split into numbered clauses: spec.clauses, spec.hash
    inputs = Inputs({"amount": (0, 20000), "country": ["DE", "FR", "US"]})    # what a decision reads, and its values
    writer = generator(URL, "openai/gpt-oss-120b", max_tokens=24000, extra_body={"reasoning": {"effort": "medium"}})
    c = compile_spec(spec, [Question("decision", "...", Answer.choice(["approve", "refuse"]))], inputs, writer)
    c.accepted, c.reason, c.parts                             # each part: kind and the clauses it implements
    system = c.system()                                        # an ordinary System; refused when not accepted

    c2 = recompile(c, spec.revise(new_text), inputs, writer)  # only what the change touches; c2.changes
    decision_diff(c, c2, store=store)                         # which stored decisions change, and the clauses why

What the writer produces. A module of plain functions: each catalog part is a function whose name is the fact it sets
and whose argument names are what it reads, and a literal `PARTS` dict gives each part's kind ("fn", "check", "rule"),
for a hard check its `then`, for a rule its question, and the clauses it implements. Clauses no part implements are
listed in `NOT_NORMATIVE` with the reason. The module is checked by `solvi.sandbox` (pure functions, standard-library
imports only) and run there, in a subprocess with limits, before anything of it enters this process.

Acceptance needs no labels. Two drafts are written independently (by default two samples of one writer: the first at
temperature 0, the second at 0.7 with seed 1; pass two writers for two models). A draft is accepted when:
1. it passes the sandbox and the module contract (every part listed, every clause cited or declared not normative);
2. the two drafts give the same answer to every question on every input of the pool — inputs generated from
   `Inputs` (with the numbers of the specification and of both drafts, ±1, as boundary values), the samples you give,
   and the tests' inputs. A disagreement is shown to both writers: the input, both answers, the clauses the deciding
   parts cite;
3. both pass the tests a separate call derived from the specification — it never sees the code, and each test names
   the clause it checks. A test both drafts fail with the same answer goes back once to the test writer, which keeps,
   corrects or drops it (recorded);
4. both match the labelled `examples` and the `reference`, when you give them.
A draft with failures is rewritten from its module and the failures, up to `rounds` rounds; then the compilation is not
accepted, `Compiled.catalog()` refuses it, and `reason` says why. The record keeps the spec's hash and clauses, every
prompt and reply, the writer, the tests, and each check's outcome per round.

What it does not do. Agreement of two drafts is not correctness: two samples of one model can share a misreading, and
the tests come from the same model. Coverage is checked by citation, not by meaning — a part may cite a clause it
implements wrongly. The inputs pool decides what "agree" covers: a region of inputs nobody generates is not compared.
Labels, when you have them, are the stronger check. Nothing here writes extractors from text or searches; the
compiled parts read structured inputs.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import sandbox
from .core import Catalog, Question
from .refine import Fail

DEFAULT_MODEL = "openai/gpt-oss-120b"
SANDBOX_EXTRA = {"Fail": Fail}                        # trusted names a compiled module may use without an import
KINDS = ("fn", "check", "rule")
META = ("PARTS", "NOT_NORMATIVE", "REMOVE")


class Rejected(RuntimeError):
    """A compilation that was not accepted was asked for its catalog. `.reason` says why it was not accepted."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"the compilation was not accepted: {reason}")


# ───────────────────────────────────────────────────────────── the specification as clauses
@dataclass
class Clause:
    id: str
    text: str
    section: str = ""


_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_SEP = re.compile(r"^:?-{2,}:?$")


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
        h = _HEADING.match(s)
        if h:
            flush()
            header = None
            level = len(h.group(1))
            path[:] = [p for p in path if p[0] < level] + [(level, h.group(2))]
            continue
        if s.startswith("|"):
            flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(_SEP.match(c) for c in cells if c):
                continue
            if header is None:
                header = cells
                continue
            out.append((" / ".join(t for _, t in path), "; ".join(f"{a}: {b}" for a, b in zip(header, cells) if b)))
            continue
        header = None
        if not s:
            flush()
        elif _ITEM.match(line):
            flush()
            buf.append(_ITEM.match(line).group(1).strip())
        else:
            buf.append(s)
    flush()
    return [(sec, t) for sec, t in out if t]


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


# ───────────────────────────────────────────────────────────── the inputs a decision reads
def _is_range(d):
    return isinstance(d, tuple) and len(d) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in d)


class Inputs:
    """What a compiled decision reads, and where the inputs that compare two drafts come from.

    fields: {name: domain} — a list of the values that occur (drawn from), a `(lo, hi)` tuple of numbers (a range: ints
    when both ends are ints), or a type (`str`, `dict`, a pydantic model: not generated; give samples or `generate`).
    describe: the inputs in words for the writer (default: made from `fields`). samples: real inputs (unlabelled),
    all put in the pool. generate(rng) → one input: your generator, called `n` times. Without `generate`, `n` inputs are
    drawn from the domains, when every field has one. Boundary values: for every number field, every number of the
    specification and the drafts inside its range, -1 / exactly / +1, on a random input each."""

    def __init__(self, fields: dict | None = None, *, describe: str | None = None, samples=(), generate=None,
                 n: int = 1000, seed: int = 0):
        self.fields = dict(fields or {})
        self.describe = describe
        self.samples = [dict(x) for x in samples]
        self.generate = generate
        self.n, self.seed = int(n), int(seed)
        if not self.fields and not self.samples and generate is None:
            raise ValueError("Inputs needs fields, samples or a generator")

    def text(self) -> str:
        """The inputs in words for the writer."""
        if self.describe:
            return self.describe.strip()
        lines = []
        for k, d in self.fields.items():
            if isinstance(d, list):
                lines.append(f"- {k}: one of " + ", ".join(json.dumps(v, ensure_ascii=False) for v in d))
            elif _is_range(d):
                kind = "integer" if all(isinstance(x, int) for x in d) else "number"
                lines.append(f"- {k}: {kind} from {d[0]} to {d[1]}")
            else:
                lines.append(f"- {k}: {getattr(d, '__name__', d)}")
        return "\n".join(lines)

    def validate(self, x) -> list[str]:
        """Why `x` is not a complete input ([] — it is): a missing or unknown field, a value outside a listed domain."""
        if not isinstance(x, dict):
            return [f"the input is a {type(x).__name__}, not an object"]
        if not self.fields:
            return []
        why = [f"missing field {k}" for k in self.fields if k not in x]
        why += [f"unknown field {k}" for k in x if k not in self.fields]
        for k, d in self.fields.items():
            if k not in x:
                continue
            v = x[k]
            if isinstance(d, list) and v not in d:
                why.append(f"{k} = {v!r} is not one of its values")
            elif _is_range(d) and (not isinstance(v, (int, float)) or isinstance(v, bool)):
                why.append(f"{k} = {v!r} is not a number")
            elif isinstance(d, type) and d in (str, int, float, dict, list, bool) and not isinstance(v, d):
                why.append(f"{k} = {v!r} is not a {d.__name__}")
        return why

    def _draw(self, rng):
        x = {}
        for k, d in self.fields.items():
            if isinstance(d, list):
                x[k] = rng.choice(d)
            elif _is_range(d):
                lo, hi = d
                x[k] = rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else rng.uniform(lo, hi)
            else:
                raise ValueError(f"field {k} has no domain to draw from: give samples or generate=")
        return x

    def pool(self, numbers=()) -> list[dict]:
        """The inputs two drafts are compared on: the samples, `n` generated inputs, and the boundary inputs around
        `numbers` (the specification's and the drafts'). Deterministic for the same numbers."""
        rng = random.Random(self.seed)
        out = [dict(x) for x in self.samples]
        drawable = bool(self.fields) and all(isinstance(d, list) or _is_range(d) for d in self.fields.values())
        if self.generate is not None:
            out += [self.generate(rng) for _ in range(self.n)]
        elif drawable:
            out += [self._draw(rng) for _ in range(self.n)]
        if drawable:
            nums = sorted({float(v) for v in numbers})
            for k, d in self.fields.items():
                if not _is_range(d):
                    continue
                lo, hi = d
                ints = isinstance(lo, int) and isinstance(hi, int)
                for v in nums:
                    for w in (v - 1, v, v + 1):
                        if lo <= w <= hi and (not ints or float(w).is_integer()):
                            x = self._draw(rng)
                            x[k] = int(w) if ints else w
                            out.append(x)
        if not out:
            raise ValueError("no inputs to compare the drafts on: give samples, generate=, or fields with domains")
        return out


def numbers_in(text: str) -> list[float]:
    """The number literals of a text or a module (thousands separators allowed)."""
    return [float(m.replace(",", "")) for m in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)]


# ───────────────────────────────────────────────────────────── the module contract
def options_of(q: Question):
    a = q.answer
    if a is None:
        return None
    return ["yes", "no"] if a.kind == "yes_no" else list(a.options)


def read_module(source: str, spec: Spec, questions, patch=False) -> tuple[dict, dict, dict, list[str]]:
    """A module's PARTS, NOT_NORMATIVE (and REMOVE for a patch) → (parts, not_normative, remove, problems). The checks of
    the contract: literals, kinds, every part a top-level function, a known question and an allowed answer in `then`, a
    rule per question (not for a patch), cited clauses that exist."""
    why = sandbox.check(source)
    if why:
        return {}, {}, {}, ["the sandbox refuses the module: " + w for w in why]
    tree = ast.parse(source)
    lits, funcs = {}, set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in META:
            try:
                lits[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                return {}, {}, {}, [f"{node.targets[0].id} must be a literal (no names or calls in it)"]
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in META \
                and node.value.func.attr == "update" and len(node.value.args) == 1:
            try:                                       # PARTS.update({...}): read as entries of PARTS
                lits.setdefault(node.value.func.value.id, {}).update(ast.literal_eval(node.value.args[0]))
            except (ValueError, TypeError):
                return {}, {}, {}, [f"{node.value.func.value.id}.update(...) must take a literal dict"]
    problems = []
    parts = lits.get("PARTS")
    if not isinstance(parts, dict) or (not parts and not patch):
        problems.append("the module has no PARTS dict (a literal at the top level)")
        parts = {}
    nn = lits.get("NOT_NORMATIVE", {})
    if not isinstance(nn, dict):
        problems.append("NOT_NORMATIVE must be a dict {clause id: reason}")
        nn = {}
    rm = lits.get("REMOVE", {})
    if not isinstance(rm, dict):
        problems.append("REMOVE must be a dict {part name: the changed clause that removes it}")
        rm = {}
    qs = {q.name: q for q in questions}
    for name, p in parts.items():
        if not isinstance(p, dict):
            problems.append(f"PARTS[{name!r}] must be a dict")
            continue
        if name not in funcs and not patch:
            problems.append(f"PARTS names {name!r}, which is not a top-level function of the module")
        kind = p.get("kind")
        if kind not in KINDS:
            problems.append(f"{name}: kind must be one of {KINDS}, not {kind!r}")
        cl = p.get("clauses")
        if not isinstance(cl, list) or (not cl and kind != "rule"):      # a rule giving only a default may cite none
            problems.append(f"{name}: 'clauses' must list the clauses it implements")
        else:
            problems += [f"{name} cites {c!r}, which is not a clause" for c in cl if c not in spec.clauses]
        if kind == "check":
            then = p.get("then") or {}
            if p.get("hard") and not then:
                problems.append(f"{name}: a hard check needs 'then' ({{question: answer}})")
            for q, a in then.items():
                if q not in qs:
                    problems.append(f"{name}: then names {q!r}, which is not a question")
                elif options_of(qs[q]) is not None and a not in options_of(qs[q]):
                    problems.append(f"{name}: then gives {q!r} the answer {a!r}, not one of {options_of(qs[q])}")
        if kind == "rule" and p.get("question") not in qs:
            problems.append(f"{name}: a rule needs 'question', one of {sorted(qs)}")
        if name in qs and kind != "rule":
            problems.append(f"{name}: a part may not be named like a question")
    if not patch:
        for q in qs:
            n = [k for k, p in parts.items() if isinstance(p, dict) and p.get("kind") == "rule" and p.get("question") == q]
            if len(n) != 1:
                problems.append(f"question {q!r} needs exactly one rule (found {len(n)})")
    problems += [f"NOT_NORMATIVE names {c!r}, which is not a clause" for c in nn if c not in spec.clauses]
    return parts, nn, rm, problems


def coverage(spec: Spec, parts: dict, nn: dict) -> list[str]:
    """Clauses neither cited by a part nor declared not normative."""
    cited = {c for p in parts.values() for c in (p.get("clauses") or [])}
    return [c for c in spec.clauses if c not in cited and c not in nn]


def given_names(inputs) -> set | None:
    """The names of the given facts: the declared fields, else the keys of the samples (None: unknown)."""
    if inputs.fields:
        return set(inputs.fields)
    if inputs.samples:
        return {k for x in inputs.samples for k in x}
    return None


def unknown_reads(source: str, parts: dict, given) -> list[str]:
    """Parts that read a name which is neither a given fact nor a fact another part sets (a rule sets none)."""
    if given is None:
        return []
    facts = {n for n, p in parts.items() if p.get("kind") in ("fn", "check")}
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in parts:
            if node.args.vararg or node.args.kwarg:
                out.append(f"part {node.name} takes *args / **kwargs: a part names each fact it reads as an argument")
                continue
            args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            bad = [a for a in args if a not in given and a not in facts]
            if bad:
                out.append(f"part {node.name} reads {', '.join(bad)}: neither an input field nor a fact set by a part "
                           f"(the inputs are {', '.join(sorted(given))})")
    return out


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


def _answer_json(a):
    if a is None:
        return None
    return a if isinstance(a, (str, int, float, bool)) else str(a)


def decide_rows(system, inputs, alarm=None, item_s=5):
    """Ask the system on every input → [{"answers": {q: answer or None (abstained)}, "ran": [part names], "error"}]."""
    rows = []
    for x in inputs:
        if alarm is not None:
            alarm.alarm(item_s)
        try:
            res = system.ask(dict(x))
            ans = {q: (None if res[q].status == "abstain" else _answer_json(res[q].answer)) for q in system.questions}
            why = {q: res[q].why for q in system.questions if res[q].status == "abstain"}
            ran = [r.name for r in res.trace.records]
            rows.append({"answers": ans, "ran": ran, "why": why})
        except BaseException as e:  # noqa: BLE001 — a crash or the time limit is a result of this input
            rows.append({"answers": {}, "ran": [], "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            if alarm is not None:
                alarm.alarm(0)
    return rows


def box_main(ns, payload, signal):
    """The sandbox driver: build the catalog from the module's namespace and decide every input of the payload."""
    from .system import System
    qs = [Question.model_validate(q) for q in payload["questions"]]
    cat, qs = build_catalog(ns, payload["parts"], qs)
    system = System(cat, qs)
    return {"rows": decide_rows(system, payload["inputs"], signal, payload.get("item_s", 5))}


# ───────────────────────────────────────────────────────────── prompts
SYSTEM_PROMPT = ("You write Python modules that an automatic harness checks and runs in a sandbox. Answer with exactly "
                 "one ```python code block holding the complete module. Explanations, if any, go before it.")

CONTRACT = """## The module you write

Plain top-level Python functions. Each catalog part is one function:
- its NAME is the fact it sets, its ARGUMENT NAMES are what it reads: input fields (below) or facts set by other
  parts. Never give a part the name of an input field or of a question.
- kinds: "fn" computes a value; "check" returns True when its condition holds; "rule" returns the answer of one
  question (one of its options; a yes/no rule may return True / False).
- a hard check that returns False forces the answer its "then" names, whatever the rule says (the first false hard check
  in PARTS order wins). A check may return Fail("why") instead of False (`Fail` needs no import).
- for each question its rule runs with the parts it reads (and theirs), and every hard check whose "then" names the
  question. A part that raises makes the questions that need it abstain.
- write one part per quantity a clause defines (for example one function per row of a points table), so the record of
  a decision shows each; helper functions that are not parts are allowed.
- deterministic and pure: no state between calls, no input / output.

At the end of the module, two literal dicts:
```python
PARTS = {
    "function_name": {"kind": "fn", "clauses": ["c4", "c5"]},
    "some_check": {"kind": "check", "hard": True, "then": {"question_name": "answer"}, "clauses": ["c9"]},
    "rule_function": {"kind": "rule", "question": "question_name", "clauses": ["c11"]},
}
NOT_NORMATIVE = {"c1": "why this clause needs no code", ...}
```
Every part is in PARTS with the clauses it implements; every clause is cited by a part or is in NOT_NORMATIVE."""

SANDBOX_RULES = """## Sandbox rules (a module that breaks one is rejected)
- Imports only from: """ + ", ".join(sorted(sandbox.ALLOWED)) + """.
- Forbidden names: """ + ", ".join(sorted(sandbox.FORBIDDEN)) + """.
- No dunder names or attributes at all (no `__name__`, no `if __name__ == "__main__":`), no global / nonlocal."""

PATCH_CONTRACT = """## What you return: only the change

The specification changed: clauses marked [changed] or [added]; removed clauses are listed above. Return a module
with ONLY what must change in the current module — everything you do not return stays exactly as it is:
- the functions (and constants) you add or replace, complete; a replaced function keeps its name. Do not return
  functions you do not change;
- PARTS = {...}: a plain literal with entries for the parts you add or replace ONLY, each citing at least one
  [changed] or [added] clause it implements (it may cite unchanged clauses too);
- REMOVE = {"part_name": "c12"} for each part to remove, with the [changed] or removed clause that removes it;
- NOT_NORMATIVE = {...} for [added] clauses that need no code.

For example, when a new clause c9 makes gift cards ship free:
```python
def gift_card_free(gift_card):
    return gift_card

def fee_rule(zone_fee, domestic_free, gift_card_free):      # replaced: it now reads gift_card_free
    return "0.00" if domestic_free or gift_card_free else f"{zone_fee:.2f}"

PARTS = {
    "gift_card_free": {"kind": "fn", "clauses": ["c9"]},
    "fee_rule": {"kind": "rule", "question": "fee", "clauses": ["c4", "c9"]},
}
REMOVE = {}
NOT_NORMATIVE = {}
```"""

TESTS_PROMPT = """# Write tests for a specification

A program will implement the specification below. You do not see the program. Write tests that a correct
implementation must pass: each test is one complete input, the answer the specification requires for each question,
the clause it checks, and why.

{spec}

## Questions the program answers
{questions}

## The inputs (every test input must have exactly these fields)
{inputs}

Cover every clause that decides something, the boundaries of every number (just below, at, just above), and the cases
where clauses interact (for example a hard rule against a sum). At most {k} tests. Answer with one ```json block:
[{{"clause": "c3", "input": {{...}}, "expect": {{"question_name": "answer"}}, "why": "..."}}, ...]
"""

REVIEW_PROMPT = """# Re-check one test against the specification

{spec}

## Questions
{questions}

## The test
{test}

Two implementations of the specification, written independently, both fail this test: they answer {got} (None: the
implementation could not answer) where the test expects {expect}. They may be wrong, or the test may be. Re-read clause
{clause} and the clauses it interacts with, and work out the expected answer step by step from the clauses and the
input before you decide. Answer with one ```json block:
{{"verdict": "keep" | "fix" | "drop", "expect": {{...}} (the corrected answers, for "fix"), "why": "..."}}
"""


def _questions_text(questions):
    lines = []
    for q in questions:
        o = options_of(q)
        lines.append(f"- {q.name}: {q.text}" + (f" — one of {', '.join(map(str, o))}" if o else ""))
    return "\n".join(lines)


def _task_text(spec, questions, inputs, marks=None, removed=()):
    rem = ""
    if removed:
        rem = "\n\nRemoved clauses (no longer in the specification):\n" + "\n".join(f"- [{c.id}] {c.text}" for c in removed)
    return (f"# Task: implement a specification as catalog parts\n\n## The specification ({spec.name})\n\n"
            f"{spec.render(marks)}{rem}\n\n## Questions to answer\n{_questions_text(questions)}\n\n"
            f"## The inputs (given facts)\n{inputs.text()}")


def code_of(reply: str) -> str | None:
    """The module in a reply: the longest ```python block that parses and defines a function (else the longest block)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, flags=re.S)
    for m in re.finditer(r"^```(?:python|py)[ \t]*\n", reply, flags=re.M):
        end = re.search(r"^```", reply[m.end():], flags=re.M)
        blocks.append(reply[m.end(): m.end() + end.start()] if end else reply[m.end():])
    if not blocks:
        return None
    ok = []
    for b in blocks:
        try:
            if any(isinstance(n, ast.FunctionDef) or isinstance(n, ast.Assign) for n in ast.parse(b).body):
                ok.append(b)
        except SyntaxError:
            pass
    return max(ok or blocks, key=len)


def _strip_json(text: str) -> str:
    """`//` and `/* */` comments outside strings and trailing commas removed (what models add to JSON)."""
    out, i, n, q = [], 0, len(text), False
    while i < n:
        ch = text[i]
        if q:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                q = False
        elif ch == '"':
            q = True
            out.append(ch)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        else:
            out.append(ch)
        i += 1
    return re.sub(r",(\s*[\]}])", r"\1", "".join(out))


def _json_block(reply: str):
    blocks = re.findall(r"```(?:json)?[ \t]*\n(.*?)```", reply, flags=re.S)
    for b in sorted(blocks, key=len, reverse=True) + [reply]:
        for s in (b, b[b.find("["):] if "[" in b else b[b.find("{"):]):
            for t in (s, _strip_json(s)):
                try:
                    return json.loads(t)
                except (json.JSONDecodeError, TypeError):
                    continue
    raise ValueError("no JSON in the reply")


# ───────────────────────────────────────────────────────────── merging a patch into a module
def _top(source):
    """Top-level statements of a module → [(key, text)]; key = function name / assigned name / None."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out = []
    for i, node in enumerate(tree.body):
        start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
        end = node.end_lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            key = node.target.id
        else:
            key = None
        out.append((key, "".join(lines[start:end])))
    return out


def _is_meta_call(text):
    return bool(re.match(r"\s*(PARTS|NOT_NORMATIVE|REMOVE)\s*(\.|\[)", text))


def _literal(name, d):
    if not d:
        return f"{name} = {{}}\n"
    body = "".join(f"    {json.dumps(k)}: {_py(v)},\n" for k, v in d.items())
    return f"{name} = {{\n{body}}}\n"


def _py(v):
    return repr(v) if not isinstance(v, (dict, list, str, bool)) else (
        "{" + ", ".join(f"{json.dumps(k)}: {_py(x)}" for k, x in v.items()) + "}" if isinstance(v, dict)
        else "[" + ", ".join(_py(x) for x in v) + "]" if isinstance(v, list)
        else ("True" if v else "False") if isinstance(v, bool) else json.dumps(v, ensure_ascii=False))


def merge(old_source: str, old_parts: dict, old_nn: dict, patch_source: str, patch_parts: dict, patch_nn: dict,
          remove: dict, spec: Spec) -> tuple[str, dict, dict]:
    """The current module with a patch applied → (source, parts, not_normative). Statements the patch does not name stay
    byte-identical and in place; replaced ones are swapped in place; new ones go before PARTS."""
    new = {k: t for k, t in _top(patch_source) if k not in META}
    plain_new = [t for k, t in _top(patch_source) if k is None and not _is_meta_call(t)]
    out, seen = [], set()
    for k, t in _top(old_source):
        if k in META or (k is None and _is_meta_call(t)):
            continue
        if k in remove:
            continue
        if k is not None and k in new:
            out.append(new[k])
            seen.add(k)
        else:
            out.append(t)
    imports = [t for t in plain_new if t not in out]
    out = imports + out + [t for k, t in new.items() if k not in seen and k is not None]
    parts = {k: v for k, v in old_parts.items() if k not in remove}
    for k, v in patch_parts.items():
        parts[k] = v
    gone = set((spec.changes or {}).get("removed", []))
    nn = {k: v for k, v in old_nn.items() if k not in gone and k in spec.clauses}
    nn.update(patch_nn)
    src = "\n\n".join(t.rstrip("\n") for t in out) + "\n\n\n" + _literal("PARTS", parts) + "\n" + _literal("NOT_NORMATIVE", nn)
    return src, parts, nn


# ───────────────────────────────────────────────────────────── the result
@dataclass
class Compiled:
    """A compilation: the module, its parts with the clauses each implements, whether it was accepted and why, and the
    record (spec, prompts and replies, writer, tests, checks per round). `catalog()` / `system()` load the module (through
    `solvi.sandbox.load`) and refuse an unaccepted compilation unless `allow_unaccepted=True`."""
    spec: Spec
    questions: list
    source: str
    parts: dict
    not_normative: dict
    accepted: bool
    reason: str
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
            self._ns = sandbox.load(self.source, SANDBOX_EXTRA)
        return build_catalog(self._ns, self.parts, self.questions)

    def system(self, allow_unaccepted: bool = False, **kw):
        """An ordinary System over the compiled catalog (kw: System's own — storage, ...)."""
        from .system import System
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
    def from_dict(cls, d) -> "Compiled":
        return cls(Spec.from_dict(d["spec"]), [Question.model_validate(q) for q in d["questions"]], d["source"],
                   d["parts"], d["not_normative"], d["accepted"], d["reason"], d.get("record") or {}, d.get("changes"))

    def save(self, path) -> Path:
        """Write `module.py` (the code, as accepted) and `compiled.json` (everything) into a folder."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "module.py").write_text(self.source)
        (p / "compiled.json").write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False, default=str))
        return p

    @classmethod
    def load(cls, path) -> "Compiled":
        d = json.loads((Path(path) / "compiled.json").read_text())
        c = cls.from_dict(d)
        if (Path(path) / "module.py").read_text() != c.source:
            raise ValueError(f"{path}/module.py differs from the source recorded in compiled.json")
        return c


# ───────────────────────────────────────────────────────────── the loop
def _writer_of(writer):
    if writer is None:
        raise ValueError("compile_spec needs a writer: a solvi.generate Generator (or a base URL string, then the model "
                         f"is {DEFAULT_MODEL})")
    if isinstance(writer, str):
        from .generate import generator
        return generator(writer, DEFAULT_MODEL, max_tokens=24000, timeout=900,
                         extra_body={"reasoning": {"effort": "medium"}})
    return writer


def _ask(gen, prompt, temperature, seed, log, what):
    """One reply's text (a cut-off or failed reply → None, its reason logged)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    t0 = time.time()
    try:
        g = gen.generate(msgs, parse=lambda s: s, temperature=temperature, seed=seed)
        text, meta, err = g.value, g.meta, None
    except Exception as e:  # noqa: BLE001 — a failed reply is a failed draft of this round, with its reason
        text, meta, err = getattr(e, "reply", None), {}, f"{type(e).__name__}: {e}"[:300]
    log.append({"what": what, "prompt": prompt, "reply": text, "error": err, "seconds": round(time.time() - t0, 1),
                "model": getattr(gen, "model_id", None), "temperature": temperature, "seed": seed,
                "usage": (meta or {}).get("usage"), "request": (meta or {}).get("request")})
    return text, err


def _norm(a):
    if isinstance(a, bool):
        return "yes" if a else "no"
    return None if a is None else str(a)


def _valid_tests(raw, spec, questions, inputs):
    """The tests of a reply that can be used, and the others with why."""
    tests, bad = [], []
    qs = {q.name: q for q in questions}
    for i, t in enumerate(raw if isinstance(raw, list) else []):
        why = []
        if not isinstance(t, dict) or not isinstance(t.get("input"), dict) or not isinstance(t.get("expect"), dict):
            bad.append({"index": i, "test": t, "why": ["not a {clause, input, expect} object"]})
            continue
        why += inputs.validate(t["input"])
        cited = [c for c in re.findall(r"\bc\d+\b", json.dumps(t.get("clause", t.get("clauses", ""))))
                 if c in spec.clauses]
        if not cited:
            why.append(f"cites {t.get('clause', t.get('clauses'))!r}, no clause of the specification")
        t["clause"] = ", ".join(dict.fromkeys(cited))
        for q, a in t["expect"].items():
            if q not in qs:
                why.append(f"expects an answer to {q!r}, not a question")
            elif options_of(qs[q]) is not None and _norm(a) not in [_norm(o) for o in options_of(qs[q])]:
                why.append(f"expects {q} = {a!r}, not one of its options")
        if why:
            bad.append({"index": i, "test": t, "why": why})
        else:
            tests.append({"id": f"t{i + 1}", **t, "expect": {q: _norm(a) for q, a in t["expect"].items()}})
    return tests, bad


def _write_tests(spec, questions, inputs, gen, log, k):
    """One call for the tests; a reply that cannot be read, or holds no usable test, is asked once more with why."""
    prompt = TESTS_PROMPT.format(spec=spec.render(), questions=_questions_text(questions), inputs=inputs.text(), k=k)
    tests, bad, err = [], [], None
    for attempt in range(2):
        p = prompt if attempt == 0 else (prompt + f"\n\nYour previous answer could not be used ({err}). Answer with "
                                         "plain JSON: no comments, no trailing commas, complete inputs with exactly "
                                         "the fields above.")
        t0 = time.time()
        try:
            g = gen.generate([{"role": "user", "content": p}], parse=_json_block, temperature=0.0)
            raw, err, meta = g.value, None, g.meta
        except Exception as e:  # noqa: BLE001
            raw, err, meta = [], f"{type(e).__name__}: {e}"[:300], {"text": getattr(e, "reply", None)}
        log.append({"what": "tests" if attempt == 0 else "tests again", "prompt": p, "reply": (meta or {}).get("text"),
                    "error": err, "seconds": round(time.time() - t0, 1), "model": getattr(gen, "model_id", None),
                    "usage": (meta or {}).get("usage")})
        if isinstance(raw, dict):                         # {"tests": [...]}: the list inside
            raw = next((v for v in raw.values() if isinstance(v, list)), [])
        tests, bad = _valid_tests(raw, spec, questions, inputs)
        if tests:
            return tests[:k], bad, None
        err = err or ("no list of tests in the reply" if not bad else
                      f"none of its {len(bad)} tests could be used, e.g. " + "; ".join(bad[0]["why"][:3]))
    return [], bad, err


def _review(spec, questions, test, got, gen, log):
    prompt = REVIEW_PROMPT.format(spec=spec.render(), questions=_questions_text(questions),
                                  test=json.dumps({k: test[k] for k in ("clause", "input", "expect", "why") if k in test},
                                                  ensure_ascii=False, indent=1),
                                  got=json.dumps(got), expect=json.dumps(test["expect"]), clause=test.get("clause"))
    try:
        g = gen.generate([{"role": "user", "content": prompt}], parse=_json_block, temperature=0.0)
        v, err, meta = g.value, None, g.meta
    except Exception as e:  # noqa: BLE001
        v, err, meta = {}, f"{type(e).__name__}: {e}"[:300], {}
    log.append({"what": f"review {test['id']}", "prompt": prompt, "reply": (meta or {}).get("text"), "error": err,
                "model": getattr(gen, "model_id", None), "usage": (meta or {}).get("usage")})
    verdict = v.get("verdict") if isinstance(v, dict) else None
    return (verdict if verdict in ("keep", "fix", "drop") else "keep"), (v if isinstance(v, dict) else {})


def _clauses_text(spec, ids, n=10):
    ids = [i for i in dict.fromkeys(ids) if i in spec.clauses][:n]
    return "\n".join(f'  [{i}] {spec.clauses[i].text[:300]}' for i in ids)


def _short(x, n=700):
    s = json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n - 1] + "…"


@dataclass
class _Draft:
    index: int
    source: str | None = None             # the full module (a patch merged in, for a recompile)
    patch: str | None = None
    parts: dict = field(default_factory=dict)
    nn: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    rows: list | None = None
    feedback: str | None = None
    caught: list = field(default_factory=list)


def _run_draft(d, questions, inputs, mem_mb, item_s):
    payload = {"parts": d.parts, "questions": [q.model_dump() for q in questions], "inputs": inputs, "item_s": item_s}
    r = sandbox.run(d.source, "solvi.compile:box_main", payload, mem_mb=mem_mb, cpu_s=60 + len(inputs),
                    wall_s=120 + 2 * len(inputs))
    if "rows" not in r:
        return None, r.get("load_error") or r.get("crash") or "no result"
    return r["rows"], None


def _ran_clauses(c_parts, ran):
    out = []
    for step in ran:
        name = step
        if step.startswith("answer:"):
            q = step.split(":", 1)[1]
            name = next((n for n, p in c_parts.items() if p.get("kind") == "rule" and p.get("question") == q), step)
        out += (c_parts.get(name) or {}).get("clauses") or []
    return out


def _loop(spec, questions, inputs, gens, *, rounds, tests, examples, reference, write, log, mem_mb, item_s, extra_check,
          need_tests=False):
    """The write → check → rewrite loop over two drafts → (accepted draft or None, reason, per-round summary, tests)."""
    review_gen = gens[0]
    test_list, bad_tests, terr = tests
    if need_tests and not test_list:
        return None, None, ("not accepted: no valid tests were derived from the specification ("
                            + (terr or f"{len(bad_tests)} invalid") + ")"), [], {}
    reviewed = {}
    history = []
    drafts = [_Draft(0), _Draft(1)]
    for rnd in range(1, rounds + 1):
        for d in drafts:
            if rnd > 1 and not d.feedback:
                continue
            src, patch, why = write(d, rnd)
            d.caught = []
            d.rows = None
            d.source, d.patch = src, patch
            d.problems = list(why)
            if src is not None and not why:
                parts, nn, _, problems = read_module(src, spec, questions)
                d.parts, d.nn = parts, nn
                d.problems += problems
                if not problems:
                    miss = coverage(spec, parts, nn)
                    if miss:
                        d.problems.append("clauses neither cited by a part nor in NOT_NORMATIVE: " + ", ".join(miss))
                if not d.problems:
                    d.problems += unknown_reads(src, parts, given_names(inputs))
                if not d.problems and extra_check is not None:
                    d.problems += extra_check(d)
            if d.problems:
                d.caught.append("contract" if src is not None else "no module")
        nums = numbers_in(spec.text) + [v for d in drafts if d.source for v in numbers_in(d.source)]
        pool = inputs.pool(nums)
        tin = [t["input"] for t in test_list]
        exin = [x for x, _ in examples]
        allin = pool + tin + exin
        for d in drafts:
            if d.problems:
                continue
            rows, err = _run_draft(d, questions, allin, mem_mb, item_s)
            if rows is None:
                d.problems.append(f"the module did not run in the sandbox: {err}")
                d.caught.append("sandbox")
            else:
                d.rows = rows
        ok = [d for d in drafts if d.rows is not None]
        # a test both drafts fail (whatever they answer) goes back once to the test writer: keep, fix or drop
        if len(ok) == 2:
            for j, t in enumerate(test_list):
                if t["id"] in reviewed:
                    continue
                k = len(pool) + j
                a = [{q: _norm(r) for q, r in d.rows[k]["answers"].items()} for d in ok]
                fails = [any(a[i].get(q) != e for q, e in t["expect"].items()) for i in range(2)]
                if all(fails):
                    got = [{q: a[i].get(q) for q in t["expect"]} for i in range(2)]
                    verdict, v = _review(spec, questions, t, got, review_gen, log)
                    reviewed[t["id"]] = {"verdict": verdict, "why": v.get("why"), "was": dict(t["expect"]),
                                         "drafts_gave": got}
                    if verdict == "fix" and isinstance(v.get("expect"), dict):
                        t["expect"] = {q: _norm(x) for q, x in v["expect"].items() if q in t["expect"]} or t["expect"]
                        reviewed[t["id"]]["now"] = dict(t["expect"])
                    elif verdict == "drop":
                        t["dropped"] = True
        live = [(j, t) for j, t in enumerate(test_list) if not t.get("dropped")]
        summary = {"round": rnd, "pool": len(pool), "tests": len(live), "drafts": []}
        fb = {d.index: [] for d in drafts}
        for d in drafts:
            if d.problems:
                fb[d.index].append("The module could not be used:\n- " + "\n- ".join(d.problems[:12]))
        for d in ok:
            failed = []
            for j, t in live:
                got = {q: _norm(r) for q, r in d.rows[len(pool) + j]["answers"].items()}
                if any(got.get(q) != e for q, e in t["expect"].items()):
                    failed.append((t, got, d.rows[len(pool) + j]))
            exfail = []
            for j, (x, want) in enumerate(examples):
                row = d.rows[len(pool) + len(tin) + j]
                got = {q: _norm(r) for q, r in row["answers"].items()}
                if any(got.get(q) != _norm(w) for q, w in want.items()):
                    exfail.append((x, want, got))
            refail = []
            if reference is not None:
                for k, x in enumerate(pool):
                    want = reference(dict(x))
                    got = {q: _norm(r) for q, r in d.rows[k]["answers"].items()}
                    if any(got.get(q) != _norm(w) for q, w in want.items()):
                        refail.append((x, want, got))
            silent = [k for k in range(len(pool)) if d.rows[k].get("error") or any(
                v is None for v in d.rows[k]["answers"].values()) or not d.rows[k]["answers"]]
            if silent:
                d.caught.append("abstained")
                lines = [f"Your module gave no answer on {len(silent)} of {len(pool)} inputs (a part raised, or read a "
                         "fact nothing gives). Examples:"]
                for k in silent[:5]:
                    row = d.rows[k]
                    lines.append(f"- input {_short(pool[k], 400)}\n  {row.get('error') or row.get('why')}")
                fb[d.index].append("\n".join(lines))
            if failed:
                d.caught.append("tests")
                lines = [f"{len(failed)} of {len(live)} tests derived from the specification fail (each names the clause "
                         "it checks):"]
                for t, got, row in failed[:8]:
                    lines.append(f"- test {t['id']} (clause {t.get('clause')}): input {_short(t['input'])}\n  expected "
                                 f"{t['expect']}; your module: {got}" + (f" ({row.get('error') or row.get('why')})"
                                                                         if row.get('error') or row.get('why') else "")
                                 + f"\n  why the test expects it: {t.get('why', '')[:300]}")
                lines.append("Clauses involved:\n" + _clauses_text(spec, [c for t, _, _ in failed[:8]
                                                                     for c in t.get("clause", "").split(", ")]))
                fb[d.index].append("\n".join(lines))
            for name, fl in (("labelled examples", exfail), ("the reference", refail)):
                if fl:
                    d.caught.append(name)
                    fb[d.index].append(f"{len(fl)} inputs differ from {name}:\n" + "\n".join(
                        f"- input {_short(x)}: expected {w}; your module: {g}" for x, w, g in fl[:8]))
        agree = None
        if len(ok) == 2:
            dis = []
            for k in range(len(allin)):
                a = {q: _norm(r) for q, r in ok[0].rows[k]["answers"].items()}
                b = {q: _norm(r) for q, r in ok[1].rows[k]["answers"].items()}
                if a != b or ok[0].rows[k].get("error") or ok[1].rows[k].get("error"):
                    dis.append(k)
            agree = {"inputs": len(allin), "disagree": len(dis)}
            if dis:
                for d, other in ((ok[0], ok[1]), (ok[1], ok[0])):
                    d.caught.append("disagreement")
                    lines = [f"Another implementation of the same specification, written independently, decides "
                             f"differently on {len(dis)} of {len(allin)} inputs. Examples:"]
                    cl = []
                    for k in dis[:8]:
                        mine, theirs = d.rows[k], other.rows[k]
                        cm, ct = _ran_clauses(d.parts, mine["ran"]), _ran_clauses(other.parts, theirs["ran"])
                        cl += cm + ct
                        lines.append(f"- input {_short(allin[k])}\n  yours: {mine['answers']}"
                                     + (f" ({mine.get('error') or mine.get('why')})" if mine.get('error') or mine.get('why') else "")
                                     + f", your parts cite {sorted(set(cm))}\n  the other: {theirs['answers']}, its parts "
                                       f"cite {sorted(set(ct))}")
                    lines.append("Clauses involved:\n" + _clauses_text(spec, cl, 12))
                    lines.append("Re-read these clauses. Where your module follows the specification, keep it; where it "
                                 "does not, fix it.")
                    fb[d.index].append("\n".join(lines))
        for d in drafts:
            d.feedback = "\n\n".join(fb[d.index]) or None
            summary["drafts"].append({"draft": d.index, "problems": d.problems[:5], "caught": d.caught,
                                      "feedback": bool(d.feedback)})
        summary["agreement"] = agree
        history.append(summary)
        if len(ok) == 2 and not any(d.feedback for d in drafts):
            return drafts[0], drafts[1], f"accepted in round {rnd}", history, reviewed
    last = "; ".join(f"draft {d.index}: " + ", ".join(dict.fromkeys(d.caught)) for d in drafts if d.feedback)
    history.append({"last_drafts": [{"source": d.source, "parts": d.parts, "problems": d.problems} for d in drafts]})
    return None, None, f"not accepted after {rounds} round(s): {last}", history, reviewed


def _gens(writer):
    ws = list(writer) if isinstance(writer, (list, tuple)) else [writer]
    ws = [_writer_of(w) for w in ws]
    return ws if len(ws) == 2 else [ws[0], ws[0]]


def _settings(i, writers):
    """Draft 0: temperature 0; draft 1: 0.7 with seed 1 — unless two writers are given (each at temperature 0)."""
    if writers[0] is not writers[1]:
        return 0.0, None
    return (0.0, None) if i == 0 else (0.7, 1)


def compile_spec(spec: Spec, questions, inputs: Inputs, writer=None, *, rounds: int = 4, tests: int = 40,
                 examples=(), reference: Callable | None = None, mem_mb: int = 1024, item_s: int = 5) -> Compiled:
    """Compile `spec` into catalog parts answering `questions` over `inputs` (see the module docs). `writer`: a
    solvi.generate Generator (or two, for two models), or a base URL (model openai/gpt-oss-120b). `examples`: labelled
    [(input, {question: answer})] that both drafts must match; `reference(input) → {question: answer}`, compared on the
    whole pool. Never raises for a bad draft: the result says whether it was accepted."""
    questions = list(questions)
    gens = _gens(writer)
    log = []
    test_list, bad, terr = _write_tests(spec, questions, inputs, gens[0], log, tests) if tests else ([], [], None)
    base = _task_text(spec, questions, inputs)

    def write(d, rnd):
        t, s = _settings(d.index, gens)
        if d.source is None or rnd == 1:
            prompt = f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}\n\nWrite the complete module now, in one ```python block."
            if d.feedback:
                prompt += f"\n\n## Your previous answer\n\n{d.feedback}"
        else:
            prompt = (f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}\n\n## Your previous module (round {rnd - 1})\n\n"
                      f"```python\n{d.source.rstrip()}\n```\n\n## What the harness found\n\n{d.feedback}\n\n"
                      "Rewrite the complete module so that these problems go away; keep what already works. Answer with "
                      "the whole module in one ```python block.")
        text, err = _ask(gens[d.index], prompt, t, s, log, f"round {rnd} draft {d.index}")
        code = code_of(text) if text else None
        if code is None:
            return d.source if d.source and rnd > 1 else None, None, [
                f"no module in the reply ({err or 'no ```python block'})" + ("; write the module sooner, keep the "
                                                                             "reasoning short" if err and "cut off" in err
                                                                             else "")]
        return code, None, []

    a, b, reason, history, reviewed = _loop(spec, questions, inputs, gens, rounds=rounds,
                                            tests=(test_list, bad, terr), examples=list(examples), reference=reference,
                                            write=write, log=log, mem_mb=mem_mb, item_s=item_s, extra_check=None,
                                            need_tests=bool(tests))
    return _result(spec, questions, a, b, reason, history, reviewed, test_list, bad, terr, log, gens, None)


def _result(spec, questions, a, b, reason, history, reviewed, test_list, bad, terr, log, gens, changes):
    record = {"spec_hash": spec.hash, "writer": [g.model_id for g in gens], "writer_fp": [g.fingerprint() for g in gens],
              "rounds": history, "tests": test_list, "tests_invalid": bad, "tests_error": terr, "reviews": reviewed,
              "calls": log, "time": time.time()}
    if a is not None:
        record["other_source"] = b.source
        if a.patch is not None:
            record["patch"] = a.patch
        return Compiled(spec, questions, a.source, a.parts, a.nn, True, reason, record, changes)
    if history and "last_drafts" in history[-1]:          # not accepted: the last drafts, for a reader (never loaded)
        record["last_drafts"] = history.pop()["last_drafts"]
    return Compiled(spec, questions, "", {}, {}, False, reason, record, changes)


def _funcs(source):
    return {n.name for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _effective(patch, pparts, old_parts):
    """A patch's PARTS without the entries that repeat a current part unchanged (no new function, same declaration)."""
    fs = _funcs(patch)
    return {n: p for n, p in pparts.items() if n in fs or old_parts.get(n) != p}


def recompile(old: Compiled, spec: Spec, inputs: Inputs, writer=None, *, rounds: int = 4, tests: int = 40,
              examples=(), reference: Callable | None = None, mem_mb: int = 1024, item_s: int = 5) -> Compiled:
    """Recompile an accepted compilation for a revised specification (`old.spec.revise(...)`): the writer returns only
    the parts it adds, replaces or removes — each citing a changed or added clause — and the rest stays byte-identical.
    Accepted by the same checks as compile_spec (two independent patches agree, tests written for the new
    specification). The result's `changes`: the clauses' changes and the parts added / replaced / removed / kept."""
    if spec.changes is None:
        raise ValueError("recompile needs a revised specification: old.spec.revise(new_text)")
    if not old.accepted:
        raise Rejected(old.reason)
    questions = list(old.questions)
    gens = _gens(writer)
    log = []
    ch = spec.changes
    if not (ch["changed"] or ch["added"] or ch["removed"]):
        c = Compiled(spec, questions, old.source, dict(old.parts), dict(old.not_normative), True,
                     "the specification did not change", {"spec_hash": spec.hash, "calls": []},
                     {"clauses": ch, "parts": {"added": [], "replaced": [], "removed": [], "kept": list(old.parts)}})
        return c
    test_list, bad, terr = _write_tests(spec, questions, inputs, gens[0], log, tests) if tests else ([], [], None)
    marks = {**{c: "changed" for c in ch["changed"]}, **{c: "added" for c in ch["added"]}}
    removed = [old.spec.clauses[c] for c in ch["removed"] if c in old.spec.clauses]
    base = _task_text(spec, questions, inputs, marks, removed)
    touched = set(ch["changed"]) | set(ch["added"])
    gone = set(ch["removed"])
    current = (f"## The current module (implements the previous version)\n\n```python\n{old.source.rstrip()}\n```")

    def write(d, rnd):
        t, s = _settings(d.index, gens)
        prompt = f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}\n\n{current}\n\n{PATCH_CONTRACT}"
        if d.patch is not None and rnd > 1:
            prompt += (f"\n\n## Your previous change (round {rnd - 1})\n\n```python\n{d.patch.rstrip()}\n```\n\n"
                       f"## What the harness found\n\n{d.feedback}\n\nWrite the change again, complete (it replaces your "
                       "previous change), so that these problems go away. One ```python block.")
        else:
            if d.feedback:
                prompt += f"\n\n## Your previous answer\n\n{d.feedback}"
            prompt += "\n\nWrite the change now, in one ```python block."
        text, err = _ask(gens[d.index], prompt, t, s, log, f"round {rnd} draft {d.index}")
        patch = code_of(text) if text else None
        if patch is None:
            return None, None, [f"no change in the reply ({err or 'no ```python block'})"]
        pparts, pnn, rm, problems = read_module(patch, spec, questions, patch=True)
        if problems:
            return None, patch, problems
        pparts = _effective(patch, pparts, old.parts)
        why = [f"PARTS names {n!r}, which is neither a function of the change nor a part of the current module"
               for n in pparts if n not in old.parts and n not in _funcs(patch)]
        for n, p in pparts.items():
            if not set(p.get("clauses") or []) & touched:
                why.append(f"{n} is added or replaced but cites no [changed] or [added] clause")
        for n, c in rm.items():
            if n not in old.parts:
                why.append(f"REMOVE names {n!r}, which is not a part of the current module")
            if c not in touched | gone:
                why.append(f"REMOVE[{n!r}] gives {c!r}, which is not a changed, added or removed clause")
        if why:
            return None, patch, why
        src, parts, nn = merge(old.source, old.parts, old.not_normative, patch, pparts, pnn, rm, spec)
        stale = [f"{n} still cites the removed clause {c}" for n, p in parts.items() for c in p.get("clauses") or []
                 if c in gone]
        return src, patch, stale

    a, b, reason, history, reviewed = _loop(spec, questions, inputs, gens, rounds=rounds,
                                            tests=(test_list, bad, terr), examples=list(examples), reference=reference,
                                            write=write, log=log, mem_mb=mem_mb, item_s=item_s, extra_check=None,
                                            need_tests=bool(tests))
    changes = {"clauses": ch}
    if a is not None:
        pparts, _, rm, _ = read_module(a.patch, spec, questions, patch=True)
        pparts = _effective(a.patch, pparts, old.parts)
        changes["parts"] = {"added": [n for n in pparts if n not in old.parts],
                            "replaced": [n for n in pparts if n in old.parts], "removed": list(rm),
                            "kept": [n for n in old.parts if n not in pparts and n not in rm]}
    c = _result(spec, questions, a, b, reason, history, reviewed, test_list, bad, terr, log, gens, changes)
    c.record["parent"] = {"spec_hash": old.spec.hash, "source_sha": hashlib.sha256(old.source.encode()).hexdigest()}
    return c


# ───────────────────────────────────────────────────────────── what a change does to decisions
@dataclass
class DecisionDiff:
    """The decisions a new compilation changes: `changed` = [{"id", "question", "old", "new", "causes": [{"step",
    "part", "clauses", "why"}]}]; `report` is solvi.diff's own report."""
    changed: list
    report: Any
    total: int

    def __str__(self):
        from collections import Counter
        c = Counter((x["question"], _norm(x["old"]), _norm(x["new"])) for x in self.changed)
        lines = [f"{len(self.changed)} of {self.total} decisions change"]
        lines += [f"  {q}: {o} → {n}: {k}" for (q, o, n), k in c.most_common()]
        cl = Counter(tuple(sorted({c for cause in x["causes"] for c in cause["clauses"]})) for x in self.changed)
        lines += ["  because of clauses " + ", ".join(k) + f": {v}" for k, v in cl.most_common()]
        return "\n".join(lines)


def _plain_answer(a):
    return a.get("answer") if isinstance(a, dict) else a


def _status(a):
    return a.get("status") if isinstance(a, dict) else None


def decision_diff(old: Compiled, new: Compiled, *, store=None, inputs=None, **filters) -> DecisionDiff:
    """Which decisions `new` would change: the stored ones (`store`, decided by `old`'s catalog — filters as for
    solvi.diff.diff) or `inputs` (each decided by `old` first, in a temporary store). Each change's causes are the steps
    solvi.diff names, with the clauses their parts implement (in the new compilation, else the old one)."""
    from .diff import diff
    from .storage import JSONLStorage
    tmp = None
    if store is None:
        if inputs is None:
            raise ValueError("decision_diff needs a store of decisions or inputs")
        tmp = tempfile.mkdtemp(prefix="solvi_ddiff_")
        store = JSONLStorage(str(Path(tmp) / "decisions.jsonl"))
        s_old = old.system(storage=store)
        for x in inputs:
            s_old.ask(dict(x))
    try:
        rep = diff(store, new.system(), **filters)
        out = []
        for ch in rep.changed:
            for q, d in ch["questions"].items():
                if not d.get("changed"):
                    continue
                causes = []
                for c in d.get("causes") or []:
                    part = new.part_of(c["name"]) or old.part_of(c["name"])
                    src = new if new.part_of(c["name"]) else old
                    causes.append({"step": c["name"], "part": part, "clauses": src.clauses_of(part) if part else [],
                                   "why": c.get("why")})
                out.append({"id": ch["id"], "seq": ch.get("seq"), "question": q, "old": _plain_answer(d["old"]),
                            "new": _plain_answer(d["new"]), "status": [_status(d["old"]), _status(d["new"])],
                            "causes": causes})
        total = len(list(store.iter()))
        return DecisionDiff(out, rep, total)
    finally:
        if tmp is not None:
            store.close()
            shutil.rmtree(tmp, ignore_errors=True)


# ───────────────────────────────────────────────────────────── versions
class Versions:
    """Compiled catalogs as numbered versions in a folder (`v1/`, `v2/`, ... each `module.py` + `compiled.json`).
    `system(n)` builds a version's System; `replay_all(store)` replays every stored decision with the version whose
    catalog fingerprint it recorded — an old decision is checked against the rules it was made with."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._fps: dict = {}

    def __len__(self):
        return len(self.numbers())

    def numbers(self) -> list[int]:
        return sorted(int(p.name[1:]) for p in self.path.glob("v*") if p.name[1:].isdigit() and (p / "compiled.json").exists())

    def add(self, compiled: Compiled, label: str | None = None) -> int:
        """Store an accepted compilation as the next version → its number."""
        if not compiled.accepted:
            raise Rejected(compiled.reason)
        n = (self.numbers() or [0])[-1] + 1
        p = compiled.save(self.path / f"v{n}")
        (p / "version.json").write_text(json.dumps({"version": n, "label": label, "fingerprint": compiled.fingerprint,
                                                    "spec_hash": compiled.spec.hash, "time": time.time()}, indent=1))
        return n

    def get(self, n: int | None = None) -> Compiled:
        """Version n (the latest when None)."""
        n = n if n is not None else self.numbers()[-1]
        return Compiled.load(self.path / f"v{n}")

    def info(self, n: int) -> dict:
        return json.loads((self.path / f"v{n}" / "version.json").read_text())

    def system(self, n: int | None = None, **kw):
        return self.get(n).system(**kw)

    def find(self, fingerprint: str) -> int | None:
        """The version whose catalog has this fingerprint."""
        for n in self.numbers():
            if self.info(n)["fingerprint"] == fingerprint:
                return n
        return None

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


# ───────────────────────────────────────────────────────────── into an agent guard
def to_guard(compiled: Compiled, guard, tools=None, *, question: str | None = None, on_fail: str = "deny") -> list:
    """Register the compiled hard checks as policies of a `solvi.agents.Guard` → their names. Each policy reads the
    compiled catalog's inputs (they must be facts the guard gives: tool_name, tool_arguments, conversation, ... or your
    declared facts), runs the compiled System on them and fails with the check's reasons when the check is false.
    `question`: the compiled question whose hard checks become policies (default: the only one)."""
    from .system import System
    cat, qs = compiled.catalog()
    q = question or (qs[0].name if len(qs) == 1 else None)
    if q is None:
        raise ValueError("to_guard: name the compiled question (question=...)")
    system = System(cat, qs)
    given = sorted({i for p in cat.parts.values() for i in p.inputs} - set(cat.parts))
    fp = compiled.fingerprint
    names = []
    import inspect
    for name, p in compiled.parts.items():
        if p.get("kind") != "check" or not p.get("hard") or q not in (p.get("then") or {}):
            continue

        def policy(_name=name, _fp=fp, **facts):
            res = system.ask({k: facts[k] for k in given if k in facts})
            v = res.values.get(_name)
            if v is None:
                return Fail(f"{_name} could not be evaluated")
            return v if not v else True
        policy.__signature__ = inspect.Signature([inspect.Parameter(g, inspect.Parameter.KEYWORD_ONLY) for g in given])
        policy.__name__ = name
        policy.__doc__ = "; ".join(compiled.spec.clauses[c].text for c in p.get("clauses") or [] if c in compiled.spec.clauses)[:300]
        guard.policy(tools, on_fail=on_fail)(policy)
        names.append(name)
    return names


__all__ = ["Clause", "Compiled", "DecisionDiff", "Inputs", "Rejected", "Spec", "Versions", "compile_spec", "coverage",
           "decision_diff", "merge", "numbers_in", "read_module", "recompile", "split_clauses", "to_guard"]
