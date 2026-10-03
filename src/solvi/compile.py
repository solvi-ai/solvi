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
   the clause it checks. A test that every draft which answers it fails goes back once to the test writer,
   which works the answer out again and keeps, corrects or drops it (recorded). Every input of the pool must get an
   answer;
4. both match the labelled `examples` and the `reference`, when you give them.
A draft with failures is rewritten from its module and the failures, up to `rounds` rounds; then the compilation is not
accepted, `Compiled.catalog()` refuses it, and `reason` says why. A draft that has not run for two rounds in a row
(refused by the contract or the sandbox, or no module in the reply) is replaced by a fresh one, written from the task
again with a seed of its own and told only what the stuck one was refused for — at most `fresh_drafts` times, each
replacement recorded; the fresh draft meets the same checks. The record keeps the spec's hash and clauses, every
prompt and reply, the writer, the tests, and each check's outcome per round.

    c = compile_spec(spec, questions, inputs, writer, review=ask_a_person)   # a person resolves what drafts dispute
    c.record["person"]                                    # every question asked, every answer, the tests they became

A person in the loop (`review=`) is asked, within a budget, about the inputs two drafts decide differently (one input
of each of the largest kinds of disagreement) and about disputed tests (instead of the test writer's re-check). An
answer becomes a test (source "person"), never code: both drafts must pass it, and the other conditions stay; the
answers are trusted like labels (a wrong one is a wrong test). An answer saying the specification does not decide the
input is a gap, and blocks acceptance. The person sees only what the drafts dispute, not a misreading they share.

    c = compile_groups(spec, questions, inputs, writer, by="tool_name")   # a large specification, group by group
    c.record["groups"]                                    # per group: its values, clauses, accepted or why not

A large specification compiles in groups: the clauses that govern each kind of decision (the values of one input
field) are compiled on their own, with the shared clauses, under the same acceptance; the accepted groups are assembled
into one module that routes each input to its group, and the whole is checked again on the full pool.

What it does not do. Agreement of two drafts is not correctness: two samples of one model can share a misreading, and
the tests come from the same model. Coverage is checked by citation, not by meaning — a part may cite a clause it
implements wrongly. The inputs pool decides what "agree" covers: a region of inputs nobody generates is not compared.
Labels, when you have them, are the stronger check. Nothing here writes extractors from text or searches; the
compiled parts read structured inputs. In groups, a clause the grouping leaves out of a group is not compiled for it:
the group's agreement and tests cannot see what its specification does not say — read the grouping.
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

    def subset(self, ids, name: str | None = None) -> "Spec":
        """The clauses `ids` of this specification (their ids, sections and order kept) as a specification of its own."""
        keep = set(ids)
        unknown = sorted(keep - set(self.clauses))
        if unknown:
            raise ValueError(f"not clauses of this specification: {', '.join(unknown)}")
        s = Spec.__new__(Spec)
        s.name, s.changes = name or self.name, None
        s.clauses = {i: c for i, c in self.clauses.items() if i in keep}
        s.text = "\n\n".join(c.text for c in s.clauses.values())
        if not s.clauses:
            raise ValueError("the subset has no clauses")
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
            problems.append(f"PARTS names {name!r}, which is not a top-level function of the module (define it at the "
                            "top level, not inside a class or another function)")
        kind = p.get("kind")
        if kind not in KINDS:
            problems.append(f"{name}: kind must be one of {KINDS}, not {kind!r}")
        cl = p.get("clauses")
        if not isinstance(cl, list) or (not cl and kind == "check"):     # a rule giving only a default, or a fact that
            problems.append(f"{name}: 'clauses' must list the clauses it implements")  # only reads an input, cites none
        else:
            problems += [f"{name} cites {c!r}, which is not a clause" for c in cl if c not in spec.clauses]
        if kind == "check":
            then = p.get("then") or {}
            if p.get("hard") and not then:
                problems.append(f"{name}: a hard check needs 'then' ({{question: answer}})")
            for q, a in list(then.items()):                # True / False for a yes/no question, as a rule may return
                if isinstance(a, bool) and q in qs and options_of(qs[q]) == ["yes", "no"]:
                    then[q] = "yes" if a else "no"
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
    nn = {c: r for c, r in nn.items() if c in spec.clauses}    # a name that is no clause declares nothing: left out
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
    """Parts that read a name which is neither a given fact nor a fact another part sets (a rule sets none), and parts
    named like a given fact."""
    if given is None:
        return []
    facts = {n for n, p in parts.items() if p.get("kind") in ("fn", "check")}
    out = [f"part {n} has the name of an input: a part is named after what it computes (e.g. {n}_value), the input "
           f"keeps its name" for n in parts if n in given]
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in parts:
            if node.args.vararg or node.args.kwarg:
                out.append(f"part {node.name} takes *args / **kwargs: a part names each fact it reads as an argument")
                continue
            args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            bad = [a for a in args if a not in given and a not in facts]
            if bad:
                near = {b: difflib.get_close_matches(b, sorted(given | facts), n=1, cutoff=0.6)
                        or [f for f in sorted(given | facts) if b in f][:1] for b in bad}
                hint = "; ".join(f"{b} — did you mean {m[0]}?" for b, m in near.items() if m)
                out.append(f"part {node.name} reads {', '.join(bad)}: neither an input field nor a fact set by a part. "
                           f"An argument of a part must be named exactly as an input ({', '.join(sorted(given))}) or "
                           f"as the part that sets the fact" + (f" ({hint})" if hint else "")
                           + "; a value inside an input is read through a fn part that takes the input, or worked "
                             "out inside the part")
    return out


def demote_helpers(source: str, parts: dict, given) -> tuple[dict, list[str]]:
    """Parts that are really helpers → (parts without them, notes). A "fn" part (or a soft check) that reads a name
    neither an input nor a part gives, and that other parts call as a plain function, is a helper the writer listed in
    PARTS: it leaves PARTS, and the clauses it cited go to every part that calls it (directly or through helpers).
    The decisions do not change — the callers computed with it anyway; only the record gets coarser. A hard check is
    never demoted (that would drop its forcing), nor a part no other part calls."""
    if given is None:
        return parts, []
    tree = ast.parse(source)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = {f: {c.func.id for c in ast.walk(n) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id in funcs and c.func.id != f} for f, n in funcs.items()}

    def closure(f):
        seen, todo = set(), [f]
        while todo:
            for g in calls.get(todo.pop(), ()):
                if g not in seen:
                    seen.add(g)
                    todo.append(g)
        return seen
    parts = {k: dict(v) if isinstance(v, dict) else v for k, v in parts.items()}
    notes = []
    while True:
        facts = {n for n, p in parts.items() if p.get("kind") in ("fn", "check")}
        cand = None
        for name, p in parts.items():
            node = funcs.get(name)
            if node is None or p.get("kind") not in ("fn", "check") or (p.get("kind") == "check" and p.get("hard")):
                continue
            args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if node.args.vararg or node.args.kwarg or any(a not in given and a not in facts for a in args):
                users = [u for u in parts if u != name and u in funcs and name in closure(u)]
                if users:
                    cand = (name, users)
                    break
        if cand is None:
            return parts, notes
        name, users = cand
        cl = parts.pop(name).get("clauses") or []
        for u in users:
            parts[u]["clauses"] = list(dict.fromkeys(list(parts[u].get("clauses") or []) + list(cl)))
        notes.append(f"{name} reads names no input or part gives and is called by {', '.join(users)}: treated as a "
                     f"helper, its clauses ({', '.join(cl) or 'none'}) go to {', '.join(users)}")


def inner_fields(inputs) -> dict:
    """{key: input field} for the keys inside dict-valued inputs (seen in the samples, else in the pool), each key found
    in one field only — what `add_accessors` may read for a part."""
    xs = inputs.samples
    if not xs:
        try:
            xs = inputs.pool()[:200]
        except ValueError:
            xs = []
    seen = {}
    for x in xs:
        for f, v in x.items():
            if isinstance(v, dict):
                for k in v:
                    if isinstance(k, str) and k.isidentifier():
                        seen.setdefault(k, set()).add(f)
    return {k: next(iter(fs)) for k, fs in seen.items() if len(fs) == 1}


def add_accessors(source: str, parts: dict, given, inner: dict, questions=()) -> tuple[str, dict, list[str]]:
    """Parts that read a key of a dict-valued input by its own name (`friends` inside `facts`) get that fact: an
    accessor part `def friends(facts): return facts["friends"]` is added to the module (citing no clause) →
    (source, parts, notes). Only keys found in one input field, never a name an input, a part or a question has. A
    missing key raises in the accessor, so the decision abstains — and an abstention is a failure of the draft."""
    if given is None or not inner:
        return source, parts, []
    taken = set(given) | set(parts) | {q.name for q in questions} | _funcs(source)
    want = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in parts:
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if a.arg not in taken and a.arg in inner:
                    want.setdefault(a.arg, inner[a.arg])
    if not want:
        return source, parts, []
    defs = "\n\n".join(f"def {k}({f}):\n    return {f}[{k!r}]\n" for k, f in want.items())
    parts = {**parts, **{k: {"kind": "fn", "clauses": [], "accessor": f} for k, f in want.items()}}
    notes = [f"{k} read inside the input {f}: accessor part added" for k, f in want.items()]
    return with_parts(source.rstrip() + "\n\n\n" + defs, parts), parts, notes


def with_parts(source: str, parts: dict) -> str:
    """The module with its PARTS literal (and PARTS.update calls) replaced by `parts`."""
    keep = [t for k, t in _top(source) if k != "PARTS" and not (k is None and re.match(r"\s*PARTS\s*\.", t))]
    return "\n\n".join(t.rstrip("\n") for t in keep) + "\n\n\n" + _literal("PARTS", parts)


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
  parts, spelled exactly — `def member_fee(base_fee, member)` reads the fact the part `base_fee` sets and the input
  `member` (it may not call them `base` or `is_member`). Never give a part the name of an input field or of a question.
  The harness calls the parts and passes the facts; inside a part, call helper functions as you like.
- kinds: "fn" computes a value; "check" returns True when its condition holds; "rule" returns the answer of one
  question (one of its options; a yes/no rule may return True / False).
- a hard check that returns False forces the answer its "then" names, whatever the rule says (the first false hard check
  in PARTS order wins). A check may return Fail("why") instead of False (`Fail` needs no import).
- for each question its rule runs with the parts it reads (and theirs), and every hard check whose "then" names the
  question. A part that raises makes the questions that need it abstain.
- write one part per quantity a clause defines (for example one function per row of a points table), so the record of
  a decision shows each; helper functions that are not parts are allowed.
- deterministic and pure: no state between calls, no input / output.
- no classes: data are dicts, lists and tuples (a class needs `__init__`, a dunder name the sandbox refuses).
- no type annotations on a part's arguments or result: solvi checks them against the facts it is given, and an inexact
  one (say `list` for an input that may be None) makes the part refuse that input.

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
- REMOVE = {"part_name": "c18"} for each part to remove, naming the clause OF THE CHANGE that removes it (a
  [changed] or [added] clause, or a removed one) — not the clause the part implemented;
- NOT_NORMATIVE = {...} for [added] clauses that need no code, and for clauses that no part implements any more
  (for example an old rule a new clause abolishes: "superseded by c18"). After the change every clause must still be
  cited by a part or be in NOT_NORMATIVE.
- a replaced part cites the [changed] or [added] clause that changes it, besides the clauses it implemented.

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

The implementations of the specification written so far, independently, all fail this test: they answer {got}
(None: the implementation could not answer) where the test expects {expect}. They may be wrong, or the test may be. Re-read clause
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
    groups: dict | None = None            # compile_groups: {group name: its own Compiled}
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

    def reviewer(self) -> Callable:
        """The person's recorded answers as a reviewer (`compile_spec(..., review=c.reviewer())` reruns this
        compilation with them): a question asked before gets the same ruling; a new one gets none (None)."""
        logs = list((self.record.get("person") or {}).get("log") or [])
        for g in (self.groups or {}).values():
            logs += (g.record.get("person") or {}).get("log") or []
        table = {_key(e["kind"], e["input"], e.get("test")): e["ruling"] for e in logs if e.get("ruling")}

        def review(d):
            r = table.get(_key(d.kind, d.input, d.test))
            return Ruling(**r) if r else None
        review.reviewer_name = "replay"
        return review

    def to_dict(self) -> dict:
        return {"spec": self.spec.to_dict(), "questions": [q.model_dump() for q in self.questions], "source": self.source,
                "parts": self.parts, "not_normative": self.not_normative, "accepted": self.accepted,
                "reason": self.reason, "record": self.record, "changes": self.changes,
                "groups": {n: g.to_dict() for n, g in self.groups.items()} if self.groups is not None else None}

    @classmethod
    def from_dict(cls, d) -> "Compiled":
        return cls(Spec.from_dict(d["spec"]), [Question.model_validate(q) for q in d["questions"]], d["source"],
                   d["parts"], d["not_normative"], d["accepted"], d["reason"], d.get("record") or {}, d.get("changes"),
                   {n: cls.from_dict(g) for n, g in d["groups"].items()} if d.get("groups") is not None else None)

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


# ───────────────────────────────────────────────────────────── a person in the loop
@dataclass
class Dispute:
    """What a person is asked during a compilation.

    kind: "disagreement" — the two drafts answer `input` differently (`answers[i]` is draft i's answer, `clauses[i]`
    the clauses its deciding parts cite; `similar`: how many inputs of the pool disagree the same way); "test" — every
    draft that answers the test `test` (its clause, input, expected answer and why) fails it (`answers`: what they
    gave). `round`: the round; `spec`: the specification's name (a group's, in compile_groups); `clause_text`: the
    texts of the clauses involved."""
    kind: str
    input: dict
    answers: list
    clauses: list
    round: int
    spec: str = ""
    test: dict | None = None
    similar: int = 1
    clause_text: dict = field(default_factory=dict)

    def text(self) -> str:
        """The question as a person reads it."""
        lines = [f"[{self.spec}] round {self.round}: " + (
            "two drafts decide this input differently" if self.kind == "disagreement" else
            "every draft fails this test derived from the specification")]
        lines.append("input: " + _short(self.input, 2000))
        if self.kind == "test":
            lines.append(f"the test expects {self.test.get('expect')} (clause {self.test.get('clause')}): "
                         f"{self.test.get('why', '')}")
        for i, a in enumerate(self.answers):
            cl = self.clauses[i] if i < len(self.clauses) else []
            lines.append(f"draft {i}: {a}" + (f" — citing {', '.join(cl)}" if cl else ""))
        if self.similar > 1:
            lines.append(f"({self.similar} inputs of the pool disagree this way)")
        lines += [f"  [{c}] {t}" for c, t in self.clause_text.items()]
        return "\n".join(lines)

    def to_dict(self):
        return {"kind": self.kind, "input": self.input, "answers": self.answers, "clauses": self.clauses,
                "round": self.round, "spec": self.spec, "test": self.test, "similar": self.similar}


@dataclass
class Ruling:
    """A person's answer to a Dispute. verdict: "draft" (draft `draft` is right), "answer" (the right answer is
    `expect`, {question: answer}), "neither" (the specification does not decide this input: a gap — or, with `expect`,
    the right answer neither draft gave), "keep" / "drop" (a disputed test is right / is not a test of the
    specification). A plain value is read too: an int picks a draft, a dict is an answer, "keep" / "drop" / "neither"."""
    verdict: str
    draft: int | None = None
    expect: dict | None = None
    note: str = ""

    @classmethod
    def pick(cls, draft: int, note: str = ""):
        return cls("draft", draft=int(draft), note=note)

    @classmethod
    def answer(cls, expect: dict, note: str = ""):
        return cls("answer", expect=dict(expect), note=note)

    @classmethod
    def neither(cls, note: str = "", expect: dict | None = None):
        return cls("neither", expect=dict(expect) if expect else None, note=note)

    @classmethod
    def keep(cls, note: str = ""):
        return cls("keep", note=note)

    @classmethod
    def drop(cls, note: str = ""):
        return cls("drop", note=note)

    def to_dict(self):
        return {"verdict": self.verdict, "draft": self.draft, "expect": self.expect, "note": self.note}


def _as_ruling(x):
    if x is None or isinstance(x, Ruling):
        return x
    if isinstance(x, bool):
        raise ValueError("a ruling is a Ruling, a draft's index, an answer dict, or 'keep' / 'drop' / 'neither'")
    if isinstance(x, int):
        return Ruling.pick(x)
    if isinstance(x, dict):
        return Ruling.answer(x)
    if isinstance(x, str) and x in ("keep", "drop", "neither"):
        return Ruling(x)
    raise ValueError(f"not a ruling: {x!r}")


def _key(kind, x, test=None):
    return json.dumps([kind, x, (test or {}).get("expect")], sort_keys=True, ensure_ascii=False, default=str)


def reference_reviewer(reference: Callable) -> Callable:
    """A simulated person for experiments: `reference(input) → {question: answer}` (a hand-written reference). On a
    disagreement it picks the draft whose answers equal the reference's, else gives the reference's answer (an answer
    neither draft gave); on a disputed test it keeps the test when the reference agrees, else corrects it. An input the
    reference cannot read is skipped (a test on it: dropped)."""
    def review(d: Dispute):
        try:
            want = {q: _norm(v) for q, v in reference(json.loads(json.dumps(d.input))).items()}
        except Exception as e:  # noqa: BLE001 — the reference cannot read this input
            return Ruling.drop(f"the reference cannot read this input: {type(e).__name__}") if d.kind == "test" else None
        if d.kind == "test":
            exp = d.test.get("expect") or {}
            w = {q: want.get(q) for q in exp}
            return Ruling.keep("the reference agrees") if w == exp else Ruling.answer(w, "the reference")
        for i, a in enumerate(d.answers):
            if all(_norm((a or {}).get(q)) == v for q, v in want.items()):
                return Ruling.pick(i, "the reference")
        return Ruling.answer(want, "the reference (neither draft)")
    review.reviewer_name = "reference"
    return review


class _Person:
    """The person of one compilation: asks within the budget, keeps every question and answer for the record."""

    def __init__(self, review, budget, per_round, spec, questions):
        self.review, self.budget, self.per_round = review, budget, per_round
        self.spec, self.qs = spec, {q.name: q for q in questions}
        self.decisions, self.gaps, self.skipped, self.invalid = [], [], 0, []
        self.round, self.this_round, self.n = 0, 0, 0
        self.decided = {}                              # json input → expect (the person tests' inputs)
        self.sigs = set()                              # the kinds of disagreement asked about

    def can_ask(self):
        return self.review is not None and (self.budget is None or self.n < self.budget) and \
            (self.per_round is None or self.this_round < self.per_round)

    def new_round(self, rnd):
        self.round, self.this_round = rnd, 0

    def _clean(self, expect, keys=None):
        """The answer as a test expects it (normalised; only questions and allowed options) → (expect, why not)."""
        if not isinstance(expect, dict) or not expect:
            return None, "no answer given"
        out = {}
        for q, a in expect.items():
            if q not in self.qs:
                return None, f"{q!r} is not a question"
            o = options_of(self.qs[q])
            if o is not None and _norm(a) not in [_norm(x) for x in o]:
                return None, f"{q} = {a!r} is not one of its options"
            if a is None:
                return None, f"{q}: no answer"
            out[q] = _norm(a)
        return out, None

    def ask(self, d: Dispute):
        """→ (ruling, expect or None, gap: bool); None when the person did not answer."""
        self.n += 1
        self.this_round += 1
        t0 = time.time()
        try:
            r = _as_ruling(self.review(d))
            err = None
        except Exception as e:  # noqa: BLE001 — a reviewer that fails is recorded, the loop goes on
            r, err = None, f"{type(e).__name__}: {e}"[:300]
        entry = {"n": self.n, **d.to_dict(), "ruling": r.to_dict() if r else None, "error": err,
                 "seconds": round(time.time() - t0, 2)}
        if r is None:
            self.skipped += 1
            entry["outcome"] = "no answer"
            self.decisions.append(entry)
            return None, None, False
        expect, gap, why = None, False, None
        if r.verdict == "draft":
            if r.draft not in range(len(d.answers)):
                why = f"draft {r.draft} is not one of the drafts asked about"
            else:
                expect, why = self._clean({q: v for q, v in (d.answers[r.draft] or {}).items() if v is not None})
        elif r.verdict in ("answer", "neither") and r.expect:
            expect, why = self._clean(r.expect)
        elif r.verdict == "neither":
            gap = True
        elif r.verdict not in ("keep", "drop"):
            why = f"unknown verdict {r.verdict!r}"
        elif d.kind != "test":
            why = f"{r.verdict!r} answers a disputed test, not a disagreement"
        entry["expect"] = expect
        entry["neither"] = bool(expect) and all(
            any(_norm((a or {}).get(q)) != v for q, v in expect.items()) for a in d.answers)
        entry["gap"] = gap
        if why:
            entry["outcome"] = "invalid: " + why
            self.invalid.append(entry)
        else:
            entry["outcome"] = "gap" if gap else r.verdict
        self.decisions.append(entry)
        if why:
            return None, None, False
        if gap:
            self.gaps.append({"input": d.input, "kind": d.kind, "note": r.note, "round": d.round})
        if expect:
            self.decided[json.dumps(d.input, sort_keys=True, default=str)] = expect
        return r, expect, gap

    def record(self):
        dec = [x for x in self.decisions if x.get("ruling") and not str(x.get("outcome", "")).startswith("invalid")]
        return {"reviewer": getattr(self.review, "reviewer_name", getattr(self.review, "__name__", None)),
                "budget": self.budget, "per_round": self.per_round, "asked": self.n, "decisions": len(dec),
                "neither": sum(bool(x.get("neither")) for x in dec), "gaps": self.gaps, "skipped": self.skipped,
                "invalid": len(self.invalid),
                "by_kind": {k: sum(x["kind"] == k for x in dec) for k in ("disagreement", "test")},
                "log": self.decisions}


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
    generation: int = 0                   # 0: the first writer's draft; n: the n-th fresh draft that replaced it
    fresh: bool = False                   # the next write starts over (a replacement), not from this module
    stuck: int = 0                        # rounds in a row this draft did not run (contract, sandbox, no module)
    pitfalls: list = field(default_factory=list)   # what the replaced drafts were refused for (told to a fresh one)
    notes: list = field(default_factory=list)      # parts treated as helpers (demote_helpers), this round


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
          need_tests=False, fresh_drafts=0, stuck_after=2, person=None):
    """The write → check → rewrite loop over two drafts → (accepted draft or None, reason, per-round summary, tests).
    A draft that did not run for `stuck_after` rounds in a row is replaced by a fresh one (at most `fresh_drafts` in
    all): its next write starts over from the task, told only what the replaced drafts were refused for. `person`: a
    _Person asked about disputed tests and disagreements; each answer becomes a test (source "person")."""
    review_gen = gens[0]
    test_list, bad_tests, terr = tests
    if need_tests and not test_list:
        return None, None, ("not accepted: no valid tests were derived from the specification ("
                            + (terr or f"{len(bad_tests)} invalid") + ")"), [], {}
    reviewed = {}
    history = []
    drafts = [_Draft(0), _Draft(1)]
    used = 0
    inner = inner_fields(inputs)
    qnames = [q.name for q in questions]

    def person_test(x, expect, clauses, note, rnd, k):
        """A person's answer as a test; writer tests on the same input that contradict it are corrected."""
        pid = f"p{sum(1 for t in test_list if t.get('source') == 'person') + 1}"
        t = {"id": pid, "clause": ", ".join(c for c in dict.fromkeys(clauses) if c in spec.clauses),
             "input": x, "expect": dict(expect), "why": "a person decided" + (f": {note}" if note else ""),
             "source": "person", "round": rnd}
        key = json.dumps(x, sort_keys=True, default=str)
        for o in test_list:
            if o.get("source") != "person" and not o.get("dropped") and \
                    json.dumps(o["input"], sort_keys=True, default=str) == key:
                clash = {q: e for q, e in o["expect"].items() if q in expect and expect[q] != e}
                if clash:
                    reviewed[o["id"]] = {"verdict": "fix", "by": "person", "why": f"the person's answer on {pid}",
                                         "was": dict(o["expect"]), "now": {**o["expect"], **{q: expect[q] for q in clash}}}
                    o["expect"] = reviewed[o["id"]]["now"]
        test_list.append(t)
        row_of[pid] = k
    for rnd in range(1, rounds + 1):
        if person is not None:
            person.new_round(rnd)
        for d in drafts:
            if rnd > 1 and not d.feedback:
                d.caught = []
                continue
            src, patch, why = write(d, rnd)
            d.fresh = False
            d.caught = []
            d.rows = None
            d.source, d.patch = src, patch
            d.problems = list(why)
            d.notes = []
            if src is not None and not why:
                parts, nn, _, problems = read_module(src, spec, questions)
                if not problems:
                    src, parts, d.notes = add_accessors(src, parts, given_names(inputs), inner, questions)
                    parts, demoted = demote_helpers(src, parts, given_names(inputs))
                    if demoted:
                        src = with_parts(src, parts)
                    d.notes += demoted
                    d.source = src
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
        row_of = {t["id"]: len(pool) + j for j, t in enumerate(test_list)}
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
        asked = {"tests": 0, "disagreements": 0}
        # a test every running draft that answers it fails goes back once — to the person when there is one (within
        # the budget; deferred to a later round when this round's questions are used up), else to the test writer:
        # keep, fix or drop (a draft that abstains or raises on it says nothing about the test — that goes back to
        # the draft)
        if ok:
            for t in list(test_list):
                if t["id"] in reviewed or t.get("source") == "person" or t.get("dropped"):
                    continue
                k = row_of[t["id"]]
                a = [{q: _norm(r) for q, r in d.rows[k]["answers"].items()} for d in ok]
                fails = [any(a[i].get(q) != e for q, e in t["expect"].items()) for i in range(len(ok))]
                answered = [not d.rows[k].get("error") and all(a[i].get(q) is not None for q in t["expect"])
                            for i, d in enumerate(ok)]
                if any(answered) and all(f for f, x in zip(fails, answered) if x):
                    got = [{q: a[i].get(q) for q in t["expect"]} for i in range(len(ok)) if answered[i]]
                    if person is not None and person.review is not None and \
                            (person.budget is None or person.n < person.budget):
                        if not person.can_ask():
                            continue                       # this round's questions are used up: ask in the next
                        cl = [c for c in re.findall(r"\bc\d+\b", t.get("clause", "")) if c in spec.clauses]
                        dsp = Dispute("test", t["input"], got, [cl] * len(got), rnd, spec.name,
                                      test={x: t[x] for x in ("id", "clause", "expect", "why") if x in t},
                                      clause_text={c: spec.clauses[c].text for c in cl})
                        r, expect, gap = person.ask(dsp)
                        asked["tests"] += 1
                        if r is None:
                            continue
                        entry = {"verdict": None, "by": "person", "why": r.note, "was": dict(t["expect"]),
                                 "drafts_gave": got}
                        if gap or r.verdict == "drop":
                            t["dropped"] = True
                            entry["verdict"] = "drop"
                        elif expect:
                            t["expect"] = {q: expect[q] for q in t["expect"] if q in expect} or dict(expect)
                            t["source"] = "person"
                            entry["verdict"], entry["now"] = "fix", dict(t["expect"])
                        else:
                            t["confirmed_by"] = "person"
                            entry["verdict"] = "keep"
                        reviewed[t["id"]] = entry
                        continue
                    verdict, v = _review(spec, questions, t, got, review_gen, log)
                    reviewed[t["id"]] = {"verdict": verdict, "why": v.get("why"), "was": dict(t["expect"]),
                                         "drafts_gave": got}
                    if verdict == "fix" and isinstance(v.get("expect"), dict):
                        t["expect"] = {q: _norm(x) for q, x in v["expect"].items() if q in t["expect"]} or t["expect"]
                        reviewed[t["id"]]["now"] = dict(t["expect"])
                    elif verdict == "drop":
                        t["dropped"] = True
        # the inputs the two drafts decide differently; the person is asked about one input of each of the largest
        # kinds of disagreement (both answers and the clauses cited), and each answer becomes a test
        dis = []
        if len(ok) == 2:
            for k in range(len(allin)):
                a = {q: _norm(r) for q, r in ok[0].rows[k]["answers"].items()}
                b = {q: _norm(r) for q, r in ok[1].rows[k]["answers"].items()}
                if a != b or ok[0].rows[k].get("error") or ok[1].rows[k].get("error"):
                    dis.append(k)
            if dis and person is not None and person.can_ask():
                kinds = {}
                for k in dis:
                    if json.dumps(allin[k], sort_keys=True, default=str) in person.decided:
                        continue                           # decided already: its test tells the drafts
                    ans = [{q: _norm(r) for q, r in d.rows[k]["answers"].items()} for d in ok]
                    cls = [sorted(set(_ran_clauses(d.parts, d.rows[k]["ran"]))) for d in ok]
                    sig = json.dumps([ans, cls], sort_keys=True, default=str)
                    kinds.setdefault(sig, []).append((k, ans, cls))
                # new kinds first (the largest first); a kind asked about in an earlier round that still divides the
                # drafts comes after them, with another input of it
                order = sorted(kinds.items(), key=lambda v: (v[0] in person.sigs, -len(v[1]), v[1][0][0]))
                for sig, group in order:
                    if not person.can_ask():
                        break
                    person.sigs.add(sig)
                    k, ans, cls = group[0]
                    if json.dumps(allin[k], sort_keys=True, default=str) in person.decided:
                        continue
                    for i, d in enumerate(ok):
                        if d.rows[k].get("error"):
                            ans[i] = {q: None for q in qnames}
                    cl = list(dict.fromkeys(cls[0] + cls[1]))
                    dsp = Dispute("disagreement", allin[k], ans, cls, rnd, spec.name, similar=len(group),
                                  clause_text={c: spec.clauses[c].text for c in cl[:12] if c in spec.clauses})
                    r, expect, gap = person.ask(dsp)
                    asked["disagreements"] += 1
                    if r is not None and expect:
                        person_test(allin[k], expect, cl, r.note, rnd, k)
        live = [(row_of[t["id"]], t) for t in test_list if not t.get("dropped")]
        summary = {"round": rnd, "pool": len(pool), "tests": len(live), "drafts": []}
        if person is not None and person.review is not None:
            summary["person"] = dict(asked)
        fb = {d.index: [] for d in drafts}
        for d in drafts:
            if d.problems:
                fb[d.index].append("The module could not be used:\n- " + "\n- ".join(d.problems[:12]))
        for d in ok:
            failed = []
            for k, t in live:
                got = {q: _norm(r) for q, r in d.rows[k]["answers"].items()}
                if any(got.get(q) != e for q, e in t["expect"].items()):
                    failed.append((t, got, d.rows[k]))
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
                failed.sort(key=lambda f: f[0].get("source") != "person")
                lines = [f"{len(failed)} of {len(live)} tests derived from the specification fail (each names the clause "
                         "it checks):"]
                if any(t.get("source") == "person" for t, _, _ in failed):
                    lines.append("(A test whose why says \"a person decided\" is the answer of a person who read the "
                                 "specification for this input: it is right. Change your module so it gives that answer, "
                                 "and decide inputs like it the same way.)")
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
        ptests = [t for t in test_list if t.get("source") == "person" and not t.get("dropped")]
        if ptests:                                     # what a person decided so far, to every draft sent back: so a
            block = ["What a person who read the specification decided so far (each is a test; decide inputs like "
                     "these the same way):"]           # draft generalises from all of it, not only what it failed
            for t in ptests[-20:]:
                block.append(f"- input {_short(t['input'], 500)}\n  answer {t['expect']}"
                             + (f" ({t['why'][len('a person decided: '):][:200]})"
                                if t.get("why", "").startswith("a person decided: ") else ""))
            for d in drafts:
                if fb[d.index]:
                    fb[d.index].append("\n".join(block))
        for d in drafts:
            d.feedback = "\n\n".join(fb[d.index]) or None
            summary["drafts"].append({"draft": d.index, "generation": d.generation, "problems": d.problems[:5],
                                      "caught": list(d.caught), "feedback": bool(d.feedback),
                                      **({"notes": list(d.notes)} if d.notes else {})})
        summary["agreement"] = agree
        history.append(summary)
        if len(ok) == 2 and not any(d.feedback for d in drafts):
            if person is not None and person.gaps:
                history.append({"last_drafts": [{"source": d.source, "parts": d.parts, "problems": d.problems}
                                                for d in drafts]})
                return None, None, (f"not accepted: a person found {len(person.gaps)} input(s) the specification does "
                                    "not decide (record['person']['gaps']): amend the specification"), history, reviewed
            return drafts[0], drafts[1], f"accepted in round {rnd}", history, reviewed
        # a draft refused before it ran, round after round, is replaced by a fresh one (bounded, recorded): a stuck
        # draft must not block a partner that works. The fresh draft meets the same checks; nothing is relaxed.
        for d in drafts:
            d.stuck = 0 if d.rows is not None else d.stuck + 1
            if d.stuck >= stuck_after and used < fresh_drafts and rnd < rounds:
                used += 1
                d.generation += 1
                d.fresh, d.stuck = True, 0
                d.pitfalls = list(dict.fromkeys(d.pitfalls + d.problems[:6]))[-10:]
                summary.setdefault("replaced", []).append({"draft": d.index, "generation": d.generation,
                                                           "after_rounds_not_run": stuck_after,
                                                           "problems": d.problems[:5]})
    last = "; ".join(f"draft {d.index}: " + ", ".join(dict.fromkeys(d.caught)) for d in drafts if d.feedback)
    history.append({"last_drafts": [{"source": d.source, "parts": d.parts, "problems": d.problems} for d in drafts]})
    return None, None, f"not accepted after {rounds} round(s): {last}", history, reviewed


def _gens(writer):
    ws = list(writer) if isinstance(writer, (list, tuple)) else [writer]
    ws = [_writer_of(w) for w in ws]
    return ws if len(ws) == 2 else [ws[0], ws[0]]


def _settings(i, writers, generation=0):
    """Draft 0: temperature 0; draft 1: 0.7 with seed 1 — unless two writers are given (each at temperature 0). A fresh
    draft that replaces a stuck one: 0.7 with a seed of its own (100 × generation + draft + 1), so it is a new sample."""
    if generation:
        return 0.7, 100 * generation + i + 1
    if writers[0] is not writers[1]:
        return 0.0, None
    return (0.0, None) if i == 0 else (0.7, 1)


def _pitfalls(d):
    """What a fresh draft is told: the problems the drafts it replaces were refused for (never their code)."""
    if not d.pitfalls:
        return ""
    return ("\n\n## Mistakes earlier attempts at this module made (the harness refused them; avoid them)\n- "
            + "\n- ".join(p[:400] for p in d.pitfalls))


def compile_spec(spec: Spec, questions, inputs: Inputs, writer=None, *, rounds: int = 4, tests: int = 40,
                 examples=(), reference: Callable | None = None, mem_mb: int = 1024, item_s: int = 5,
                 fresh_drafts: int = 2, stuck_after: int = 2, review: Callable | None = None,
                 review_budget: int | None = 20, review_per_round: int | None = 5) -> Compiled:
    """Compile `spec` into catalog parts answering `questions` over `inputs` (see the module docs). `writer`: a
    solvi.generate Generator (or two, for two models), or a base URL (model openai/gpt-oss-120b). `examples`: labelled
    [(input, {question: answer})] that both drafts must match; `reference(input) → {question: answer}`, compared on the
    whole pool. `fresh_drafts`: how many times in all a draft that did not run (refused by the contract or the sandbox,
    or no module in the reply) for `stuck_after` rounds in a row is replaced by a fresh draft, written from the task
    again with a seed of its own (0: never). `review(dispute) → Ruling`: a person in the loop — asked about the drafts'
    disagreements and the disputed tests (at most `review_per_round` questions a round, `review_budget` in all; None:
    no limit); each answer becomes a test both drafts must pass (see `Dispute`, `Ruling`). Never raises for a bad
    draft: the result says whether it was accepted."""
    questions = list(questions)
    gens = _gens(writer)
    log = []
    person = _Person(review, review_budget, review_per_round, spec, questions) if review is not None else None
    test_list, bad, terr = _write_tests(spec, questions, inputs, gens[0], log, tests) if tests else ([], [], None)
    base = _task_text(spec, questions, inputs)

    def write(d, rnd):
        t, s = _settings(d.index, gens, d.generation)
        if d.fresh:
            prompt = (f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}{_pitfalls(d)}\n\nWrite the complete module now, in one "
                      "```python block.")
        elif d.source is None or rnd == 1:
            prompt = f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}\n\nWrite the complete module now, in one ```python block."
            if d.feedback:
                prompt += f"\n\n## Your previous answer\n\n{d.feedback}"
        else:
            prompt = (f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}\n\n## Your previous module (round {rnd - 1})\n\n"
                      f"```python\n{d.source.rstrip()}\n```\n\n## What the harness found\n\n{d.feedback}\n\n"
                      "Rewrite the complete module so that these problems go away; keep what already works. Answer with "
                      "the whole module in one ```python block.")
        text, err = _ask(gens[d.index], prompt, t, s, log, f"round {rnd} draft {d.index}"
                         + (f" (fresh {d.generation})" if d.generation else ""))
        code = code_of(text) if text else None
        if code is None:
            return d.source if d.source and rnd > 1 and not d.fresh else None, None, [
                f"no module in the reply ({err or 'no ```python block'})" + ("; write the module sooner, keep the "
                                                                             "reasoning short" if err and "cut off" in err
                                                                             else "")]
        return code, None, []

    a, b, reason, history, reviewed = _loop(spec, questions, inputs, gens, rounds=rounds,
                                            tests=(test_list, bad, terr), examples=list(examples), reference=reference,
                                            write=write, log=log, mem_mb=mem_mb, item_s=item_s, extra_check=None,
                                            need_tests=bool(tests), fresh_drafts=fresh_drafts, stuck_after=stuck_after,
                                            person=person)
    return _result(spec, questions, a, b, reason, history, reviewed, test_list, bad, terr, log, gens, None, person)


def _result(spec, questions, a, b, reason, history, reviewed, test_list, bad, terr, log, gens, changes, person=None):
    record = {"spec_hash": spec.hash, "writer": [g.model_id for g in gens], "writer_fp": [g.fingerprint() for g in gens],
              "rounds": history, "tests": test_list, "tests_invalid": bad, "tests_error": terr, "reviews": reviewed,
              "replaced": [{"round": h["round"], **r} for h in history for r in h.get("replaced", [])],
              "calls": log, "time": time.time()}
    if person is not None:
        record["person"] = person.record()
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
              examples=(), reference: Callable | None = None, mem_mb: int = 1024, item_s: int = 5,
              fresh_drafts: int = 2, stuck_after: int = 2, review: Callable | None = None,
              review_budget: int | None = 20, review_per_round: int | None = 5) -> Compiled:
    """Recompile an accepted compilation for a revised specification (`old.spec.revise(...)`): the writer returns only
    the parts it adds, replaces or removes — each citing a changed or added clause — and the rest stays byte-identical.
    Accepted by the same checks as compile_spec (two independent patches agree, tests written for the new
    specification; `review=` a person in the loop, as for compile_spec). The result's `changes`: the clauses' changes
    and the parts added / replaced / removed / kept."""
    if spec.changes is None:
        raise ValueError("recompile needs a revised specification: old.spec.revise(new_text)")
    if not old.accepted:
        raise Rejected(old.reason)
    if old.record.get("grouped"):
        raise ValueError("recompile does not patch a compilation made in groups: run compile_groups on the revised "
                         "specification")
    questions = list(old.questions)
    gens = _gens(writer)
    log = []
    ch = spec.changes
    if not (ch["changed"] or ch["added"] or ch["removed"]):
        c = Compiled(spec, questions, old.source, dict(old.parts), dict(old.not_normative), True,
                     "the specification did not change", {"spec_hash": spec.hash, "calls": []},
                     {"clauses": ch, "parts": {"added": [], "replaced": [], "removed": [], "kept": list(old.parts)}})
        return c
    person = _Person(review, review_budget, review_per_round, spec, questions) if review is not None else None
    test_list, bad, terr = _write_tests(spec, questions, inputs, gens[0], log, tests) if tests else ([], [], None)
    marks = {**{c: "changed" for c in ch["changed"]}, **{c: "added" for c in ch["added"]}}
    removed = [old.spec.clauses[c] for c in ch["removed"] if c in old.spec.clauses]
    base = _task_text(spec, questions, inputs, marks, removed)
    touched = set(ch["changed"]) | set(ch["added"])
    gone = set(ch["removed"])
    current = (f"## The current module (implements the previous version)\n\n```python\n{old.source.rstrip()}\n```")

    def write(d, rnd):
        t, s = _settings(d.index, gens, d.generation)
        prompt = f"{base}\n\n{CONTRACT}\n\n{SANDBOX_RULES}\n\n{current}\n\n{PATCH_CONTRACT}"
        if d.fresh:
            prompt += _pitfalls(d) + "\n\nWrite the change now, in one ```python block."
        elif d.patch is not None and rnd > 1:
            prompt += (f"\n\n## Your previous change (round {rnd - 1})\n\n```python\n{d.patch.rstrip()}\n```\n\n"
                       f"## What the harness found\n\n{d.feedback}\n\nWrite the change again, complete (it replaces your "
                       "previous change), so that these problems go away. One ```python block.")
        else:
            if d.feedback:
                prompt += f"\n\n## Your previous answer\n\n{d.feedback}"
            prompt += "\n\nWrite the change now, in one ```python block."
        text, err = _ask(gens[d.index], prompt, t, s, log, f"round {rnd} draft {d.index}"
                         + (f" (fresh {d.generation})" if d.generation else ""))
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
                why.append(f"{n} is added or replaced but cites no [changed] or [added] clause: add the one that makes "
                           f"this change to its clauses ({', '.join(sorted(touched))})")
        for n, c in rm.items():
            if n not in old.parts:
                why.append(f"REMOVE names {n!r}, which is not a part of the current module")
            cs = c if isinstance(c, list) else [c]
            if not set(cs) & (touched | gone):
                why.append(f"REMOVE[{n!r}] gives {c!r}: name the [changed], [added] or removed clause of the change that "
                           f"removes it ({', '.join(sorted(touched | gone))})")
        if why:
            return None, patch, why
        src, parts, nn = merge(old.source, old.parts, old.not_normative, patch, pparts, pnn, rm, spec)
        stale = [f"{n} still cites the removed clause {c}" for n, p in parts.items() for c in p.get("clauses") or []
                 if c in gone]
        return src, patch, stale

    a, b, reason, history, reviewed = _loop(spec, questions, inputs, gens, rounds=rounds,
                                            tests=(test_list, bad, terr), examples=list(examples), reference=reference,
                                            write=write, log=log, mem_mb=mem_mb, item_s=item_s, extra_check=None,
                                            need_tests=bool(tests), fresh_drafts=fresh_drafts, stuck_after=stuck_after,
                                            person=person)
    changes = {"clauses": ch}
    if a is not None:
        pparts, _, rm, _ = read_module(a.patch, spec, questions, patch=True)
        pparts = _effective(a.patch, pparts, old.parts)
        changes["parts"] = {"added": [n for n in pparts if n not in old.parts],
                            "replaced": [n for n in pparts if n in old.parts], "removed": list(rm),
                            "kept": [n for n in old.parts if n not in pparts and n not in rm]}
    c = _result(spec, questions, a, b, reason, history, reviewed, test_list, bad, terr, log, gens, changes, person)
    c.record["parent"] = {"spec_hash": old.spec.hash, "source_sha": hashlib.sha256(old.source.encode()).hexdigest()}
    return c


# ───────────────────────────────────────────────────────────── a large specification, compiled in groups
@dataclass
class Groups:
    """Which clauses govern which decisions. `by`: the input field that names the decision (a tool's name, the kind of
    a request); `groups`: {name: {"values": [values of `by` the group decides], "clauses": [clause ids]}}; `shared`:
    clauses that govern every decision (or state no rule: a preamble, a definition), put into every group's
    specification. A clause may be in several groups; a value of `by` in exactly one. `record`: how the grouping was
    made (the prompt and reply when `group_clauses` wrote it)."""
    by: str
    groups: dict
    shared: list = field(default_factory=list)
    record: dict = field(default_factory=dict)

    def problems(self, spec: Spec, values=None) -> list[str]:
        """Why this grouping cannot be used with `spec` ([] — it can): unknown clauses, a clause placed nowhere, a
        value in two groups or not a value of `by`, a group name that is not an identifier."""
        why, seen = [], {}
        if not isinstance(self.groups, dict) or not self.groups:
            return ["no groups"]
        for g, d in self.groups.items():
            if not isinstance(g, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,40}", g):
                why.append(f"group name {g!r} is not a short identifier")
            if not isinstance(d, dict) or not isinstance(d.get("values"), list) or not isinstance(d.get("clauses"), list):
                why.append(f"group {g}: needs a 'values' list and a 'clauses' list")
                continue
            if not d["values"]:
                why.append(f"group {g} decides no value of {self.by}")
            for v in d["values"]:
                if v in seen:
                    why.append(f"{self.by} = {v!r} is in groups {seen[v]} and {g}: each value goes to one group")
                seen.setdefault(v, g)
                if values is not None and v not in values:
                    why.append(f"group {g}: {v!r} is not a value of {self.by}")
            why += [f"group {g} names {c!r}, which is not a clause" for c in d["clauses"] if c not in spec.clauses]
        if not isinstance(self.shared, list):
            return why + ["shared must be a list of clause ids"]
        why += [f"shared names {c!r}, which is not a clause" for c in self.shared if c not in spec.clauses]
        placed = set(self.shared) | {c for d in self.groups.values() if isinstance(d, dict) for c in d.get("clauses") or []}
        miss = [c for c in spec.clauses if c not in placed]
        if miss:
            why.append("clauses in no group and not shared: " + ", ".join(miss))
        return why

    def of(self, value) -> str | None:
        """The group that decides this value of `by` (None: no group)."""
        return next((g for g, d in self.groups.items() if value in d["values"]), None)

    def to_dict(self):
        return {"by": self.by, "groups": self.groups, "shared": self.shared, "record": self.record}


GROUP_PROMPT = """# Group the clauses of a specification by the decision they govern

{spec}

## The decisions
Every input names its decision in the field `{by}`, one of: {values}.

Group the values of `{by}` into at most {k} groups of decisions governed by mostly the same clauses (one group per
value, or one for values alike), and put each clause into the groups whose decisions it governs. A clause that governs
every decision, or states no rule (a preamble, a definition every group uses), goes into "shared"; a clause that
governs several groups but not all goes into each of them. Every value of `{by}` is in exactly one group; a group
whose decisions only the shared clauses govern has no clauses of its own. Group names are short identifiers.
Answer with one ```json block:
{{"shared": ["c1", ...], "groups": {{"group_name": {{"values": [...], "clauses": ["c7", ...]}}, ...}}}}
"""


def _by_values(inputs, by, values):
    if values is not None:
        return list(values)
    d = inputs.fields.get(by)
    if isinstance(d, list):
        return list(d)
    out = list(dict.fromkeys(x.get(by) for x in inputs.samples if by in x))
    if not out:
        raise ValueError(f"no values of {by!r} to group by: declare the field's values, give samples, or values=")
    return out


def group_clauses(spec: Spec, by: str, values, writer=None, *, max_groups: int = 8) -> Groups:
    """One LLM call (temperature 0) puts the clauses of `spec` into groups by the value of the input field `by` they
    govern → a `Groups`. A reply that cannot be used is asked once more with why; still unusable → ValueError. The
    grouping only decides what each sub-compilation reads: every group is accepted on its own, and the assembled whole
    is checked again (see compile_groups) — a clause left out of a group shows as a disagreement or a failed test, or
    not at all if no input exercises it, so read `groups.groups` before you rely on it."""
    gen = _gens(writer)[0]
    values = list(values)
    prompt = GROUP_PROMPT.format(spec=spec.render(), by=by, values=", ".join(json.dumps(v, ensure_ascii=False)
                                                                             for v in values), k=max_groups)
    log, err = [], None
    for attempt in range(2):
        p = prompt if attempt == 0 else prompt + f"\n\nYour previous answer could not be used:\n- {err}\nAnswer again."
        t0 = time.time()
        try:
            g = gen.generate([{"role": "user", "content": p}], parse=_json_block, temperature=0.0)
            raw, meta, e = g.value, g.meta, None
        except Exception as ex:  # noqa: BLE001 — an unusable reply is asked once more
            raw, meta, e = {}, {}, f"{type(ex).__name__}: {ex}"[:300]
        log.append({"what": "grouping" if attempt == 0 else "grouping again", "prompt": p,
                    "reply": (meta or {}).get("text"), "error": e, "seconds": round(time.time() - t0, 1),
                    "model": getattr(gen, "model_id", None), "usage": (meta or {}).get("usage")})
        raw = raw if isinstance(raw, dict) else {}
        grp = Groups(by, raw.get("groups") or {}, raw.get("shared") or [], {"calls": log})
        why = grp.problems(spec, values) if not e else [e]
        if len(grp.groups) > max_groups:
            why.append(f"{len(grp.groups)} groups, at most {max_groups}")
        missing = [v for v in values if grp.of(v) is None] if not e and isinstance(grp.groups, dict) else []
        if missing:
            why.append(f"values in no group: {', '.join(map(str, missing))}")
        if not why:
            return grp
        err = "\n- ".join(why[:12])
    raise ValueError(f"the grouping could not be used: {err}")


class _Scoped(Inputs):
    """The inputs of one group: the samples and drawn inputs whose `by` value the group decides."""

    def __init__(self, base: Inputs, by, values):
        fields = dict(base.fields)
        if isinstance(fields.get(by), list):
            fields[by] = [v for v in fields[by] if v in values]
        note = (f"\n\nThis part of the specification decides only the inputs whose `{by}` is one of: "
                + ", ".join(json.dumps(v, ensure_ascii=False) for v in values)
                + ". Other inputs are decided elsewhere: your module is never asked about them.")
        super().__init__(fields, describe=base.text() + note, samples=[x for x in base.samples if x.get(by) in values],
                         generate=base.generate, n=base.n, seed=base.seed)
        self.by, self.values = by, list(values)

    def validate(self, x):
        why = super().validate(x)
        if isinstance(x, dict) and x.get(self.by) not in self.values:
            why.append(f"{self.by} = {x.get(self.by)!r} is not decided by this group")
        return why

    def pool(self, numbers=()):
        try:
            out = super().pool(numbers)
        except ValueError:
            out = []
        out = [x for x in out if x.get(self.by) in self.values]
        if not out:
            raise ValueError(f"no input of the pool has {self.by} in {self.values}")
        return out


def _bound(tree):
    """Names a module binds at its top level (a dotted `import a.b` binds `a`, left as it is)."""
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.Import):
            out |= {a.asname for a in node.names if a.asname} | {a.name for a in node.names
                                                                if not a.asname and "." not in a.name}
        elif isinstance(node, ast.ImportFrom):
            out |= {a.asname or a.name for a in node.names if a.name != "*"}
    return out


class _Prefix(ast.NodeTransformer):
    """Every use of a module's own top-level names gets a prefix (names, arguments, keyword arguments of calls to its
    own functions, imports by alias); attributes, strings and method names stay."""

    def __init__(self, names, pre):
        self.names, self.pre = names, pre

    def _p(self, n):
        return self.pre + n if n in self.names else n

    def visit_Name(self, node):
        node.id = self._p(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._p(node.arg)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._p(node.name)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.name = self._p(node.name)
        for i, st in enumerate(node.body):
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = st.name
                node.body[i] = self.visit(st)
                node.body[i].name = name
            else:
                node.body[i] = self.visit(st)
        for f in ("bases", "keywords", "decorator_list"):
            setattr(node, f, [self.visit(x) for x in getattr(node, f)])
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.names:
            for kw in node.keywords:
                if kw.arg is not None:
                    kw.arg = self._p(kw.arg)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        for a in node.names:
            if a.asname:
                a.asname = self._p(a.asname)
            elif "." not in a.name and a.name in self.names:
                a.asname = self.pre + a.name
        return node

    def visit_ImportFrom(self, node):
        for a in node.names:
            if a.name != "*" and (a.asname or a.name) in self.names:
                a.asname = self.pre + (a.asname or a.name)
        return node


def _group_module(source, parts, pre, by, given, questions):
    """One group's module for the assembly → (statements, parts, rule of each question). Its top-level names get the
    prefix; each part is gated — outside the group's values a check is True and a fact or the group's rule is None,
    before anything else runs — and loses its type annotations (an out-of-scope None must not be refused); the
    group's rules become facts the assembled rule routes to."""
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in META for t in n.targets))
        or (isinstance(n, ast.Expr) and _is_meta_call(ast.unparse(n))))]
    names = (_bound(tree) | set(parts)) - set(given or ()) - {"Fail"}
    tree = _Prefix(names, pre).visit(tree)
    out_parts, rules = {}, {}
    for name, p in parts.items():
        new = pre + name if name in names else name
        p = dict(p)
        if p["kind"] == "rule":
            rules[p["question"]] = new
            p = {"kind": "fn", "clauses": list(p.get("clauses") or []), "rule_of": p["question"]}
        out_parts[new] = p
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in out_parts:
            a = node.args
            for x in a.posonlyargs + a.args + a.kwonlyargs:
                x.annotation = None
            node.returns = None
            neutral = "True" if out_parts[node.name]["kind"] == "check" else "None"
            if by not in [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]:
                # the harness gives `by`; a call from the group's own code (a part used as a helper) does not, and
                # gets the default: such a call is never gated
                a.args.append(ast.arg(arg=by))
                a.defaults.append(ast.Name(id=f"{pre}DIRECT", ctx=ast.Load()))
                gate = f"if {by} is not {pre}DIRECT and {by} not in {pre}SCOPE:\n    return {neutral}"
            else:
                gate = f"if {by} not in {pre}SCOPE:\n    return {neutral}"
            node.body.insert(0, ast.parse(gate).body[0])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), out_parts, rules


def assemble(by: str, members: list, questions, given=None, refused=()) -> tuple[str, dict]:
    """The groups' modules as one module → (source, parts). `members`: [(group name, values, source, parts)] of
    accepted groups; `refused`: [(group name, values, reason)] — inputs with those values abstain with the reason. Each
    question's rule routes by `by` to the rule of the group that decides the value; a value no group decides abstains.
    A part that did not read `by` gets it as a last argument with a default, so the group's own calls of it as a
    function (a rule calling a check) still work and are not gated."""
    chunks, parts, rules = [], {}, {}
    for name, values, source, gparts in members:
        pre = f"{name}__"
        chunks.append(f"{pre}SCOPE = frozenset({list(values)!r})\n{pre}DIRECT = object()")
        text, gp, gr = _group_module(source, gparts, pre, by, given, questions)
        chunks.append(text)
        parts.update(gp)
        rules[name] = gr
    for name, values, _ in refused:
        chunks.append(f"{name}__SCOPE = frozenset({list(values)!r})")
    for q in questions:
        args = [rules[n][q.name] for n, *_ in members if q.name in rules[n]]
        lines = [f"def decide_{q.name}({', '.join([by] + args)}):"]
        for name, *_ in members:
            if q.name in rules[name]:
                lines.append(f"    if {by} in {name}__SCOPE:\n        return {rules[name][q.name]}")
        for name, _, why in refused:
            lines.append(f"    if {by} in {name}__SCOPE:\n        raise ValueError({('the clauses of group ' + name + ' were not accepted: ' + why)[:300]!r})")
        lines.append(f"    raise ValueError('no group of the specification decides {by} = ' + repr({by}))")
        chunks.append("\n".join(lines))
        parts[f"decide_{q.name}"] = {"kind": "rule", "question": q.name, "clauses": []}
    src = "\n\n\n".join(c.rstrip("\n") for c in chunks) + "\n\n\n" + _literal("PARTS", parts)
    return src, parts


def compile_groups(spec: Spec, questions, inputs: Inputs, writer=None, *, by: str, groups: Groups | dict | None = None,
                   values=None, max_groups: int = 8, partial: bool = False, **kw) -> Compiled:
    """Compile a large specification in groups: the clauses that govern each kind of decision (the values of the input
    field `by`, such as a tool's name) are compiled on their own, with the shared clauses, by `compile_spec` and its
    acceptance — two independent drafts agree on the group's inputs, tests derived from the group's clauses pass —
    then assembled into one module whose rule routes each input to its group, and the whole is checked again:
    both assembled modules (the groups' accepted drafts, and their independent partners) agree on the full pool, give
    every input of an accepted group an answer, decide each group's inputs as the group's own draft did, and pass every
    group's tests. Nothing is rewritten at this step: a failure is a refusal, with the reason.

    groups: a `Groups` (or {"shared": [...], "groups": {name: {"values", "clauses"}}}); None — `group_clauses` writes it
    with the writer. values: the values of `by` (default: the field's declared values, else those of the samples).
    partial: False — the whole is accepted only when every group is; True — the accepted groups are assembled, and an
    input of a refused group abstains with the reason (the result says which groups). kw: compile_spec's options
    (rounds, tests, examples, reference, mem_mb, item_s, fresh_drafts, stuck_after, review, review_budget,
    review_per_round — a person in the loop is asked per group, each group with its own budget).
    The result is an ordinary `Compiled` (system(), to_guard, Versions); `c.groups` holds each group's own compilation
    and `c.record["groups"]` what each accepted."""
    questions = list(questions)
    vals = _by_values(inputs, by, values)
    if groups is None:
        groups = group_clauses(spec, by, vals, writer, max_groups=max_groups)
    elif isinstance(groups, dict):
        groups = Groups(by, groups.get("groups") or {}, groups.get("shared") or [], {"given": True})
    why = groups.problems(spec)
    if why:
        raise ValueError("the grouping cannot be used: " + "; ".join(why[:8]))
    plan = {g: {"values": [v for v in d["values"]], "clauses": list(d["clauses"])} for g, d in groups.groups.items()}
    rest = [v for v in vals if groups.of(v) is None]
    if rest:                                          # values no group names: decided by the shared clauses alone
        plan["other" if "other" not in plan else "other_values"] = {"values": rest, "clauses": []}
    subs, rec = {}, {}
    examples = list(kw.pop("examples", ()))
    for g, d in plan.items():
        ids = list(dict.fromkeys(list(groups.shared) + d["clauses"]))
        sub_inputs = _Scoped(inputs, by, d["values"])
        r = {"values": d["values"], "clauses": d["clauses"], "shared": len(groups.shared)}
        if not ids:
            subs[g] = Compiled(spec, questions, "", {}, {}, False, "the group has no clauses", {})
        else:
            sub_spec = spec.subset(ids, name=f"{spec.name} — {g}")
            try:
                sub_inputs.pool()
            except ValueError as e:
                subs[g] = Compiled(sub_spec, questions, "", {}, {}, False, f"not compiled: {e}", {})
            else:
                subs[g] = compile_spec(sub_spec, questions, sub_inputs, writer,
                                       examples=[(x, a) for x, a in examples if x.get(by) in d["values"]], **kw)
        c = subs[g]
        r.update({"accepted": c.accepted, "reason": c.reason, "rounds": len([h for h in c.record.get("rounds", [])
                                                                             if "round" in h]),
                  "replaced": len(c.record.get("replaced", [])), "tests": len(c.record.get("tests", [])),
                  "parts": len(c.parts)})
        if c.record.get("person"):
            p = c.record["person"]
            r["person"] = {k: p[k] for k in ("asked", "decisions", "neither", "skipped", "by_kind")}
            r["person"]["gaps"] = len(p["gaps"])
        rec[g] = r
    record = {"grouped": True, "by": by, "grouping": groups.to_dict(), "groups": rec,
              "spec_hash": spec.hash, "calls": groups.record.get("calls", [])}
    ok = [g for g in plan if subs[g].accepted]
    refused = [g for g in plan if not subs[g].accepted]
    if not ok or (refused and not partial):
        reason = "not accepted: groups not accepted — " + "; ".join(f"{g}: {subs[g].reason}" for g in refused)
        return Compiled(spec, questions, "", {}, {}, False, reason, record, None, subs)
    given = given_names(inputs) or set()
    a_members = [(g, plan[g]["values"], subs[g].source, subs[g].parts) for g in ok]
    b_members = []
    for g in ok:
        other = subs[g].record.get("other_source")
        bparts, _, _, _ = read_module(other, subs[g].spec, questions)
        b_members.append((g, plan[g]["values"], other, bparts))
    ref = [(g, plan[g]["values"], subs[g].reason) for g in refused]
    src_a, parts_a = assemble(by, a_members, questions, given, ref)
    src_b, parts_b = assemble(by, b_members, questions, given, ref)
    nn = {}
    cited = {c for p in parts_a.values() for c in p.get("clauses") or []}
    for g in ok:
        for c, why_nn in subs[g].not_normative.items():
            if c not in cited:
                nn.setdefault(c, why_nn)
    for g in refused:
        for c in plan[g]["clauses"]:
            if c not in cited:
                nn.setdefault(c, f"not compiled: group {g} was not accepted, its inputs abstain")
    whole = _check_whole(spec, questions, inputs, by, plan, subs, ok, refused, src_a, parts_a, src_b, parts_b, nn,
                         given, kw.get("mem_mb", 1024), kw.get("item_s", 5))
    record["whole"] = whole
    record["other_source"] = src_b
    if whole["problems"]:
        reason = "not accepted: the assembled whole fails its check — " + "; ".join(whole["problems"][:5])
        return Compiled(spec, questions, "", {}, {}, False, reason, record, None, subs)
    reason = f"accepted: {len(ok)} group(s) assembled, the whole agrees on {whole['inputs']} inputs"
    if refused:
        reason += ("; not accepted, so their inputs abstain: " + ", ".join(refused))
        record["refused_groups"] = {g: {"values": plan[g]["values"], "reason": subs[g].reason} for g in refused}
    return Compiled(spec, questions, src_a, parts_a, nn, True, reason, record, None, subs)


def _check_whole(spec, questions, inputs, by, plan, subs, ok, refused, src_a, parts_a, src_b, parts_b, nn, given, mem_mb,
                 item_s):
    """The checks of the assembled whole (see compile_groups) → {"inputs", "problems", per-check counts}."""
    problems = []
    for src, parts, side in ((src_a, parts_a, "A"), (src_b, parts_b, "B")):
        why = sandbox.check(src)
        why += unknown_reads(src, {n: p for n, p in parts.items()}, given)
        miss = coverage(spec, parts, nn) if side == "A" else []
        if miss:
            why.append("clauses no group covers: " + ", ".join(miss))
        problems += [f"assembled {side}: {w}" for w in why]
    if problems:
        return {"inputs": 0, "problems": problems}
    nums = numbers_in(spec.text) + numbers_in(src_a) + numbers_in(src_b)
    try:
        pool = inputs.pool(nums)
    except ValueError:
        pool = []
    tests = [(g, t) for g in ok for t in subs[g].record.get("tests", []) if not t.get("dropped")]
    allin = pool + [t["input"] for _, t in tests]
    rows = {}
    for side, src, parts in (("A", src_a, parts_a), ("B", src_b, parts_b)):
        r, err = _run_draft(_Draft(0, source=src, parts=parts), questions, allin, mem_mb, item_s)
        if r is None:
            return {"inputs": len(allin), "problems": [f"assembled {side} did not run: {err}"]}
        rows[side] = r
    ans = {s: [{q: _norm(v) for q, v in row["answers"].items()} for row in rows[s]] for s in rows}
    out = {"inputs": len(allin), "pool": len(pool), "tests": len(tests), "disagree": 0, "unanswered": 0,
           "not_abstaining": 0, "unfaithful": 0, "tests_failed": 0, "no_group": 0}
    for k, x in enumerate(allin):
        g = next((g for g in plan if x.get(by) in plan[g]["values"]), None)
        a, b = ans["A"][k], ans["B"][k]
        if g is None or g in refused:
            if g is None:
                out["no_group"] += 1
            if any(v is not None for v in list(a.values()) + list(b.values())):
                out["not_abstaining"] += 1
            continue
        if any(v is None for v in list(a.values()) + list(b.values())) or not a or not b:
            out["unanswered"] += 1
        elif a != b:
            out["disagree"] += 1
    # each group's own drafts on the same inputs: the assembly must not change a decision
    for g in ok:
        idx = [k for k, x in enumerate(allin) if x.get(by) in plan[g]["values"]]
        if not idx:
            continue
        sub_in = [allin[k] for k in idx]
        for side, src in (("A", subs[g].source), ("B", subs[g].record.get("other_source"))):
            gparts, _, _, _ = read_module(src, subs[g].spec, questions)
            r, err = _run_draft(_Draft(0, source=src, parts=gparts), questions, sub_in, mem_mb, item_s)
            if r is None:
                problems.append(f"group {g} ({side}) did not run on the full pool: {err}")
                continue
            for k, row in zip(idx, r):
                if {q: _norm(v) for q, v in row["answers"].items()} != ans[side][k]:
                    out["unfaithful"] += 1
    for j, (g, t) in enumerate(tests):
        k = len(pool) + j
        for s in ("A", "B"):
            if any(ans[s][k].get(q) != e for q, e in t["expect"].items()):
                out["tests_failed"] += 1
    for key, text in (("disagree", "the two assembled modules disagree on {} inputs"),
                      ("unanswered", "{} inputs of accepted groups get no answer"),
                      ("not_abstaining", "{} inputs of refused groups or of no group are answered"),
                      ("unfaithful", "{} decisions differ from the group's own draft"),
                      ("tests_failed", "{} group tests fail on the assembled modules")):
        if out[key]:
            problems.append(text.format(out[key]))
    out["problems"] = problems
    return out


# ───────────────────────────────────────────────────────────── what a change does to decisions
@dataclass
class DecisionDiff:
    """The decisions a new compilation changes: `changed` = [{"id", "question", "old", "new", "causes": [{"step",
    "part", "clauses", "why"}]}]; `report` is solvi.diff's own report."""
    changed: list
    report: Any
    total: int

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
def to_guard(compiled: Compiled, guard, tools=None, *, question: str | None = None, on_fail: str = "deny",
             allow=None, name: str | None = None) -> list:
    """Register the compiled policy with a `solvi.agents.Guard` → the names of the policies added. The policies read
    the compiled catalog's inputs (they must be facts the guard gives: tool_name, tool_arguments, conversation, ... or
    your declared facts) and run the compiled System on them.

    allow=None: each compiled hard check that names `question` becomes a policy, which fails with the check's reasons
    when the check is false. allow="yes" (the answer that lets a call through): one policy (`name`, default
    `compiled_<question>`) that asks the compiled question and fails when the answer is anything else — with the
    clauses of the parts that decided (the false hard checks, else the question's rule) as the reasons — or when the
    compiled policy cannot answer (it abstains: an input it cannot read, or a group that was not accepted).
    `question`: the compiled question (default: the only one)."""
    from .system import System
    cat, qs = compiled.catalog()
    q = question or (qs[0].name if len(qs) == 1 else None)
    if q is None:
        raise ValueError("to_guard: name the compiled question (question=...)")
    system = System(cat, qs)
    given = sorted({i for p in cat.parts.values() for i in p.inputs} - set(cat.parts))
    import inspect
    sig = inspect.Signature([inspect.Parameter(g, inspect.Parameter.KEYWORD_ONLY) for g in given])
    texts = compiled.spec.clauses
    if allow is not None:
        hard = [n for n, p in compiled.parts.items() if p.get("kind") == "check" and p.get("hard")
                and q in (p.get("then") or {})]
        rules = [n for n, p in compiled.parts.items() if (p.get("kind") == "rule" and p.get("question") == q)
                 or p.get("rule_of") == q]

        def policy(**facts):
            res = system.ask({k: facts[k] for k in given if k in facts}, [q])
            a = res[q]
            if a.status == "abstain":
                return Fail(f"the compiled policy cannot decide this call: {a.why}"[:300])
            if _norm(a.answer) == _norm(allow):
                return True
            ran = {compiled.part_of(r.name) or r.name for r in res.trace.records}
            failed = [n for n in hard if n in ran and res.values.get(n) is not None and not res.values.get(n)]
            by = failed if a.status == "forced" and failed else [n for n in rules if n in ran]
            cl = list(dict.fromkeys(c for n in by for c in compiled.parts[n].get("clauses") or [] if c in texts))
            if not cl:
                return Fail(f"{q} = {a.answer}: {a.why}"[:300])
            return Fail("; ".join(f"[{c}] {texts[c].text}" for c in cl[:6])[:600])
        policy.__signature__ = sig
        policy.__name__ = name or f"compiled_{q}"
        policy.__doc__ = f"The compiled {compiled.spec.name} must answer {q} = {allow}."
        policy.show_fail_reasons = True               # the guard's reason names the clauses that decided this call
        guard.policy(tools, on_fail=on_fail)(policy)
        return [policy.__name__]
    fp = compiled.fingerprint
    names = []
    for pname, p in compiled.parts.items():
        if p.get("kind") != "check" or not p.get("hard") or q not in (p.get("then") or {}):
            continue

        def policy(_name=pname, _fp=fp, **facts):
            res = system.ask({k: facts[k] for k in given if k in facts})
            v = res.values.get(_name)
            if v is None:
                return Fail(f"{_name} could not be evaluated")
            return v if not v else True
        policy.__signature__ = sig
        policy.__name__ = pname
        policy.__doc__ = "; ".join(texts[c].text for c in p.get("clauses") or [] if c in texts)[:300]
        guard.policy(tools, on_fail=on_fail)(policy)
        names.append(pname)
    return names


__all__ = ["Clause", "Compiled", "DecisionDiff", "Dispute", "Groups", "Inputs", "Rejected", "Ruling", "Spec", "Versions",
           "assemble", "compile_groups", "compile_spec", "coverage", "decision_diff", "group_clauses", "merge",
           "numbers_in", "read_module", "recompile", "reference_reviewer", "split_clauses", "to_guard"]
