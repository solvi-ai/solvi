"""`solvi check`: catalog lint — mistakes a catalog can carry for a long time before a decision shows them.

    from solvi.check import lint
    rep = lint(system)                 # a System (or a Catalog: the question checks are skipped)
    print(rep); rep.ok                 # errors fail; warnings fail with strict=True (solvi check --strict)

    solvi check myapp.decisions:system [--strict] [--json]     # exit 0: no errors; 1: errors; 2: usage errors

Errors (a decision is, or can be, wrong or impossible):
  then_not_in_flow      a hard check sets `then=` for a question whose flow never runs it (the question's rule does not
                        read it, through any fact, and the question does not list it in `checkpoints`): when the check
                        fails, the question is answered as if it had passed
  then_unknown / then_bad_answer   `then=` names no question / an answer that is not one of the question's options
  cycle                 facts that need each other: none of them can be computed unless one is given (facts derived
                        from each other, each with a producer outside the loop, are a cycle for the deterministic
                        strategist only; with System(strategist=...) they are the note `mutual_producers`)
  unanswerable          a question no input can answer (its flow needs a fact nothing can compute; a span / rank /
                        estimate question without a rule — a learned head cannot answer it; a missing checkpoint)
  type_conflict         a producer's return type that its consumer cannot read; a System(inputs=...) field that a typed
                        reader cannot read
  constraint_never_holds / constraints_conflict   constraints between answers that no combination of answers satisfies
                        (brute force over the answers' finite domains)
  constraint_unknown_question      a constraint that reads a name that is not a question: it never applies
  validate_reads_unknown           a producer's `validate` requires an argument no producer of its fact takes as an input:
                        it cannot run, so every output of that producer is rejected (a System refuses such a catalog)
  hard_check_untyped    a hard check without `-> bool` with a `return` that is plainly not True or False (`return 0`,
                        `return None`, a bare `return`, a string): on that path the check is rejected and every
                        question it governs abstains — return a bool and declare it `-> bool`
  rule_returns_non_option   a rule with `return <literal>` (also in `a if c else b`) that is not one of its question's
                        options (`return "aprove"`): on that path the question abstains
Warnings:
  unused_part           a part no question's flow uses (a question without a rule, fit or `uses` uses everything computable:
                        those are the candidate features of the head it will be fitted with)
  then_on_soft_check    `then=` on a check that is not hard: it is ignored
  question_as_fact      a rule reads a question's name: answers are not facts, so it must be given in the input
  reader_types_differ   typed readers of a given fact, or alternative producers of a fact, that no value can satisfy both
  dead_option           an option the constraints rule out whatever the other answers are
  constraint_raises     a constraint that raises on some combination of answers (it counts as broken)
  unused_rule           a rule registered for a question that is not among the System's questions: it never runs
  input_not_declared    with System(inputs=Model): a part or a rule reads a name that is neither computed by a part nor
                        a field of the model (`amout` for `amount`) — it can only arrive as an extra key of the input;
                        when the model forbids extra keys it can never arrive
  uses_unknown          with System(inputs=Model): a question's `uses` hint names something that is neither a part nor
                        a field of the model. (Without an input model a `uses` name that is not a part is taken for a
                        given fact — a learned head may read given facts directly — so a typo there cannot be told;
                        a misspelled `checkpoints` entry is an error: `unanswerable`)
  silent_default        in a function that reads the input (a given fact): `x or <literal>` or `.get(k, <literal>)` on
                        that input (`amount or 0`, `order.get("total", 0)`; not a lookup in a constant table or a
                        default on a computed value), which turns a missing, empty or null input into a value nobody
                        gave — say so explicitly (check
                        for None and abstain, or declare the default in System(inputs=...)); `# solvi: ok` on the line
                        accepts it
Notes (never fail):
  mutual_producers      facts derived from each other (net from gross and gross from net), each with a producer
                        outside the loop: the system's strategist plans around it
  no_rule               a question without a rule answers only after fit / fit_fast (it abstains until then)

The flows are planned with every given fact present (`solvi.strategist.given_facts`), as `solvi serve` does, by the
system's own strategist (System(strategist=)) — the deterministic one when none is set."""
from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from dataclasses import asdict, dataclass, field, replace

LEVELS = ("error", "warning", "note")


@dataclass
class Finding:
    level: str                  # error | warning | note
    code: str                   # see the module docs
    where: str                  # a part, a question, a constraint or file:line
    message: str


@dataclass
class Report:
    findings: list = field(default_factory=list)
    strict: bool = False

    def add(self, level, code, where, message):
        self.findings.append(Finding(level, code, where, message))

    def of(self, level):
        return [f for f in self.findings if f.level == level]

    @property
    def errors(self):
        return self.of("error")

    @property
    def warnings(self):
        return self.of("warning")

    @property
    def ok(self):
        """No errors (and, with strict, no warnings)."""
        return not self.errors and not (self.strict and self.warnings)

    def codes(self, level=None):
        return [f.code for f in self.findings if level is None or f.level == level]

    def to_dict(self):
        return {"ok": self.ok, "errors": len(self.errors), "warnings": len(self.warnings),
                "findings": [asdict(f) for f in self.findings]}

    def __str__(self):
        if not self.findings:
            return "catalog check: no problems found"
        lines = []
        for lv in LEVELS:
            for f in self.of(lv):
                lines.append(f"{lv:7s} {f.code:22s} {f.where}: {f.message}")
        lines.append(f"{len(self.errors)} error(s), {len(self.warnings)} warning(s), {len(self.of('note'))} note(s)")
        return "\n".join(lines)


def lint(obj, strict=False, max_combos=100_000):
    """A System (or a Catalog) → Report. `max_combos`: the largest number of answer combinations tried per group of
    constraints that share questions."""
    from .strategist import given_facts
    if hasattr(obj, "tools") and hasattr(obj, "system") and not hasattr(obj, "questions"):   # a solvi.agents.Guard
        rep = Report(strict=strict)
        for name, t in obj.tools.items():
            if t.model is None:
                rep.add("note", "no_schema", f"{name}", "the tool has no argument schema yet (adopted from an MCP server)")
                continue
            for f in lint(obj.system(name), strict, max_combos).findings:
                rep.findings.append(replace(f, where=f"{name}: {f.where}"))
        return rep
    system = obj if hasattr(obj, "catalog") and hasattr(obj, "questions") else None
    cat = system.catalog if system is not None else obj
    questions = dict(system.questions) if system is not None else {}
    heads = system.heads if system is not None else {}
    rep = Report(strict=strict)
    given = given_facts(cat, questions.values())
    if system is not None:                            # planned as the system's ask plans (its strategist, if any)
        planner = system._plan
    else:
        from .strategist import plan

        def planner(qs, keys):
            return plan(cat, qs, keys, heads)
    in_cycle = _cycles(cat, rep, getattr(system, "strategist", None))
    flows, counted = _questions(cat, questions, heads, given, rep, planner)
    _hard_checks(cat, questions, flows, rep)
    if questions:
        used = set().union(*counted.values()) if counted else set()
        for name, p in cat.parts.items():
            if name not in used and name not in in_cycle:
                rep.add("warning", "unused_part", name, f"no question's flow uses this {p.kind}")
        _rules(cat, questions, rep)
        _names(cat, system, questions, rep)
    _types(cat, system, given, rep)
    for fact, name, lost in cat.unreadable_validates():
        rep.add("error", "validate_reads_unknown", f"{fact} ({name})",
                f"validate reads {', '.join(lost)}, which no producer of {fact} takes as an input: it cannot run, so every "
                f"output of {name} is rejected")
    if questions:
        _constraints(cat, questions, rep, max_combos)
    elif cat.constraints:
        rep.add("note", "constraints_not_checked", ", ".join(cat.constraints),
                "constraints are checked against a System's questions: lint the System")
    _silent_defaults(cat, given, rep)
    return rep


# --- cycles
def _cycles(cat, rep, strategist=None):
    """Strongly connected components of the fact graph (a part → the facts it reads) with more than one fact, or a part
    that reads its own fact → the facts in them. Facts derivable from each other (net from gross, gross from net) where
    each also has a producer that reads nothing inside the loop are a cycle only for the deterministic strategist, which
    needs the inputs of every producer: with System(strategist=...) they are a note."""
    graph = {n: [x for x in p.inputs if x in cat.parts] for n, p in cat.parts.items()}
    index, low, stack, on, out, counter = {}, {}, [], set(), [], [0]

    def visit(v):                                     # Tarjan, iteratively (deep catalogs must not hit the recursion limit)
        work = [(v, iter(graph[v]))]
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        while work:
            node, it = work[-1]
            nxt = next(it, None)
            if nxt is not None:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on.add(nxt)
                    work.append((nxt, iter(graph[nxt])))
                elif nxt in on:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                out.append(comp)
    for v in graph:
        if v not in index:
            visit(v)
    found = set()
    for comp in out:
        if len(comp) > 1 or comp[0] in graph[comp[0]]:
            comp = sorted(comp)
            found.update(comp)
            where = " → ".join(comp + [comp[0]])
            if not _breakable(cat, set(comp)):
                rep.add("error", "cycle", where,
                        "these facts need each other: none of them can be computed unless one of them is given")
            elif strategist is None:
                rep.add("error", "cycle", where,
                        "these facts are derived from each other; each has a producer outside the loop, but the "
                        "deterministic strategist needs the inputs of every producer of a fact, so none of them can be "
                        "computed — plan with System(..., strategist=solvi.strategy.ModelStrategist()), which uses the "
                        "producers whose inputs are there")
            else:
                rep.add("note", "mutual_producers", where,
                        "these facts are derived from each other; each has a producer outside the loop, which the "
                        "system's strategist uses when its inputs are given")
    return found


def _breakable(cat, comp):
    """Can every fact of a loop be computed without going round it: some producer of it reads only facts outside the loop
    or facts of the loop already settled that way?"""
    from .strategy import alternatives
    done, changed = set(), True
    while changed:
        changed = False
        for f in comp - done:
            if any(all(x not in comp or x in done for x in a.inputs) for a in alternatives(cat.parts[f])):
                done.add(f)
                changed = True
    return done == comp


# --- questions
def _questions(cat, questions, heads, given, rep, planner):
    """Plan each question with every given fact present → ({question: facts in its flow}, {question: the facts that count
    as used}); unanswerable questions, rules reading question names, questions without a rule. planner(questions, given
    facts) → Flow: the system's own (System._plan)."""
    from .core import PRIMITIVES
    from .strategist import PlanError
    flows, counted = {}, {}
    for name, q in questions.items():
        try:
            flow = planner([q], given)
        except PlanError as e:
            rep.add("error", "unanswerable", name, str(e))
            flows[name] = counted[name] = set()
            continue
        flows[name] = set(flow.per_question.get(name, ()))
        counted[name] = flows[name]
        if name not in cat.rules and name not in heads and not q.uses and q.answer is not None \
                and q.answer.kind in PRIMITIVES:
            # no rule, and no head can answer it: its flow (everything computable) runs for nothing — only the
            # checkpoints count as used. A question a head can answer uses everything computable: its candidate features
            counted[name] = set(planner([replace(q, uses=list(q.checkpoints))], given).per_question.get(name, ())) \
                if q.checkpoints else set()
        missing = flow.unresolved.get(name)
        if missing:
            rep.add("error", "unanswerable", name, "no input can answer it: nothing can compute " + ", ".join(missing))
        rule = cat.rules.get(name)
        if rule is not None:
            asked = [x for x in rule.inputs if x in questions and x not in cat.parts]
            if asked:
                rep.add("warning", "question_as_fact", name,
                        f"its rule reads {', '.join(asked)}, which {'is a question' if len(asked) == 1 else 'are questions'}: "
                        "answers are not facts, so the input must give it")
        elif name not in heads:
            if q.answer is not None and q.answer.kind in PRIMITIVES:
                rep.add("error", "unanswerable", name, f"a {q.answer.kind} question without a rule: a learned head cannot "
                                                       "answer it (give it a rule or a model decision)")
            else:
                rep.add("note", "no_rule", name, "no rule: it abstains until an answer head is fitted (fit / fit_fast)"
                        + ("" if q.uses else "; without `uses` its flow is everything computable"))
    return flows, counted


def _hard_checks(cat, questions, flows, rep):
    for name, p in cat.parts.items():
        if p.kind == "check" and p.hard and p.returns is None and p.func is not None:
            for line, vals in _returns(p.func) or ():
                bad = [v for v in vals if not isinstance(v, bool)]
                if bad:
                    rep.add("error", "hard_check_untyped", _where(p.func, line),
                            f"hard check {name} returns {bad[0]!r}: a check answers True or False — on that path it is "
                            "rejected and every question it governs abstains (return a bool and declare it `-> bool`)")
        if p.kind != "check" or not p.then:
            continue
        if not p.hard:
            rep.add("warning", "then_on_soft_check", name, "`then=` is set but the check is not hard: it is ignored "
                                                          "(@cat.check(hard=True, then=...))")
            continue
        for qn, ans in p.then.items():
            q = questions.get(qn)
            if q is None:
                if questions:
                    rep.add("error", "then_unknown", name, f"`then=` names {qn!r}, which is not a question")
                continue
            try:
                q.answer.normalize(ans)
            except (ValueError, TypeError) as e:
                rep.add("error", "then_bad_answer", name, f"`then=` answers {qn!r} with {ans!r}: {e}")
            if name not in flows.get(qn, ()):
                rep.add("error", "then_not_in_flow", name,
                        f"`then=` sets {qn!r}, but {qn!r}'s flow never runs this check (its rule does not read it through "
                        f"any fact and it is not in the question's checkpoints): when it fails, {qn!r} is answered as if "
                        f"it had passed — add checkpoints=[{name!r}] to the question")


# --- what a function plainly returns
def _returns(func):
    """The `return` statements of a function's own body (not of functions defined inside it) → [(line, the returned
    literals)] where a literal is a constant, also in the branches of `a if c else b`; a bare `return` is the literal
    None; a return of anything else gives no literal. None when the source is not available."""
    f = inspect.unwrap(func)
    try:
        lines, start = inspect.getsourcelines(f)
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    def literals(e):
        if e is None:
            return [None]
        if isinstance(e, ast.Constant):
            return [e.value]
        if isinstance(e, ast.IfExp):
            return literals(e.body) + literals(e.orelse)
        return []
    out, todo = [], list(tree.body[0].body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return):
            out.append((start + node.lineno - 1, literals(node.value)))
        todo.extend(ast.iter_child_nodes(node))
    return sorted(out, key=lambda t: t[0])


def _where(func, line):
    f = inspect.unwrap(func)
    return f"{_short_path(getattr(getattr(f, '__code__', None), 'co_filename', '?'))}:{line} ({getattr(f, '__name__', '?')})"


def _rules(cat, questions, rep):
    """Rules nobody asks; rules that plainly return an answer outside their question's options."""
    for qn, rule in cat.rules.items():
        q = questions.get(qn)
        if q is None:
            rep.add("warning", "unused_rule", qn, f"a rule is registered for {qn!r}, which is not among the system's "
                                                  "questions: it never runs")
            continue
        at = q.answer
        if at is None or at.kind not in ("yes_no", "choice", "ordinal", "multi") or rule.func is None \
                or getattr(rule.func, "__solvi_decision__", None) is not None:
            continue
        for line, vals in _returns(rule.func) or ():
            for v in vals:
                if v is None or v is Ellipsis:           # None: the rule abstains on purpose
                    continue
                try:
                    at.normalize(v)
                except (ValueError, TypeError):
                    rep.add("error", "rule_returns_non_option", _where(rule.func, line),
                            f"returns {v!r}, which is not one of {qn!r}'s options {list(at.options)}: on that path the "
                            "question abstains")


def _names(cat, system, questions, rep):
    """With System(inputs=Model), names only a typo (or an extra key) explains: an argument, or a `uses` hint, that
    nothing computes and the model does not declare. Without a model every such name is a given fact: nothing to tell."""
    m = getattr(system, "inputs", None)
    fields = getattr(m, "model_fields", None)
    readers = {}
    for p in list(cat.parts.values()) + list(cat.rules.values()):
        for a in (p.alternatives or [p]):
            for x in a.inputs:
                readers.setdefault(x, []).append(getattr(a.func, "__name__", a.name) if a.func is not None else a.name)
    if fields is None:
        return
    forbid = (getattr(m, "model_config", None) or {}).get("extra") == "forbid"
    how = ": the model forbids extra keys, so it can never be given" if forbid else \
        ": it can only arrive as an extra key of the input"
    for x, by in readers.items():
        if x in cat.parts or x in fields or x in questions:
            continue
        rep.add("warning", "input_not_declared", x,
                f"{', '.join(dict.fromkeys(by))} read{'s' if len(set(by)) == 1 else ''} {x}, which no part computes and "
                f"which is not a field of System(inputs={m.__name__})" + how)
    for qn, q in questions.items():
        for x in q.uses or ():
            if x not in cat.parts and x not in fields:
                rep.add("warning", "uses_unknown", qn, f"`uses` names {x!r}, which is neither a part of the catalog nor a "
                                                       f"field of System(inputs={m.__name__})" + how)


# --- types
def _types(cat, system, given, rep):
    from .typed import compatible, producers, type_name
    for fact, readers in cat.readers.items():
        for reader, t in readers.items():
            for prod, pt in producers(cat, fact):
                if not compatible(pt, t):
                    rep.add("error", "type_conflict", fact, f"{prod} returns {type_name(pt)}, but {reader} reads "
                                                            f"{fact}: {type_name(t)}")
        if fact in given:
            items = list(readers.items())
            for (ra, ta), (rb, tb) in itertools.combinations(items, 2):
                if ta != tb and not compatible(ta, tb) and not compatible(tb, ta):
                    rep.add("warning", "reader_types_differ", fact, f"given fact read as {type_name(ta)} by {ra} and as "
                                                                    f"{type_name(tb)} by {rb}")
        m = getattr(system, "inputs", None)
        if m is not None and fact in m.model_fields:
            ft = m.model_fields[fact].annotation
            for reader, t in readers.items():
                if not compatible(ft, t):
                    rep.add("error", "type_conflict", fact, f"System(inputs={m.__name__}) gives {fact}: {type_name(ft)}, "
                                                            f"but {reader} reads {type_name(t)}")
    for fact, g in cat.parts.items():
        typed = [(a.name, a.returns) for a in (g.alternatives or ()) if a.returns is not None]
        for (na, ta), (nb, tb) in itertools.combinations(typed, 2):
            if ta != tb and not compatible(ta, tb) and not compatible(tb, ta):
                rep.add("warning", "reader_types_differ", fact, f"alternative producers return {type_name(ta)} ({na}) "
                                                                f"and {type_name(tb)} ({nb})")


# --- constraints between answers
def _domain(at, max_multi=10):
    """The finite set of answers a question can have (what a constraint receives), or None."""
    from .core import Unknown
    if at is None:
        return None
    if at.kind in ("yes_no", "choice", "ordinal"):
        d = list(at.options)
    elif at.kind == "multi" and len(at.options) <= max_multi:
        d = [tuple(o for o, b in zip(at.options, bits) if b) for bits in itertools.product([0, 1], repeat=len(at.options))]
    else:
        return None
    return d + [Unknown] if at.unknown else d


def _holds(c, assign):
    """→ True / False, or the exception it raised."""
    try:
        return bool(c.func(**{q: assign[q] for q in c.inputs}))
    except Exception as e:  # noqa: BLE001
        return e


def _constraints(cat, questions, rep, max_combos):
    cons = []
    for name, c in cat.constraints.items():
        bad = [x for x in c.inputs if x not in questions]
        if bad:
            rep.add("error", "constraint_unknown_question", name, f"reads {', '.join(bad)}, which "
                    f"{'is not a question' if len(bad) == 1 else 'are not questions'}: it never applies")
        else:
            cons.append(c)
    doms = {q: _domain(questions[q].answer) for q in {x for c in cons for x in c.inputs}}
    finite = [c for c in cons if all(doms[q] is not None for q in c.inputs)]
    groups = []                                       # constraints that share questions, checked together
    for c in finite:
        qs = set(c.inputs)
        hit = [g for g in groups if g[0] & qs]
        for g in hit:
            groups.remove(g)
            qs |= g[0]
        groups.append((qs, [c] + [x for g in hit for x in g[1]]))
    for qs, group in groups:
        qs = sorted(qs)
        group = sorted(group, key=lambda c: list(cat.constraints).index(c.name))
        n = 1
        for q in qs:
            n *= len(doms[q])
        if n > max_combos:
            rep.add("note", "constraints_not_checked", ", ".join(c.name for c in group),
                    f"{n} combinations of answers (more than {max_combos})")
            continue
        holds_any = {c.name: False for c in group}
        raised = {}
        allowed = {q: set() for q in qs}
        n_ok = 0
        for combo in itertools.product(*[range(len(doms[q])) for q in qs]):
            assign = {q: doms[q][i] for q, i in zip(qs, combo)}
            all_ok = True
            for c in group:
                h = _holds(c, assign)
                if isinstance(h, Exception):
                    raised.setdefault(c.name, (assign, h))
                    h = False
                holds_any[c.name] |= h
                all_ok = all_ok and h
            if all_ok:
                n_ok += 1
                for q, i in zip(qs, combo):
                    allowed[q].add(i)
        for name, (assign, e) in raised.items():
            rep.add("warning", "constraint_raises", name, f"raises {type(e).__name__}: {e} on "
                    + ", ".join(f"{q}={v!r}" for q, v in assign.items() if q in cat.constraints[name].inputs)
                    + " (a raising constraint counts as broken)")
        never = [c for c, h in holds_any.items() if not h]
        for c in never:
            rep.add("error", "constraint_never_holds", c, "no combination of answers satisfies it")
        if n_ok == 0 and not never:
            rep.add("error", "constraints_conflict", ", ".join(c.name for c in group),
                    "each can hold, but no combination of answers satisfies them all")
        if n_ok:
            for q in qs:
                dead = [doms[q][i] for i in range(len(doms[q])) if i not in allowed[q]]
                for v in dead:
                    rep.add("warning", "dead_option", q, f"the constraints ({', '.join(c.name for c in group if q in c.inputs)}) "
                                                         f"rule out {v!r} whatever the other answers are")


# --- silent defaults in functions that read the input
def _literal(node):
    """A literal that is not None: a constant, or a list / tuple / set / dict display of literals (possibly empty)."""
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_literal(e) or (isinstance(e, ast.Constant)) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and isinstance(k, ast.Constant) for k in node.keys) and \
            all(_literal(v) or isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    return False


def silent_defaults(func, names=None):
    """[(line, snippet, why)] in a function's source: `x or <literal>` and `.get(k, <literal>)` (the literal not None),
    except on lines marked `# solvi: ok`. [] when the source is not available. names: the function's arguments that are
    given inputs — then only a default on one of them counts (`amount or 0`, `order.get("total", 0)`, `order["x"] or
    0`), not one on a constant table (`{...}.get(kind, 1)`), a computed value (`math.sqrt(v) or 1e-9`) or an
    exception's attribute (`e.lineno or 0`)."""
    f = inspect.unwrap(func)
    try:
        lines, start = inspect.getsourcelines(f)
    except (OSError, TypeError):
        return []
    src = textwrap.dedent("".join(lines))
    try:
        tree = ast.parse(src)
    except SyntaxError:                               # a lambda in the middle of an expression
        return []
    out = []

    def given(expr):                                  # is the value an input's own (the input, or a field / key of it)?
        if names is None:
            return True
        while True:
            if isinstance(expr, (ast.Attribute, ast.Subscript)):
                expr = expr.value
            elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "get":
                expr = expr.func.value                # order.get("x") or 0
            else:
                return isinstance(expr, ast.Name) and expr.id in names
    for node in ast.walk(tree):
        hit = None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and _literal(node.values[-1]) \
                and any(given(v) for v in node.values[:-1]):
            hit = f"`{ast.unparse(node)}` gives {ast.unparse(node.values[-1])} for a missing, empty or null value"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and len(node.args) == 2 and _literal(node.args[1]) and given(node.func.value):
            hit = f"`{ast.unparse(node)}` gives {ast.unparse(node.args[1])} for a missing key"
        if hit:
            line = start + node.lineno - 1
            text = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
            if "solvi: ok" not in text:
                out.append((line, text.strip(), hit))
    return sorted(set(out))


def _silent_defaults(cat, given, rep):
    seen = set()
    parts = list(cat.parts.values()) + list(cat.rules.values())
    for p in parts:
        for a in (p.alternatives or [p]):
            if a.func is None or not set(a.inputs) & given or id(a.func) in seen:
                continue
            seen.add(id(a.func))
            f = inspect.unwrap(a.func)
            path = _short_path(getattr(getattr(f, "__code__", None), "co_filename", "?"))
            for line, _, why in silent_defaults(a.func, set(a.inputs) & given):
                rep.add("warning", "silent_default", f"{path}:{line} ({getattr(f, '__name__', a.name)})",
                        why + ": say what a missing input means (check for None and abstain, or declare the default in "
                              "System(inputs=...)); `# solvi: ok` accepts it")


def _short_path(p):
    import os
    try:
        r = os.path.relpath(p)
    except ValueError:                                # another drive (Windows)
        return p
    return p if r.startswith("..") else r


# --- the command
def cmd_check(a):
    """`solvi check` (see solvi.cli) → 0: no errors (with --strict: no warnings either); 1: problems. The target is
    module:attr / file.py:attr, or — as for `solvi test` — a task file or a directory holding task.py (its System:
    `task.system()`, else System(task.cat, task.QUESTIONS))."""
    obj = _target(a.target)
    if not (hasattr(obj, "parts") and hasattr(obj, "rules")) and not hasattr(obj, "catalog"):
        from .cli import _fail
        _fail(f"check {a.target}: not a solvi System, Catalog or Guard")
    rep = lint(obj, strict=a.strict, max_combos=a.max_combos)
    if a.json:
        from .schema import dumps
        print(dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(rep)
    return 0 if rep.ok else 1


def _target(spec):
    from pathlib import Path
    p = Path(spec)
    task = p / "task.py" if p.is_dir() else p if p.is_file() and p.suffix == ".py" else None
    if task is not None and task.is_file():
        from .honesty import load_task, system_of
        mod = load_task(task)
        if not callable(getattr(mod, "system", None)) and not (hasattr(mod, "cat") and hasattr(mod, "QUESTIONS")):
            from .cli import _fail
            _fail(f"check {spec}: a task module defines system() or cat and QUESTIONS (else give module:attr)")
        return system_of(mod)
    from .cli import load_object
    return load_object(spec)


def add_parser(sub):
    c = sub.add_parser("check", help="lint a catalog: hard checks outside their question's flow, unused parts, cycles, "
                                     "type conflicts, constraints that cannot hold, silent defaults")
    c.add_argument("target", help="module:attr or file.py:attr — a System (or a function returning one), a Catalog, or a "
                                  "solvi.agents.Guard (each tool's checks); or a task file / a directory with task.py, "
                                  "as for solvi test")
    c.add_argument("--strict", action="store_true", help="warnings fail too (exit status 1)")
    c.add_argument("--max-combos", type=int, default=100_000,
                   help="the most answer combinations tried per group of constraints (default 100000)")
    c.add_argument("--json", action="store_true", help="print JSON")
    return c
