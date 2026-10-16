"""Narrows / widens proved over a rule's text (opt-in, the `smt` extra: pip install "solvi[smt]" → z3-solver).

    from solvi.experimental.changes import prove_direction
    prove_direction("def ok(amount, country): return amount < 1000 and country in {'DE', 'FR'}",
                    "def ok(amount, country): return amount < 1500 and country in {'DE', 'FR'}")
    # → {"proved": "widens", "witness_widens": {"amount": 1000, "country": "DE"}, "ms": ..., ...}

A rule accepts an input when it returns a value that is not a refusal: None, False and a refusal word (is_refusal) are
refusals; True and any other constant are answers. With `outcome=` the rule accepts an input when it returns that value
(e.g. "high": the inputs that get high priority). Both texts are translated to SMT and two questions are asked: is there an
input the new rule accepts and the old one refused (widens), and one the old rule accepted and the new one refuses
(narrows)? Both → "both"; neither → "same" (the two accept exactly the same inputs); a found input is the witness.

The supported subset: one function (`def` or `lambda`) or a bare expression over the facts (its free names); its
parameters are the facts; `if` / `elif` / `else`, conditional expressions, local assignments and `return`; comparisons
(chained too), `and` / `or` / `not`, + - * / and unary minus on numbers, abs / min / max, a boolean used as 0 / 1 in a
sum, membership in a literal set, list or tuple (`in`, `not in`), and constants — literals, module-level literal
assignments in the text, and literal globals of a function given as an object. Anything else (a call, a loop, an
attribute, a subscript, a name outside the inputs, a non-linear product the solver gives up on) → "unknown" with the
reason, and the caller falls back to measuring (Consistency(probes=)).

Sorts are inferred from use: a fact compared with a string is a string, used as a condition a boolean, else a real
number; `types={"age": int}` makes a fact an integer (x > 3 and x >= 4 are the same rule over integers, not over reals).
A fact used with two sorts → "unknown". The sorts are returned ("sorts") so the reader sees what was proved over."""
from __future__ import annotations

import ast
import inspect
import textwrap
import time
from typing import Any, Callable

SMT_HINT = 'narrows / widens over the rule text needs z3: pip install "solvi[smt]" (the z3-solver package)'


class Unsupported(Exception):
    """A construct outside the subset translated to SMT."""


def require_z3():
    """The z3 module, or an ImportError that says how to install it."""
    try:
        import z3
    except ImportError as e:
        raise ImportError(SMT_HINT) from e
    return z3


def _literal_globals(fn) -> dict:
    out = {}
    for k in fn.__code__.co_names:
        v = fn.__globals__.get(k, inspect.Parameter.empty)
        if isinstance(v, (bool, int, float, str)) or (isinstance(v, (set, frozenset, tuple, list))
                                                      and all(isinstance(x, (bool, int, float, str)) for x in v)):
            out[k] = v
    return out


def _source(rule) -> tuple[str, dict]:
    """A rule's text and the literal globals it reads (a function object); a string is the text itself."""
    if isinstance(rule, str):
        return textwrap.dedent(rule), {}
    if callable(rule) and hasattr(rule, "__code__"):
        try:
            return textwrap.dedent(inspect.getsource(rule)), _literal_globals(rule)
        except (OSError, TypeError) as e:
            raise Unsupported(f"no source text for {rule!r}: {e}") from None
    raise Unsupported(f"a rule is its text or a function, not {type(rule).__name__}")


def _parse(text: str):
    """Text → (params or None for a bare expression, body statements or an expression node, module constants)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        raise Unsupported("the rule text does not parse") from None
    consts, fns = {}, []
    for st in tree.body:
        if isinstance(st, ast.FunctionDef):
            fns.append(st)
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            try:
                consts[st.targets[0].id] = ast.literal_eval(st.value)
            except ValueError:
                lam = [n for n in ast.walk(st.value) if isinstance(n, ast.Lambda)]
                if len(lam) == 1:
                    fns.append(lam[0])
        elif isinstance(st, ast.Expr):
            lam = [n for n in ast.walk(st.value) if isinstance(n, ast.Lambda)]
            if len(lam) == 1:
                fns.append(lam[0])
            elif not lam and len(tree.body) == 1:
                return None, st.value, consts
        elif not isinstance(st, (ast.Import, ast.ImportFrom)):
            raise Unsupported(f"a top-level {type(st).__name__} in the rule text")
    if len(fns) != 1:
        raise Unsupported(f"the text holds {len(fns)} functions, not one")
    f = fns[0]
    a = f.args
    if a.vararg or a.kwarg or a.posonlyargs:
        raise Unsupported("*args / **kwargs / positional-only parameters")
    params = [p.arg for p in a.args + a.kwonlyargs]
    body = f.body if isinstance(f, ast.FunctionDef) else [ast.Return(f.body)]
    return params, body, consts


_NONE = object()                                       # the constant None (a refusal when returned)


class _Translator:
    """One rule's text → its accept condition as a z3 boolean; the fact variables are shared across rules."""

    def __init__(self, z3, facts: dict, types: dict, refused: Callable, outcome):
        self.z3, self.facts, self.types, self.refused, self.outcome = z3, facts, types, refused, outcome

    # ---------------------------------------------------------------------------------- facts and constants
    def fact(self, name, sort):
        z3 = self.z3
        want = self.types.get(name)
        if want is not None:
            sort = {int: "Int", float: "Real", bool: "Bool", str: "String"}.get(want, want)
        if name in self.facts:
            v = self.facts[name]
            have = self.sort_of(v)
            if sort not in (None, have) and not (sort == "Real" and have == "Int"):
                raise Unsupported(f"the fact {name!r} is used as {have} and as {sort}")
            return v
        sort = sort or "Real"
        mk = {"Int": z3.Int, "Real": z3.Real, "Bool": z3.Bool, "String": z3.String}.get(sort)
        if mk is None:
            raise Unsupported(f"sort {sort!r} of {name!r}")
        self.facts[name] = v = mk(name)
        return v

    def sort_of(self, v):
        z3 = self.z3
        if isinstance(v, (bool, int, float, str)) or v is _NONE:
            return {bool: "Bool", int: "Real", float: "Real", str: "String"}.get(type(v))
        if z3.is_bool(v):
            return "Bool"
        if z3.is_int(v):
            return "Int"
        if z3.is_real(v):
            return "Real"
        if z3.is_string(v):
            return "String"
        raise Unsupported(f"a value of sort {v.sort()}")

    def const(self, c):
        z3 = self.z3
        if isinstance(c, bool):
            return z3.BoolVal(c)
        if isinstance(c, int):
            return z3.IntVal(c)
        if isinstance(c, float):
            return z3.RealVal(c)
        if isinstance(c, str):
            return z3.StringVal(c)
        raise Unsupported(f"a constant {c!r}")

    def num(self, v):
        """A value used as a number (a boolean counts 0 / 1)."""
        z3 = self.z3
        if self.sort_of(v) == "Bool":
            return z3.If(v, z3.IntVal(1), z3.IntVal(0))
        if self.sort_of(v) not in ("Int", "Real"):
            raise Unsupported("arithmetic on a string")
        return v

    # ---------------------------------------------------------------------------------- expressions
    def expr(self, n, env, hint=None):
        """An expression node → a z3 term (or a Python literal tuple for a literal set). hint: the sort a bare fact
        takes when this is its first use."""
        z3 = self.z3
        if isinstance(n, ast.Constant):
            if n.value is None:
                return _NONE
            return self.const(n.value)
        if isinstance(n, ast.Name):
            if n.id in env:
                v = env[n.id]
                if _is_fact(v):
                    return self.fact(n.id, hint)
                return v if hasattr(v, "sort") else self.const_value(v)
            if n.id in ("True", "False"):
                return z3.BoolVal(n.id == "True")
            raise Unsupported(f"the name {n.id!r} is not an input of the rule")
        if isinstance(n, ast.BoolOp):
            xs = [self.boolean(v, env) for v in n.values]
            return z3.And(*xs) if isinstance(n.op, ast.And) else z3.Or(*xs)
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return z3.Not(self.boolean(n.operand, env))
            if isinstance(n.op, ast.USub):
                return -self.num(self.expr(n.operand, env, "Real"))
            if isinstance(n.op, ast.UAdd):
                return self.num(self.expr(n.operand, env, "Real"))
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = self.num(self.expr(n.left, env, "Real")), self.num(self.expr(n.right, env, "Real"))
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return z3.ToReal(a) / z3.ToReal(b) if z3.is_int(a) and z3.is_int(b) else a / b
        if isinstance(n, ast.IfExp):
            c = self.boolean(n.test, env)
            a, b = self.expr(n.body, env, hint), self.expr(n.orelse, env, hint)
            if a is _NONE or b is _NONE:
                raise Unsupported("None inside an expression")
            return z3.If(c, a, b)
        if isinstance(n, ast.Compare):
            return self.compare(n, env)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("abs", "min", "max") \
                and n.func.id not in env and not n.keywords:
            xs = [self.num(self.expr(a, env, "Real")) for a in n.args]
            if n.func.id == "abs" and len(xs) == 1:
                return z3.If(xs[0] >= 0, xs[0], -xs[0])
            if n.func.id in ("min", "max") and len(xs) >= 2:
                out = xs[0]
                for x in xs[1:]:
                    out = z3.If(x < out, x, out) if n.func.id == "min" else z3.If(x > out, x, out)
                return out
        raise Unsupported(f"{type(n).__name__} ({ast.unparse(n)[:60]})")

    def const_value(self, v):
        if v is None:
            return _NONE
        if isinstance(v, (set, frozenset, tuple, list)):
            raise Unsupported("a collection used as a value")
        return self.const(v)

    def boolean(self, n, env):
        v = self.expr(n, env, "Bool")
        if self.sort_of(v) != "Bool":
            raise Unsupported(f"a non-boolean condition ({ast.unparse(n)[:60]})")
        return v

    def literal_set(self, n, env):
        if isinstance(n, (ast.Set, ast.List, ast.Tuple)):
            try:
                return [ast.literal_eval(e) for e in n.elts]
            except ValueError:
                raise Unsupported("membership in a set that is not literal") from None
        if isinstance(n, ast.Name) and isinstance(env.get(n.id), (set, frozenset, tuple, list)):
            return list(env[n.id])
        raise Unsupported(f"membership in {ast.unparse(n)[:60]} (only a literal set, list or tuple)")

    def sort_hint(self, n, env):
        """The sort an operand fixes for the other side (None when it does not)."""
        if isinstance(n, ast.Constant) and n.value is not None:
            return {bool: "Bool", str: "String"}.get(type(n.value), "Real")
        if isinstance(n, ast.Name) and n.id in env and not _is_fact(env[n.id]):
            v = env[n.id]
            if isinstance(v, (bool, int, float, str)):
                return {bool: "Bool", str: "String"}.get(type(v), "Real")
            if hasattr(v, "sort"):
                return self.sort_of(v)
        if isinstance(n, ast.Name) and n.id in self.facts and _is_fact(env.get(n.id)):
            return self.sort_of(self.facts[n.id])
        if isinstance(n, (ast.BinOp, ast.UnaryOp)) and not isinstance(getattr(n, "op", None), ast.Not):
            return "Real"
        return None

    def compare(self, n, env):
        z3 = self.z3
        parts, left = [], n.left
        for op, right in zip(n.ops, n.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                vals = self.literal_set(right, env)
                kinds = {("String" if isinstance(v, str) else "Bool" if isinstance(v, bool) else "Real") for v in vals}
                if len(kinds) > 1:
                    raise Unsupported("a set of mixed constants")
                x = self.expr(left, env, next(iter(kinds), "Real"))
                eq = z3.Or(*[x == self.const(v) for v in vals]) if vals else z3.BoolVal(False)
                parts.append(eq if isinstance(op, ast.In) else z3.Not(eq))
            else:
                hint = self.sort_hint(right, env) or self.sort_hint(left, env) or "Real"
                a, b = self.expr(left, env, hint), self.expr(right, env, hint)
                if a is _NONE or b is _NONE:
                    raise Unsupported("a comparison with None")
                sa, sb = self.sort_of(a), self.sort_of(b)
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    if {sa, sb} <= {"Int", "Real"} or sa == sb:
                        r = a == b
                    elif "Bool" in (sa, sb) and {sa, sb} <= {"Bool", "Int", "Real"}:
                        r = self.num(a) == self.num(b)
                    else:
                        raise Unsupported(f"{sa} compared with {sb}")
                    parts.append(r if isinstance(op, ast.Eq) else z3.Not(r))
                elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    a, b = self.num(a), self.num(b)
                    parts.append({ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b}[type(op)])
                else:
                    raise Unsupported(f"the comparison {type(op).__name__}")
            left = right
        return parts[0] if len(parts) == 1 else z3.And(*parts)

    # ---------------------------------------------------------------------------------- statements
    def accepts(self, n, env):
        """A returned expression → does it accept the input?"""
        z3 = self.z3
        if isinstance(n, ast.IfExp):
            return z3.If(self.boolean(n.test, env), self.accepts(n.body, env), self.accepts(n.orelse, env))
        if n is None or (isinstance(n, ast.Constant) and n.value is None):
            return z3.BoolVal(self.outcome is None)             # None refuses unless it is the outcome asked
        if isinstance(n, ast.Constant):
            if self.outcome is not _ANY:
                return z3.BoolVal(n.value == self.outcome)
            if isinstance(n.value, bool):
                return z3.BoolVal(n.value)
            return z3.BoolVal(not self.refused(n.value))
        v = self.expr(n, env, "Bool")
        if self.sort_of(v) == "Bool":
            if self.outcome is _ANY:
                return v
            if isinstance(self.outcome, bool):
                return v if self.outcome else z3.Not(v)
        raise Unsupported(f"a returned value that is not a constant or a condition ({ast.unparse(n)[:60]})")

    def block(self, stmts, env):
        z3 = self.z3
        for i, st in enumerate(stmts):
            if isinstance(st, ast.Pass) or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
                continue
            if isinstance(st, ast.Return):
                return self.accepts(st.value, env)
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                env = {**env, st.targets[0].id: self.expr(st.value, env)}
                continue
            if isinstance(st, ast.If):
                rest = stmts[i + 1:]
                c = self.boolean(st.test, env)
                return z3.If(c, self.block(st.body + rest, env), self.block(st.orelse + rest, env))
            raise Unsupported(f"the statement {type(st).__name__} ({ast.unparse(st).splitlines()[0][:60]})")
        return self.accepts(None, env)                 # fell off the end: returns None

    def rule(self, rule):
        text, globs = _source(rule)
        params, body, consts = _parse(text)
        env: dict[str, Any] = {**globs, **consts}
        if params is None:                             # a bare expression: its free names are the facts
            params = sorted({x.id for x in ast.walk(body) if isinstance(x, ast.Name)} - set(env) - {"abs", "min", "max"})
            body = [ast.Return(body)]
        for p in params:
            env[p] = "\0fact:" + p
        return self.block(body, env)


def _is_fact(v):
    return isinstance(v, str) and v.startswith("\0fact:")


_ANY = object()                                        # no outcome given: accepted = not a refusal


def _value(z3, v):
    if z3.is_true(v) or z3.is_false(v):
        return z3.is_true(v)
    if z3.is_int_value(v):
        return v.as_long()
    if z3.is_rational_value(v):
        f = v.as_fraction()
        return int(f) if f.denominator == 1 else float(f)
    if z3.is_string_value(v):
        return v.as_string()
    return str(v)


def prove_direction(before, after, *, types: dict | None = None, outcome=_ANY, refused: Callable | None = None,
                    timeout_ms: int = 2000) -> dict:
    """Does the rule `after` accept more inputs than `before` (widens), fewer (narrows), both, or exactly the same?
    Proved over the two rules' text with z3; outside the supported subset (module docstring) → "unknown".

    before, after: a rule's text (a `def`, a `lambda`, or an expression over the facts) or its function.
    types: {fact: int | float | bool | str} for a fact whose sort the use does not tell (a number is real by default).
    outcome: the rule accepts an input when it returns this value (default: when it returns a value that is not a
             refusal — None, False, or what `refused` says; default is_refusal).
    → {"proved": "widens" | "narrows" | "both" | "same" | "unknown", "reason", "witness_widens" / "witness_narrows"
       (an input that shows it), "sorts", "ms"}. Raises ImportError (how to install it) when z3 is missing."""
    z3 = require_z3()
    if refused is None:
        from ._refusal import is_refusal
        refused = is_refusal
    t0 = time.perf_counter()
    facts: dict = {}
    tr = _Translator(z3, facts, dict(types or {}), refused, outcome)

    def done(proved, reason="", **kw):
        return {"proved": proved, "reason": reason, **kw,
                "sorts": {k: tr.sort_of(v) for k, v in sorted(facts.items())},
                "ms": round((time.perf_counter() - t0) * 1000, 3)}

    try:
        a, b = tr.rule(before), tr.rule(after)
    except Unsupported as e:
        return done("unknown", str(e))
    except (RecursionError, z3.Z3Exception) as e:
        return done("unknown", f"translation failed: {e}")
    found = {}
    for key, q in (("widens", z3.And(b, z3.Not(a))), ("narrows", z3.And(a, z3.Not(b)))):
        s = z3.Solver()
        s.set("timeout", int(timeout_ms))
        s.add(q)
        r = s.check()
        if r == z3.unknown:
            return done("unknown", f"the solver gave up ({s.reason_unknown()})")
        if r == z3.sat:
            m = s.model()
            found[key] = {k: _value(z3, m.eval(v, model_completion=True)) for k, v in sorted(facts.items())}
    proved = "both" if len(found) == 2 else next(iter(found), "same")
    return done(proved, **{f"witness_{k}": v for k, v in found.items()})


__all__ = ["prove_direction", "require_z3", "SMT_HINT", "Unsupported"]
