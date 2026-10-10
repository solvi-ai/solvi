"""A contract on an answer: what it must be before it counts, checked by code (stable since 1.1).

A check that an answerer runs correctly — a program that passes its examples, a part whose replay matches — says
nothing about whether it was the right answerer for this input: a program that counts words, run on a question that
asks for a name, returns a number, and returns it correctly. A certificate certifies execution, not reading. The
contract states what reading the question right looks like, in three declarative checks:

  returns     the answer's type matches what the question asks: a Python type (or a tuple of them; an int is a float,
              a bool is not a number) or an answer type (`Answer.choice(...)`, `Answer.estimate()`, ...); None — the
              question's declared answer type
  echo        the answer is not the input it was read from (a "summary" that is the text, a "label" that is the
              field it was given): True — every given fact the question's rule reads, directly or through computed
              facts; a list of fact names; False — off; None (the default) — by the answer's kind: checked for free
              text (a str answer, a span, `returns=str`), not for a declared choice, yes / no, enumeration or number,
              which may rightly equal an input field. An explicit value always overrides. Compared after
              normalisation (case, spacing; a number and its digits are one value)
  applicable  a precondition `applicable(state) -> bool` over the given facts, checked BEFORE the answerer is called:
              an answerer is called because its question applies, not because the input has the right form. One that
              raises counts as not met.

Where it applies (each violation is a hard refusal with a named reason, whatever the confidence):

    cat.contract("first_word", returns=str, applicable=one_line)     # in a System: the guard "contract"
    AskPath(system2, contract=Contract(applicable=asks_a_count))               # a slow path: not called, not accepted
    class MySpecialist(Specialist):
        def applicable(self, source, question): ...                           # a specialist: blocked, not proposed

In a System an answer that breaks the contract abstains (`Result.guard == "contract"`, the violations in
`Result.extra["contract"]`); a question whose precondition is not met is not computed (`guard == "not_applicable"`).
In the dispatcher a slow path whose contract does not apply is not called — a person gets the input, and the
decision records "not applicable" — and an answer of the slow path that breaks it is not accepted
(`Thought.contract`); both replay. Hard checks forcing an answer (`then=`) are the author's answers and are not
checked against the contract."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

CODES = ("wrong_type", "echo", "not_applicable")


@dataclasses.dataclass(frozen=True)
class Violation:
    """One broken clause of a contract: `code` ("wrong_type", "echo") and `message` (for people)."""
    code: str
    message: str

    def to_dict(self):
        return {"code": self.code, "message": self.message}

    def __str__(self):
        return f"{self.code}: {self.message}"


def _short(v, n=60):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _number(v):
    """A number, or a string that is exactly a number written in digits → its float; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().replace(",", "")
        if t and all(c in "0123456789.-+eE" for c in t) and any(c.isdigit() for c in t):
            try:
                return float(t)
            except ValueError:
                return None
    return None


def canon(v):
    """A value as the echo check compares it: a number (or its digits) by value, a string by its words without case,
    a list or tuple item by item; None, a bool and an empty string compare to nothing."""
    import enum
    if isinstance(v, enum.Enum):
        v = v.value
    if v is None or isinstance(v, bool):
        return None
    n = _number(v)
    if n is not None:
        return ("number", n)
    if isinstance(v, str):
        words = " ".join(v.casefold().split())
        return ("text", words) if words else None
    if isinstance(v, (list, tuple)):
        items = tuple(canon(x) for x in v)
        return ("items", items) if items else None
    text = getattr(v, "text", None)                  # a Quote: its text
    if isinstance(text, str):
        return canon(text)
    return ("repr", repr(v))


def _type_name(t):
    if isinstance(t, tuple):
        return " or ".join(_type_name(x) for x in t)
    return getattr(t, "__name__", str(t))


def _is_answer_type(t):
    return hasattr(t, "normalize") and hasattr(t, "kind")


def _type_ok(answer, t):
    """Does the answer have the type t (a Python type or tuple of types, or an answer type)? → (ok, what was expected)."""
    if _is_answer_type(t):
        try:
            t.normalize(answer)
            return True, f"an answer of kind {t.kind}" + (f" (one of {list(t.options)})" if t.options else "")
        except (ValueError, TypeError) as e:
            return False, f"an answer of kind {t.kind}: {str(e).splitlines()[0][:120]}"
    types = t if isinstance(t, tuple) else (t,)
    ok = any((isinstance(answer, x) and not (isinstance(answer, bool) and x in (int, float)))
             or (x is float and isinstance(answer, int) and not isinstance(answer, bool)) for x in types)
    return ok, _type_name(t)


@dataclasses.dataclass(frozen=True)
class Contract:
    """The contract of an answer (see the module docs). returns: a Python type, a tuple of types or an answer type
    (None: the question's own); echo: True, a list of fact names, False, or None (by the answer's kind: free text
    only); applicable: a function of the given facts
    → bool (None: always applies)."""
    returns: Any = None
    echo: Any = None
    applicable: Callable | None = None

    def __post_init__(self):
        if self.applicable is not None and not callable(self.applicable):
            raise TypeError("applicable is a function (state) -> bool")
        if self.echo is not None and not isinstance(self.echo, bool):
            names = (self.echo,) if isinstance(self.echo, str) else tuple(self.echo)
            if not all(isinstance(x, str) for x in names):
                raise TypeError("echo is True, False, None or a list of fact names")
            object.__setattr__(self, "echo", names)
        r = self.returns
        if r is not None and not _is_answer_type(r) and not isinstance(r, type) and not (
                isinstance(r, tuple) and r and all(isinstance(x, type) for x in r)):
            raise TypeError("returns is a type, a tuple of types or an answer type (Answer.choice(...), ...)")

    # --- the clauses
    def applies(self, state) -> tuple:
        """Is the precondition met on these given facts? → (True, None) or (False, why)."""
        f = self.applicable
        if f is None:
            return True, None
        name = getattr(f, "__name__", type(f).__name__)
        try:
            ok = f(dict(state))
        except Exception as e:  # noqa: BLE001 — a precondition that cannot be evaluated is not met
            return False, f"{name} could not be evaluated: {type(e).__name__}: {e}"
        if ok is True:
            return True, None
        doc = ((getattr(f, "__doc__", None) or "").strip().splitlines() or [""])[0]
        return False, f"{name} is false" + (f" ({doc})" if doc else "")

    def echo_inputs(self, given, read=None):
        """The given facts an answer must not equal: all those the answerer read (`read`; None — every given fact) when
        echo is True or None, the named ones when echo names facts, none when False."""
        if self.echo is False:
            return {}
        names = self.echo if isinstance(self.echo, tuple) else (read if read is not None else list(given))
        return {k: given[k] for k in names if k in given}

    def echoes(self, answer, answer_type=None) -> bool:
        """Is the echo clause checked for this answer? An explicit echo= decides; else (None) by the answer's kind:
        free text — a span or a str answer type (`returns`, else the question's answer type, else the answer's own
        value) — is checked; a declared choice, yes / no, ordinal, multi, rank, estimate, an enumeration, a number or
        a bool is not."""
        import enum
        if self.echo is not None:
            return self.echo is not False
        t = self.returns if self.returns is not None else answer_type
        if t is not None:
            if _is_answer_type(t):
                return t.kind == "span" and t.type in (None, str)
            types = t if isinstance(t, tuple) else (t,)
            return all(isinstance(x, type) and issubclass(x, str) and not issubclass(x, enum.Enum) for x in types)
        return isinstance(answer, str) and not isinstance(answer, enum.Enum)

    def violations(self, answer, inputs=None, answer_type=None, *, echo_type=None) -> list:
        """The clauses an answer breaks → [Violation] ([] when it keeps the contract; an answer of None — an
        abstention — is not checked). inputs: {fact: value} the answer must not equal (see echo_inputs);
        answer_type: the question's, used for the type clause when `returns` is None and for the echo default;
        echo_type: the question's answer type for the echo default only (a System, whose answers already hold their
        type)."""
        if answer is None:
            return []
        out = []
        t = self.returns if self.returns is not None else answer_type
        if t is not None:
            ok, expected = _type_ok(answer, t)
            if not ok:
                out.append(Violation("wrong_type", f"the answer {_short(answer)} ({type(answer).__name__}) is not "
                                                   f"{expected}"))
        if self.echoes(answer, echo_type if echo_type is not None else answer_type):
            a = canon(answer)
            if a is not None:
                same = sorted(k for k, v in (inputs or {}).items() if canon(v) == a)
                if same:
                    out.append(Violation("echo", f"the answer {_short(answer)} is the input {', '.join(same)} it was "
                                                 "read from"))
        return out

    def fingerprint(self):
        """What the contract's verdicts depend on: the type, the echo setting and the code of `applicable` (its source,
        or its bytecode when there is none) → a short hash. Changing any of them changes the fingerprint of the
        System or slow path that carries the contract, so a decision made under another contract does not replay."""
        import hashlib
        import inspect
        r = self.returns
        rt = repr(r) if _is_answer_type(r) else _type_name(r) if r is not None else None
        code = None
        f = self.applicable
        if f is not None:
            f = inspect.unwrap(f)
            try:
                code = inspect.getsource(f)
            except (OSError, TypeError):
                co = getattr(f, "__code__", None)
                code = co.co_code.hex() + repr(co.co_consts) if co is not None else type(f).__qualname__
            code = f"{getattr(f, '__qualname__', type(f).__qualname__)}\n{code}"
        echo = list(self.echo) if isinstance(self.echo, tuple) else "auto" if self.echo is None else self.echo
        blob = repr((rt, echo, code))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def refusal(violations, answer):
    """The reason an answer is refused for its violations (one line)."""
    return "contract: " + "; ".join(str(v) for v in violations) + f"; would have answered {_short(answer)}"


__all__ = ["canon", "CODES", "Contract", "refusal", "Violation"]
