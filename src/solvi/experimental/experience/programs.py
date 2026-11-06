"""Finite, typed scalar programs and a deterministic proposal-only enumerator.

Observational deduplication is incomplete: expressions agreeing on proposal
rows may disagree elsewhere (or have different depth/overflow behaviour).
A found program requires independent admission by the caller. ``if`` is eager:
all three children are evaluated, typed and charged, including the unused arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, product

from ._common import BudgetStop, WorkBudget, digest, identifier, natural

_ARITY = {"add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2, "lt": 2, "eq": 2, "and": 2, "or": 2, "not": 1, "if": 3}
_NUMERIC = {"add", "sub", "mul", "min", "max", "lt"}


def _scalar(value, maximum):
    if type(value) not in (int, bool) or (type(value) is int and abs(value) > maximum):
        raise ValueError("expected a bounded strict int or bool")
    return value


def _primitive(op, args, maximum):
    if op in _NUMERIC and any(type(v) is not int for v in args):
        raise ValueError("numeric operator requires strict integers")
    if op in ("and", "or", "not") and any(type(v) is not bool for v in args):
        raise ValueError("boolean operator requires booleans")
    if op == "eq" and type(args[0]) is not type(args[1]):
        raise ValueError("equality requires matching types")
    if op == "if" and (type(args[0]) is not bool or type(args[1]) is not type(args[2])):
        raise ValueError("if requires bool condition and matching branches")
    if op == "add":
        value = args[0] + args[1]
    elif op == "sub":
        value = args[0] - args[1]
    elif op == "mul":
        value = args[0] * args[1]
    elif op == "min":
        value = min(args)
    elif op == "max":
        value = max(args)
    elif op == "lt":
        value = args[0] < args[1]
    elif op == "eq":
        value = args[0] == args[1]
    elif op == "and":
        value = args[0] and args[1]
    elif op == "or":
        value = args[0] or args[1]
    elif op == "not":
        value = not args[0]
    else:
        value = args[1] if args[0] else args[2]
    return _scalar(value, maximum)


@dataclass(frozen=True)
class Grammar:
    inputs: tuple[str, ...]
    constants: tuple[int, ...] = (0, 1)
    operators: tuple[str, ...] = tuple(_ARITY)
    max_depth: int = 3
    max_nodes: int = 15
    max_candidates: int = 10000
    max_abs: int = 1000000

    def __post_init__(self):
        for name in ("inputs", "constants", "operators"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in self.inputs:
            identifier(name, "input name")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("duplicate inputs")
        for name in ("max_depth", "max_nodes"):
            natural(getattr(self, name), name, positive=True)
        for name in ("max_candidates", "max_abs"):
            natural(getattr(self, name), name)
        if self.max_depth > 64 or self.max_nodes > 4096:
            raise ValueError("grammar exceeds implementation structural limits")
        if any(type(c) is not int or abs(c) > self.max_abs for c in self.constants):
            raise ValueError("constants must be bounded strict integers")
        if any(type(op) is not str or op not in _ARITY for op in self.operators):
            raise ValueError("unknown operator")
        if len(set(self.operators)) != len(self.operators):
            raise ValueError("duplicate operators")

    def to_dict(self):
        return {
            "inputs": list(self.inputs),
            "constants": list(self.constants),
            "operators": list(self.operators),
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "max_candidates": self.max_candidates,
            "max_abs": self.max_abs,
        }

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls(()).to_dict()):
            raise ValueError("invalid grammar snapshot")
        return cls(**value)


def _freeze(ast, level=1, counter=None):
    if counter is None:
        counter = [0]
    counter[0] += 1
    if level > 64 or counter[0] > 4096:
        raise ValueError("AST exceeds implementation structural limits")
    if type(ast) not in (list, tuple) or not ast or type(ast[0]) is not str:
        raise ValueError("malformed AST")
    op = ast[0]
    if op in ("input", "const"):
        if len(ast) != 2:
            raise ValueError("malformed terminal")
        if op == "input":
            identifier(ast[1], "input name")
        elif type(ast[1]) not in (int, bool):
            raise ValueError("constant must be strict int or bool")
        return (op, ast[1])
    if op not in _ARITY or len(ast) != _ARITY[op] + 1:
        raise ValueError("unknown operator or incorrect arity")
    return (op, *(_freeze(c, level + 1, counter) for c in ast[1:]))


def _json(ast):
    return [ast[0], ast[1]] if ast[0] in ("const", "input") else [ast[0], *(_json(c) for c in ast[1:])]


def _metadata(ast):
    if ast[0] in ("input", "const"):
        return 1, 1, {ast[1]} if ast[0] == "input" else set()
    children = [_metadata(c) for c in ast[1:]]
    return 1 + sum(c[0] for c in children), 1 + max(c[1] for c in children), set().union(*(c[2] for c in children))


@dataclass(frozen=True, init=False)
class Program:
    ast: tuple
    nodes: int
    depth: int
    input_names: tuple[str, ...]
    fingerprint: str

    def __init__(self, ast):
        frozen = _freeze(ast)
        nodes, depth, names = _metadata(frozen)
        for key, value in [
            ("ast", frozen),
            ("nodes", nodes),
            ("depth", depth),
            ("input_names", tuple(sorted(names))),
            ("fingerprint", digest(_json(frozen))),
        ]:
            object.__setattr__(self, key, value)

    def to_dict(self):
        return {"v": 1, "ast": _json(self.ast), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value):
        if (
            type(value) is not dict
            or set(value) != {"v", "ast", "fingerprint"}
            or type(value["v"]) is not int
            or value["v"] != 1
        ):
            raise ValueError("invalid program snapshot")
        program = cls(value["ast"])
        if value["fingerprint"] != program.fingerprint:
            raise ValueError("program fingerprint mismatch")
        return program

    def evaluate(self, inputs, budget: WorkBudget, grammar: Grammar):
        # Debit before even input/cap failures; the root visit is paid here.
        budget.charge("program_node")
        if self.nodes > grammar.max_nodes or self.depth > grammar.max_depth:
            raise ValueError("program exceeds grammar structural limits")
        if type(inputs) is not dict or set(inputs) != set(grammar.inputs):
            raise ValueError("input schema mismatch")
        values = {}
        for name in grammar.inputs:
            budget.charge("program_input")
            values[name] = _scalar(inputs[name], grammar.max_abs)

        def visit(node, root=False):
            if not root:
                budget.charge("program_node")
            op = node[0]
            if op == "input":
                if node[1] not in values:
                    raise ValueError("undeclared input")
                return values[node[1]]
            if op == "const":
                value = _scalar(node[1], grammar.max_abs)
                if type(value) is int and value not in grammar.constants:
                    raise ValueError("constant outside grammar")
                return value
            if op not in grammar.operators:
                raise ValueError("operator outside grammar")
            args = [visit(c) for c in node[1:]]
            return _primitive(op, args, grammar.max_abs)

        return visit(self.ast, root=True)


@dataclass(frozen=True)
class SearchResult:
    candidate: Program | None
    tried: int
    reason: str


def _partitions(total, arity):
    if arity == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - arity + 2):
        for tail in _partitions(total - first, arity - 1):
            yield (first, *tail)


def synthesize(examples, grammar: Grammar, budget: WorkBudget) -> SearchResult:
    """Find first exact proposal match; never certifies generalization.

    Inputs have a fixed name schema and consistent per-name types across rows.
    Generated candidates (including rejected ones), comparison work, and every
    primitive/terminal evaluation and cached child read are debited before work.
    Paid child output vectors are reused; public Program evaluation is unchanged.
    Search is bounded and incomplete.
    Reasons: found, no_examples, exhausted, candidate_limit, budget_exhausted.
    Invalid proposal data raises ValueError after its validation debit.
    """
    tried = 0
    persistence_failure = False

    def charge(kind, units=1):
        # A BudgetStop from a durable sink follows a successful debit. It is
        # a persistence failure, not an ordinary refusal for insufficient work.
        nonlocal persistence_failure
        before = budget.spent
        try:
            budget.charge(kind, units)
        except BudgetStop:
            persistence_failure = budget.spent != before
            raise

    try:
        rows = []
        schema = None
        target_type = None
        for example in examples:
            charge("search_example")
            if type(example) not in (tuple, list) or len(example) != 2:
                raise ValueError("expected input/label pair")
            inputs, expected = example
            if type(inputs) is not dict or set(inputs) != set(grammar.inputs):
                raise ValueError("proposal input schema mismatch")
            copied = {}
            for name in grammar.inputs:
                charge("search_input")
                copied[name] = _scalar(inputs[name], grammar.max_abs)
            _scalar(expected, grammar.max_abs)
            types = tuple(type(copied[n]) for n in grammar.inputs)
            if schema is not None and (schema != types or target_type is not type(expected)):
                raise ValueError("inconsistent proposal types")
            schema, target_type = types, type(expected)
            rows.append((copied, expected))
        if not rows:
            return SearchResult(None, tried, "no_examples")
        pools = {}
        seen = set()

        def consider(ast, *, charged=False, children=None):
            nonlocal tried
            if not charged:
                if tried >= grammar.max_candidates:
                    return SearchResult(None, tried, "candidate_limit")
                charge("search_generate")
                tried += 1
            p = Program(ast)
            if p.depth > grammar.max_depth or p.nodes > grammar.max_nodes:
                return None
            outputs = []
            matches = True
            for row_index, (inputs, expected) in enumerate(rows):
                # The proposals were validated once above. Each new root
                # does exactly one primitive/terminal operation per row.
                charge("program_node")
                if children is None:
                    if ast[0] == "input":
                        value = inputs[ast[1]]
                    else:
                        try:
                            value = _scalar(ast[1], grammar.max_abs)
                        except ValueError:
                            return None
                else:
                    args = []
                    for child in children:
                        charge("search_cache_lookup")
                        args.append(child[2][row_index][1])
                    try:
                        value = _primitive(ast[0], args, grammar.max_abs)
                    except ValueError:
                        return None
                charge("search_compare")
                matches = matches and type(value) is type(expected) and value == expected
                # No permitted parent can consume a child of maximum size.
                # A mismatching terminal-size candidate needs no output vector;
                # matching candidates must still visit every proposal row.
                if p.nodes == grammar.max_nodes and not matches:
                    return None
                outputs.append((type(value).__name__, value))
            if matches:
                return SearchResult(p, tried, "found")
            charge("search_compare", len(outputs))
            signature = tuple(outputs)
            if signature not in seen:
                seen.add(signature)
                pools.setdefault(p.nodes, []).append((p, type(outputs[0][1]), signature))
            return None

        terminals = chain(
            (("input", n) for n in grammar.inputs),
            (("const", c) for c in grammar.constants),
            (("const", False), ("const", True)),
        )
        for ast in terminals:
            result = consider(ast)
            if result is not None:
                return result
        for size in range(2, grammar.max_nodes + 1):
            for op in grammar.operators:
                for sizes in _partitions(size - 1, _ARITY[op]):
                    charge("search_partition")
                    for entries in product(*(pools.get(n, ()) for n in sizes)):
                        if tried >= grammar.max_candidates:
                            return SearchResult(None, tried, "candidate_limit")
                        charge("search_generate")
                        tried += 1
                        types = tuple(entry[1] for entry in entries)
                        invalid = (
                            (op in _NUMERIC and any(t is not int for t in types))
                            or (op in ("and", "or", "not") and any(t is not bool for t in types))
                            or (op == "eq" and types[0] is not types[1])
                            or (op == "if" and (types[0] is not bool or types[1] is not types[2]))
                        )
                        if invalid:
                            continue
                        result = consider((op, *(entry[0].ast for entry in entries)), charged=True, children=entries)
                        if result is not None:
                            return result
        return SearchResult(None, tried, "exhausted")
    except BudgetStop:
        if persistence_failure:
            raise
        return SearchResult(None, tried, "budget_exhausted")


__all__ = ["Grammar", "Program", "SearchResult", "synthesize"]
