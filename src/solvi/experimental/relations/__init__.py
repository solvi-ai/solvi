"""Relations over a state and rules found over them (experimental): a relation-algebra enumerator proposes relational
atoms, the held-out test of a law set judges them, and a found rule runs as plain set code after a differential check.

    from solvi.experimental.relations import (Relations, RelAlgProposer, PathTree, Library, compile_program,
                                              check_program, enumerate_programs, synthesize, top_programs, show)

    r = Relations(["PARENT", "NAME=secret"], 4)               # one state: nodes 0..3, packed-bit matrices
    r.update("PARENT", [(1, 2), (2, 0), (3, 1)]); r.mark("NAME=secret", [1])
    ancestor_secret = ("COMP", ("PLUS", ("T", ("R", "PARENT"))), ("R", "NAME=secret"))
    show(ancestor_secret)                                     # '((PARENTᵀ)⁺ ∘ NAME=secret)'
    found = synthesize(examples, targets, size=7)             # the first program equal to the targets on every example
    top = top_programs(examples, y, size=4, k=20)             # ranked by the readout's precision; noise-tolerant
    lib = Library(); lib.add(("PLUS", ("T", ("R", "PARENT"))), "ancestor")    # a terminal of later searches
    fn = compile_program(ancestor_secret)                     # plain set code: fn(r.sets(), r.n) → {(x, y), ...}
    check_program(ancestor_secret, worlds).differences        # 0: compiled == interpreted, pair for pair
    laws = LawSet(grammar, base=...)
    laws.proposer = RelAlgProposer(grammar, k=20, encoder=PathTree("path", names=("secret",)), rows=held_out)

Relations. `Relations(names, n)`: named binary relations over nodes 0..n−1 of one state, stored as packed bits
(uint8 (K, n, ⌈n/8⌉)); a unary predicate is a diagonal relation (`mark`). Node 0 is the subject (the player, the path
acted on). The algebra on boolean matrices: `compose(a, b, k=1)` (at least k paths), `converse`, `union`,
`intersect`, `minus`, `complement(a, mask)` (bounded by the real pairs), `closure` (transitive, to a fixed point),
`count_at_least(a, k)` (a unary: the nodes with at least k successors).

Programs are tuples: terminals ("R", name), ("I",), ("U",) (every real pair), ("L", name) (a library entry); unary
("T", x) converse, ("NOT", x), ("PLUS", x) closure, ("CNT2", x) / ("CNT3", x); binary ("OR" | "AND" | "MINUS" | "XOR"
| "COMP" | "C2" | "C3", a, b), C2 / C3 = at least 2 / 3 paths. `show`, `size` (nodes), `evaluate(ast, relations)`,
`readout(value)` — a relation as a state bit: does node 0 relate to anything.

The enumerator. `enumerate_programs(examples, size, library=(), ops=...)` builds programs bottom-up by size and keeps
one per behaviour: two programs are equivalent when their full relations on every example are equal (the packed bits,
a 16-byte hash) — never when only a readout agrees, which would merge relations that differ off node 0 and lose what
composes from them. Deterministic (a fixed order); caps on behaviours, evaluations or seconds. `synthesize(examples,
target, size)` returns the first program equal to the target on every example (exact). `score(ast, examples, y)` is
the support and precision of the readout against target bits; `top_programs` / `rank` score every behaviour (no stop
at the first match) and keep the top k by the lower bound of the precision, at most `per_readout` per readout pattern:
labels with noise still rank the rule first.

The library. `Library`: accepted programs (`Entry(program, name, examples, test)` — the examples it was accepted on
and its test record), each a terminal ("L", name) of later searches, so a composite rule is a small program over
earlier ones; `expand` inlines them.

Compiled rules. `compile_ast(ast)` writes plain Python over sets of pairs (an index per relation; composition a join;
closure joins of the new pairs to a fixed point; counts per pair) that imports nothing; `compile_program` executes it.
`backend="arrays"` writes the same over index arrays for large graphs: a relation is a sorted int64 array of keys
x·n + y (`to_keys`, `from_keys`), composition a join by binary search, closure the new pairs joined to a fixed point.
A dense answer — a complement or U outside ∩ / ∖ with a sparse side — is refused (`DenseAnswer`) unless the domain is
declared bounded (`dense=True`; a state's window is). `check_program(ast, worlds)` is the differential check against
the interpreter, pair for pair; the bar is 0 differences.

Encoders and the laws adapter. `encode(state, window=3)` / `Egocentric`: a grid state seen from the player (node 0;
a 3×3 or 5×5 window; EAST, SOUTH, AT, FACE; diagonal MAT_<m> and CRT_<k>); the inventory stays flat atoms.
`encode_paths(root, focus=, names=)` / `PathTree(key, names)`: a path tree (PARENT, SAME_DIR, IS_DIR, NAME=x).
`relational_atom(ast, encoder)` is a laws `Atom("rel:<program>", ..., feature="rel")` whose code is the compiled
program, so a law over it compiles with `LawSet.to_module` and passes `check_compiled`. `RelAlgProposer(grammar,
base=None, k=20, encoder=, rows=)`: for a mismatch, enumerate on the rows of the mismatched transition's own group,
score each behaviour against "the effect predicts this row", and hand the top k as candidates (rel_atom,) + flat atoms
to the law set's unchanged held-out test (other groups only) — the test does the work.

Measured: benchmarks/relations_battery.py (the enumerator on a pre-registered battery of random rules over three
relations) and benchmarks/relations_compile.py (compiled rules on a forest of 100,000 nodes and on 100,000 law rows);
the numbers are in their docstrings and tests/test_relations_gate.py (`pytest -m slow`).

Pitfalls built in. Equivalence holds on the examples only: the program kept for a class is its smallest member and
may not generalise, so every find needs a held-out check on other sizes and families (the proposer's candidates go
through `LawSet.test` on other groups; the battery script re-checks on larger graphs). The order of terminals and the
one-sided commutative operators only choose which equivalent program represents a class. A complement is taken within
each example's real pairs, so padding stays empty. Memory: a behaviour is B·n² bits; keys are 16-byte hashes, never
the bytes, and `max_bytes` caps the store — with many or large examples search a subset and re-check on all
(`synthesize(..., check=(examples, targets))`). With noisy labels compare by agreement (`score`, `top_programs`), never
by the exact target hash; deduplication stays exact. A rule that needs four variables at once (x and y in a common
4-clique) is out of reach, and the battery script's control checks it is never accepted.

What is not done:
- the enumeration grows as (terminals × operators)^size: with the twenty-odd relations of a grid window, programs of
  four or five nodes are the practical depth, and the behaviour cap bounds memory;
- three-variable logic with counting: a rule that needs four variables at once (a 4-clique) is out of reach;
- the proposer enumerates on one group's rows; with few rows many programs share a readout, and only the held-out
  test separates them;
- relational atoms stay in the grammar once proposed (every row evaluates them); nothing prunes unused ones yet;
- the compiled atom of a state's graph allows complements (its domain, the window or the path's chain, is bounded);
  `compile_ast` refuses them by default."""
from __future__ import annotations

from .. import warn_on_import
from ._adapters import (CREATURES, MATERIALS, PATHS, PROPOSER_OPS, Egocentric, PathTree, RelAlgProposer, encode,
                        encode_paths, relational_atom)
from ._algebra import (BINARY, OPS, UNARY, Relations, closure, complement, compose, converse, count_at_least,
                       evaluate, intersect, minus, readout, show, size, union, xor)
from ._compile import (BACKENDS, RUNTIME, RUNTIME_ARRAYS, DenseAnswer, ProgramCheck, check_program, compile_ast,
                       compile_program, from_keys, to_keys)
from ._enumerate import (Entry, Enumerator, Found, Library, Program, Score, Scored, enumerate_programs, lower_bound,
                         rank, score, synthesize, top_programs)

warn_on_import(__name__)

__all__ = ["BACKENDS", "BINARY", "CREATURES", "DenseAnswer", "Egocentric", "Entry", "Enumerator", "Found", "Library", "MATERIALS",
           "OPS", "PATHS", "PROPOSER_OPS", "PathTree", "Program", "ProgramCheck", "RUNTIME", "RUNTIME_ARRAYS", "RelAlgProposer",
           "Relations", "Score", "Scored", "UNARY", "check_program", "closure", "compile_ast", "compile_program",
           "complement", "compose", "converse", "count_at_least", "encode", "encode_paths", "enumerate_programs",
           "evaluate", "from_keys", "intersect", "lower_bound", "minus", "rank", "readout", "relational_atom", "score",
           "show", "size", "synthesize", "to_keys", "top_programs", "union", "xor"]
