"""Rules over relations (experimental): programs over the relations of a state — a path tree, a family forest, the
cells around a player — found by enumeration and judged by a held-out test.

    from solvi.experimental.relations.library import HeldOut, find_rules

    r = find_rules(worlds, targets, base=["PARENT", "IS_DIR", "NAME=secret"], size=6, budget=50_000, library=(),
                   test=HeldOut(other_worlds, targets))
    r.found["denied"].program            # e.g. (under_secret ∩ depth3), over earlier finds in r.library

`library`: composites by enumeration with a library — `find_rules` enumerates small programs bottom-up (observational
equivalence on the examples), admits only what passes the held-out test, and enumerates again with the admitted rules as
new terminals, so a rule three levels deep is a small program over earlier finds. A learned proposer has a slot
(`proposer=`) as a fallback when the budget runs out; none ships.

Measured on two toys (benchmarks/relations_library_toy.py, five seeds each): see the `relations` entry of
solvi.experimental.STATUS.

What is not done:
- the intermediate concepts are given as targets; find_rules finds the order and the programs, not the concepts;
- the enumerator inside `library` is a minimal one, to be replaced by the full relation enumerator;
- no stand beyond the toys (a file server's path policy, a game simulator's cells)."""
from __future__ import annotations

from .. import warn_on_import
from .library import HeldOut, Library, RuleProposer, Rules, TargetRecord, find_rules

warn_on_import(__name__)

__all__ = ["HeldOut", "Library", "RuleProposer", "Rules", "TargetRecord", "find_rules"]
