"""Where a label may come from: the trusted sources (a person, an outcome, a rule), the "verified" System 2 answer that
only some channels take, and the check every channel runs (check_source). Moved out of solvi.core.store in 1.0, which
re-exports every name, so that the guarantees and the memory of corrections check a label without importing the store."""
from __future__ import annotations


TRUSTED_SOURCES = ("human", "outcome", "rule")   # where a label may come from: never the system's own answers
VERIFIED = "verified"         # a System 2 answer that passed the checks and a guarantee: a label only where accepted
# What a channel says when it refuses a verified label: measured on three stand tasks replayed as a stream (Abt-Buy,
# CUAD, Banking77; docs/best_practices.md), System 2's verified answers fed to it did not make System 1 answer more
# within its promise on at least two of the three.
VERIFIED_REFUSED = {
    "memory": "a memory of corrections fed verified System 2 answers broke System 1's promise on a contract task (P(alone "
              "and wrong) 3.1% against 3%) and helped on one task of three",
    "head": "a head refitted on verified System 2 answers answered more within its promise on one task of three (contracts; "
            "not on product matching, where it already answered 90%) — feed it human or outcome labels",
    "learning": "the learning loop's ladder (fit, memory, adapter) was not shown to gain from verified System 2 answers on "
                "two tasks of three; it learns from human, outcome and rule labels",
}


class UntrustedLabel(ValueError):
    """A label from a source outside TRUSTED_SOURCES (the model, the system itself, an unknown process), or a
    "verified" label offered to a channel that does not take it."""


def check_source(source, accept=(), channel=None):
    """A label's source → itself, when it is trusted: "human", "outcome" or "rule" — or "verified" when the channel
    lists it in `accept` (only System.guarantee(..., sources=) does: see VERIFIED); else UntrustedLabel, with the
    measured reason when a channel (`channel`, a key of VERIFIED_REFUSED) refuses a verified label."""
    if source in TRUSTED_SOURCES or (source == VERIFIED and VERIFIED in tuple(accept)):
        return source
    if source == VERIFIED:
        why = VERIFIED_REFUSED.get(channel, "this channel does not take verified System 2 answers")
        raise UntrustedLabel(f"label source {VERIFIED!r} is not taken here: {why}; a verified label is stored and can "
                             "recalibrate a guarantee (System.guarantee(question, examples, corrections=store, "
                             f"sources=TRUSTED_SOURCES + ({VERIFIED!r},)))")
    raise UntrustedLabel(f"label source {source!r} is not trusted: labels come only from outside the model "
                         f"({', '.join(TRUSTED_SOURCES)}; {VERIFIED!r} for a System 2 answer that passed the checks and "
                         "a guarantee, where a channel takes it); the system's own answers are never labels")


__all__ = ["TRUSTED_SOURCES", "UntrustedLabel", "VERIFIED", "VERIFIED_REFUSED", "check_source"]
