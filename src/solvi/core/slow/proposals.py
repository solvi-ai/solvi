"""One count of what a proposer offered and what the checks did with it, named the same wherever a model proposes and
code checks: laws (`solvi.experimental.laws.LawSet`), the refine loop (`Refinement.proposals`) and compiled
specifications (`Compiled.proposals`, solvi.experimental.compile).

    proposed = dropped + refuted + admitted

dropped: refused before any check could judge it (laws: a condition the grammar refuses; refine: a reply the generator
rejected, or a proposer that failed; compile: a draft with no module, one that breaks the module contract, or one the
sandbox could not run); refuted: a check judged it and said no (laws: a candidate proposed for a mismatch and not
accepted; refine: a round not accepted; compile: a draft that ran and was sent back — tests, examples, a disagreement,
an abstention); admitted: it passed (laws: accepted; refine: the accepted round; compile: a draft that passed every
check in its round). `proposals_of(x)` reads it off any of the three; the share the checks refused is
`(dropped + refuted) / proposed`."""
from __future__ import annotations

from dataclasses import asdict, dataclass

_REFUSED_BEFORE_RUN = ("contract", "no module", "sandbox")


@dataclass(frozen=True)
class Proposals:
    """proposed = dropped + refuted + admitted (see the module docs)."""
    proposed: int
    dropped: int
    refuted: int
    admitted: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def refused(self) -> float | None:
        """The share of the proposals the checks refused (dropped or refuted); None with no proposal."""
        return (self.dropped + self.refuted) / self.proposed if self.proposed else None

    def text(self) -> str:
        return (f"{self.proposed} proposed: {self.dropped} dropped before a check, {self.refuted} refuted by a check, "
                f"{self.admitted} admitted")


def _make(dropped, refuted, admitted) -> Proposals:
    d, r, a = int(dropped), int(refuted), int(admitted)
    return Proposals(d + r + a, d, r, a)


def of_rounds(rounds) -> Proposals:
    """A refine loop's rounds (solvi.core.slow.refine.Round)."""
    d = r = a = 0
    for rd in rounds:
        if rd.accepted:
            a += 1
        elif rd.error is not None:
            d += 1
        else:
            r += 1
    return _make(d, r, a)


def of_compile_record(record: dict) -> Proposals:
    """A compilation's record (Compiled.record): its drafts, round by round."""
    d = r = a = 0
    for rnd in record.get("rounds") or ():
        for dr in rnd.get("drafts") or ():
            caught = list(dr.get("caught") or ())
            if any(c in _REFUSED_BEFORE_RUN for c in caught):
                d += 1
            elif caught or dr.get("feedback"):
                r += 1
            else:
                a += 1
    return _make(d, r, a)


def of_lawset(lawset) -> Proposals:
    """A law set: its proposer's grammar drops (`dropped`, CallableProposer) and its search-cost record — candidates
    proposed for mismatches, accepted (probes and hand-added laws included)."""
    cost = lawset.cost
    d = int(getattr(lawset.proposer, "dropped", 0) or 0)
    return _make(d, max(0, int(cost.get("candidates", 0)) - int(cost.get("accepted", 0))), int(cost.get("accepted", 0)))


def proposals_of(x) -> Proposals:
    """The count of a LawSet, a Refinement or a Compiled (anything with a `proposals` attribute, or a law set's
    `cost` and `proposer`)."""
    got = getattr(x, "proposals", None)
    if isinstance(got, Proposals):
        return got
    if isinstance(getattr(x, "cost", None), dict) and hasattr(x, "proposer"):
        return of_lawset(x)
    if hasattr(x, "rounds") and hasattr(x, "question"):
        return of_rounds(x.rounds)
    if isinstance(getattr(x, "record", None), dict):
        return of_compile_record(x.record)
    raise TypeError(f"no proposal count for {type(x).__name__}: a LawSet, a Refinement or a Compiled")


__all__ = ["Proposals", "proposals_of"]
