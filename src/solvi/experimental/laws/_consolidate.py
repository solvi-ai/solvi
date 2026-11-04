"""Transactional, opt-in LawSet compaction; ordinary admission remains in _core.test."""
from copy import deepcopy

from ._core import Candidate, _key


def _scoped_confirm(owner, confirm, candidate, evidence):
    """Roll back callback writes to the owning book, including writes through existing Law references.

    World/grammar/buffer side effects remain the callback's responsibility. Confirmation is a read-only
    predicate, not a second opportunity to edit the book being transactionally replaced.
    """
    laws = owner.laws + owner.retired + owner.suspended
    # Keep adapters and in-flight world objects in place. Only book containers and Law state are owned here.
    memo = {id(value): value for value in owner.__dict__.values()
            if not isinstance(value, (dict, list, set))}
    memo.update({id(law): law for law in laws})
    memo[id(evidence)] = evidence
    state = deepcopy(owner.__dict__, memo)
    payloads = {id(value): value for law in laws for value in (law.action, law.effect, law.born)}
    law_states = [(law, deepcopy(law.__dict__, payloads.copy())) for law in laws]
    owner._consolidation_confirming = True
    try:
        return confirm(candidate)
    finally:
        for law, saved in law_states:
            law.__dict__.clear()
            law.__dict__.update(saved)
        owner.__dict__.clear()
        owner.__dict__.update(state)


def consolidate(owner, candidate, buffer, exclude, *, commit):
    if getattr(owner, "_consolidation_confirming", False):
        raise RuntimeError("consolidate cannot be re-entered from confirmation")
    c = Candidate(candidate.action, owner.grammar.check(candidate.condition), candidate.effect, candidate.proposer)
    key = (_key(c.action), tuple(c.condition), _key(c.effect))
    if any(owner._ck(law) == key for law in owner.laws):
        return None
    targets = [law for law in owner.laws
               if _key(law.action) == key[0] and _key(law.effect) == key[2]
               and set(c.condition) < set(law.condition)]
    if len(targets) < 2:
        return None
    # Copy mutable state together so memo weakrefs cannot make the trial mutate the owner's book.
    trial = deepcopy(owner)
    trial.grammar = owner.grammar
    # Trial state is private; world adapters and user callbacks keep their identity.
    for name, value in owner.__dict__.items():
        if callable(value):
            setattr(trial, name, value)
    trial._memo, trial._memo_at, trial._memo_buffer = {}, None, None
    evidence = buffer if commit else deepcopy(buffer)
    target_ids = {law.id for law in targets}
    original_net = trial.score(c, evidence, exclude)[2]
    allowed, _, old_ok, _ = owner._view(evidence, c.action, exclude)
    for law in list(trial.laws):
        if law.id in target_ids:
            trial.retire(law, why="pending consolidation")
    trial.support_on = "matched"
    # Falsification is a separate gate, also for books using readmit='any'.
    at = trial._falsified_at.get(key)
    if at is not None and at >= 0 and trial._blocked(at, evidence):
        return None
    if not commit:
        trial.confirm = None
    elif owner.confirm is not None:
        confirm = owner.confirm
        trial.confirm = lambda candidate: _scoped_confirm(owner, confirm, candidate, evidence)
    try:
        accepted = trial.test([c], evidence, exclude)
    finally:
        if commit:
            owner.cost["tests"] = trial.cost["tests"]
            owner._last_tested = trial._last_tested
    if not accepted:
        return None
    replacement = accepted[0]
    retained = {law.id for law in owner.laws if law.id not in target_ids}
    active = {law.id for law in trial.laws}
    if replacement.id not in active or not retained <= active:
        return None
    _, _, new_ok, _ = trial._view(evidence, c.action, exclude)
    if old_ok & allowed & ~new_ok:
        return None
    proof = {"replaced": tuple(law.id for law in targets), "sources": replacement.sources,
             "original_net": original_net, "counterfactual_net": replacement.net,
             "old_correct": (old_ok & allowed).bit_count(),
             "new_correct": (new_ok & allowed).bit_count(), "exclude": exclude,
             "support_on": "matched", "domain_guarantee": False}
    replacement.consolidation = proof
    replacement.admitted = "consolidation"
    for law in trial.retired:
        if law.id in target_ids:
            law.why = f"subsumed by {replacement.id}"
    trial._stamp([law for law in trial.retired if law.id in target_ids], trial._clock(evidence))
    if not commit:
        replacement.status = "proposed"
        return replacement
    # Preserve existing Law identities, including references held by the caller.
    originals = {law.id: law for law in owner.laws + owner.retired + owner.suspended}
    for name in ("laws", "retired", "suspended", "last_retired"):
        remapped = []
        for law in getattr(trial, name):
            existing = originals.get(law.id)
            if existing is not None:
                existing.status, existing.why, existing.held = law.status, law.why, law.held
                law = existing
            remapped.append(law)
        setattr(trial, name, remapped)
    # Do not replace grammar, adapters, a pending transition/proposal or unrelated caller state.
    for name in ("laws", "retired", "suspended", "_seq", "_laws_version", "_retired_at",
                 "cost", "_last_tested", "refused_readmit", "refused_rival", "capped"):
        setattr(owner, name, getattr(trial, name))
    owner._memo, owner._memo_at, owner._memo_buffer = {}, None, None
    owner.last_retired = [originals[law.id] for law in targets]
    return replacement
