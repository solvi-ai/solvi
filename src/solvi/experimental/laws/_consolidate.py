"""Transactional, opt-in LawSet compaction; ordinary admission remains in _core.test."""
from copy import deepcopy

from ._core import _key


def consolidate(owner, candidate, buffer, exclude, *, commit):
    c = candidate
    if not owner.grammar.allows(c.condition):
        return None
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
    accepted = trial.test([c], evidence, exclude)
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
