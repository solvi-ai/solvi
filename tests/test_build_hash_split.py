"""build(split="hash"): each example's share is set by a hash of its state (salted by seed=), not by its place in a
shuffled list — adding examples keeps every old example in its share, and the order of the examples does not matter.
The default (the seeded shuffle) is unchanged."""

import pytest

from solvi import build
from test_auto import WORDS, _learner, _texts


def _with_ids(ex, start=0):
    return [({**st, "id": start + i}, y) for i, (st, y) in enumerate(ex)]


def _fit_ids(ex, **kw):
    seen = []

    def learner(fit):
        if not seen:
            seen.append({st["id"] for st, _ in fit})
        return _learner(fit)
    ds = build("topic", ex, learner=learner, max_error=0.10, novel=False, **kw)
    return seen[0], ds


def test_added_examples_leave_every_old_example_in_its_share():
    old = _with_ids(_texts(600, WORDS))
    new = old + _with_ids(_texts(60, WORDS, seed=5), start=600)
    fit_old, ds_old = _fit_ids(old, split="hash")
    fit_new, ds_new = _fit_ids(new, split="hash")
    assert fit_new & set(range(600)) == fit_old                 # no old example moved into or out of the fit share
    assert "hash" in ds_new.choices["split_by"]
    # the seeded shuffle moves old examples between shares when examples are added
    rnd_old, _ = _fit_ids(old)
    rnd_new, _ = _fit_ids(new)
    assert rnd_new & set(range(600)) != rnd_old


def test_the_order_of_the_examples_does_not_matter_with_the_hash_split():
    ex = _with_ids(_texts(600, WORDS))
    rev = list(reversed(ex))
    a, ds_a = _fit_ids(ex, split="hash")
    b, ds_b = _fit_ids(rev, split="hash")
    assert a == b and ds_a.choices["split"] == ds_b.choices["split"]
    probe = [st for st, _ in _texts(80, WORDS, seed=9)]
    assert [ds_a.ask({**st, "id": -1}).by for st in probe] == [ds_b.ask({**st, "id": -1}).by for st in probe]


def test_the_shares_hold_about_their_proportions_and_seed_salts_the_hash():
    ex = _with_ids(_texts(800, WORDS))
    fit0, ds = _fit_ids(ex, split="hash")
    assert abs(len(fit0) / 800 - 0.75) < 0.06
    fit1, _ = _fit_ids(ex, split="hash", seed=1)
    assert fit0 != fit1


def test_split_takes_shares_or_hash_only():
    with pytest.raises(ValueError, match="hash"):
        build("topic", _texts(100, WORDS), learner=_learner, max_error=0.1, split="random")
