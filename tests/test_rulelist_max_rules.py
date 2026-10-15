"""RuleList says when max_rules cut the list short: a warning and `truncated`; a list that ends on its own is silent."""
import warnings

import pytest

from solvi.core.deciders.rulelist import RuleList


def _data(k=6, n=5):
    rows, ans = [], []
    for j in range(k):
        for _ in range(n):
            rows.append({"t": f"code{j}"})
            ans.append(f"a{j}")
    return rows, ans


def test_max_rules_hit_warns_and_counts_the_rest():
    rows, ans = _data()
    with pytest.warns(UserWarning, match="max_rules=3"):
        rl = RuleList(["t"], max_rules=3).fit(rows, ans)
    assert len(rl.rules) == 3 and rl.truncated == 15


def test_a_finished_list_is_silent():
    rows, ans = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rl = RuleList(["t"], max_rules=6).fit(rows, ans)
    assert len(rl.rules) == 6 and rl.truncated == 0
    assert rl.predict({"t": "code4"})[0] == "a4"
