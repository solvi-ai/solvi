"""Reproductions of the audit rows fixed for 1.1 (each test failed before its fix)."""
import warnings
from datetime import date

import pytest

from solvi import Answer, Catalog, Question, Quote, System
from solvi.core.provenance import matches


class _M:
    version = "1"


def _quoting(val, text="hello world"):
    cat = Catalog()

    @cat.extract(model=_M())
    def v(doc):
        return Quote(val, 0, 5)

    @cat.rule("q")
    def q(v):
        return bool(v)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return System(cat, [Question("q", "?", Answer.yes_no())]).ask({"doc": text})


# --- row 44 (A7): a model quote holding a bool is compared with the text
def test_a_bool_quote_over_unrelated_text_is_not_grounded():
    r = _quoting(True)
    assert r["q"].status == "abstain"
    assert any(s["kind"] == "grounding" for s in r.safeguards)


def test_a_bool_quote_over_its_word_is_grounded():
    r = _quoting(True, "yes, paid in full")
    assert r["q"].status == "ok" and r["q"].answer == "yes"
    assert _quoting(False, "No.  ")["q"].status == "ok"


def test_bool_matching_words():
    assert matches(True, "Yes") and matches(True, " true ") and matches(False, "нет")
    assert matches(True, "no") is False and matches(False, "hello") is False
    np = pytest.importorskip("numpy")
    assert matches(np.bool_(True), "yes") and matches(np.bool_(False), "yes") is False
