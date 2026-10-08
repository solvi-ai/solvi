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


# --- row 127 (E10): Question(uses=[...]) that does nothing is reported by solvi check
def test_uses_on_a_question_with_a_rule_is_reported():
    from solvi.check import lint
    cat = Catalog()

    @cat.rule("approve")
    def approve(amount):
        return "approve"

    s = System(cat, [Question("approve", "?", Answer.choice(["approve", "reject"]), uses=["nope"])])
    assert "uses_ignored" in lint(s).codes()


def test_uses_naming_nothing_is_a_note_without_a_rule():
    from solvi.check import lint
    cat = Catalog()

    @cat.fn
    def total(amount):
        return amount

    s = System(cat, [Question("risk", "?", Answer.choice(["low", "high"]), uses=["totl"])])
    rep = lint(s)
    assert "uses_unknown" in rep.codes() and "uses_ignored" not in rep.codes()


# --- row 73 (F3): a store path that cannot be a store is a usage error (status 2, one line), not a traceback
def _cli(capsys, *argv):
    from solvi.cli import main
    capsys.readouterr()
    try:
        code = main([str(a) for a in argv])
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().err


@pytest.mark.parametrize("store", ["nodir/sub/d.db", "nodir/d.jsonl"])
def test_ask_store_into_a_missing_directory_is_a_usage_error(tmp_path, capsys, store):
    from solvi.cli import main
    d = tmp_path / "app"
    assert main(["init", str(d), "--template", "support"]) == 0
    code, err = _cli(capsys, "ask", f"{d}/catalog.py:system", d / "example.json", "--store", tmp_path / store)
    assert code == 2 and "does not exist" in err and "Traceback" not in err


def test_verify_on_a_directory_is_a_usage_error(tmp_path, capsys):
    code, err = _cli(capsys, "verify", tmp_path)
    assert code == 2 and "is a directory, not a store" in err


# --- row 83 (E19): the signature timing says what it times, and gives the time through a real store
def test_signature_timing_names_items_and_store():
    import pathlib
    root = pathlib.Path(__file__).parents[1]
    guide = (root / "docs" / "guide.md").read_text()
    assert "| Stored decisions | SQLite: `store.signature()` / `verify(signature=)`" in guide
    assert "code alone over 40-byte items" in guide
    assert "def store_times" in (root / "benchmarks" / "trace_signature.py").read_text()
