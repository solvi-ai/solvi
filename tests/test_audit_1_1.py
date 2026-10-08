"""Reproductions of the audit rows fixed for 1.1 (each test failed before its fix)."""
import warnings

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
    assert "syndrome code alone: sign / locate" in guide
    assert "def store_times" in (root / "benchmarks" / "trace_signature.py").read_text()


# --- row 76 (G8, docs part): the guide says what scan_user does with a field label, and states the measured rates
def test_scan_user_does_not_escalate_a_field_label_and_the_guide_says_so():
    import pathlib
    from solvi.solutions.guard import Guard
    g = Guard(scan_user=True)

    @g.tool(ground=["order"], ground_from=("user",))
    def refund(order: str) -> str:
        return "ok"

    msg = "Model: XPS 13 9310. Please refund order A-10457."
    assert g.check({"name": "refund", "arguments": {"order": "A-10457"}}, [{"role": "user", "content": msg}]).outcome \
        == "allow"
    guide = (pathlib.Path(__file__).parents[1] / "docs" / "guide.md").read_text()
    assert "is escalated too: turn `scan_user` on" not in guide
    assert "flagged **11 of the 12" in guide and "missed 14 of the\n20 injections" in guide


# --- rows 48 (A19) and 135 (H-13): every public module has an API page, in the index and the nav
def test_every_public_module_has_an_api_page():
    import pathlib
    import re
    root = pathlib.Path(__file__).parents[1]
    documented = set()
    for p in (root / "docs" / "api").glob("*.md"):
        documented |= set(re.findall(r"^:::\s*([\w.]+)", p.read_text(), re.M))
    mods = []
    for p in sorted((root / "src" / "solvi").rglob("*.py")):
        parts = list(p.relative_to(root / "src").with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not any(x.startswith("_") for x in parts):
            mods.append(".".join(parts))
    assert [m for m in mods if m not in documented] == []
    index, nav = (root / "docs" / "api" / "index.md").read_text(), (root / "mkdocs.yml").read_text()
    for mod in ("solvi.core.runtime", "solvi.core.types", "solvi.core.provenance", "solvi.core.schema"):
        assert f"[`{mod}`](" in index and f"- {mod}: api/" in nav, mod


def test_answer_yes_no_has_a_docstring_and_a_response_detaches_without_a_private_name():
    from solvi import Answer
    from solvi.core.response import Response
    assert Answer.yes_no.__doc__ and Answer.__doc__
    assert "_system = None" not in Response.__doc__
    cat = Catalog()

    @cat.rule("q")
    def q(x):
        return x > 1

    r = System(cat, [Question("q", "?", Answer.yes_no())]).ask({"x": 2})
    assert r.detach() is r and r._system is None


def test_public_docs_name_no_experiment_codes_in_prose():
    """docs/decide_format.md named formats by experiment codes in prose; the literal format strings stay (in backticks
    or quotes: checkpoints carry them)."""
    import pathlib
    import re
    root = pathlib.Path(__file__).parents[1]
    for page in (root / "docs").glob("*.md"):
        text = re.sub(r"`[^`\n]*`|\"[^\"\n]*\"|'[^'\n]*'|```.*?```", "", page.read_text(), flags=re.S)
        assert not re.search(r"\bl14[a-z]?\b", text, re.I), (page.name, re.search(r".{30}\bl14[a-z]?\b.{30}", text).group())


# --- row 147 (B21): one name, one meaning: the guard kind and the reason prefix have different names
def test_type_rejected_is_one_name_with_one_value():
    from solvi.core import primitives, provenance
    assert getattr(primitives, "TYPE_REJECTED", provenance.TYPE_REJECTED) == provenance.TYPE_REJECTED == "type rejected"
    assert primitives.GUARD_TYPE_REJECTED == "type_rejected"


# --- row 146 (B18): CandidateHead.n counts taught steps; CostBook.get takes the name observe takes
def test_candidate_head_counts_taught_steps_and_costbook_reads_by_name():
    from solvi.core.costs import CostBook
    from solvi.core.deciders.heads import CandidateHead
    steps = [([{"d": 1.0}, {"d": 5.0}], 0), ([{"d": 2.0}, {"d": 9.0}], 0), ([{"d": 7.0}, {"d": 0.5}], 1)]
    h = CandidateHead(["d"]).fit(steps)
    assert h.n == 3
    h.teach([{"d": 1.5}, {"d": 4.0}], 0)
    assert h.n == 4
    book = CostBook()
    book.observe("total", 4.0)
    assert book.get("total") == 4.0 and book.get("other", default=2.0) == 2.0


# --- row 148 (C25): without torch / safetensors, reading an adapter refuses with what to install (runs without torch)
def test_lora_read_without_its_libraries_says_what_to_install(monkeypatch, tmp_path):
    import sys
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental import lora
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(ImportError, match=r'solvi\[lora\]'):
        lora.read(tmp_path / "adapter.safetensors")


# --- row 150 (G10): Guard(lang=) reaches the decision's message; the recorded reasons stay as recorded
def test_guard_lang_translates_the_message_not_the_recorded_reasons():
    from solvi.solutions.guard import Guard
    out = {}
    for lang in ("en", "ru"):
        g = Guard(lang=lang)

        @g.tool(ground=["order"], ground_from=("user",))
        def refund(order: str) -> str:
            return "ok"

        d = g.check({"name": "refund", "arguments": {"order": "B-999"}}, [{"role": "user", "content": "refund A-1"}])
        out[lang] = d
    en, ru = out["en"], out["ru"]
    assert en.outcome == ru.outcome == "deny" and en.reasons == ru.reasons
    assert en.message().startswith("refund denied: ")
    assert ru.message().startswith("refund: отклонено: ") and ru.message() != en.message().replace("refund denied", "")
    assert ru.message(lang="en") == en.message()


# --- row 126 (G14): the playground is checked against the gallery; the realms Space cites only files it has
def test_the_playground_presets_are_in_sync_with_the_gallery():
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("sync_gallery", root / "tools" / "sync_gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.stale() == [], "run: python tools/sync_gallery.py"


def test_the_realms_readme_cites_files_that_exist_and_none_is_tracked_against_gitignore():
    import pathlib
    import re
    import shutil
    import subprocess
    root = pathlib.Path(__file__).parents[1]
    realms = root / "spaces" / "realms"
    for line in (realms / "README.md").read_text().splitlines():
        if line.startswith("uv run"):
            continue                                  # a command's output path
        for ref in re.findall(r"`(results/[^`]+?)`", line):
            ref = re.sub(r"\{(\w+),[^}]*\}", r"\1", ref.replace("<seed>", "1"))     # one instance of a pattern
            assert (realms / ref).exists(), ref
    if shutil.which("git") and (root / ".git").exists():
        out = subprocess.run(["git", "ls-files", "-ci", "--exclude-standard"], cwd=root, capture_output=True, text=True)
        assert out.returncode != 0 or out.stdout.strip() == ""


# --- rows 92 (M2) / 118 (D-22): no module imports a private name of another subpackage where a public one exists
def test_charts_and_the_planners_import_public_names():
    import pathlib
    root = pathlib.Path(__file__).parents[1] / "src" / "solvi"
    charts = (root / "experimental" / "charts" / "check.py").read_text()
    assert "from ...core.textin import _" not in charts and "core.textin import NUMBER_WORDS" in charts
    for f in ("core/plan/cost.py", "core/runtime.py", "experimental/counterfactual.py", "experimental/learning.py"):
        assert "_group_func" not in (root / f).read_text(), f
    from solvi.core import textin
    from solvi.core.catalog import group_func  # noqa: F401
    assert {"NUMBER_WORDS", "NUMBER_SCALES", "CURRENCY_AFTER", "digits"} <= set(textin.__all__)
    assert textin.digits("1,250.50") == textin._digits("1,250.50")
