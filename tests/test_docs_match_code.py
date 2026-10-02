"""Sentences of the README and the guide that restate the code — a signature, a list of keys, what a published checkpoint
does — checked against the code, so they cannot drift apart again."""
import inspect
import json
import re
from pathlib import Path

from solvi.audit import STATS
from solvi.decide import DecideModel

ROOT = Path(__file__).resolve().parents[1]
GUIDE = (ROOT / "docs" / "guide.md").read_text()
README = (ROOT / "README.md").read_text()


def _flat(text):
    return re.sub(r"\s+", " ", text)


def test_the_guide_gives_the_whole_signature_of_model_decision():
    sig = str(inspect.signature(DecideModel.decision)).replace("(self, ", "(").replace("'", '"')
    assert f"`model.decision{sig}`" in _flat(GUIDE)


def test_the_guide_lists_every_lifetime_stat():
    para = _flat(GUIDE[GUIDE.index("### Lifetime stats"):GUIDE.index("`system.safeguard_report()` prints them")])
    assert [k for k in STATS if f"`{k}`" not in para] == []


def test_the_readme_does_not_promise_one_forward_pass_for_the_published_checkpoints():
    """Both published checkpoints declare their multi-question pass and ship it switched off."""
    flat = _flat(README)
    assert "several in one forward pass when the checkpoint can" not in flat and "four answers from one forward pass" not in flat
    assert "one question per forward pass with the published checkpoints" in flat and "multi_question=True" in flat
    assert "the shared pass, what escalated" not in flat           # the audit of that snippet prints no shared pass
    example = (ROOT / "examples" / "15_typed_decisions.py").read_text()
    assert "model.batchable" in example                              # the heading says what the model does
    assert json.dumps("four answers from one forward pass")[1:-1] not in (ROOT / "examples" / "README.md").read_text()


def test_decide_format_says_what_a_bare_solvi_decide_v3_gets():
    """docs/decide_format.md promised the answer-primitives defaults for "format": "solvi_decide v3"; the code gives them
    with the l14g format or subformat only (solvi.llm and solvi.systemone declare v3 and their own fields, and their
    fingerprints rest on that)."""
    from solvi.decide import capabilities
    bare = capabilities({"format": "solvi_decide v3"})
    assert bare["modes"] == ["single", "multi"] and bare["unknown"] is None and bare["pointer"] is None and bare["act"] is None
    typed = capabilities({"format": "solvi_decide v3", "subformat": "l14g typed v2"})
    assert "span" in typed["modes"] and typed["unknown"]["column"] == 3 and typed["pointer"]["start"] == 4
    doc = _flat((ROOT / "docs" / "decide_format.md").read_text())
    assert "| `solvi_decide v3` (without that subformat) |" in doc and "no \"not stated\", no pointer, no act head unless declared" in doc
    assert "(also `format` `l14g typed v2` or `solvi_decide v3`)" not in doc


def test_every_module_the_docs_import_from_has_an_api_page_in_the_index_and_the_nav():
    """29 of 63 modules had no API page, among them solvi.drift, solvi.many, solvi.episode and solvi.worldmap that the
    guide tells users to import from."""
    docs = README + GUIDE + "".join(p.read_text() for p in (ROOT / "docs").glob("*.md"))
    used = set(re.findall(r"from (solvi\.[a-z_]+)(?:\.[a-z_]+)* import", docs))
    used |= {"solvi.drift", "solvi.many", "solvi.episode", "solvi.worldmap"}
    index = (ROOT / "docs" / "api" / "index.md").read_text()
    nav = (ROOT / "mkdocs.yml").read_text()
    for mod in sorted(used):
        page = ROOT / "docs" / "api" / f"{mod.split('.')[1]}.md"
        assert page.exists() and f"::: {mod}" in page.read_text(), mod
        assert f"[`{mod}`]({page.name})" in index and f"{mod}: api/{page.name}" in nav, mod
    assert "[`solvi.lora`](lora.md)" in index


def test_the_api_pages_render_what_the_guide_tells_users_to_call_on_combinations_and_memories():
    """act_guard, conformal, decide, fit ... of Cascade / Vote / Route live on a private base and rendered nowhere;
    a memory's save / load had no docstring, so the page left them out."""
    import ast
    assert "inherited_members: true" in (ROOT / "docs" / "api" / "multi.md").read_text()
    tree = ast.parse((ROOT / "src" / "solvi" / "multi.py").read_text())
    base = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "_Combination")
    for name in ("act_guard", "conformal", "decide", "usage", "fit", "teach", "save_calibration", "fingerprint"):
        f = next(f for f in base.body if isinstance(f, ast.FunctionDef) and f.name == name)
        assert ast.get_docstring(f), name
    from solvi.memory import CorrectionMemory
    for name in ("save", "load", "load_dict", "to_dict", "apply"):
        assert getattr(CorrectionMemory, name).__doc__, name


def test_the_llm_docs_give_the_measured_advice_not_the_llm_as_the_last_stage_of_a_cascade():
    """solvi.llm's docstring and the guide's first LLM sample recommended "the LLM as the last, most expensive stage"
    of a Cascade, against the guide's own measurement ("Do not make 'a small model first, the LLM second' the default")."""
    import solvi.llm
    assert "last" not in solvi.llm.__doc__.split("Cost and latency")[1] and "most expensive stage" not in solvi.llm.__doc__
    assert "Cascade([small, large, part])" not in GUIDE
