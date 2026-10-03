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
    sig = inspect.signature(DecideModel.decision)
    sig = sig.replace(parameters=[p for p in sig.parameters.values() if not p.name.startswith("_")])   # private: not shown
    sig = str(sig).replace("(self, ", "(").replace("'", '"')
    assert f"`model.decision{sig}`" in _flat(GUIDE)


def test_the_guide_lists_every_lifetime_stat():
    para = _flat(GUIDE[GUIDE.index("### Lifetime stats"):GUIDE.index("`system.safeguard_summary()` prints them")])
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
    """29 of 63 modules had no API page, among them solvi.drift, solvi.episode and solvi.worldmap that the
    guide tells users to import from."""
    docs = README + GUIDE + "".join(p.read_text() for p in (ROOT / "docs").glob("*.md"))
    used = set(re.findall(r"from (solvi\.[a-z_]+)(?:\.[a-z_]+)* import", docs))
    used |= {"solvi.drift", "solvi.episode", "solvi.worldmap"}
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
    base = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Combination")
    for name in ("act_guard", "conformal", "decide", "calls", "fit", "teach", "save_calibration", "fingerprint"):
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


def test_every_python_block_in_the_docs_is_python():
    """Two guide blocks were not Python: `res.model_dump("json") / res.to_json()` (a TypeError when pasted) and the
    LangGraph `...compile(` line (a SyntaxError)."""
    import ast
    bad = []
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text()
        for m in re.finditer(r"```python\n(.*?)```", text, re.S):
            try:
                ast.parse(m.group(1))
            except SyntaxError as e:
                bad.append(f"{path.name}:{text[:m.start()].count(chr(10)) + 1}: {e}")
    assert not bad, "\n".join(bad)
    assert 'res.model_dump("json") / res.to_json()' not in GUIDE


def test_best_practices_says_where_its_numbers_come_from_and_is_linked():
    """The page said its scripts "are named in the changelog entries" (one script was, for a number the page does not
    quote) and was reachable only from the site's navigation."""
    page = (ROOT / "docs" / "best_practices.md").read_text()
    assert "named in the changelog" not in page and "benchmarks/" in page
    assert "docs/best_practices.md" in README and "best_practices.md" in GUIDE


def test_benchmarks_says_the_store_overhead_was_measured_in_memory_and_what_a_disk_costs():
    """The documented 2.45 ms per ask with SQLite holds only for a store on tmpfs (the script's temporary folder); on a
    disk it is 15-19 ms. The page also counted twelve gallery entries where the script runs fifteen."""
    page = (ROOT / "docs" / "benchmarks.md").read_text()
    assert "RAM disk (tmpfs)" in page and "NVMe disk" in page and "twelve gallery entries" not in page
    assert len([p for p in (ROOT / "gallery").iterdir() if (p / "cases.json").exists()]) == 15


def test_the_readme_speed_table_names_its_version_and_machine_and_the_run_all_column():
    """The table was the v0.1.0 measurement, never re-measured, without version or machine; the parts it runs are
    trivial, which the reader could not see."""
    speed = README.split("## Speed")[1].split("## ")[0]
    assert "0.7.1" in speed and "i7-12700H" in speed and "run all, no plan" in speed
    assert "| 10 000 | 6 ms |" not in speed and "about 6 ms" not in README


def test_stale_sentences_of_the_docs_stay_corrected():
    """A batch of stale sentences (each checked against the code): "solvi answers yes/no and choice questions",
    "the only command that downloads", decide-* ids, the guide's System signature, the install lists, and `model`
    rebound to a hosted service in the middle of a chapter."""
    import inspect as _inspect

    from solvi import System
    fmt = (ROOT / "docs" / "decide_format.md").read_text()
    assert "solvi answers yes/no and choice questions" not in GUIDE and "decide-*" not in fmt
    assert "proposed `solvi_decide v3`" not in fmt and "model_block*.onnx" in fmt
    for text in (README, GUIDE, (ROOT / "src" / "solvi" / "scaffold.py").read_text()):
        assert "the only command that downloads" not in text
    sig = _flat(GUIDE.split("`System(catalog, questions,")[1].split(")`")[0])
    assert all(f"{p}=" in sig for p in list(_inspect.signature(System).parameters)[2:]), sig
    extras = ["model", "onnx", "serve", "mcp", "otel", "duckdb", "lora"]
    for text in (README, GUIDE):
        assert all(f'"solvi[{e}]"' in text for e in extras)
    assert not re.search(r"^model = systemone\(", GUIDE, re.M)


def test_the_guides_printed_samples_of_examples_12_and_21_are_what_they_print(tmp_path):
    """The audit sample of example 12 lacked its guarantee line, and the chart sample's trace hash was stale."""
    import subprocess
    import sys

    def run(name, *args):
        return subprocess.run([sys.executable, str(ROOT / "examples" / name), *args], capture_output=True, text=True,
                              check=True, cwd=tmp_path).stdout
    out12 = run("12_grounded_audit.py")
    start = GUIDE.index("approve = 'yes'  [ok]  confidence 0.60  ← computed by approve\n")
    sample = GUIDE[start:GUIDE.index("```", start)]
    assert sample in out12 + "\n"
    out21 = run("21_verified_chart.py", str(tmp_path))
    line = next(x for x in GUIDE.splitlines() if x.startswith("charts 1: rendered · trace "))
    assert line in out21.splitlines()


def test_the_readme_shows_the_published_extractor_and_says_what_it_does_without_labels():
    """The README advertised extraction "by description" and said it "does not work yet" without relating the two,
    and its code block used MultiSpanExtractor, which cannot be saved and is used by nothing."""
    block = README.split("## Extract from documents")[1].split("## ")[0]
    assert "LongSpanExtractor.load(\"solvi-ai/extract-base\")" in block and "ex.save(" in block
    assert "from solvi.extract_multi import MultiSpanExtractor" not in block and "46.5%" in block
    assert "does not work yet" not in README
    from solvi.extract_long import LongSpanExtractor
    assert all(hasattr(LongSpanExtractor, n) for n in ("load", "save", "fit", "field", "tune_threshold"))


def test_every_measured_number_names_a_script_or_a_model_card():
    """benchmarks.md said "All numbers come from pre-registered experiments" while most numbers in the README and the
    docs had no script here. Since 0.8 a number either comes from a script in benchmarks/ (or an example) or a model
    card, or it is not in the docs: no page needs to say that its numbers cannot be reproduced."""
    pages = [README, GUIDE] + [p.read_text() for p in (ROOT / "docs").glob("*.md")]
    for text in pages:
        flat = _flat(text)
        assert "not in this repository" not in flat and "cannot be reproduced from it" not in flat
    results = README.split("## Results")[1].split("## ")[0]
    assert "comes from a script in [benchmarks/](benchmarks/) or from a published model card" in _flat(results)
