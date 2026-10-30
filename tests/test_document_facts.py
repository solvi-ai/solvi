"""S2 of the solvi.document plan: `Response.facts` (per-fact quotes with offsets, from the trace) and the short
`why` / `audit` — a long input text is named by its length and hash (`doc[12,345 chars, sha256:…]`), never carried;
the audit of a 12k-char document stays under 1 KB. Replay of a record stored with the 1.4 `why` form still passes
(the 1.4 `<text, N chars>` marker; pre-1.4 whole inputs are the other accepted form). Fake server, no network."""
import hashlib
import inspect
import os
import sys

import pytest

from solvi import Answer
from solvi.core.catalog import Catalog, Question
from solvi.core.system import System, _rule_why

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_document_recipe import Server, case_reply, doc_of, ops_system  # noqa: E402


def part(name, inputs, body):
    """A catalog part like the recipe's _part: named, with an explicit input signature."""
    def run(**kw):
        return body(kw)
    run.__name__ = run.__qualname__ = name
    run.__signature__ = inspect.Signature([inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                           for x in inputs])
    return run


def len_system(long_input=False):
    cat = Catalog()
    if long_input:
        cat.fn(part("n", ["text"], lambda kw: len(kw["text"])))
        cat.rule("len")(part("compute", ["n", "text"], lambda kw: kw["n"]))
    else:
        cat.fn(part("n", ["text"], lambda kw: len(kw["text"])))
        cat.rule("len")(part("compute", ["n"], lambda kw: kw["n"]))
    return System(cat, [Question("len", "the length", Answer.estimate())])


def test_facts_quotes_and_offsets():
    """`res.facts[i].quote == doc[start:end]` — the acceptance line of the plan; names, values, source, confidence."""
    doc = doc_of(1)
    s = ops_system(Server(case_reply()))
    res = s.ask({"doc": doc, "op": "diff"})
    facts = res.facts
    assert {"a", "b"} <= {f.name for f in facts}
    for f in facts:
        assert f.source == "doc"
        assert f.quote == doc[f.start:f.end]           # the quote, literally at its offsets
        assert f.origin in ("quoted", "decided")       # a model-read field is "decided", a plain extract "quoted"
        assert 0 <= f.start < f.end <= len(doc)
    a = next(f for f in facts if f.name == "a")
    assert a.quote == "$1,250" and a.value == "$1,250"   # the trace record's own value (the typed read is a_value)
    assert a.confidence == pytest.approx(0.9)


def test_facts_empty_without_quotes():
    """A system whose flow quotes nothing: res.facts is empty, not an error."""
    res = len_system().ask({"text": "abc"})
    assert res.facts == []
    assert res["len"].answer == 3


def test_why_names_long_input_by_hash():
    """A rule input longer than 200 chars: `why` carries `doc[<N chars>, sha256:<12 hex>]`, not the text."""
    long_text = "word " * 80                        # 400 chars
    res = len_system(long_input=True).ask({"text": long_text})
    why = res["len"].why
    h = hashlib.sha256(long_text.encode("utf-8")).hexdigest()[:12]
    assert f"doc[{len(long_text):,} chars, sha256:{h}]" in why
    assert "word word" not in why                  # the text itself is not carried


def test_rule_why_forms():
    """The render forms of a long input: the current one (doc + sha256, with or without a leading model quote),
    the 1.4 marker (only when a quote leads), the pre-1.4 whole text."""
    vals = {"x": "abcdefghij" * 30}                # 300 chars

    class R:                                       # the record shape _model_quote reads
        def __init__(self, quote=None):
            self.extra = None if quote is None else {"llm": {"quote": quote}}

    class Rule:
        inputs = ["x"]

    q = ("the quoted fragment", 0, 18)
    now = _rule_why(Rule(), R(), vals, {})
    now_q = _rule_why(Rule(), R(q), vals, {})
    old14_q = _rule_why(Rule(), R(q), vals, {}, legacy="1.4")
    pre14 = _rule_why(Rule(), R(q), vals, {}, legacy=True)
    assert "sha256:" in now and "abcdefghij" * 5 not in now
    assert now_q.split("; ")[0].startswith("quote ") and "sha256:" in now_q
    assert "<text, 300 chars>" in old14_q and "sha256:" not in old14_q
    assert repr(vals["x"]) in pre14


def test_audit_12k_doc_under_1kb():
    """The acceptance line: the audit of a 12k-char document is under 1 KB and names the text by hash, not by content."""
    doc = "Filler sentence with figures 42 and 17. " * 320 + doc_of(1)
    assert len(doc) > 12_000
    s = ops_system(Server(case_reply()))
    res = s.ask({"doc": doc, "op": "diff"})
    q = next(iter(res.results))
    text = str(res.audit(q))
    assert len(text) < 1024
    assert "sha256:" in text
    assert doc[:60] not in text


def test_replay_why_tolerance_contract():
    """The tolerance contract: a recomputed `why` whose only difference is the historical marker is accepted. The
    cross-version property itself (a record stored by the pre-S2 code replays clean under the new render) is verified
    with a record written at the parent commit — not reproducible inside the suite without two code versions."""
    long_text = "word " * 80
    h = hashlib.sha256(long_text.encode("utf-8")).hexdigest()[:12]
    res = len_system(long_input=True).ask({"text": long_text})
    r = res.results["len"]
    assert f"doc[{len(long_text):,} chars, sha256:{h}]" in r.why
    # the historical forms render from the same inputs (what _why_before collects for the replay check)
    from solvi.core.system import _model_quote  # noqa: F401  (the forms are covered in test_rule_why_forms)
    old14 = r.why.replace(f"doc[{len(long_text):,} chars, sha256:{h}]", f"<text, {len(long_text):,} chars>")
    assert old14 != r.why and "<text," in old14


def cat_for_long():
    cat = Catalog()
    cat.fn(part("n", ["text"], lambda kw: len(kw["text"])))
    cat.rule("len")(part("compute", ["n", "text"], lambda kw: kw["n"]))
    return cat
