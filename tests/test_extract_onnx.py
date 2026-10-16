"""LongSpanExtractor's ONNX backend (solvi[onnx]: onnxruntime + tokenizers, no torch) on a tiny exported checkpoint
(tests/fixtures/extract_onnx_tiny: a word-level tokenizer and a lookup-table "encoder" with the published export's
interface), and on the published solvi-ai/extract-base when it is downloaded."""
import os
import shutil
import sys
from pathlib import Path

import pytest

from solvi import Catalog, Question, System
from solvi.core.extract import LongSpanExtractor

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
TINY = str(Path(__file__).parent / "fixtures" / "extract_onnx_tiny")


def tiny(**kw):
    return LongSpanExtractor.load(TINY, backend="onnx", **kw)


def test_the_onnx_backend_quotes_a_field_at_its_characters_and_says_no_answer_below_the_threshold():
    ex = tiny()
    assert (ex.backend, ex.max_len, ex.stride, ex.max_span, ex.thr_default, ex.thr) == ("onnx", 32, 8, 16, 0.5, {"iban": 0.9})
    doc = "invoice 7 TOTAL due 42 EUR thanks"
    s, e, sc, null = ex.predict(doc, "the total amount")
    assert doc[s:e] == "TOTAL due 42 EUR" and 0.5 < sc <= 1.0 and 0.0 < null < sc
    q = ex.field("total", "the total amount")(doc)
    assert (q.value, doc[q.start:q.end]) == ("TOTAL due 42 EUR", "TOTAL due 42 EUR") and q.confidence == pytest.approx(sc)
    assert ex.field("iban", "the IBAN")("invoice 7 thanks").value == ""          # no TOTAL: [CLS] wins, below 0.5


def test_an_answer_in_a_late_window_of_a_long_document_comes_back_at_its_characters():
    words = [f"w{i}" for i in range(200)]
    doc = " ".join(words[:150]) + " TOTAL 15.50 EUR " + " ".join(words[150:])
    ex = tiny()
    w = ex._windows("the total", doc)
    assert len(w["input_ids"]) > 5 and {len(x) for x in w["input_ids"]} == {32}
    q = ex.field("total", "the total")(doc)
    assert (q.value, doc[q.start:q.end]) == ("TOTAL 15.50 EUR", "TOTAL 15.50 EUR") and 0.5 < q.confidence <= 1.0


def test_the_tokenizer_gives_ids_offsets_and_special_tokens_of_tokenizer_json():
    ex = tiny()
    t = ex.tok("the  TOTAL", return_offsets_mapping=True)
    assert t["offset_mapping"] == [(0, 3), (5, 10)] and len(t["input_ids"]) == 2
    assert (ex.tok.cls_token_id, ex.tok.sep_token_id, ex.tok.pad_token_id) == (2, 3, 0)


def test_a_trace_records_the_onnx_model_and_replays():
    ex = tiny()
    fp = ex.fingerprint()
    assert fp == tiny().fingerprint() and isinstance(fp, str)
    cat = Catalog()
    cat.extract(ex.field("total", "the total amount"))

    @cat.rule("found")
    def found(total) -> bool:
        return bool(total)
    res = System(cat, [Question("found", "", None)]).ask({"doc": "invoice TOTAL 42 EUR"})
    assert res["found"].answer == "yes" and res.trace.replay(cat)["ok"]


def test_training_and_embeddings_need_the_torch_backend():
    ex = tiny()
    for call in (lambda: ex.fit([("a TOTAL b EUR", "t", (2, 13))]), lambda: ex.save("x"), lambda: ex.embed("a b")):
        with pytest.raises(RuntimeError, match=r"needs the torch backend .*backend='onnx'"):
            call()


def test_auto_takes_onnx_when_torch_is_absent_and_errors_name_what_is_missing(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='backend must be "torch", "onnx" or "auto", not \'tf\''):
        LongSpanExtractor.load(TINY, backend="tf")
    bare = tmp_path / "no_onnx"
    shutil.copytree(TINY, bare)
    shutil.rmtree(bare / "onnx")
    with pytest.raises(FileNotFoundError, match=r"no ONNX export in .*tools/export_onnx.py"):
        LongSpanExtractor.load(str(bare), backend="onnx")
    monkeypatch.setitem(sys.modules, "torch", None)                # as if it were not installed
    assert LongSpanExtractor.load(TINY).backend == "onnx"
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r'needs a runtime: pip install "solvi\[model\]" \(torch\) or "solvi\[onnx\]"'):
        LongSpanExtractor.load(TINY)
    with pytest.raises(ImportError, match=r"needs onnxruntime: pip install 'solvi\[onnx\]'"):
        LongSpanExtractor.load(TINY, backend="onnx")


def _published():
    from solvi import models
    p = models.cached_path("solvi-ai/extract-base")
    return str(p) if p is not None and os.path.isdir(os.path.join(str(p), "onnx")) else None


@pytest.mark.model
def test_the_published_extractor_on_onnx_if_present():
    path = _published()
    if path is None:
        pytest.skip("solvi-ai/extract-base is not downloaded with its onnx/ folder")
    ex = LongSpanExtractor.load(path, backend="onnx")
    doc = "ACME Store\n12 Main Street\nCoffee 3.50\nCake 4.20\nTOTAL 7.70\nThank you!"
    q = ex.field("total", "the total amount paid")(doc)
    assert "7.70" in q.value and doc[q.start:q.end] == q.value
