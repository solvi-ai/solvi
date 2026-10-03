"""LongSpanExtractor's own arithmetic — windows, the best span, the "no answer" threshold, confidences — with torch and
the tokenizer replaced by small stand-ins (torch is not imported, no model is loaded)."""
import contextlib
import re
import types

import numpy as np
import pytest

from solvi import Catalog, Question, System
from solvi.core.extract import LongSpanExtractor


class T:
    """The few tensor operations predict() uses, over numpy."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, k):
        return T(self.a[k])

    def float(self):
        return T(self.a.astype(float))

    def bool(self):
        return T(self.a.astype(bool))

    def __invert__(self):
        return T(~self.a)

    def masked_fill(self, m, v):
        b = self.a.astype(float).copy()
        b[m.a] = v
        return T(b)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def _softmax(x, dim):
    e = np.exp(x.a - x.a.max(dim, keepdims=True))
    return T(e / e.sum(dim, keepdims=True))


FAKE_TORCH = types.SimpleNamespace(no_grad=contextlib.nullcontext, autocast=lambda *a, **k: contextlib.nullcontext(),
                                   bfloat16=None, tensor=lambda x, device=None: T(x), softmax=_softmax)


class Tok:
    """Whitespace words; offsets include the leading space, as BPE offsets do."""
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs = [], []
        for m in re.finditer(r"\s*\S+", text):
            ids.append(self.vocab.setdefault(m.group().strip(), len(self.vocab) + 10))
            offs.append((m.start(), m.end()))
        return {"input_ids": ids, "offset_mapping": offs}


def extractor(start_word=None, end_word=None, strength=8.0):
    ex = LongSpanExtractor.__new__(LongSpanExtractor)
    ex.torch, ex.device, ex.tok = FAKE_TORCH, "cpu", Tok()
    ex.max_len, ex.stride, ex.max_span, ex.thr, ex.thr_default = 32, 8, 16, {}, 0.5
    ex._cache, ex._tok_key, ex._tok_val, ex._fp_weights, ex.model_name = {}, None, None, "fp", "fake"
    ex.model_id = "fake"
    ex.enc = lambda input_ids, attention_mask: types.SimpleNamespace(last_hidden_state=input_ids)

    def head(h):                                       # start logit on start_word, end logit on end_word, some on [CLS]
        lg = np.zeros(h.a.shape + (2,))
        lg[..., 0][h.a == ex.tok.vocab.get(start_word, -1)] = strength
        lg[..., 1][h.a == ex.tok.vocab.get(end_word, -1)] = strength
        lg[:, 0, :] = 2.0
        return T(lg)
    ex.head = head
    ex.fingerprint = lambda: "fake-fp"
    return ex


def test_a_field_of_an_empty_document_is_not_found_with_a_confidence_inside_0_and_1():
    """The best span started at score -1.0 and the "not found" confidence is 1 - score: an empty document gave
    Quote("", 0, 0, confidence=2.0), and that went into the trace record."""
    ex = extractor("TOTAL", "42")
    for doc in ("", "   "):
        assert ex.predict(doc, "the total")[:3] == (0, 0, 0.0)
        q = ex.field("total", "the total")(doc)
        assert (q.value, q.start, q.end, q.confidence) == ("", 0, 0, 1.0)
    cat = Catalog()
    cat.extract(ex.field("total", "the total"))

    @cat.rule("found")
    def found(total) -> bool:
        return bool(total)
    res = System(cat, [Question("found", "", None)]).ask({"doc": ""})
    assert all(0.0 <= r.confidence <= 1.0 for r in res.trace.records) and res["found"].answer == "no"
    assert res.trace.replay(cat)["ok"]


def test_a_field_is_quoted_with_its_span_score_and_not_found_below_the_threshold():
    ex = extractor("TOTAL", "42")
    doc = "invoice 7 TOTAL due 42 thanks"
    q = ex.field("total", "the total")(doc)
    assert (q.value, doc[q.start:q.end]) == ("TOTAL due 42", "TOTAL due 42") and 0.5 < q.confidence <= 1.0
    ex.thr["total"] = 1.0
    ex._cache = {}
    q = ex.field("total", "the total")(doc)
    assert q.value == "" and 0.0 <= q.confidence < 0.5
    weak = extractor("TOTAL", "42", strength=0.1).field("total", "the total")(doc)
    assert weak.value == "" and 0.5 < weak.confidence <= 1.0


def test_an_answer_in_a_late_window_of_a_long_document_comes_back_at_its_characters():
    words = [f"w{i}" for i in range(200)]
    doc = " ".join(words[:150]) + " TOTAL 15.50 EUR " + " ".join(words[150:])
    ex = extractor("15.50", "EUR")
    ex.tok(doc)                                        # fill the vocabulary before the head looks words up
    w = ex._windows("the total", doc)
    assert len(w["input_ids"]) > 5 and {len(x) for x in w["input_ids"]} == {32}
    covered = {offs[j] for offs, ctx in zip(w["offset_mapping"], w["ctx"]) for j in ctx}
    assert len(covered) == len(ex.tok(doc)["input_ids"])           # every token of the document is in some window
    q = ex.field("total", "the total")(doc)
    assert (q.value, doc[q.start:q.end]) == ("15.50 EUR", "15.50 EUR") and 0.5 < q.confidence <= 1.0
    absent = extractor("nowhere", "nowhere")
    absent.tok(doc)
    assert absent.field("iban", "the IBAN")(doc).value == ""


def test_a_labelled_span_maps_to_the_tokens_that_cover_it():
    from solvi.core.extract.multi import MultiSpanExtractor
    offs = [(0, 0), (0, 5), (6, 10), (11, 15), (0, 0)]             # [CLS], three tokens, [SEP]
    assert MultiSpanExtractor._tok_span(offs, (11, 15)) == (3, 3)
    assert MultiSpanExtractor._tok_span(offs, (0, 10)) == (1, 2) and MultiSpanExtractor._tok_span(offs, (7, 9)) == (2, 2)


def test_a_missing_optional_dependency_names_the_extra_to_install(monkeypatch):
    """The extractors failed with a bare ModuleNotFoundError (with backend="auto" and no
    runtime: "No module named 'torch'"), while storage and serve name the extra."""
    import sys
    from solvi.core.extract import LongSpanExtractor
    from solvi.core.extract.multi import MultiSpanExtractor
    monkeypatch.setitem(sys.modules, "torch", None)                # as if it were not installed
    for make in (lambda: MultiSpanExtractor(["total"], "m"), lambda: LongSpanExtractor("m")):
        with pytest.raises(ImportError, match=r"Extractor needs torch: pip install 'solvi\[model\]'"):
            make()


def test_one_extractor_protocol_and_no_training_target_past_the_encoded_text():
    """MultiSpanExtractor had fit(docs, spans), predict_doc and no save / load; a labelled span past the truncated window
    became (first token, last token), a wrong target; LongSpanExtractor's cache never shrank and keyed by hash(text)."""
    import inspect

    from solvi.core.extract import LongSpanExtractor
    from solvi.core.extract.multi import MultiSpanExtractor
    offs = [(0, 0), (0, 5), (6, 10), (11, 15), (0, 0)]
    assert MultiSpanExtractor._tok_span(offs, (40, 45)) is None and MultiSpanExtractor._tok_span(offs, (6, 10)) == (2, 2)
    for cls in (MultiSpanExtractor, LongSpanExtractor):
        for name in ("fit", "predict", "field", "save", "load", "fingerprint"):
            assert callable(getattr(cls, name)), (cls, name)
    assert list(inspect.signature(MultiSpanExtractor.predict).parameters)[1:3] == ["text", "field"]
