"""Defects an outside user hit building one document decision with an LLM decider (a fake server here, no network):
act_guard crashed on an invalid reply, a gateway's mixed probability sources were refused only after every call was
paid, an answer echoing "option: description" was rejected, a stated 0.9 escalated by a float artefact, `why` printed
the whole document, calibration was silent, the README's links did not resolve on PyPI, and the store kept the input
twice."""
import io
import json
import logging
import pathlib
import re
import warnings

import pytest

from solvi import Catalog, System
from solvi.core.deciders import Facts
from solvi.core.deciders.llm import llm
from solvi.core.store import JSONLStorage

ROOT = pathlib.Path(__file__).parents[1]
OPTS = {"entailment": "the contract states or clearly implies it",
        "contradiction": "the contract states the opposite of the hypothesis",
        "neutral": "the contract does not address it"}
DOC = "The Receiving Party shall not disclose any Confidential Information to third parties. " + "Other terms. " * 60
P90 = [{"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05},
       {"entailment": 0.9, "contradiction": 0.07, "neutral": 0.03},
       {"entailment": 0.9, "contradiction": 0.06, "neutral": 0.04}]


def reply(answer="entailment", probs=None, quote="shall not disclose"):
    return json.dumps({"answer": answer, "probabilities": probs or P90[0], "quote": quote})


class Server:
    """A chat-completions server: content(i, body) gives the i-th reply (1-based); lp(i) → True adds log-probabilities."""

    def __init__(self, content, lp=lambda i: False):
        self.content, self.lp, self.calls = content, lp, 0

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.calls += 1
        c = self.content(self.calls, body)
        ch = {"index": 0, "message": {"role": "assistant", "content": c}, "finish_reason": "stop"}
        if body.get("logprobs") and self.lp(self.calls):
            ch["logprobs"] = {"content": [{"token": x, "logprob": -0.01, "top_logprobs": []} for x in c]}
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                      "usage": {"prompt_tokens": 50, "completion_tokens": 9}}).encode())


def part_of(server, options=OPTS, **kw):
    m = llm("http://fake/v1", "fake-model", opener=server, workers=1, **kw)
    return m.decision("label", "Does the contract entail the hypothesis?", ["hypothesis", "contract"], options)


def examples(n, label="entailment"):
    return [(Facts(hypothesis=f"H{i}: no disclosure", contract=DOC + str(i)), label) for i in range(n)]


# 1 --------------------------------------------------------------------------------------------------------------------
def test_act_guard_counts_an_invalid_reply_as_escalated_not_a_crash():
    """`ValueError: None is not one of [...]`: a calibration decision escalated as invalid LLM output (value None)."""
    srv = Server(lambda i, b: "not json at all" if i % 10 == 0 else reply())
    part = part_of(srv)
    info = part.act_guard(examples(60), max_risk=0.10)
    assert info["unusable"] == 6 and info["n"] == 60
    assert any("6 of 60 calibration decisions gave no usable output" in w for w in info["warnings"])
    assert info["answered"] == pytest.approx(54 / 60)     # the invalid ones are never answered alone
    assert info["risk"] == 0.0 and info["base_error"] == pytest.approx(0.1)
    c = part.calibrate_for(examples(20), max_error=0.05)
    assert c["unusable"] == 2
    assert part.conformal(examples(20), coverage=0.9)["n"] == 20


# 2 --------------------------------------------------------------------------------------------------------------------
def test_mixed_probability_sources_are_refused_after_the_first_calls_not_after_all():
    srv = Server(lambda i, b: reply(), lp=lambda i: i % 2 == 0)    # a gateway: every other provider gives logprobs
    part = part_of(srv)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="first 16 calibration examples .of 300. come from two sources"):
            part.act_guard(examples(300), max_risk=0.10)
    assert srv.calls == 16
    assert any("two sources" in str(w.message) for w in got)    # the scorer warned at the first mixed reply


def test_one_source_calibrates_in_chunks_with_the_same_numbers():
    def answer(i, b):                                  # right or wrong by the example, sure or unsure by the example
        k = int(re.search(r"H(\d+):", b["messages"][1]["content"]).group(1))
        ans = "entailment" if k % 7 else "neutral"
        p = 0.6 + 0.39 * ((k * 37) % 70) / 70 if k % 7 else 0.5 + 0.2 * (k % 3) / 3   # the wrong ones mostly unsure
        rest = {o: (1 - p) / 2 for o in OPTS if o != ans}
        return reply(ans, {ans: p, **rest})
    chunked, whole = part_of(Server(answer)), part_of(Server(answer))
    whole.model.deterministic = True                   # read as a local model is: every example in one batch
    a = chunked.act_guard(examples(70), max_risk=0.05)
    b = whole.act_guard(examples(70), max_risk=0.05)
    assert a == b and a["n"] == 70 and 0 < a["answered"] < 1 and "unusable" not in a
    assert chunked.calibrate_for(examples(70), max_error=0.1) == whole.calibrate_for(examples(70), max_error=0.1)


# 3 --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("said, value", [
    ("contradiction: the contract states the opposite of the hypothesis", "contradiction"),   # its own description
    ("contradiction: the contract states the opposite…", "contradiction"),                    # cut short
    ("contradiction — the contract states the opposite of the hypothesis here", "contradiction"),   # and more words
    ("neutral — not addressed anywhere", None),                     # a description, but not its own
    ("entailment: the contract does not address it", None),         # another option's description
    ("entailment: the contract", None),                             # starts like every description
    ("neutral:", None),                                             # nothing after the separator
    ("neutral: ", None),
    ("neutral: neither entailment nor contradiction", None),       # names other options
    ("neutral: or maybe entailment", None),
    ("contradiction or neutral", None),                             # no separator
    ("Contradiction: the contract states the opposite", None),      # not the option as written
    ("contradiction: the contract states the opposite of the hypothesis, or rather entailment", None),   # a tail that
    ("contradiction: the contract states the opposite of the hypothesis. Actually neutral.", None),      # names another
    ("contradiction: the contract states the opposite of the hypothesis; entailment: the contract states or clearly "
     "implies it", None),
])
def test_an_answer_echoing_option_and_description(said, value):
    srv = Server(lambda i, b: reply(said, {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05}
                                    if value != "neutral" else {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9}))
    d = part_of(srv).decide(Facts(hypothesis="H", contract=DOC))
    if value is None:
        assert d.value is None and "invalid LLM output" in d.escalate and said[:20] in d.escalate
    else:
        assert d.value == value and d.escalate is None and d.extra["llm"]["answer_echo"] == said


@pytest.mark.parametrize("said, value", [
    ("yes: it applies", "yes"),
    ("yes: it does not", None),                                     # no's description
    ("yes - it does not apply", None),                              # starts like no's description
    ("no: it does not apply", "no"),
    ("yes: it applies, no it does not", None),
    ("yes: it applies. it does not", None),                         # holds no's description
])
def test_an_echo_with_another_options_description_is_not_read(said, value):
    srv = Server(lambda i, b: reply(said, {"yes": 0.9, "no": 0.1} if said.startswith("yes") else {"yes": 0.1, "no": 0.9}))
    m = llm("http://fake/v1", "fake-model", opener=srv, workers=1)
    d = m.decision("q", "Does it apply?", "contract", {"yes": "it applies", "no": "it does not"}).decide(DOC)
    assert d.value == value


@pytest.mark.parametrize("said, value", [("neutral - not addressed", "neutral"), ("neutral: or maybe entailment", None),
                                         ("neutral:", None)])
def test_an_echo_over_options_without_descriptions(said, value):
    srv = Server(lambda i, b: reply(said, {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9}))
    d = part_of(srv, options=list(OPTS)).decide(Facts(hypothesis="H", contract=DOC))
    assert d.value == value


def test_two_options_sharing_a_prefix_are_not_guessed():
    opts = {"yes": "it applies", "yes - partly": "it applies in part", "no": "it does not"}
    srv = Server(lambda i, b: reply("yes - partly: it applies in part", {"yes": 0.1, "yes - partly": 0.8, "no": 0.1}))
    m = llm("http://fake/v1", "fake-model", opener=srv, workers=1)
    d = m.decision("q", "Does it apply?", "contract", opts).decide(DOC)
    assert d.value is None and "invalid LLM output" in d.escalate


# 4 --------------------------------------------------------------------------------------------------------------------
def test_a_stated_point_nine_is_point_nine_whatever_the_other_options():
    """0.9000000000000001 (the threshold) against 0.8999999999999999 (a decision): every stated 0.9 escalated."""
    srv = Server(lambda i, b: reply(probs=P90[i % 3]))
    part = part_of(srv)
    ds = part.decide([Facts(hypothesis=f"H{i}", contract=DOC) for i in range(6)])
    assert {d.conf for d in ds} == {0.9}
    info = part.act_guard(examples(60), max_risk=0.10)
    assert info["threshold"] == 0.9 and info["answered"] == 1.0
    assert part.decide(Facts(hypothesis="H new", contract=DOC)).escalate is None


# 5 --------------------------------------------------------------------------------------------------------------------
def test_why_and_audit_show_the_quote_not_the_whole_document():
    part = part_of(Server(lambda i, b: reply()))
    cat = Catalog()
    s = System(cat, [part.question(cat)])
    res = s.ask({"hypothesis": "H: no disclosure", "contract": DOC})
    why = res["label"].why
    at = DOC.index("shall not disclose")
    assert why.startswith(f"quote 'shall not disclose' (contract[{at}:{at + 18}])")
    assert f"contract = <text, {len(DOC):,} chars>" in why and "hypothesis = 'H: no disclosure'" in why
    assert "Other terms. Other terms." not in why and "Other terms. Other terms." not in str(res.audit("label"))


def test_why_without_a_quote_is_as_before():
    part = part_of(Server(lambda i, b: reply(quote="")))
    cat = Catalog()
    res = System(cat, [part.question(cat)]).ask({"hypothesis": "H", "contract": "short text"})
    assert res["label"].why == "hypothesis = 'H'; contract = 'short text'"


def test_a_record_stored_with_the_earlier_why_still_replays(tmp_path, monkeypatch):
    import solvi.core.system as system_mod
    part = part_of(Server(lambda i, b: reply()))
    cat = Catalog()
    s = System(cat, [part.question(cat)], storage=JSONLStorage(tmp_path / "d.jsonl"))
    new = system_mod._rule_why
    monkeypatch.setattr(system_mod, "_rule_why", lambda rule, r, vals, by, legacy=False: new(rule, r, vals, by, True))
    s.ask({"hypothesis": "H: no disclosure", "contract": DOC})                # stored as an earlier version wrote it
    monkeypatch.setattr(system_mod, "_rule_why", new)
    st = JSONLStorage(tmp_path / "d.jsonl")
    assert "Other terms. Other terms." in json.loads((tmp_path / "d.jsonl").read_text())["response"]["results"]["label"]["why"]
    assert st.verify()["ok"]
    assert st.replay_all(s) == []


# 8 --------------------------------------------------------------------------------------------------------------------
def test_calibration_progress_is_logged_only_when_asked(caplog, capsys):
    part = part_of(Server(lambda i, b: reply()))
    part.act_guard(examples(50), max_risk=0.10)
    assert not [r for r in caplog.records if r.name == "solvi.calibration"]
    assert capsys.readouterr() == ("", "")
    with caplog.at_level(logging.INFO, logger="solvi.calibration"):
        part_of(Server(lambda i, b: reply())).act_guard(examples(50), max_risk=0.10)
    lines = [r.getMessage() for r in caplog.records if r.name == "solvi.calibration"]
    assert [re.match(r"label: (\d+)/50 calibration examples decided", x).group(1) for x in lines] == ["16", "48", "50"]
    assert "left" in lines[0] and "left" not in lines[-1]


# 9 --------------------------------------------------------------------------------------------------------------------
def test_the_store_keeps_a_long_input_once_and_the_record_verifies_and_replays(tmp_path):
    part = part_of(Server(lambda i, b: reply()))
    cat = Catalog()
    s = System(cat, [part.question(cat)], storage=JSONLStorage(tmp_path / "d.jsonl"))
    res = s.ask({"hypothesis": "H: no disclosure", "contract": DOC})
    line = (tmp_path / "d.jsonl").read_text()
    assert line.count("Confidential Information to third parties. Other terms.") == 1
    st = JSONLStorage(tmp_path / "d.jsonl")
    back = st.get(res.stored_id, s)
    assert back.values == res.values and back["label"].answer == "entailment"
    assert st.verify()["ok"] and st.replay_all(s) == []


# 6, 7 -----------------------------------------------------------------------------------------------------------------
def test_the_input_docstrings_name_where_facts_come_from():
    from solvi.core.deciders import DecisionPart
    assert "from solvi.core.deciders import Facts" in DecisionPart.decide.__doc__
    assert "from solvi.core.deciders import Facts" in DecisionPart.act_guard.__doc__


def test_the_readme_links_resolve_outside_the_repository():
    """PyPI shows the README without the repository: a relative link (docs/guide.md) leads nowhere there."""
    text = (ROOT / "README.md").read_text()
    links = re.findall(r"\]\(([^)\s]+)\)", text)
    rel = [x for x in links if not re.match(r"https?:|#|mailto:", x)]
    assert not rel, rel
    base = "https://github.com/solvi-ai/solvi/blob/main/"
    missing = [x for x in links if x.startswith(base) and not (ROOT / x[len(base):].split("#")[0]).exists()]
    assert not missing, missing


# second outside-user test (numbers out of financial tables) ------------------------------------------------------------
@pytest.mark.parametrize("passage, text, want", [
    ("7", "In 2017 sales rose by 7 percent.", "7"),           # not the 7 of 2017
    ("10", "There were 109 stores, 10 of them new.", "10"),
    ("1.0", "Ratio 41.0 in total, 1.0 per share.", "1.0"),
    ("6", "Revenue 13,064 and 6 segments.", "6"),
    ("7", "In 2017 and 1997.", None),                         # only inside numbers: not found
    ("7.", "Up 7.5 percent, then 7. And more", "7."),          # not the 7. of 7.5
    ("1,", "Total 1,300; items 1, 2", "1,"),
    ("12 3", "codes 112 3 and 12 3", "12 3"),                  # overlapping matches of the words pattern
    ("charged", "he was charged twice", "charged"),           # words as before
])
def test_locating_a_quoted_number_never_lands_inside_a_longer_number(passage, text, want):
    from solvi.core.deciders.llm import locate
    got = locate(passage, text)
    assert (None if got is None else text[got[0]:got[1]]) == want
    if got is not None:
        from solvi.core.catalog import cuts_number
        assert not cuts_number(text, *got)


def test_an_llm_span_number_is_read_from_its_own_occurrence():
    doc = "Fiscal 2017 results. The margin rose by 7 points."
    srv = Server(lambda i, b: json.dumps({"answer": "7", "confidence": 0.9, "quote": ""}))
    m = llm("http://fake/v1", "fake-model", opener=srv, workers=1)
    from solvi import Span
    cat = Catalog()
    part = m.decision("margin", "By how many points did the margin rise?", "doc", type=Span[float])
    res = System(cat, [part.question(cat)]).ask({"doc": doc})
    assert res["margin"].answer == 7.0 and res["margin"].status == "ok"
    assert res["margin"].span.start == doc.index(" 7 ") + 1


def test_a_computed_number_names_its_answer_type():
    from solvi import Answer, Question
    cat = Catalog()

    @cat.rule("total")
    def total(x: float) -> float:
        return x + 1
    with pytest.raises(TypeError, match=r"Answer\.estimate\(\)"):
        System(cat, [Question("total", "Total?")])
    r = System(cat, [Question("total", "Total?", Answer.estimate())]).ask({"x": 2.0})["total"]
    assert r.answer == 3.0 and r.extra["interval"] == [3.0, 3.0]


def _guarded(tmp_path=None):
    part = part_of(Server(lambda i, b: reply(probs=P90[i % 3])))
    cat = Catalog()
    s = System(cat, [part.question(cat)], storage=None if tmp_path is None else JSONLStorage(tmp_path / "d.jsonl"))
    return s


def test_a_system_guarantee_saves_and_loads(tmp_path):
    s = _guarded(tmp_path)
    ex = [({"hypothesis": f"H{i}", "contract": DOC}, "entailment" if i % 9 else "neutral") for i in range(120)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = s.guarantee("label", ex, max_risk=0.15, weak="warn")
    res = s.ask({"hypothesis": "H new", "contract": DOC})
    path = s.save_guarantee("label", tmp_path / "label.guarantee.json")
    fresh = _guarded()
    got = fresh.load_guarantee(path)
    assert got["threshold"] == rep["threshold"] and got["n"] == 120
    assert fresh.guards["label"].fingerprint() == s.guards["label"].fingerprint()
    assert JSONLStorage(tmp_path / "d.jsonl").replay_all(fresh) == []        # replays without calibrating again
    assert res.stored_id
    rec = json.loads(pathlib.Path(path).read_text())
    rec["catalog"] = "0" * 16
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="calibrated on catalog"):
        _guarded().load_guarantee(bad)
    rec = json.loads(pathlib.Path(path).read_text())
    rec["promise"]["threshold"] = 0.0                                       # a forged threshold: not the saved guard
    bad.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="rebuilt guarantee"):
        _guarded().load_guarantee(bad)
    with pytest.raises(KeyError):
        _guarded().save_guarantee("label", tmp_path / "x.json")
    with pytest.raises(ValueError, match="only for a signal that was a function"):
        _guarded().load_guarantee(path, signal=lambda r, v: 1.0)


def test_one_store_two_systems_replays_each_record_against_its_own(tmp_path):
    from solvi import Question
    store = tmp_path / "d.jsonl"
    a = _guarded(tmp_path)
    cat = Catalog()

    @cat.rule("double")
    def double(n: int) -> bool:
        return n % 2 == 0
    b = System(cat, [Question("double", "Even?")], storage=JSONLStorage(store))
    for i in range(3):
        a.ask({"hypothesis": f"H{i}", "contract": DOC})
        b.ask({"n": i})
    st = JSONLStorage(store)
    assert st.replay_all([a, b]) == []
    alone = st.replay_all(a)                                            # b's records: one clear line each
    assert len(alone) == 3 and all(x["kinds"] == {"other_system": 1} for x in alone)
    assert all("another system's record" in x["summary"] for x in alone)
    assert st.replay_all(a, question="label") == []
    lines = store.read_text().splitlines()                              # a tampered record of b is still caught
    rec = json.loads(lines[1])
    rec["response"]["trace"]["init"]["n"] = 7
    lines[1] = json.dumps(rec)
    store.write_text("\n".join(lines) + "\n")
    st = JSONLStorage(store)
    hit = [x for x in st.replay_all(a) if x["seq"] == rec["seq"]]
    assert hit and "integrity" in hit[0]["kinds"]
    hit = [x for x in st.replay_all([a, b]) if x["seq"] == rec["seq"]]
    assert hit and "integrity" in hit[0]["kinds"]


def test_grounding_refuses_a_quote_ending_in_a_separator_inside_a_number():
    from solvi.core.catalog import Quote, check_evidence, cuts_number
    t = "Up 7.5 percent."
    assert cuts_number(t, 3, 5) and cuts_number(t, 4, 6) and not cuts_number(t, 3, 6)
    assert check_evidence([Quote("7.", 3, 5, "doc")], {"doc": t})


def test_a_foreign_record_is_never_given_a_benign_verdict_it_was_not_checked_for(tmp_path):
    from solvi import Question
    store = tmp_path / "d.jsonl"
    a = _guarded(tmp_path)
    cat = Catalog()

    @cat.rule("double")
    def double(n: int) -> bool:
        return n % 2 == 0
    b = System(cat, [Question("double", "Even?")], storage=JSONLStorage(store, record="compact"))
    a.ask({"hypothesis": "H", "contract": DOC})
    b.ask({"n": 2})
    st = JSONLStorage(store)
    [x] = st.replay_all(a)                                      # b's compact record: no trace to check, and it says so
    assert x["kinds"] == {"not_kept": 1, "other_system": 1} and "no verdict" in x["summary"]
    assert "hashes verify" not in x["summary"]
    lines = store.read_text().splitlines()
    rec = json.loads(lines[0])                                  # a's own record, its question renamed to look foreign
    rec["answers"] = {"other": rec["answers"]["label"]}
    lines[0] = json.dumps(rec)
    store.write_text("\n".join(lines) + "\n")
    hit = [x for x in JSONLStorage(store).replay_all([a, b]) if x["seq"] == rec["seq"]]
    assert hit and "integrity" in hit[0]["kinds"] and hit[0]["summary"].startswith("data damaged")


def test_a_saved_guarantee_is_refused_for_another_model(tmp_path):
    s = _guarded()
    ex = [({"hypothesis": f"H{i}", "contract": DOC}, "entailment" if i % 9 else "neutral") for i in range(120)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.guarantee("label", ex, max_risk=0.15, weak="warn")
    path = s.save_guarantee("label", tmp_path / "g.json")
    rec = json.loads(pathlib.Path(path).read_text())
    assert rec["models"] == s.fingerprint()["models"] and rec["models"]
    cat = Catalog()
    part = llm("http://fake/v1", "another-model", opener=Server(lambda i, b: reply()), workers=1).decision(
        "label", "Does the contract entail the hypothesis?", ["hypothesis", "contract"], OPTS)
    other = System(cat, [part.question(cat)])
    with pytest.raises(ValueError, match="calibrated with other models than this system's — answer:label"):
        other.load_guarantee(path)
    assert other.load_guarantee(path, strict=False)["n"] == 120          # strict=False: attached anyway
    for bad in (None, "absent"):                                           # one deleted key does not bypass it
        if bad is None:
            rec["models"] = None
        else:
            del rec["models"]
        old = tmp_path / "old.json"
        old.write_text(json.dumps(rec))
        with pytest.raises(ValueError, match="names no models it was calibrated with"):
            _guarded().load_guarantee(old)
        assert _guarded().load_guarantee(old, strict=False)["n"] == 120
