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
    srv = Server(lambda i, b: reply(probs=P90[i % 3]))
    info = part_of(srv).act_guard(examples(70), max_risk=0.10)
    assert srv.calls == 70 and info["n"] == 70 and "unusable" not in info


# 3 --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("said, value", [
    ("contradiction: the contract states the opposite of the hypothesis", "contradiction"),   # its own description
    ("contradiction: the contract states the opposite…", "contradiction"),                    # cut short
    ("neutral — not addressed anywhere", "neutral"),                                          # names no other option
    ("neutral: neither entailment nor contradiction", None),       # names other options: ambiguous
    ("neutral: or maybe entailment", None),
    ("contradiction or neutral", None),                             # no separator
    ("Contradiction: the contract states the opposite", None),      # not the option as written
])
def test_an_answer_echoing_option_and_description(said, value):
    srv = Server(lambda i, b: reply(said, {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05}
                                    if value != "neutral" else {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9}))
    d = part_of(srv).decide(Facts(hypothesis="H", contract=DOC))
    if value is None:
        assert d.value is None and "invalid LLM output" in d.escalate and said[:20] in d.escalate
    else:
        assert d.value == value and d.escalate is None and d.extra["llm"]["answer_echo"] == said


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
