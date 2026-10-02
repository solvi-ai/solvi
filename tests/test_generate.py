"""solvi.generate against a fake chat-completions server (no network): a text or a validated structure, quotes checked
literally in a text, an invalid reply escalated (never repaired), the client settings of solvi.llm reused, K samples, a
catalog part recorded as a model's output, and the replay that re-reads the recorded reply instead of calling the model."""
import io
import json
import urllib.error

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.generate import Generated, Generator, Schema, Unanswered, generator, json_errors, several
from solvi.llm import InvalidOutput, LLMError, llm

KEY = "sk-secret-123"


class FakeServer:
    """Replies in turn from `replies` (the last one repeats); a reply is a string (the content), a dict (the whole
    choice: content, finish_reason, refusal), a callable of the request body, or an int (raise that HTTP code)."""

    def __init__(self, *replies):
        self.replies, self.bodies, self.headers = list(replies), [], []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.bodies.append(body)
        self.headers.append(dict(req.header_items()))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        r = r(body) if callable(r) else r
        if isinstance(r, int):
            raise urllib.error.HTTPError(req.full_url, r, "error", {}, io.BytesIO(b'{"error": {"message": "nope"}}'))
        ch = {"message": {"role": "assistant", "content": r}, "finish_reason": "stop"} if isinstance(r, str) else \
            {"message": {"role": "assistant", "content": r.get("content"), "refusal": r.get("refusal")},
             "finish_reason": r.get("finish_reason", "stop")}
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                      "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode())


def gen(*replies, **kw):
    srv = FakeServer(*replies)
    return generator("http://127.0.0.1:9/v1", "m-1", api_key=KEY, opener=srv, sleep=lambda s: None, **kw), srv


class Slot(BaseModel):
    day: str
    start: int


def test_generate_returns_the_text_and_records_the_request_without_the_key():
    g, srv = gen("SELECT 1;")
    out = g.generate("Write a query")
    assert isinstance(out, Generated) and out.value == "SELECT 1;" and out.text == "SELECT 1;"
    assert srv.bodies[0]["messages"] == [{"role": "user", "content": "Write a query"}]
    assert srv.bodies[0]["temperature"] == 0.0 and "seed" not in srv.bodies[0] and "response_format" not in srv.bodies[0]
    assert srv.headers[0]["Authorization"] == f"Bearer {KEY}"
    meta = out.meta
    assert meta["model"] == "llm:m-1@http://127.0.0.1:9/v1/chat/completions" and len(meta["request"]) == 16
    assert meta["usage"] == {"input_tokens": 11, "output_tokens": 7} and meta["finish"] == "stop"
    assert KEY not in json.dumps(meta) and KEY not in g.fingerprint() and KEY not in repr(g)
    assert g.usage["output_tokens"] == 7 and g.calls == 1


def test_a_structured_reply_is_validated_against_a_pydantic_model_or_a_json_schema():
    g, _ = gen('Here: ```json\n{"day": "Mon", "start": 9}\n```')
    out = g.generate("Propose", schema=Slot)
    assert out.value == Slot(day="Mon", start=9) and out.text.startswith("Here:")
    g, _ = gen('{"day": "Mon", "start": 9}')
    js = {"type": "object", "properties": {"day": {"enum": ["Mon", "Tue"]}, "start": {"type": "integer", "minimum": 8}},
          "required": ["day", "start"], "additionalProperties": False}
    assert g.generate("Propose", schema=js).value == {"day": "Mon", "start": 9}
    assert json_errors(js, {"day": "Sun", "start": 9}) == "$.day = 'Sun' is not one of ['Mon', 'Tue']"
    assert json_errors(js, {"day": "Mon", "start": 7}) == "$.start = 7 breaks minimum 8"
    assert json_errors(js, {"day": "Mon"}) == "$.start is missing"
    assert json_errors(js, {"day": "Mon", "start": 9, "x": 1}) == "$.x is not allowed"


def test_json_schema_keywords_that_are_not_checked_are_refused_at_construction():
    with pytest.raises(ValueError, match=r"\['pattern'\].*not checked"):
        Schema({"type": "object", "properties": {"iban": {"type": "string", "pattern": "^DE"}}})
    with pytest.raises(ValueError, match=r"\$ref"):
        Schema({"$ref": "#/defs/x"})


@pytest.mark.parametrize("reply, why", [
    ("not json at all", "the reply is not JSON"),
    ('{"day": "Mon"}', "does not match Slot: start: Field required"),
    ('{"day": "Mon", "start": "nine"}', "does not match Slot: start"),
    ({"content": '{"day": "Mon", "st', "finish_reason": "length"}, "cut off"),
    ("   ", "the reply is empty"),
    ({"content": None, "refusal": "I cannot help"}, "the model refused"),
])
def test_an_invalid_reply_raises_invalid_output_with_the_reason_and_is_never_repaired(reply, why):
    g, _ = gen(reply)
    with pytest.raises(InvalidOutput, match=why):
        g.generate("Propose", schema=Slot)


def test_a_parse_function_that_raises_or_finds_nothing_rejects_the_reply():
    g, _ = gen("no code block here")
    with pytest.raises(InvalidOutput, match="nothing could be read"):
        g.generate("q", parse=lambda t: None)
    with pytest.raises(InvalidOutput, match="could not be read: ValueError"):
        g.generate("q", parse=lambda t: int(t))


class Table(BaseModel):
    rows: list[str]


def test_quotes_must_be_in_the_text_as_written_whole_words_and_numbers():
    text = "North Beach to Chinatown: 30.\nChinatown to Bayview: 12."
    g, _ = gen('{"rows": ["North Beach to Chinatown: 30", "Chinatown to Bayview: 12"]}')
    out = g.generate("copy the rows", schema=Table, text=text, quotes=["rows"])
    assert out.evidence == ["North Beach to Chinatown: 30", "Chinatown to Bayview: 12"]
    g, _ = gen('{"rows": ["North Beach to Chinatown: 3"]}')                  # "3" is not in "30"
    with pytest.raises(InvalidOutput, match=r"the quote 'North Beach to Chinatown: 3' \(rows\) is not in the text"):
        g.generate("copy the rows", schema=Table, text=text, quotes=["rows"])
    with pytest.raises(InvalidOutput, match="no 'cells' in the reply"):
        gen('{"rows": []}')[0].generate("x", schema=dict, text=text, quotes=["cells"])
    with pytest.raises(ValueError, match="needs text="):
        gen('{"rows": ["a"]}')[0].generate("x", schema=Table, quotes=["rows"])


def test_server_errors_are_retried_then_unanswered_and_a_wrong_key_raises_llm_error():
    g, srv = gen(503, 503, "fine")
    assert g.generate("q").value == "fine" and len(srv.bodies) == 3
    g, _ = gen(503)
    with pytest.raises(Unanswered, match="after 3 attempts"):
        g.generate("q")
    g, _ = gen(400)
    with pytest.raises(Unanswered, match="the LLM server refused the request: HTTP 400 — nope"):
        g.generate("q")
    g, _ = gen(401)
    with pytest.raises(LLMError, match="check the URL"):
        g.generate("q")


def test_the_client_settings_are_solvi_llms_extra_body_is_sent_and_cannot_set_what_solvi_sets():
    g, srv = gen("ok", extra_body={"reasoning": {"effort": "low"}}, headers={"X-Title": "t"})
    g.generate("q", max_tokens=99, temperature=0.3, seed=5)
    b = srv.bodies[0]
    assert b["reasoning"] == {"effort": "low"} and b["max_tokens"] == 99 and b["temperature"] == 0.3 and b["seed"] == 5
    assert srv.headers[0]["X-title"] == "t"
    with pytest.raises(ValueError, match="cannot set"):
        generator("http://x/v1", "m", extra_body={"temperature": 1})
    srv2 = FakeServer("shared")
    decider = llm("http://127.0.0.1:9/v1", "m-2", api_key=KEY, opener=srv2, extra_body={"provider": {"order": ["a"]}})
    g2 = Generator.of(decider, max_tokens=50)
    assert g2.generate("q").value == "shared" and srv2.bodies[0]["provider"] == {"order": ["a"]}
    assert g2.client is decider.scorer and g2.model_id == decider.scorer.model_id


def test_json_schema_response_format_is_sent_only_when_asked_and_the_reply_is_still_validated():
    g, srv = gen('{"day": "Mon", "start": "x"}', response_format="json_schema")
    with pytest.raises(InvalidOutput):
        g.generate("q", schema=Slot)
    assert srv.bodies[0]["response_format"]["json_schema"]["schema"]["required"] == ["day", "start"]


def test_sample_asks_the_greedy_reply_first_then_seeded_samples_and_keeps_a_failed_one_as_none():
    g, srv = gen("A", "", "B")                       # the fake server answers in call order: ask one at a time so the
    g.workers = 1                                     # order is the requests' (in use, each request has its own seed)
    out = g.sample("q", k=3, temperature=0.9)
    assert out.value == ["A", None, "B"]
    assert [(b["temperature"], b.get("seed")) for b in sorted(srv.bodies, key=lambda b: b.get("seed") or 0)] == \
        [(0.0, None), (0.9, 1), (0.9, 2)]
    assert "the reply is empty" in out.meta[1]["error"] and out.meta[0]["request"] != out.meta[2]["request"]
    with pytest.raises(InvalidOutput, match="all 2 replies failed"):
        gen("")[0].sample("q", k=2)


def test_several_generators_give_one_candidate_each():
    a, _ = gen("from a")
    b, _ = gen(503)
    out = several([a, b], "q")
    assert out.value == ["from a", None] and "Unanswered" in out.meta[1]["error"]


def _sql_system(g, **kw):
    cat = Catalog()
    cat.fn(g.part("sql", lambda question: f"Write SQL for: {question}", **kw))

    @cat.check(hard=True, then={"ok": "no"})
    def selects(sql) -> bool:
        return sql.upper().startswith("SELECT")

    @cat.rule("ok")
    def ok(selects) -> bool:
        return True
    return System(cat, [Question("ok", "Is the query fine?", Answer.yes_no(), requires=["selects"])]), cat


def test_a_generation_part_is_recorded_as_a_proposed_model_output_and_replay_does_not_call_the_model():
    g, srv = gen("SELECT 1")
    s, cat = _sql_system(g)
    res = s.ask({"question": "how many?"})
    rec = next(r for r in res.trace.records if r.name == "sql")
    assert res["ok"].answer == "yes" and rec.origin == "proposed" and rec.model["type"] == "GenerationPart"
    assert rec.extra["generated"]["request"] and rec.value == "SELECT 1"
    rep = res.trace.replay(s)
    assert rep["ok"] and rep["models"] == [(rec.step, "sql", "trusted")] and len(srv.bodies) == 1
    assert "1 from models" in str(res.audit("ok"))


def test_an_invalid_reply_in_a_catalog_makes_the_question_abstain_with_the_cause():
    g, _ = gen({"content": "SEL", "finish_reason": "length"})
    s, _ = _sql_system(g)
    r = s.ask({"question": "q"})["ok"]
    assert r.status == "abstain" and "caused by sql: InvalidOutput: the reply was cut off" in r.why


def test_replay_rereads_the_recorded_reply_through_the_schema_and_names_a_value_that_does_not_follow():
    g, _ = gen('{"day": "Mon", "start": 9}')
    cat = Catalog()
    cat.fn(g.part("slot", lambda ask: ask, schema=Slot))

    @cat.rule("ok")
    def ok(slot) -> bool:
        return slot.start < 12
    s = System(cat, [Question("ok", "?", Answer.yes_no())])
    res = s.ask({"ask": "a slot"})
    assert res["ok"].answer == "yes" and res.trace.replay(s)["ok"]
    from solvi import Response
    stored = Response.model_validate(json.loads(res.to_json()), catalog=cat)    # a stored response loads back and replays
    assert stored.values["slot"] == Slot(day="Mon", start=9) and stored.trace.replay(s)["ok"]
    rec = next(r for r in res.trace.records if r.name == "slot")
    model = cat.parts["slot"].model
    assert model.check_record(rec) == []
    rec.extra["generated"]["text"] = '{"day": "Tue", "start": 9}'
    assert "reads as Slot(day='Tue', start=9), not the recorded" in model.check_record(rec)[0]
    rec.extra["generated"]["text"] = '{"day": "Tue"}'
    assert "not accepted any more" in model.check_record(rec)[0]


def test_a_changed_prompt_shows_on_replay_as_a_changed_model():
    g, _ = gen("SELECT 1")
    s, _ = _sql_system(g)
    res = s.ask({"question": "q"})
    s2, _ = _sql_system(g, max_tokens=10)                      # another setting of the part
    rep = res.trace.replay(s2)
    assert not rep["ok"] and rep["kinds"] == {"model_changed": 1}


def test_quotes_of_a_part_become_evidence_located_in_the_given_text_and_checked_on_replay():
    text = "Paris and Rome. from Rome to Oslo."
    g, _ = gen('{"rows": ["Paris and Rome", "from Rome to Oslo"]}')
    cat = Catalog()
    cat.fn(g.part("flights", lambda problem: problem, schema=Table, text="problem", quotes=["rows"]))

    @cat.rule("n")
    def n(flights) -> bool:
        return len(flights.rows) == 2
    s = System(cat, [Question("n", "?", Answer.yes_no())])
    res = s.ask({"problem": text})
    rec = next(r for r in res.trace.records if r.name == "flights")
    assert rec.extra["evidence"] == [[0, 14, "problem", "Paris and Rome"], [16, 33, "problem", "from Rome to Oslo"]]
    assert res.trace.replay(s)["ok"]
    with pytest.raises(ValueError, match="must be a parameter"):
        g.part("x", lambda doc: doc, text="problem")
