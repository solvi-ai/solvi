"""The document recipe end to end: one model read per field of a pydantic schema, every fact grounded in the text
(quote at its offsets), the answer computed by your code, a calibrated abstention threshold, a hash-chained record
that verifies and replays — all against a fake chat-completions endpoint, no network. Point `model=` at a real
endpoint (solvi.models.llm(URL, NAME)) and the same code runs on your documents.

The document: a short company note ("Revenue was $1,250 and cost was (1,250)").
  - the schema asks for the two amounts as Span[float]; the model reads each field, the value must stand in the
    text at its offsets — `(1,250)` is an accounting negative, `facts.units` keeps the scale the text states;
  - `compute` is the answer's rule in plain code (a − b), a hard check refuses a zero cost;
  - `calibrate` sets the abstention threshold from labelled examples (workers=4 asks them in parallel; fewer than
    100 examples warn — the threshold rests on them);
  - the answer carries res.facts (value, offsets, the quoted text itself) and a checkable why: a long input text
    is named by its length and hash (`doc[12,345 chars, sha256:…]`), never carried — print the audit: it stays
    small on a 12k-char document;
  - storage keeps every ask; verify() checks the chain, replay_all() re-executes each decision.

    uv run python examples/27_document_decision.py
"""
import io
import json
import re
import tempfile
import warnings

import solvi
from pydantic import BaseModel, Field
from solvi import Span
from solvi.core.deciders.llm import llm


class Server:
    """A chat-completions endpoint that reads the fields by regex from the text it is shown (a stand-in for a real
    model; the recipe neither knows nor cares)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        body = json.loads(req.data.decode())
        msg = body["messages"][-1]["content"]
        nums = re.search(r"Revenue was (\S+) and cost was (\S+)\.", msg)
        field = "a" if "revenue" in msg.split("<text>")[0] else "b"
        ans = nums.group(1) if field == "a" else nums.group(2)
        conf = 0.9 if field == "a" else 0.95
        c = json.dumps({"answer": ans, "confidence": conf, "quote": ""})
        ch = {"index": 0, "message": {"role": "assistant", "content": c}, "finish_reason": "stop"}
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                      "usage": {"prompt_tokens": 40, "completion_tokens": 9}}).encode())


class Note(BaseModel):
    a: Span[float] = Field(description="The revenue, exactly as written.")
    b: Span[float] = Field(description="The cost, exactly as written.")


def doc_of(i, a="$1,250", b="(1,250)"):
    return f"Case {i}. Revenue was {a} and cost was {b}. Other figure: 77."


with warnings.catch_warnings():
    warnings.simplefilter("ignore")                  # a 12-example calibration warns: below 100
    with tempfile.TemporaryDirectory() as d:
        model = llm("http://fake/v1", "fake-model", opener=Server(), workers=1, logprobs=False, ask="confidence")
        recipe = solvi.document(Note, model, task="A short company note.",
                                compute=lambda f, s: f.a - f.b,
                                checks={"b_not_zero": lambda f, s: f.b != 0},
                                storage=f"{d}/decisions.jsonl")
        RES = recipe.ask({"doc": doc_of(1)})         # res["answer"]: answer, confidence, why
        FACTS = RES.facts                            # quoted facts: value, offsets, the quoted text itself
        THRESHOLD = recipe.calibrate([({"doc": doc_of(i), "op": "diff"}, 2500.0)
                                      for i in range(12)], max_risk=0.10, workers=4)["threshold"]
        VERIFY_OK = recipe.verify()["ok"]
        REPLAYS = recipe.replay_all()                # [] — every stored decision re-executes clean

if __name__ == "__main__":
    print("answer:", RES["answer"].answer, RES["answer"].status, f"confidence {RES['answer'].confidence:.2f}")
    for f in FACTS:
        print(f"  {f.name} = {f.value}  at doc[{f.start}:{f.end}] = {f.quote!r}")
    print("calibrated threshold:", THRESHOLD, "| store verifies:", VERIFY_OK, "| replays:", len(REPLAYS))
    print(RES.audit("answer"))
