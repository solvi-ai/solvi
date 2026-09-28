"""Fixes from the adversarial re-check before 0.7 outside the guard: exception texts in solvi serve's answers, the specs
of a hand-built TextRead, negations after a yes / no cue, and small edges (async max_inflight, MCP string arguments,
a failed forward)."""
import json

import pytest

from solvi import Answer, Catalog, Question, System

SECRET = "/srv/secret/models/risk.bin"


# ---------------------------------------------------------------------------------------------------- B7
def leaky(tmp_path=None):
    cat = Catalog()

    @cat.fn(provides="rate")
    def rate_remote(amount: float) -> float:
        raise ConnectionError(f"cannot open {SECRET}")

    @cat.fn(provides="rate")
    def rate_local(amount: float) -> float:
        raise OSError(f"{SECRET}: permission denied")

    @cat.fn
    def risk(amount: float) -> float:
        raise FileNotFoundError(f"{SECRET} not found")

    @cat.check(hard=True, then={"approve": "no"})
    def low_risk(risk) -> bool:
        return risk < 1

    @cat.rule("approve")
    def approve(amount) -> str:
        return "yes"

    @cat.rule("price")
    def price(amount, rate) -> str:
        return "ok"

    @cat.rule("limit")
    def limit(amount) -> str:
        raise RuntimeError(f"config at {SECRET} is broken")
    qs = [Question("approve", "Approve?", Answer.choice(["yes", "no"]), checkpoints=["low_risk"]),
          Question("price", "Priced?", Answer.choice(["ok", "no"])),
          Question("limit", "Within limit?", Answer.choice(["ok", "no"]))]
    s = System(cat, qs)
    if tmp_path is not None:
        from solvi.storage import open_storage
        s.storage = open_storage(tmp_path / "t.jsonl", s)
    return s


def test_serve_answers_never_carry_exception_texts(tmp_path, caplog):
    from solvi.serve import Service
    svc = Service(leaky(tmp_path))
    with caplog.at_level("ERROR", logger="solvi.serve"):
        d = svc.ask({"amount": 5})
    body = json.dumps(d)
    assert SECRET not in body and "permission denied" not in body
    errs = {r["name"]: r["error"] for r in d["trace"]["records"] if r.get("error")}
    assert errs["risk"].startswith("FileNotFoundError (incident ")
    assert "RuntimeError (incident " in d["results"]["limit"]["why"]
    tried = [t for r in d["trace"]["records"] if r.get("tried") for t in r["tried"]]
    assert tried and all(SECRET not in t[1] for t in tried) and any("ConnectionError (incident" in t[1] for t in tried)
    assert SECRET in caplog.text                                   # the server log keeps the text
    stored = svc.system.storage.record(d["stored_id"])
    assert SECRET in json.dumps(stored)                            # and so does the stored trace
    for name in ("approve", "price", "limit"):
        assert SECRET not in json.dumps(svc.tool(name, {"amount": 5})), name


def test_serve_http_ask_hides_exception_texts():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    c = TestClient(create_app(system=leaky()), raise_server_exceptions=False)
    r = c.post("/ask", json={"state": {"amount": 5}})
    assert r.status_code == 200 and SECRET not in r.text and "(incident " in r.text


def test_redact_leaves_answers_without_exceptions_alone():
    from solvi.serve import redact
    d = {"results": {"q": {"why": "hard check x is false"}},
         "trace": {"records": [{"name": "a", "error": "missing inputs: b"}, {"name": "c", "error": "timed out after 1s"}]}}
    assert redact(json.loads(json.dumps(d))) == d
