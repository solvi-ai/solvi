import io
import json
from pathlib import Path
import sys
import subprocess
from urllib.request import Request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import invoice_workflow_run as runner


def reply(content, finish="stop"):
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": finish}], "usage": {"cost": 0.0001}}
    ).encode()


def test_cache_resume_budget_and_failure(tmp_path):
    calls = []

    def opener(req, timeout):
        calls.append(json.loads(req.data))
        return io.BytesIO(reply({"answer": "approve", "confidence": 0.9}))

    req = Request("https://example.invalid", data=b'{"messages": []}')
    tr = runner.Transport(tmp_path / "ledger", opener=opener)
    assert tr(req).read() == tr(req).read()
    assert len(calls) == 1
    assert calls[0]["provider"] == runner.PROVIDER
    assert tr.events[0] == tr.events[1]
    tr.db.close()
    tr = runner.Transport(tmp_path / "ledger", opener=opener, budget=0.0001)
    assert tr(req).read()
    with pytest.raises(runner.BudgetStop):
        tr(Request("https://example.invalid", data=b'{"messages": [1]}'))
    assert len(calls) == 1
    tr.db.close()

    def fail(req, timeout):
        raise OSError("secret")

    tr = runner.Transport(tmp_path / "fail", opener=fail)
    with pytest.raises(runner.RecordedFailure):
        tr(req)
    with pytest.raises(runner.RecordedFailure):
        tr(req)
    assert tr.db.execute("select status from calls").fetchone()[0] == "failed"


def test_policy_and_arm_validation(tmp_path):
    case = runner.make_case("x", "doc", {"vendor": "v", "total": "1.00", "rendered_total": "2.00", "currency": "EUR"})
    assert case["gold"] == "reject"
    assert "rendered_total" not in json.dumps(case["state"])

    def opener(req, timeout):
        body = json.loads(req.data)
        text = body["messages"][0]["content"]
        assert "rendered_total" not in text and "gold" not in text
        return io.BytesIO(
            reply({"invoice_number": "x", "total": case["state"]["order_total"], "currency": "EUR", "confidence": 0.8})
        )

    tr = runner.Transport(tmp_path / "ledger", opener=opener)
    result = runner.run_arm("plain", case, tr, "secret", tmp_path)
    assert result["answer"] == "approve" and result["cost_usd"] == 0.0001
    assert runner.split("v") == runner.split("v")


def test_resume_smoke_and_evaluator_contract(tmp_path, monkeypatch):
    calls = []

    def fake(name, case, transport, key, out):
        calls.append((name, case["id"]))
        return {"answer": None, "confidence": 0.0, "latency_s": 0.0, "cost_usd": 0.0}

    monkeypatch.setattr(runner, "run_arm", fake)
    tr = runner.Transport(tmp_path / "ledger", opener=None)
    runner.run(tmp_path, smoke=True, transport=tr, key="x")
    runner.run(tmp_path, smoke=True, transport=tr, key="x")
    assert len(calls) == 6
    rows = [json.loads(line) for line in (tmp_path / "smoke/predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(set(row["arms"]) == set(runner.ARMS) for row in rows)
    with pytest.raises(ValueError):
        runner.run(tmp_path, limit=1, transport=tr, key="x")


def test_parallel_documents_resume_without_repeating_saved_arms(tmp_path, monkeypatch):
    calls = []

    def fake(name, case, transport, key, out):
        calls.append((case["id"], name))
        return {"answer": case["gold"], "confidence": 1., "cost_usd": 0., "latency_s": 0.}

    monkeypatch.setattr(runner, "run_arm", fake)
    runner.run(tmp_path, smoke=True, key="offline")
    runner.run(tmp_path, smoke=True, key="offline")
    assert len(calls) == len(set(calls)) == 6
    rows = [json.loads(line) for line in (tmp_path / "smoke/predictions.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [c["id"] for c in runner.smoke_cases()]


def test_solvi_transport_smoke(tmp_path):
    case = runner.smoke_cases()[0]

    def opener(req, timeout):
        body = json.loads(req.data)
        msg = body["messages"][-1]["content"]
        field = "SMOKE-1" if "invoice number" in msg else "EUR" if "ISO currency" in msg else "12.00"
        return io.BytesIO(reply({"answer": field, "confidence": 0.99, "quote": field}))

    tr = runner.Transport(tmp_path / "ledger", opener=opener)
    result = runner.run_arm("solvi", case, tr, "secret", tmp_path)
    assert result["cost_usd"] > 0
    assert result["answer"] == case["gold"]
    assert "error" not in result


def test_solvi_budget_stop_propagates(tmp_path):
    tr = runner.Transport(tmp_path / "ledger", opener=lambda *args, **kwargs: pytest.fail("network called"), budget=0)
    with pytest.raises(runner.BudgetStop):
        runner.run_arm("solvi", runner.smoke_cases()[0], tr, "secret", tmp_path)


def test_prepare_resumes_and_hashes(tmp_path):
    requests = []

    def fetch(url, timeout):
        requests.append(url)
        return io.BytesIO(url.encode())

    runner.prepare(tmp_path, fetch)
    assert len(requests) == 401
    manifest = json.loads((tmp_path / "dataset/manifest.json").read_text())
    assert len(manifest["sha256"]) == 401 and manifest["revision"] == runner.REV
    assert all(
        value == runner.digest((runner.SOURCE + "/" + path).encode()) for path, value in manifest["sha256"].items()
    )
    runner.prepare(tmp_path, fetch)
    assert len(requests) == 401


def test_process_killed_between_arms_resumes_without_repeating_saved_arm(tmp_path):
    script = """
import json, os, signal, sys
sys.path.insert(0, sys.argv[1])
import invoice_workflow_run as r
from pathlib import Path
root = Path(sys.argv[2])
def fake(name, case, transport, key, out):
    if sys.argv[3] == 'kill' and name == 'plain':
        os.kill(os.getpid(), signal.SIGTERM)
    with (root / 'events').open('a') as f:
        print(name, file=f)
    return {'answer': None, 'confidence': 0., 'cost_usd': 0., 'latency_s': 0.}
r.run_arm = fake
r.run(root, smoke=True, limit=1, key='offline')
"""
    args = [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "benchmarks"), str(tmp_path)]
    killed = subprocess.run([*args, "kill"], capture_output=True, timeout=30)
    assert killed.returncode < 0
    subprocess.run([*args, "resume"], capture_output=True, timeout=30, check=True)
    assert (tmp_path / "events").read_text().splitlines() == ["direct", "plain", "solvi"]


def test_reservations_atomic_across_connections(tmp_path):
    import threading

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def opener(req, timeout):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return io.BytesIO(reply({"answer": "approve", "confidence": 1}))

    path = tmp_path / "ledger"
    first = runner.Transport(path, opener=opener, budget=0.0025)
    second = runner.Transport(path, opener=opener, budget=0.0025)
    request = Request("https://example.invalid", data=b'{"messages": []}')
    thread = threading.Thread(target=lambda: first(request))
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(runner.RecordedFailure):
            second(request)
        with pytest.raises(runner.BudgetStop):
            second(Request("https://example.invalid", data=b'{"messages": [1]}'))
        assert len(calls) == 1
    finally:
        release.set()
        thread.join(5)


def test_resume_rejects_metadata_and_protocol_change_before_call(tmp_path, monkeypatch):
    tr = runner.Transport(tmp_path / "ledger")
    monkeypatch.setattr(
        runner, "run_arm", lambda *a: {"answer": None, "confidence": 0.0, "latency_s": 0.0, "cost_usd": 0.0}
    )
    runner.run(tmp_path, smoke=True, transport=tr, key="x")
    path = tmp_path / "smoke/results/SMOKE-1.json"
    row = json.loads(path.read_text())
    row["gold"] = "changed"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="metadata mismatch"):
        runner.run(tmp_path, smoke=True, transport=tr, key="x")
    protocol = tmp_path / "protocol.json"
    spec = json.loads(protocol.read_text())
    spec["model"] = "changed"
    protocol.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="protocol manifest mismatch"):
        runner.run(tmp_path, smoke=True, transport=tr, key="x")
