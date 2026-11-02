"""Invoice review with durable human hand-off. Offline demo: python examples/invoice_workflow.py --demo DIR.

This is an application example, not a new solvi API. Its file inbox is for one trusted local operator;
it is not an authenticated review service. No payment is made. Statistical guarantees need calibration.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import time
import uuid
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field

import solvi
from solvi import Answer, Span
from solvi.experimental.process import HumanAnswer, HumanChannel, Process
from solvi.models import llm

POLICY_VERSION = "invoice-match-v1"
OPTIONS = ("approve", "reject")


def publish(path, value):
    """Atomically create an immutable JSON file; a duplicate never overwrites the first answer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


class FileInbox(HumanChannel):
    """Immutable requests and answers survive a fresh Python process. Local trusted filesystem only."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, ticket, suffix):
        if not re.fullmatch(r"review-[0-9a-f]{32}", ticket):
            raise ValueError("invalid review ticket")
        return self.folder / f"{ticket}.{suffix}.json"

    def submit(self, request):
        ticket = "review-" + uuid.uuid4().hex
        publish(self._path(ticket, "request"), request.to_dict())
        return ticket

    def poll(self, ticket):
        if not self._path(ticket, "request").exists():
            raise KeyError(ticket)
        path = self._path(ticket, "outcome")
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["kind"] == "cancel":
            return None
        return HumanAnswer(value["answer"], value["reviewer"], value.get("note"), at=value["at"])

    def answer(self, ticket, answer, reviewer, note=None):
        if answer not in OPTIONS or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("answer must be approve/reject and reviewer must be named")
        if not self._path(ticket, "request").exists():
            raise KeyError(ticket)
        publish(self._path(ticket, "outcome"), {"kind": "answer", "answer": answer,
                "reviewer": reviewer, "note": note, "at": time.time()})

    def cancel(self, ticket, why):
        if not self._path(ticket, "request").exists():
            raise KeyError(ticket)
        publish(self._path(ticket, "outcome"), {"kind": "cancel", "why": str(why), "at": time.time()})

    def fingerprint(self):
        # Moving a workflow with its inbox must not change the logical channel identity.
        return "invoice-file-inbox-v1"


class Invoice(BaseModel):
    invoice_number: Span[str] = Field(description="The invoice number exactly as written.")
    total: Span[Decimal] = Field(description="The final total amount payable exactly as written, with its sign.")
    currency: Span[str] = Field(description="The ISO currency code of the total exactly as written.")


def invoice_policy(facts, state):
    """Approval means matching an independently supplied order, not arithmetic consistency of an invoice."""
    return "approve" if (facts.total == Decimal(state["order_total"])
                         and facts.currency.strip().upper() == state["order_currency"]
                         and facts.invoice_number.strip()) else "reject"


def make_recipe(model, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    return solvi.document(Invoice, model, task=f"Invoice matching, policy {POLICY_VERSION}.",
                          compute=invoice_policy, answer=Answer.choice(list(OPTIONS)),
                          storage=Path(folder) / "decisions.jsonl")


class InvoiceWorkflow:
    def __init__(self, folder, model):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.recipe = make_recipe(model, self.folder)
        self.inbox = FileInbox(self.folder / "inbox")
        specfile = self.folder / "process-spec.json"
        if specfile.exists():
            self.process = Process.from_spec(json.loads(specfile.read_text()),
                            systems={"invoice": self.recipe.system}, human=self.inbox,
                            storage=self.folder / "process.jsonl")
        else:
            self.process = Process(POLICY_VERSION, storage=self.folder / "process.jsonl", human=self.inbox)
            self.process.node("invoice", self.recipe.system, reads=["doc", "state"],
                              writes={"verdict": "answer"}, question="answer", end=True)
            publish(specfile, self.process.spec())

    def run(self, doc, order_total, currency):
        amount = Decimal(str(order_total))
        if not amount.is_finite() or not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("order total must be finite and currency an uppercase ISO code")
        given = self.recipe.given({"doc": doc, "order_total": str(amount), "order_currency": currency})
        return self.process.run(given)

    def operator_state(self, result=None):
        guard = self.recipe.system.guards.get("answer")
        promise = guard.promise.record() if guard is not None else None
        replay = self.process.replay_all()
        return {"policy_version": POLICY_VERSION, "spec_hash": self.process.spec()["hash"],
                "system_fingerprint": self.recipe.system.fingerprint(),
                "model_fingerprint": self.recipe.model.fingerprint(),
                "statistical_promise": promise, "label_source": "none" if promise is None else "calibration required",
                "drift": {"watched": False, "reason": "no live-stream monitor configured"},
                "pending": self.process.pending(), "records_verified": replay["ok"],
                "replay_failures": replay["failed"], "result": None if result is None else {
                    "run": result.run, "status": result.status, "answer": result.answer,
                    "automatic": result.automatic, "reason": result.why,
                    "ticket": result.ticket, "risk": result.risk}}


class DemoReader:
    """A deterministic stand-in, deliberately limited to labelled lines; never a real extraction benchmark."""

    def __call__(self, req, timeout=None):
        body = json.loads(req.data)
        message = body["messages"][-1]["content"]
        task, text = message.split("<text>", 1)
        field = "invoice_number" if "invoice number" in task else "currency" if "ISO currency" in task else "total"
        match = re.search(rf"^{field}: (.+)$", text, re.MULTILINE)
        content = {"answer": match.group(1) if match else "not present", "confidence": 0.99, "quote": ""}
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(content)},
                                                   "finish_reason": "stop"}]}).encode())


def demo_model():
    return llm("http://offline/v1", "invoice-demo", opener=DemoReader(), workers=1,
               logprobs=False, ask="confidence", retries=0)


def demo(folder):
    w = InvoiceWorkflow(folder, demo_model())
    auto = w.run("invoice_number: A-1\ntotal: 123.45\ncurrency: EUR\n", "123.45", "EUR")
    pending = w.run("invoice_number: A-2\ncurrency: EUR\n", "123.45", "EUR")
    restarted = InvoiceWorkflow(folder, demo_model())
    restarted.inbox.answer(pending.ticket, "reject", "demo-reviewer", "required total missing")
    reviewed = restarted.process.resume(pending.run)
    return {"automatic": restarted.operator_state(auto), "reviewed": restarted.operator_state(reviewed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(demo(args.demo), ensure_ascii=False, indent=2, default=str))
