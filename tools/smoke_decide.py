"""End-to-end smoke test of a solvi-decide checkpoint through solvi, on both backends.

Every kind the checkpoint declares goes through a real System: choice, multi, score, yes/no, "not stated" (Maybe[...]),
spans (Span[str], Span[float] — the tokenizer-offsets path of the pointer), evidence quotes (require_evidence), rank and
number (Estimate). Then several questions in one pass over a JSON state, replay, a JSON round trip, torch vs ONNX
agreement, and latency. Nothing here depends on a particular model's accuracy: the checks are about the integration (valid
answers, grounded quotes, replay, backend parity); the answers are printed for a human to judge.

    uv run --with torch --with transformers --with onnxruntime --with tokenizers --with huggingface_hub \
        python tools/smoke_decide.py ~/.cache/solvi_release/decide-typed-v2 [--backend torch|onnx|both]

Exit code 1 when an integration check fails."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from solvi import Catalog, Estimate, Maybe, Rank, Response, Scale, Span, System, Unknown
from solvi.decide import DecideModel

TEXTS = {
    "claim": ("Claim 311. The parcel arrived with a cracked screen. Repair quote: 149.90 EUR. Signed by the customer. "
              "Please call me, phone is best; email is fine too. The shop says the repair takes about 5 days."),
    "sparse": "Claim 312. Box was wet on arrival, contents look fine. Amount: 20 EUR. Write to me.",
    "ticket": ("Hi, I was charged twice for order 5521 and the app crashes when I open the invoice. This is the third time "
               "I write. Fix it today or I cancel. -- Dana Whitfield, Berlin"),
}
STATE = {"ticket": {"subject": "Double charge", "body": "Charged twice for order 5521, please refund.",
                    "customer": {"tier": "pro", "since": 2019}, "channel": "email"}}


class Triage(BaseModel):                    # several questions in one pass (block layout when the checkpoint has it)
    team: Literal["billing", "technical", "shipping"] = Field(description="Which team should handle this ticket?")
    urgency: Scale[Literal["low", "medium", "high", "critical"]] = Field(description="How urgent is it?")
    angry: bool = Field(description="Is the customer angry?")
    topics: list[Literal["refund", "delay", "bug"]] = Field(description="What does the ticket mention?")


FAILS = []


def check(ok, what):
    print(f"    [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILS.append(what)


def primitives_catalog(m):
    cat = Catalog()
    qs = [m.decision("signed", "Is the claim signed?", "doc", Maybe[bool]).question(cat),
          m.decision("amount", "What is the claimed amount?", "doc", Maybe[Span[float]]).question(cat),
          m.decision("item", "Which item is damaged?", "doc", Maybe[Span[str]]).question(cat),
          m.decision("damaged", "Is the item damaged?", "doc", bool, evidence=True).question(cat, require_evidence=True),
          m.decision("contact", "Which contact channels does the customer prefer?", "doc",
                     Rank[Literal["email", "phone", "letter"], 2]).question(cat),
          m.decision("repair", "How many days does the repair take?", "doc", Maybe[Estimate[0, 3, 7, 14]]).question(cat),
          m.decision("mood", "How upset is the customer?", "doc", Maybe[Scale[Literal["calm", "annoyed", "angry"]]]).question(cat),
          m.decision("kind", "What kind of claim is it?", "doc", Maybe[Literal["damage", "loss", "delay"]]).question(cat)]
    return cat, qs


def run(path, backend):
    print(f"\n=== {backend} ===")
    t0 = time.perf_counter()
    m = DecideModel.load(path, backend=backend, device="cpu")
    print(f"  loaded in {time.perf_counter() - t0:.1f} s: {m.backend}; format {m.meta.get('format')} / "
          f"{m.meta.get('subformat', '-')}; modes {m.caps['modes']}; not stated {m.has_unknown}; pointer {m.has_pointer}; "
          f"act {m.has_act}; batchable {m.batchable}; fp {m.weights_fingerprint()}")
    out = {}

    # 1. answer primitives from a text: the full layout, pointer over the tokenizer's offsets
    if m.has_unknown and m.has_pointer:
        cat, qs = primitives_catalog(m)
        s = System(cat, qs)
        for name in ("claim", "sparse"):
            doc = TEXTS[name]
            t0 = time.perf_counter()
            res = s.ask({"doc": doc})
            ms = (time.perf_counter() - t0) * 1000
            print(f"  [{name}] {ms:.0f} ms, overall confidence {res.confidence:.2f}")
            for q, r in res.results.items():
                ev = "; ".join(f"{e.value!r}[{e.start}:{e.end}]" for e in (r.evidence or []))
                ans = "not stated" if r.answer is Unknown else r.answer
                print(f"    {q:8s} {ans!r:32s} {r.status:8s} conf {r.confidence:.2f}  {r.guard or ''} {ev}")
                for e in r.evidence or []:          # the offsets path: every quote literally in the text at its offsets
                    check(doc[e.start:e.end] == e.value, f"{name}.{q}: quote {e.value!r} is doc[{e.start}:{e.end}]")
                out[(name, q)] = ans if not isinstance(ans, float) else round(ans, 4)
            check(all(r.status in ("ok", "abstain") for r in res.results.values()), f"{name}: every answer ok or abstain")
            check(res.trace.replay(s)["ok"], f"{name}: trace replays")
            back = Response.from_json(res.to_json(), catalog=s)
            check(all(back[q].answer == res[q].answer for q in res.results), f"{name}: JSON round trip keeps the answers")
            check(back.trace.replay(s)["ok"], f"{name}: the loaded trace replays")
        r = s.ask({"doc": TEXTS["claim"]})
        check(r["amount"].status != "ok" or isinstance(r["amount"].answer, float), "Span[float] is parsed to a float")
        check(r["damaged"].status != "ok" or bool(r["damaged"].evidence), "require_evidence: an ok answer has a quote")
    else:
        print("  (no 'not stated' / pointer: the primitives part is skipped)")

    # 2. typed decisions over a JSON state, several questions per pass (the block layout when the backend can run it; the
    # ONNX export has no block inputs, so there it is one question per pass — compared with torch in the same layout)
    layouts = [("block" if m.batchable and backend == "torch" else "single", m)]
    if layouts[0][0] == "block":
        layouts.append(("single", DecideModel.load(path, backend=backend, device="cpu", multi_question=False)))
    for layout, mm in layouts:
        cat = Catalog()
        s = System(cat, mm.questions(cat, Triage, text_fact="ticket"))
        before = mm.passes
        t0 = time.perf_counter()
        res = s.ask(STATE)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [state, {layout}] {ms:.0f} ms, {mm.passes - before} forward pass(es) for 4 questions")
        for q, r in res.results.items():
            print(f"    {q:8s} {r.answer!r:32s} {r.status:8s} conf {r.confidence:.2f}  {r.guard or ''}")
            out[(f"state/{layout}", q)] = r.answer
        check(res.trace.replay(s)["ok"], f"state ({layout}): trace replays")
        if layout == "block":
            check(mm.passes - before == 1, "state (block): one shared pass for 4 questions")

    # 3. latency: one yes/no question on a short text, warm
    p = m.decision("angry", "Is the customer angry?", "doc", bool)
    p(doc=TEXTS["ticket"])
    t0 = time.perf_counter()
    for i in range(5):
        m._cache.clear()
        p(doc=TEXTS["ticket"] + " " * i)
    print(f"  latency, one yes/no question (warm, CPU): {(time.perf_counter() - t0) / 5 * 1000:.0f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.environ.get("SOLVI_DECIDE_MODEL",
                                                               "~/.cache/solvi_release/decide-typed-v2"))
    ap.add_argument("--backend", default="both", choices=["torch", "onnx", "both"])
    a = ap.parse_args()
    path = os.path.expanduser(a.path)
    backends = ["torch", "onnx"] if a.backend == "both" else [a.backend]
    outs = {b: run(path, b) for b in backends}
    if len(outs) == 2:
        t, o = outs["torch"], outs["onnx"]
        common = set(t) & set(o)                  # the same questions in the same layout
        same = [k for k in common if t[k] == o[k]]
        diff = {k: (t[k], o[k]) for k in common if t[k] != o[k]}
        print(f"\n=== torch vs onnx: {len(same)} answers identical, {len(diff)} differ")
        for k, v in sorted(diff.items()):
            print(f"    {k}: torch {v[0]!r}  onnx {v[1]!r}")
        check(len(diff) <= max(1, len(same) // 10), "torch and ONNX agree (at most 10% differ: ONNX is fp16)")
    print(f"\n{len(FAILS)} integration check(s) failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
