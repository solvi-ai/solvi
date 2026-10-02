"""RAGTruth with solvi: is a RAG response supported by its sources — answered by an LLM judge inside solvi, alone only
where a calibrated promise allows, with a verbatim pointer at the unsupported passage.

How it works:
  1. The judge reads one text fact, `doc`: the source material, the writer's task and the response. Two wordings of the
     same yes/no question are asked of one LLM through `solvi.llm` — `judge_a` (the baseline's question) and `judge_c`
     (strict: any detail the source does not state counts as unsupported). solvi validates every reply and keeps the
     judge's quote only when it is literally in the text.
  2. The answer is the judge with the higher F1 on dev parts A and B (chosen here, on dev, never on eval).
  3. Escalation: `Vote([judge_a, judge_c], rule="all")` answers alone only when both judges agree and both are sure
     enough; `act_guard(max_risk=0.10)` on dev part B sets the threshold, so that P(answered alone and wrong) ≤ 10% of
     all responses for inputs like dev.
  4. A System asks both questions per response — `hallucinated` (the vote: an answer or an escalation) and `guess` (the
     chosen judge's answer, what a person taking an escalation is shown) — and stores every decision with its trace.

Dev is split by (kind, label) cell in file order: part A = the first 25 per cell, part B = the next 45.

    python common/proxy.py --cache cache &                          # the LLM endpoint with a cache (OPENROUTER_API_KEY)
    python ragtruth/solution.py [--llm http://127.0.0.1:8765/ragtruth/v1] [--split eval]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, DEFAULT, options, parse, read_jsonl, write_jsonl  # noqa: E402

HERE = Path(__file__).resolve().parent
KINDS = ("QA", "Summary", "Data2txt")
TASK_A = ("Does the RESPONSE contain any statement that contradicts the SOURCE MATERIAL or that the SOURCE MATERIAL does "
          "not support? If yes, the quote must be the unsupported passage, copied from the RESPONSE; if no, the quote is \"\".")
TASK_C = ("Treat as unsupported any detail, step, number, name, reason or piece of advice in the RESPONSE that the SOURCE "
          "MATERIAL does not state, even when it is plausible or common knowledge, and anything that conflicts with the "
          "SOURCE MATERIAL. Does the RESPONSE contain at least one unsupported or conflicting passage? If yes, the quote "
          "must be that passage, copied from the RESPONSE; if no, the quote is \"\".")


def load(split):
    """eval or dev whole; dev_a / dev_b: the first 25 / the next 45 responses of every (kind, label) cell of dev."""
    rows = read_jsonl(DATA / f"ragtruth/prepared/{'dev' if split.startswith('dev') else split}.jsonl")
    if split in ("eval", "dev"):
        return rows
    lo, hi = {"dev_a": (0, 25), "dev_b": (25, 70)}[split]
    return [r for k in KINDS for h in (False, True) for r in [x for x in rows if x["task_type"] == k and x["hallucinated"] == h][lo:hi]]


def doc_of(r):
    return f"SOURCE MATERIAL:\n{r['context']}\n\nTASK GIVEN TO THE WRITER:\n{r['query']}\n\nRESPONSE:\n{r['output']}"


def f1(rows, answers):
    from score import prf
    return prf([(r["hallucinated"], bool(a)) for r, a in zip(rows, answers)])["f1"]


def main():
    p = options(__doc__)
    p.add_argument("--llm", default="http://127.0.0.1:8765/ragtruth/v1", help="chat-completions endpoint (the stand's proxy)")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--split", default="eval")
    p.add_argument("--max-risk", type=float, default=0.10)
    p.add_argument("--out", default=None)
    a = parse(p)
    from solvi import Answer, Catalog, Question, System
    from solvi.llm import llm
    from solvi.multi import Vote

    # logprobs off: OpenRouter serves one model from several providers, some with log-probabilities and some without,
    # and the two kinds of confidence cannot share one threshold. Reasoning on: solvi then puts the reply contract in the
    # prompt and lets the model think before it answers.
    model = llm(a.llm, a.model, api_key="stand", timeout=300, max_tokens=1500, workers=8, logprobs=False,
                extra_body={"reasoning": {"effort": "low"}})
    judges = {"judge_a": model.decision("judge_a", TASK_A, "doc", bool), "judge_c": model.decision("judge_c", TASK_C, "doc", bool)}
    task = {"judge_a": TASK_A, "judge_c": TASK_C}

    # 2. the answer: the judge with the higher F1 on dev A + B
    dev = load("dev_a") + load("dev_b")
    said = {n: [d.value for d in j.decide([doc_of(r) for r in dev])] for n, j in judges.items()}
    dev_f1 = {n: f1(dev, v) for n, v in said.items()}
    chosen = max(dev_f1, key=dev_f1.get)
    print("F1 on dev A+B:", dev_f1, "-> the answer is", chosen)

    # 3. escalation: the vote of both judges, calibrated on dev B
    vote = Vote(list(judges.values()), rule="all", name="hallucinated")
    info = vote.act_guard([(doc_of(r), r["hallucinated"]) for r in load("dev_b")], max_risk=a.max_risk)
    print("act_guard on dev B:", {k: info[k] for k in ("threshold", "answered", "error", "risk", "n")})

    # 4. the system: the vote answers alone or escalates; `guess` is the chosen judge's answer, shown on an escalation
    cat = Catalog()
    cat.fn(model.decision("best", task[chosen], "doc", bool))

    @cat.rule("guess")
    def guess(best):
        return "yes" if best else "no"

    q = vote.question(cat, name="hallucinated", text="Does the response contain anything its sources contradict or do not support?")
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    store = runs / f"decisions_{a.split}.jsonl"
    store.unlink(missing_ok=True)
    system = System(cat, [q, Question("guess", "The best guess, shown to the person who takes an escalation", Answer.yes_no())],
                    storage=str(store))
    rows = load(a.split)
    first = {n: j.decide([doc_of(r) for r in rows]) for n, j in judges.items()}[chosen]   # in parallel first; the asks reuse them
    out = []
    for r, d in zip(rows, first):
        res = system.ask({"doc": doc_of(r)})
        yes = res["guess"].answer == "yes"
        quote = (d.extra.get("llm", {}).get("quote") or [None])[0]   # located by solvi in the text; kept if in the response
        out.append({"id": r["id"], "hallucinated": yes, "escalate": res["hallucinated"].status != "ok",
                    "quote": quote if yes and quote and quote in r["output"] else None})
    path = write_jsonl(a.out or runs / f"solution_{a.split}.jsonl", out)
    print("store verifies:", system.storage.verify()["ok"])
    from score import score
    print(json.dumps(score(path, a.split), indent=1))


if __name__ == "__main__":
    main()
