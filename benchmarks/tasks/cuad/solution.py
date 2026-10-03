"""CUAD with solvi: for a contract and a kind of clause, say whether the contract has one and quote it — every quote
literally in the contract, and a question handed to a person when the answer is not trusted enough.

How it works:
  1. Propose. One span question per kind of clause, `Maybe[Span[str]]`, asked of an LLM through `solvi.core.deciders.llm` with
     `long="retrieve"`: solvi finds the contract's sections by the words of `queries.py` (`retrieve_query`) and the
     model reads about `max_len=3000` tokens of them, not the whole contract. solvi keeps an answer only when it is
     literally in the contract; `repair.py` first cuts an almost-literal passage to its longest literal piece.
  2. Check. A yes / no question about each quoted passage with a little context around it (same LLM); a passage it
     refuses becomes "absent". Not asked for the three header kinds (document name, parties, date).
  3. Trust. A number per answer: how often this kind of answer was right for this kind of clause on dev (the
     reliability table) × the LLM's confidence × the check's p(yes) — three facts a catalog computes.
  4. Promise. `System.guarantee("clause", dev examples, signal="trust", max_risk=0.03)`: below the calibrated threshold
     the question abstains and goes to a person; P(answered alone and wrong) ≤ 3% of the questions, for questions like
     dev. On dev each answer's reliability is taken from a table built without its own contract.

Dev is the first 20 of the 25 dev contracts in sorted order (820 questions).

    python common/proxy.py --cache cache &                          # the LLM endpoint with a cache (OPENROUTER_API_KEY)
    python cuad/solution.py [--llm http://127.0.0.1:8765/cuad/v1] [--split eval]
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DATA, DEFAULT, options, parse, read_jsonl, write_jsonl  # noqa: E402
from queries import QUERY, details, task_of  # noqa: E402
from score import overlaps  # noqa: E402

D = DATA / "cuad/prepared"
HEADER = {"Document Name", "Parties", "Agreement Date"}    # not clauses: the check is not asked for them
CONTEXT = 200                                              # characters around a quoted passage the check reads
SMOOTH = 4.0                                               # the reliability table's pull towards the answer's overall share


def slug(category):
    return re.sub(r"\W+", "_", category).strip("_").lower()


def propose(model, rows):
    """→ {id: state}: the LLM's answer for every question of `rows`, checked by solvi and by the yes / no question."""
    from solvi import Maybe, Span, Unknown
    from solvi.core.deciders.llm import locate
    from repair import literal_piece
    out = {}
    for c in sorted({r["contract"] for r in rows}):
        text = (D / "contracts" / f"{c}.txt").read_text(encoding="utf-8")
        items = [r for r in rows if r["contract"] == c]

        def one(r):
            part = model.decision(slug(r["category"]), task_of(r), "contract", Maybe[Span[str]], long="retrieve",
                                  retrieve_query=QUERY[r["category"]])
            d = part(contract=text)
            v, at = d.value, None
            if v is not Unknown and v is not None and v.end > v.start:
                if not d.escalate:
                    at = (v.start, v.end)
                elif "crosses two sections" in d.escalate:         # two neighbouring sections joined by the window
                    at = locate(str(v.value), text) or literal_piece(str(v.value), text)
            quote = text[at[0]:at[1]] if at else None
            rejected = bool(d.escalate) and quote is None
            check = None
            if quote and r["category"] not in HEADER:
                q = model.decision("is_" + slug(r["category"]),
                                   f'Is there a "{r["category"]}" clause in this excerpt of a contract? '
                                   f'({" ".join(details(r["question"]).split())}) '
                                   f"Answer no when the excerpt only touches a related topic.", "excerpt", bool)
                check = q(excerpt=text[max(0, at[0] - CONTEXT): at[1] + CONTEXT]).probs.get("yes")
            answer = ("rejected" if rejected else "absent" if not quote
                      else "dropped" if check is not None and check < 0.5 else "kept")
            conf = float(d.conf)
            return {"id": r["id"], "contract": c, "category": r["category"], "answer": answer, "quote": quote,
                    "llm_confidence": conf,          # a passage: P(it is the answer); "not stated": P(the text does not state it)
                    "check": 1.0 if check is None or answer != "kept" else float(check)}
        with ThreadPoolExecutor(6) as ex:
            for s in ex.map(one, items):
                out[s["id"]] = s
        print(f"  {c[:60]:60s} {len(items)} questions", flush=True)
    return out


def right(s, g):
    """Is the answer of a state right, as score.py judges it?"""
    has = not g["is_impossible"]
    return (has and any(overlaps(s["quote"], a["text"]) for a in g["answers"])) if s["answer"] == "kept" else not has


def reliability(states, ok, without=None):
    """{(kind of clause, answer): share right on dev}, smoothed towards the answer's overall share."""
    mean, cells = {}, {}
    for s in states:
        if s["contract"] != without and s["answer"] != "rejected":
            for d, k in ((mean, s["answer"]), (cells, (s["category"], s["answer"]))):
                d.setdefault(k, [0, 0])
                d[k][0] += ok[s["id"]]
                d[k][1] += 1
    mu = {k: a / n for k, (a, n) in mean.items()}
    table = {k: (a + SMOOTH * mu[k[1]]) / (n + SMOOTH) for k, (a, n) in cells.items()}
    return lambda s: table.get((s["category"], s["answer"]), mu.get(s["answer"], 0.5))


def system():
    """The question `clause` answers the proposal's kind; a proposal solvi rejected gives no answer (the rule abstains).
    `trust` is a computed fact the guarantee reads."""
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.fn
    def trust(reliability: float, llm_confidence: float, check: float) -> float:
        return reliability * llm_confidence * check

    @cat.rule("clause")
    def clause(answer):
        return None if answer == "rejected" else answer

    return System(cat, [Question("clause", "Does the contract have this kind of clause, and where?",
                                 Answer.choice(["kept", "absent", "dropped"]))])


def given(s, rel):
    return {"answer": s["answer"], "reliability": rel(s), "llm_confidence": s["llm_confidence"], "check": s["check"]}


def main():
    p = options(__doc__)
    p.add_argument("--llm", default="http://127.0.0.1:8765/cuad/v1", help="chat-completions endpoint (the stand's proxy)")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--split", default="eval")
    p.add_argument("--max-risk", type=float, default=0.03)
    p.add_argument("--dev-contracts", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="only the first N contracts of the split (a pilot)")
    p.add_argument("--out", default=None)
    a = parse(p)
    from repair import opener
    from solvi.core.deciders.llm import llm
    model = llm(a.llm, a.model, api_key="stand", timeout=240, retries=2, response_format="json_schema", max_tokens=2000,
                extra_body={"reasoning": {"effort": "low"}}, workers=6, max_len=3000, opener=opener)

    # dev: the answers, judged against gold, give the reliability table and the calibration examples
    dev_rows = read_jsonl(D / "dev.jsonl")
    keep = set(sorted({r["contract"] for r in dev_rows})[:a.dev_contracts])
    dev_rows = [r for r in dev_rows if r["contract"] in keep]
    gold = {r["id"]: r for r in dev_rows}
    dev = list(propose(model, dev_rows).values())
    ok = {s["id"]: right(s, gold[s["id"]]) for s in dev}
    loo = {c: reliability(dev, ok, without=c) for c in keep}
    sysm = system()
    examples = [(given(s, loo[s["contract"]]), ok[s["id"]]) for s in dev]
    try:                                      # the LLM's own confidence as the signal, for comparison: refused or weak
        r = sysm.guarantee("clause", examples, max_risk=a.max_risk, signal="llm_confidence", correct=lambda result, is_right: is_right)
        print("the LLM's confidence as the signal:", {k: r.get(k) for k in ("threshold", "answered", "error")},
              "AUROC", round(r["separation"]["auroc"], 3))
    except ValueError as e:
        print("the LLM's confidence as the signal:", e)
    rep = sysm.guarantee("clause", examples, max_risk=a.max_risk,
                         signal="trust", correct=lambda result, is_right: is_right)
    print("guarantee on dev:", {k: rep.get(k) for k in ("threshold", "answered", "error", "risk")},
          "AUROC of trust", round(rep["separation"]["auroc"], 3))

    rows = read_jsonl(D / f"{a.split}.jsonl")
    if a.limit is not None:
        some = set(sorted({r["contract"] for r in rows})[:a.limit])
        rows = [r for r in rows if r["contract"] in some]
    full = reliability(dev, ok)
    out = []
    for s in propose(model, rows).values():
        res = sysm.ask(given(s, full), store=False)["clause"]
        out.append({"id": s["id"], "present": s["answer"] == "kept", "quotes": [s["quote"]] if s["answer"] == "kept" else [],
                    "escalate": res.status != "ok"})
    path = write_jsonl(a.out or Path(__file__).parent / "runs" / f"solution_{a.split}.jsonl", out)
    from score import score
    print(json.dumps(score(path, a.split), indent=1))


if __name__ == "__main__":
    main()
