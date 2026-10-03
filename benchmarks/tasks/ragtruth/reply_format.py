"""RAGTruth: the same LLM judge asked through solvi.core.deciders.llm with the reply format enforced (`response_format="json_schema"`)
and with the default for a reasoning request (the reply contract in the prompt), next to the plain call of baseline.py.

Some providers behind one model name apply an enforced format from the first token and skip the thinking; this
measures what that costs the judge. The judge is solution.py's `judge_a` (the baseline's question).

    python ragtruth/reply_format.py [--llm http://127.0.0.1:8765/ragtruth/v1] [--split dev]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DEFAULT, NotCached, chat_many, json_in, options, parse  # noqa: E402
from baseline import PROMPT  # noqa: E402
from score import prf  # noqa: E402
from solution import TASK_A, doc_of, load  # noqa: E402


def plain_call(url, model, prompt):
    """baseline.py's request, sent to the same endpoint as solvi's (so one cache answers both)."""
    import urllib.request
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500, "temperature": 0.0,
            "reasoning": {"effort": "low"}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=400) as r:
        return json.loads(r.read())["choices"][0]["message"].get("content")


def main():
    p = options(__doc__)
    p.add_argument("--llm", default="http://127.0.0.1:8765/ragtruth/v1")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--split", default="dev")
    a = parse(p)
    from solvi.core.deciders.llm import llm
    rows = load(a.split)
    gold = [r["hallucinated"] for r in rows]
    out = {}
    try:                                     # baseline.py's own cache, else the endpoint's
        plain = chat_many(a.model, [PROMPT.format(**r) for r in rows], tag="ragtruth/baseline", max_tokens=1500)
    except NotCached:
        with ThreadPoolExecutor(8) as ex:
            plain = list(ex.map(lambda r: plain_call(a.llm, a.model, PROMPT.format(**r)), rows))
    out["plain call (baseline.py)"] = prf([(g, bool((json_in(x) or {}).get("hallucinated"))) for g, x in zip(gold, plain)])
    for fmt in ("json_schema", "auto"):
        model = llm(a.llm, a.model, api_key="stand", timeout=300, max_tokens=1500, workers=8, logprobs=False,
                    response_format=fmt, extra_body={"reasoning": {"effort": "low"}})
        ds = model.decision("judge_a", TASK_A, "doc", bool).decide([doc_of(r) for r in rows])
        unthought = sum((d.extra.get("llm") or {}).get("reasoning") == "none" for d in ds)
        out[f"solvi.core.deciders.llm, response_format={fmt}"] = {**prf([(g, bool(d.value)) for g, d in zip(gold, ds)]),
                                                    "escalated": sum(bool(d.escalate) for d in ds), "no_reasoning": unthought}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
