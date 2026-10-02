"""τ-bench retail without solvi: the benchmark's tool-calling agent — the policy as the system prompt, the model picks a
tool or writes to the customer, one action per turn; the simulated customer is another model (harness.py).

    python taubench/baseline.py [--model openai/gpt-oss-120b] [--n 5] [--cache DIR]"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DATA, options, parse, write_jsonl  # noqa: E402
from harness import chat, run_task  # noqa: E402
from score import score  # noqa: E402


def make_agent(model):
    def agent(s):
        messages = [{"role": "system", "content": s.wiki}, {"role": "user", "content": s.first}]
        while not s.done:
            m = chat(model, messages, tag="taubench/baseline", max_tokens=3000, reasoning="low", full=True, tools=s.tools)
            calls = (m.get("tool_calls") or [])[:1]
            if calls:
                f = calls[0]["function"]
                try:
                    kwargs = json.loads(f["arguments"] or "{}")
                except json.JSONDecodeError:
                    kwargs = {}
                obs = s.call(f["name"], **kwargs)
                messages += [{"role": "assistant", "content": m["content"] or "", "tool_calls": calls},
                             {"role": "tool", "tool_call_id": calls[0]["id"], "name": f["name"], "content": obs}]
            else:
                obs = s.say(m["content"] or "")
                messages += [{"role": "assistant", "content": m["content"] or ""}, {"role": "user", "content": obs}]
    return agent


def main():
    p = options(__doc__)
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--n", type=int, default=None, help="the first n of the 30 test tasks")
    p.add_argument("--out", default=None)
    a = parse(p)
    ids = json.loads((DATA / "taubench/prepared/eval_30_ids.json").read_text())[: a.n]
    with ThreadPoolExecutor(5) as ex:
        rows = list(ex.map(lambda i: run_task(int(i.rsplit("_", 1)[1]), make_agent(a.model)), ids))
    out = write_jsonl(a.out or Path(__file__).parent / "runs" / "baseline.jsonl", rows)
    print(json.dumps(score(out), indent=1))


if __name__ == "__main__":
    main()
