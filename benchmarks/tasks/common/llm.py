"""The stand's plain chat-completions client: no solvi in it. Baselines call a model with `chat`; the solvi solutions
reach a model through `proxy.py`, which shares this cache, ledger and budget.

Every answer is cached on disk by the hash of the request, so a rerun of the same request costs nothing; every paid
call is a line of `<cache>/spend.jsonl` with the cost the server reports; a call is refused once the ledger reaches the
budget. The key is read from OPENROUTER_API_KEY and sent only in the Authorization header.

    from common.llm import chat, configure
    configure(cache="cache", budget=1.0)
    text = chat("openai/gpt-oss-120b", [{"role": "user", "content": "..."}], tag="ragtruth/baseline")

Cache layout: `<cache>/chat/` (what `chat` returns), `<cache>/raw/` (whole responses, for the proxy), `<cache>/spend.jsonl`.
"""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]                              # benchmarks/tasks
DATA = Path(os.environ.get("STAND_DATA") or HERE / "data")              # where fetch.sh and prepare.py put the data
DEFAULT = "openai/gpt-oss-120b"
OPENROUTER = "https://openrouter.ai/api/v1"
CONFIG = {"cache": HERE / "cache", "upstream": OPENROUTER, "budget": 2.0, "offline": False}
_lock = threading.Lock()
_total = {}


class BudgetReached(RuntimeError):
    pass


class NotCached(RuntimeError):
    """An offline run met a request that is not in the cache."""


def configure(cache=None, upstream=None, budget=None, offline=None):
    """Where answers are cached, which server answers a request that is not, the budget in dollars, and offline mode
    (a request that is not cached raises NotCached instead of being sent)."""
    for k, v in (("cache", cache), ("upstream", upstream), ("budget", budget), ("offline", offline)):
        if v is not None:
            CONFIG[k] = Path(v) if k == "cache" else v


def options(description):
    """An argument parser with the stand's common options: --cache, --upstream, --budget, --offline."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cache", default=str(CONFIG["cache"]), help="cache directory (default: benchmarks/tasks/cache)")
    p.add_argument("--upstream", default=OPENROUTER, help="chat-completions API root for requests not in the cache")
    p.add_argument("--budget", type=float, default=CONFIG["budget"], help="stop paying once the ledger reaches this (dollars)")
    p.add_argument("--offline", action="store_true", help="never send a request: one that is not cached is an error")
    return p


def parse(parser, argv=None):
    a = parser.parse_args(argv)
    configure(cache=a.cache, upstream=a.upstream, budget=a.budget, offline=a.offline)
    return a


def ledger():
    return CONFIG["cache"] / "spend.jsonl"


def spent(tag=None):
    """Dollars in the ledger (for one tag prefix, or in all)."""
    if not ledger().exists():
        return 0.0
    rows = [json.loads(x) for x in ledger().read_text().splitlines() if x.strip()]
    return round(sum(r["cost"] for r in rows if tag is None or r["tag"].startswith(tag)), 6)


def _key(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _post(body, retries, deadline):
    """POST the body to the upstream server → its JSON response. Retries network errors and 5xx, not a refusal."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and CONFIG["upstream"].startswith(OPENROUTER):
        raise RuntimeError("set OPENROUTER_API_KEY (or run --offline from a filled cache)")
    err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(CONFIG["upstream"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key or 'none'}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:           # a gateway's keep-alive bytes can hold a read open
                buf, t0 = b"", time.time()                               # for as long as a slow provider generates
                while chunk := r.read1(65536):
                    buf += chunk
                    if time.time() - t0 > deadline:
                        raise TimeoutError(f"no complete answer in {deadline} s")
                data = json.loads(buf)
            if "choices" not in data:
                raise RuntimeError(str(data.get("error", data))[:300])
            return data
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError, ConnectionError) as e:
            err = e
            code = getattr(e, "code", None)
            if code is not None and 400 <= code < 500 and code not in (408, 409, 429):
                raise RuntimeError(f"HTTP {code}: {e.read()[:300]!r}") from None
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"no answer after {retries} tries: {err}")


def _pay(tag, model, usage):
    cost = float(usage.get("cost") or 0.0)
    with open(ledger(), "a") as led:
        led.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "model": model,
                              "in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0), "cost": cost}) + "\n")
    _total[ledger()] = _total.get(ledger(), 0.0) + cost


def _check_budget():
    with _lock:
        if ledger() not in _total:
            _total[ledger()] = spent()
        if _total[ledger()] >= CONFIG["budget"]:
            raise BudgetReached(f"spent {_total[ledger()]:.3f} of {CONFIG['budget']} dollars")


def chat(model, messages, tag="", max_tokens=2000, temperature=0.0, reasoning="low", full=False, retries=4, deadline=240,
         **extra):
    """One chat completion → its text (full=True: a dict with content, reasoning, tool_calls, usage). Cached by the request."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True}, **extra}
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    h = _key(body)
    f = CONFIG["cache"] / "chat" / h[:2] / f"{h}.json"
    if f.exists():
        out = json.loads(f.read_text())
        return out if full else out["content"]
    if CONFIG["offline"]:
        raise NotCached(f"{tag}: request {h[:12]} is not in {CONFIG['cache']}")
    _check_budget()
    data = _post(body, retries, deadline)
    msg, u = data["choices"][0]["message"], data.get("usage", {})
    out = {"content": msg.get("content") or "", "reasoning": msg.get("reasoning"), "tool_calls": msg.get("tool_calls"),
           "finish": data["choices"][0].get("finish_reason"), "model": data.get("model"), "provider": data.get("provider"),
           "usage": {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0), "cost": float(u.get("cost") or 0.0)}}
    with _lock:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, ensure_ascii=False))
        _pay(tag, model, u)
    return out if full else out["content"]


def raw(body, tag="", retries=4, deadline=240):
    """An OpenAI-style request body → the server's whole response, cached by the body and counted (for proxy.py)."""
    body = {**body, "usage": {"include": True}}
    body.pop("stream", None)
    h = _key(body)
    f = CONFIG["cache"] / "raw" / h[:2] / f"{h}.json"
    if f.exists():
        return json.loads(f.read_text())
    c = CONFIG["cache"] / "chat" / h[:2] / f"{h}.json"
    if c.exists():                     # the same request was sent by `chat` (a baseline's call): its answer, not a new one
        out = json.loads(c.read_text())
        msg = {"role": "assistant", "content": out["content"], "reasoning": out.get("reasoning")}
        if out.get("tool_calls"):
            msg["tool_calls"] = out["tool_calls"]
        return {"model": out.get("model"), "choices": [{"index": 0, "message": msg, "finish_reason": out.get("finish")}],
                "usage": {"prompt_tokens": out["usage"]["in"], "completion_tokens": out["usage"]["out"]}}
    if CONFIG["offline"]:
        raise NotCached(f"{tag}: request {h[:12]} is not in {CONFIG['cache']}")
    _check_budget()
    data = _post(body, retries, deadline)
    with _lock:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, ensure_ascii=False))
        _pay(tag, body.get("model"), data.get("usage", {}))
    return data


def chat_many(model, prompts, tag="", workers=8, **kw):
    """The answers to many prompts (a string or a message list each), in order; a failed call gives None."""
    def one(p):
        try:
            return chat(model, [{"role": "user", "content": p}] if isinstance(p, str) else p, tag=tag, **kw)
        except (BudgetReached, NotCached):
            raise
        except Exception as e:                                # noqa: BLE001 - one failed call is one missing answer
            print("  call failed:", str(e)[:200])
            return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def json_in(text):
    """The first JSON object or array in a model's answer, or None."""
    if not text:
        return None
    for open_, close in (("{", "}"), ("[", "]")):
        a, b = text.find(open_), text.rfind(close)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    return None


if __name__ == "__main__":
    a = parse(options("Spend in the ledger of a cache directory, by task."))
    by = {}
    rows = [json.loads(x) for x in ledger().read_text().splitlines() if x.strip()] if ledger().exists() else []
    for r in rows:
        k = by.setdefault(r["tag"].split("/")[0], [0, 0, 0, 0.0])
        k[0] += 1; k[1] += r["in"]; k[2] += r["out"]; k[3] += r["cost"]   # noqa: E702
    for k, (n, i, o, c) in sorted(by.items()):
        print(f"{k:14s} calls {n:5d}  in {i / 1e6:6.2f}M  out {o / 1e6:6.2f}M  ${c:.4f}")
    print(f"total ${spent():.4f} of ${CONFIG['budget']}")
