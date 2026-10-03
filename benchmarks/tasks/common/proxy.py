"""A local OpenAI-compatible endpoint in front of OpenRouter, with the cache, ledger and budget of common/llm.py, for
the solvi solutions: `solvi.core.deciders.llm` and `solvi.core.slow.generate` talk to it as to any chat-completions server. The key stays here.

    OPENROUTER_API_KEY=... python common/proxy.py --cache cache          # listens on 127.0.0.1:8765
    base_url = "http://127.0.0.1:8765/<tag>/v1"     # the path before /v1 is the tag of the call in the ledger

--offline answers from the cache only: a request that is not there gets HTTP 422 (solvi escalates that decision) and is
listed in <cache>/misses.jsonl, so a dry run shows what a rerun would cost before anything is paid.
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import CONFIG, BudgetReached, NotCached, options, parse, raw  # noqa: E402

_misses = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "only /v1/chat/completions"}})
        tag = self.path.split("/v1/")[0].strip("/") or "proxy"
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        try:
            self._send(200, raw(body, tag=tag))
        except NotCached as e:
            chars = sum(len(str(m.get("content") or "")) for m in body.get("messages", []))
            with _misses:
                with open(CONFIG["cache"] / "misses.jsonl", "a") as f:
                    f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "tag": tag, "model": body.get("model"),
                                        "prompt_chars": chars, "max_tokens": body.get("max_tokens")}) + "\n")
            self._send(422, {"error": {"message": str(e)}})
        except BudgetReached as e:
            self._send(402, {"error": {"message": f"budget reached: {e}"}})
        except Exception as e:                               # noqa: BLE001 - the caller sees the reason, the proxy stays up
            self._send(502, {"error": {"message": str(e)[:300]}})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    p = options("An OpenAI-compatible caching endpoint for the solvi solutions.")
    p.add_argument("--port", type=int, default=8765)
    a = parse(p)
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    print(f"proxy on http://127.0.0.1:{a.port}/<tag>/v1, cache {a.cache}{', offline' if a.offline else ''}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
