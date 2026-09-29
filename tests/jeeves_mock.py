"""A stand-in Jeeves server (github.com/PostHog/jeeves, MIT) for tests: a real HTTP server on a free local port that
validates requests and builds replies as Jeeves's `inference/api.py` does (at the 2026-09-29 release), with keyword
probabilities in place of the 9B model.

What is Jeeves's:
- `POST /v1/systemone`: `state` required, `questions` a non-empty object, `model` a string (default "jeeves-latest");
  a question has only `type` (choice | noul | score), `instructions`, `criteria` — any other field is a 422; choice
  criteria an object of 1..255 options, noul criteria null or only {true, false}, score criteria a list of 1..255 levels;
- `options` null or an object of only think / max_think / nothink_threshold / return_reasoning (else 422), typed as
  Jeeves checks them;
- the reply: `model`, `answers` ({type, choice, confidence, probabilities} | {type: "noul", noul} | {type: "score",
  score, legend, probabilities, confidence}; numbers rounded to 2 places, choice confidence (max − 1/k) / (1 − 1/k)),
  `usage` {input_tokens, output_tokens, reasoning_tokens}, `latency_ms`, and `reasoning` {question: {thought, closed,
  tokens, text}} when `return_reasoning` is set; errors are 422 {"detail": ...}, 404 for other paths.

The reasoning text is long and ends with a sentence naming a *different* answer than the probabilities choose, so a test
can check that solvi reads the pointer head's probabilities and never the chain."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "jeeves-latest"
MAX_OPTIONS = 255
DEFAULTS = {"think": True, "max_think": 2560, "nothink_threshold": None, "return_reasoning": False}
KEYWORDS = {"billing": ("charged", "invoice", "refund"), "shipping": ("parcel", "late", "courier"),
            "yes": ("urgent", "charged"), "no": ("parcel",), "calm": ("thanks",), "frustrated": ("late",),
            "very angry": ("furious", "!!")}
CHAIN = "Let me think about the state step by step. " * 60      # ~2,600 characters: long enough to be truncated


def parse_options(raw):
    if raw is None:
        return dict(DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError("options must be an object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    o = {**DEFAULTS, **raw}
    if not isinstance(o["think"], bool) or not isinstance(o["return_reasoning"], bool):
        raise ValueError("think and return_reasoning must be booleans")
    if isinstance(o["max_think"], bool) or not isinstance(o["max_think"], int) or o["max_think"] < 0:
        raise ValueError("max_think must be a non-negative integer")
    t = o["nothink_threshold"]
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1):
        raise ValueError("nothink_threshold must be null or a number in [0, 1]")
    return o


def parse_question(qid, raw):
    if not isinstance(raw, dict):
        raise ValueError(f"question {qid!r} must be an object")
    kind = raw.get("type")
    if kind not in ("choice", "noul", "score"):
        raise ValueError(f"question {qid!r}: type must be one of ['choice', 'noul', 'score']")
    unknown = set(raw) - {"type", "instructions", "criteria"}
    if unknown:
        raise ValueError(f"question {qid!r}: unknown fields {sorted(unknown)}")
    c = raw.get("criteria")
    if kind == "choice" and not (isinstance(c, dict) and 1 <= len(c) <= MAX_OPTIONS):
        raise ValueError(f"question {qid!r}: choice criteria must be an object with 1..{MAX_OPTIONS} options")
    if kind == "noul" and c is not None and not (isinstance(c, dict) and set(c) <= {"true", "false"}):
        raise ValueError(f"question {qid!r}: noul criteria may only describe true and false")
    if kind == "score" and not (isinstance(c, list) and 1 <= len(c) <= MAX_OPTIONS):
        raise ValueError(f"question {qid!r}: score criteria must be a list of 1..{MAX_OPTIONS} levels")
    return raw


def parse_request(body):
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in body:
        raise ValueError("state is required")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise ValueError("questions must be a non-empty object")
    model = body.get("model", MODEL_ID)
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    return {k: parse_question(str(k), v) for k, v in qs.items()}, model, parse_options(body.get("options"))


def r2(x):
    return round(float(x), 2)


def _hits(text, option):
    return any(k in text for k in KEYWORDS.get(option.lower(), (option.lower(),)))


def probabilities(state, q):
    """Keyword probabilities: the options' in order (noul: [false, true]; score: its levels)."""
    text = (state if isinstance(state, str) else json.dumps(state)).lower()
    if q["type"] == "noul":
        ins = str(q.get("instructions", ""))
        if "not stated" in ins:
            p = 0.9 if text.strip() == "hello" else 0.05
        elif "Does the option" in ins:                    # solvi's multi-label: one noul per option
            opt = ins.split("Does the option ", 1)[1].split("'")[1]
            p = 0.95 if _hits(text, opt) else 0.1
        else:
            p = 0.9 if _hits(text, "yes") else 0.1
        return [1 - p, p]
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(x) for x in q["criteria"]]
    if text.strip() == "hello" and "not stated" in keys:
        raw = [8.0 if k == "not stated" else 1.0 for k in keys]
    else:
        raw = [1.0 + 8.0 * _hits(text, k) for k in keys]
    z = sum(raw)
    return [r / z for r in raw]


def answer(q, p):
    if q["type"] == "noul":
        return {"type": "noul", "noul": r2(p[1])}
    k = len(p)
    best = max(range(k), key=lambda i: p[i])
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return {"type": "choice", "choice": keys[best], "confidence": r2(1.0 if k == 1 else (p[best] - 1 / k) / (1 - 1 / k)),
                "probabilities": {kk: r2(v) for kk, v in zip(keys, p)}}
    conf = 1.0 if k == 1 else 1.0 - sum(pi * abs(i - best) for i, pi in enumerate(p)) / (k - 1)
    return {"type": "score", "score": r2(sum(i * pi for i, pi in enumerate(p))),
            "legend": {str(i): str(x) for i, x in enumerate(q["criteria"])},
            "probabilities": {str(i): r2(v) for i, v in enumerate(p)}, "confidence": r2(conf)}


def respond(body):
    """A request body → Jeeves's reply (ValueError → 422)."""
    qs, model, opts = parse_request(body)
    answers, reasoning, spent = {}, {}, 0
    for qid, q in qs.items():
        p = probabilities(body["state"], q)
        a = answers[qid] = answer(q, p)
        sure = opts["nothink_threshold"] is not None and max(p) >= opts["nothink_threshold"]
        thought = opts["think"] and not sure
        tokens = min(opts["max_think"], 640) if thought else 0
        spent += tokens
        wrong = next((k for k in (a.get("probabilities") or {}) if k != a.get("choice")), "the other one")
        reasoning[qid] = {"thought": thought, "closed": thought and tokens < opts["max_think"], "tokens": tokens,
                          "text": (CHAIN + f"So the answer must be {wrong}.") if thought else ""}
    out = {"model": model, "answers": answers,
           "usage": {"input_tokens": 40 * len(qs) + len(str(body["state"])) // 4, "output_tokens": 16 * len(qs),
                     "reasoning_tokens": spent},
           "latency_ms": 300.0 + 5.0 * spent}
    if opts["return_reasoning"]:
        out["reasoning"] = reasoning
    return out


class JeevesMock:
    """`with JeevesMock() as url:` — a Jeeves stand-in on a free local port (or `port=`); `.bodies` holds the requests
    it received, `.statuses` the HTTP status of each reply."""

    def __init__(self, port=0):
        self.port, self.bodies, self.statuses = port, [], []

    def __enter__(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def send(self, code, payload):
                data = json.dumps(payload).encode()
                mock.statuses.append(code)
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/") == "/v1/models":
                    self.send(200, {"models": [{"name": n} for n in (MODEL_ID, "jev-latest")]})
                else:
                    self.send(404, {"detail": "not found"})

            def do_POST(self):
                if self.path.rstrip("/") != "/v1/systemone":
                    self.send(404, {"detail": "not found"})
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"null")
                    mock.bodies.append(body)
                    self.send(200, respond(body))
                except (ValueError, KeyError) as e:
                    self.send(422, {"detail": str(e)})

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
