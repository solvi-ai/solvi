"""Any decision model that speaks the System One HTTP API (`POST /v1/systemone`: Jev, and open servers such as Kev,
Jeeves, Von, Laya-serve, Intern-Decision) as a solvi decider — the model proposes, solvi's checks, rules and thresholds
decide.

    from solvi.systemone import systemone
    model = systemone("http://127.0.0.1:8009", "kev-latest")            # api_key= for a hosted service
    part = model.decision("team", "Which team should handle this?", "email", {"billing": "Charges", "shipping": "Delivery"})

One question is one request: a System and decide_pass ask a remote model their questions about an input one after
another (the scorer's own `logits(items)` does put the items it is handed about one text into one request — that is
how a multi-label question's options travel). Choice questions are sent as `choice` (criteria: option → description),
yes/no questions as `noul`, scores as `choice` over their levels (the probability of each level is what solvi needs).
The returned probabilities become the decider's logits (log p), so everything built on a DecideModel works unchanged:
act_guard / conformal / calibrate_for on your labelled examples (the service gives no act signal: solvi uses the
confidence), fit / teach / adapt, the audit and the trace.

What the API has no type for is asked in its terms:

- "not stated" (`not_stated=True`, `Maybe[...]`): one more option, "not stated", with a description ("the input does not
  state it ..."); a yes/no question that allows it is asked as a choice over yes / no / not stated. Its probability
  competes with the options' in one softmax, as with solvi.llm and the checkpoints that have a "not stated" output: the
  decision is `Unknown` when it is the most probable, and a question built on it abstains ("not stated").
- multi-label questions: one `noul` per option (does this option apply?) in the same request; an option is chosen when
  its probability reaches the model's multi threshold (0.5), the confidence is the least sure option's max(p, 1 − p) —
  what act_guard calibrates on. With "not stated" allowed, one more `noul` asks whether the input leaves it unsaid.

Spans and evidence quotes are not part of the API (ValueError).

`extra_body`: server-specific request fields merged into every request (OpenRouter's `provider` routing, `user`; a
thinking decision model's controls, e.g. Jeeves's `{"options": {"max_think": 512, "nothink_threshold": 0.9}}`); the
fields solvi sets (model, state, questions) are refused, and extra_body enters the fingerprint (except Jeeves's
`options.return_reasoning`, which changes the reply, not the answers). Per decision `extra["systemone"]` records the
endpoint (the URL without credentials or query, as the fingerprint and repr show it; the request keeps the query), the
model name (and `served_by` when the service names another), the request's `ms` and, when the service
reports them, its `usage` (input / output / reasoning tokens), `cost` and `latency_ms` — for the whole request, which
answers `questions` questions at once — and the question's `reasoning` when the service returns it (Jeeves with
`return_reasoning`: the text cut to REASONING_CHARS characters, for the audit; the answer is read from the
probabilities, never from that text). The API key is sent in the Authorization header only, never recorded. A service
that does not answer (network errors, timeouts, 429, 5xx: `retries` more attempts with backoff), refuses the request
(another 4xx: its error text) or gives a reply that breaks the contract escalates the decision — never a guess, never
an exception; a failed request is not cached.

A hosted model is not replayed (`deterministic=False`, the default): replay checks the recorded output instead of calling
the service again; `deterministic=True` for a local server whose output is reproducible. The trace records the endpoint
and the model name — not the weights behind them, which the service can change: calibrate again when it does."""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from .decide import DecideModel, _unknown_caps
from .remote import NoAnswer, Refused, RemoteClient, RemoteError
from .remote import tokens as remote_tokens

EPS = 1e-6
REASONING_CHARS = 1000          # the most characters of a question's reasoning text kept in a decision's extra
NOT_STATED = "not stated"
NOT_STATED_DESCRIPTION = ("The input does not state it: the facts this question needs are missing (absent, empty or "
                          "unknown), and nothing that is given decides it.")
# the request fields solvi sets itself: the questions, their names and the reply's reading depend on them
RESERVED = ("model", "state", "questions")


def _extra_body(extra):
    """extra_body → a JSON-safe copy; ValueError for a non-dict, a value that is not JSON or a field solvi sets itself."""
    if extra is None:
        return None
    if not isinstance(extra, dict):
        raise ValueError(f"extra_body is a dict of request fields, not {type(extra).__name__}")
    taken = sorted(k for k in extra if k in RESERVED)
    if taken:
        raise ValueError(f"extra_body cannot set {taken}: solvi sets them itself (model is an argument; the state and "
                         "the questions are the request)")
    try:
        return json.loads(json.dumps(extra, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValueError(f"extra_body is not JSON: {e}") from None


def _prob(v, what):
    """A probability of a reply: a finite number in [0, 1] — else ValueError (the reply breaks the contract: the decision
    escalates). NaN compares false with every threshold, so it would pass them all."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
        raise ValueError(f"{what} is not a probability: {v!r}")
    return float(v)


def _logit(p):
    p = min(max(float(p), EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _noul(it):
    return tuple(o.lower() for o in it.options) in (("yes", "no"), ("true", "false"))


def _not_stated(it):
    """Does this question allow "not stated" as an answer of its own (an option named so is just an option)?"""
    return bool(getattr(it, "unknown", False)) and NOT_STATED not in it.options


class SystemOneError(RemoteError):
    """The service refused the request itself (a wrong key, model or URL: HTTP 401, 403, 404 and other client errors
    except 400 / 413 / 422) — raised, as solvi.llm does, rather than escalated. The message has the endpoint, never the
    key."""


class _Failed(Exception):
    """A reply that is not a JSON object: the reason, for the escalation."""


class SystemOneScorer(RemoteClient):
    """A scorer for DecideModel over `POST {base_url}/v1/systemone` (standard library HTTP, no dependencies; the shared
    client of solvi.remote)."""

    tag = "systemone"
    service = "the System One service"
    error = SystemOneError

    def __init__(self, base_url, model, api_key=None, *, timeout=30.0, opener=None, extra_body=None, retries=2,
                 backoff=1.0, sleep=None):
        super().__init__(base_url, model, api_key, path="/v1/systemone", timeout=timeout, retries=retries,
                         backoff=backoff, opener=opener, sleep=sleep)
        self.extra_body = _extra_body(extra_body)
        self.model_id = f"systemone:{model}"
        self.cost = 0.0                                  # the sum of the replies' usage.cost

    def fingerprint(self):
        fp = f"systemone|{self.endpoint}|{self.model}"
        x = self.extra_body
        if x and isinstance(x.get("options"), dict) and "return_reasoning" in x["options"]:
            # Jeeves's return_reasoning only adds the chains to the reply: the answers, and the fingerprint, stay
            x = {**x, "options": {k: v for k, v in x["options"].items() if k != "return_reasoning"}}
            x = {k: v for k, v in x.items() if k != "options" or v}
        if x:                                            # provider pinning, thinking options ... change what answers
            blob = json.dumps(x, sort_keys=True, ensure_ascii=False)
            fp += "|x:" + hashlib.sha256(blob.encode()).hexdigest()[:16]
        return fp

    # --- questions
    @staticmethod
    def questions(it):
        """An Item → its System One questions {name suffix: question}: one for a choice, yes/no or score question (a yes /
        no question that allows "not stated" is a choice over yes / no / not stated), one noul per option for a
        multi-label question (and one more for "not stated" when it is allowed)."""
        if it.pointer:
            raise ValueError("the System One API gives no spans or evidence quotes")
        ns = _not_stated(it)
        desc = it.descriptions or (None,) * len(it.options)
        if it.multi:
            qs = {f"__{j}": {"type": "noul", "instructions": f"{it.task} Does the option {o!r}"
                             + (f" ({d})" if d else "") + " apply?"} for j, (o, d) in enumerate(zip(it.options, desc))}
            if ns:
                qs["__ns"] = {"type": "noul", "instructions": f"{it.task} Is the answer not stated in the input? "
                              + NOT_STATED_DESCRIPTION}
            return qs
        if _noul(it) and not ns:
            return {"": {"type": "noul", "instructions": it.task}}
        criteria = {o: (d or None) for o, d in zip(it.options, desc)}
        if ns:
            criteria[NOT_STATED] = NOT_STATED_DESCRIPTION
        return {"": {"type": "choice", "instructions": it.task, "criteria": criteria}}

    @staticmethod
    def answer_logits(it, ans):
        """An answer to a single (not multi-label) question → log-probabilities in the item's option order."""
        if ans.get("type") == "noul" or "noul" in ans:
            p = min(max(_prob(ans["noul"], "noul"), EPS), 1 - EPS)
            yes_first = it.options[0].lower() in ("yes", "true")
            return np.log(np.array([p, 1 - p] if yes_first else [1 - p, p]))
        probs = ans.get("probabilities") or {}
        missing = [o for o in it.options if o not in probs]
        if missing:
            raise ValueError(f"the service returned no probability for {missing}")
        return np.log(np.clip(np.array([_prob(probs[o], f"the probability of {o!r}") for o in it.options]), EPS, None))

    @classmethod
    def read(cls, it, answers):
        """The answers to an Item's questions {name suffix: answer} → the scorer's output {"logits", "unknown"?}."""
        ns = _not_stated(it)
        if it.multi:
            z = np.array([_logit(answers[f"__{j}"]["noul"]) for j in range(len(it.options))])
            return {"logits": z, **({"unknown": _logit(answers["__ns"]["noul"])} if ns else {})}
        ans = answers[""]
        z = cls.answer_logits(it, ans)
        if not ns:
            return {"logits": z}
        p = (ans.get("probabilities") or {}).get(NOT_STATED)
        if p is None:
            raise ValueError(f"the service returned no probability for {NOT_STATED!r}")
        return {"logits": z, "unknown": math.log(max(float(p), EPS))}

    # --- one request
    def request(self, body):
        """→ the service's response; NoAnswer after the retries (network errors, timeouts, a broken connection, 408 /
        409 / 429 / 5xx), Refused for HTTP 400 / 413 / 422 (the reason, never the key), SystemOneError for a wrong
        key, model or URL (401, 403, 404, another 4xx)."""
        return self.send(body)

    def body(self, text, questions):
        b = json.loads(json.dumps(self.extra_body)) if self.extra_body else {}      # a fresh copy per request
        b.update({"state": text, "model": self.model, "questions": questions})
        return b

    def _info(self, resp, ms, n):
        """What a decision records of its request (extra["systemone"])."""
        info = {"endpoint": self.endpoint, "model": self.model, "ms": round(ms, 3), "questions": n}
        if resp.get("model") and resp.get("model") != self.model:
            info["served_by"] = str(resp["model"])
        u = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
        tokens = remote_tokens(u)
        if tokens:
            info["usage"] = tokens
        c = u.get("cost")
        if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c):
            info["cost"] = float(c)
        lat = resp.get("latency_ms")                     # the service's own time for the request (Jeeves)
        if isinstance(lat, (int, float)) and not isinstance(lat, bool) and math.isfinite(lat) and lat >= 0:
            info["latency_ms"] = float(lat)
        return info

    @staticmethod
    def _reasoning(it, qs, resp):
        """The reasoning a service returned for an Item's questions (Jeeves with options.return_reasoning) → what the
        decision records: {"text" (at most REASONING_CHARS characters), "tokens", "thought", "closed", "truncated"?} —
        per option for a multi-label question ({option | "not stated": ...}); None when there is none. It is recorded
        for the audit only: the answer is read from the probabilities, never from this text."""
        r = resp.get("reasoning")
        if not isinstance(r, dict):
            return None
        out = {}
        for name, suffix in qs:
            x = r.get(name)
            if not isinstance(x, dict) or not isinstance(x.get("text"), str):
                continue
            text = x["text"]
            rec = {"text": text[:REASONING_CHARS]}
            if len(text) > REASONING_CHARS:
                rec["truncated"] = len(text)
            if isinstance(x.get("tokens"), int) and not isinstance(x.get("tokens"), bool):
                rec["tokens"] = x["tokens"]
            for k in ("thought", "closed"):
                if isinstance(x.get(k), bool):
                    rec[k] = x[k]
            key = "" if not suffix else (NOT_STATED if suffix == "__ns" else it.options[int(suffix[2:])])
            out[key] = rec
        if not out:
            return None
        return out[""] if list(out) == [""] else out

    def logits(self, items):
        by_text = {}
        for i, it in enumerate(items):
            by_text.setdefault(it.text, []).append(i)
        out = [None] * len(items)
        for text, idx in by_text.items():
            names = {f"q{k}": i for k, i in enumerate(idx)}
            qs = {n: self.questions(items[i]) for n, i in names.items()}
            t0 = time.perf_counter()
            try:
                resp = self.request(self.body(text, {n + s: q for n, sub in qs.items() for s, q in sub.items()}))
                if not isinstance(resp, dict):
                    raise _Failed("the System One service's reply is not a JSON object")
            except (NoAnswer, Refused, _Failed) as e:   # escalate every question of the request; not cached: asked
                for i in idx:                            # again next time
                    out[i] = {"logits": np.zeros(len(items[i].options)), "escalate": str(e), "transient": True,
                              "info": {"systemone": {"endpoint": self.endpoint, "model": self.model, "questions": len(idx)}}}
                continue
            ms = (time.perf_counter() - t0) * 1000
            info = self._info(resp, ms, len(idx))
            self.count(info.get("usage", {}))
            self.cost += info.get("cost", 0.0)
            answers = resp.get("answers") if isinstance(resp.get("answers"), dict) else {}
            for n, i in names.items():
                try:
                    gone = [n + s for s in qs[n] if not isinstance(answers.get(n + s), dict)]
                    if gone:
                        raise ValueError(f"the service gave no answer to question {gone[0]!r}")
                    rec = dict(info)
                    why = self._reasoning(items[i], [(n + s, s) for s in qs[n]], resp)
                    if why is not None:
                        rec["reasoning"] = why
                    out[i] = {**self.read(items[i], {s: answers[n + s] for s in qs[n]}), "info": {"systemone": rec}}
                except (ValueError, TypeError, KeyError) as e:     # a reply that breaks the contract: never a guess
                    out[i] = {"logits": np.zeros(len(items[i].options)), "escalate": f"invalid System One output — {e}",
                              "info": {"systemone": dict(info)}}
        return out


def systemone(base_url, model, api_key=None, *, timeout=30.0, opener=None, extra_body=None, deterministic=False,
              retries=2, backoff=1.0, sleep=None, max_len=None):
    """A DecideModel over a System One endpoint (see the module docs).

    extra_body: request fields merged into every request's JSON, e.g. OpenRouter's provider routing and `user`; a field
    solvi sets itself (model, state, questions) is refused with ValueError, never overridden; it enters the fingerprint.
    Pinning one OpenRouter provider, with no fallback to another:

        systemone("https://openrouter.ai/api", "<model>", api_key=KEY,
                  extra_body={"provider": {"only": ["<provider>"], "allow_fallbacks": False}})

    A local Jeeves server with shorter thinking (its `options`; an option it does not know is a 422, which escalates):

        systemone("http://127.0.0.1:8009", "jeeves-latest",
                  extra_body={"options": {"max_think": 512, "nothink_threshold": 0.9}})

    deterministic: False (default) — replay checks the recorded output instead of calling the service again; True for a
    local server whose output is reproducible (replay re-runs it and compares). retries / backoff: for network errors,
    timeouts, a broken connection, 408 / 409 / 429 / 5xx (backoff · 2^k seconds between attempts); after them the
    decision escalates ("did not answer after N attempts: ...") and is not cached. HTTP 400 / 413 / 422 escalates at
    once, with the service's error text (and a gateway's wrapped cause, OpenRouter's `error.metadata.raw`); 401, 403,
    404 and any other 4xx — a wrong key, model or URL — raise SystemOneError, as solvi.llm raises LLMError (0.7
    escalated every decision instead, which read like a model that is never sure). Everything after `api_key` is
    keyword-only. A reply that breaks the contract escalates too ("invalid System One output — ..."). opener: a replacement for urllib's urlopen (tests,
    proxies); sleep: for the backoff (tests). max_len: the tokens one request reads under long="retrieve" (words and
    punctuation × 1.3, the question included; default None: 512, as for a local decider) — as llm(max_len=...)."""
    sc = SystemOneScorer(base_url, model, api_key, timeout=timeout, opener=opener, extra_body=extra_body,
                         retries=retries, backoff=backoff, sleep=sleep)
    meta = {"format": "systemone", "temperature": 1.0}
    if max_len is not None:                           # absent: 512, and the meta hashes as before
        if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 64:
            raise ValueError(f"max_len must be a number of tokens of at least 64, not {max_len!r}")
        meta["max_len"] = max_len
    m = DecideModel(sc, meta=meta, model_id=sc.model_id, backend="systemone")
    # "not stated" is asked in words (an option of its own, as solvi.llm does); set on the capabilities directly so that
    # the model's fingerprint stays what it was
    m.caps["unknown"] = _unknown_caps({"label": NOT_STATED}, m.caps["columns"])
    m.deterministic = bool(deterministic)
    return m
