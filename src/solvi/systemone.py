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

- "not stated" (`unknown=True`, `Maybe[...]`): one more option, "not stated", with a description ("the input does not
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
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .decide import DecideModel, _unknown_caps
from .llm import WHY_CHARS, _error_text

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


class _Failed(Exception):
    """The service did not answer (after the retries) or refused the request: the reason, for the escalation."""


class SystemOneScorer:
    """A scorer for DecideModel over `POST {base_url}/v1/systemone` (standard library HTTP, no dependencies)."""

    tag = "systemone"

    def __init__(self, base_url, model, api_key=None, timeout=30.0, opener=None, *, extra_body=None, retries=2,
                 backoff=1.0, sleep=None):
        if not str(base_url).lower().startswith(("http://", "https://")):
            raise ValueError(f"a System One service is an http(s):// URL, not {str(base_url)[:40]!r}")  # no file: / ftp:
        from .llm import endpoint
        sp = urllib.parse.urlsplit(str(base_url))
        path = sp.path.rstrip("/")
        path = path if path.endswith("/v1/systemone") else path + "/v1/systemone"
        self.url = urllib.parse.urlunsplit((sp.scheme, sp.netloc, path, sp.query, ""))   # a query stays a query
        self.endpoint = endpoint(self.url)               # what the trace, repr and fingerprint show: no credentials
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.extra_body = _extra_body(extra_body)
        self.retries, self.backoff = max(0, int(retries)), float(backoff)
        self.sleep = sleep or time.sleep
        self.model_id = f"systemone:{model}"
        self.requests = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}   # reasoning: thinking models
        self.cost = 0.0                                  # the sum of the replies' usage.cost

    def __repr__(self):                                  # never the key
        return f"SystemOneScorer({self.endpoint!r}, {self.model!r})"

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
    def _post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",  # noqa: S310 — http(s) only
                                     headers={"content-type": "application/json",
                                              **({"authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with self.opener(req, timeout=self.timeout) as r:
            self.requests += 1
            return json.loads(r.read().decode())

    def request(self, body):
        """→ the service's response; _Failed (with the reason, never the key) when it does not answer after the retries
        (network errors, timeouts, a broken connection, 408 / 409 / 429 / 5xx) or refuses the request (another 4xx)."""
        attempt, last = 0, None
        while True:
            try:
                return self._post(body)
            except urllib.error.HTTPError as e:
                if not (e.code in (408, 409, 429) or e.code >= 500):
                    why = _error_text(e)
                    raise _Failed(f"the System One service refused the request: HTTP {e.code}"
                                  + (f" — {why[:WHY_CHARS]}" if why else "")) from None
                why = _error_text(e)
                last = f"HTTP {e.code}" + (f" — {why[:WHY_CHARS]}" if why else "")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as e:
                # HTTPException: the connection broke mid-answer (IncompleteRead, RemoteDisconnected, BadStatusLine);
                # ValueError: a reply that is not JSON
                last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            attempt += 1
            if attempt > self.retries:
                raise _Failed(f"the System One service did not answer after {attempt} attempts: {last}")
            self.sleep(self.backoff * 2 ** (attempt - 1))

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
        tokens = {k: u[k] for k in self.usage if isinstance(u.get(k), int) and not isinstance(u.get(k), bool)}
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
            except _Failed as e:              # escalate every question of the request; not cached: asked again next time
                for i in idx:
                    out[i] = {"logits": np.zeros(len(items[i].options)), "escalate": str(e), "transient": True,
                              "info": {"systemone": {"endpoint": self.endpoint, "model": self.model, "questions": len(idx)}}}
                continue
            ms = (time.perf_counter() - t0) * 1000
            info = self._info(resp, ms, len(idx))
            for k in info.get("usage", {}):
                self.usage[k] += info["usage"][k]
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


def systemone(base_url, model, api_key=None, timeout=30.0, opener=None, *, extra_body=None, deterministic=False,
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
    decision escalates ("did not answer after N attempts: ...") and is not cached. Another 4xx escalates at once, with
    the service's error text (and a gateway's wrapped cause, OpenRouter's `error.metadata.raw`); a reply that breaks the
    contract escalates too ("invalid System One output — ..."). opener: a replacement for urllib's urlopen (tests,
    proxies); sleep: for the backoff (tests). max_len: the tokens one request reads under long="retrieve" (words and
    punctuation × 1.3, the question included; default None: 512, as for a local decider) — as llm(max_len=...)."""
    sc = SystemOneScorer(base_url, model, api_key, timeout, opener, extra_body=extra_body, retries=retries,
                         backoff=backoff, sleep=sleep)
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
