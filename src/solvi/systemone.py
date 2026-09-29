"""Any decision model that speaks the System One HTTP API (`POST /v1/systemone`: Jev, and open servers such as Kev, Von,
Laya-serve, Intern-Decision) as a solvi decider — the model proposes, solvi's checks, rules and thresholds decide.

    from solvi.systemone import systemone
    model = systemone("http://127.0.0.1:8009", "kev-latest")            # api_key= for a hosted service
    part = model.decision("team", "Which team should handle this?", "email", {"billing": "Charges", "shipping": "Delivery"})

The questions of one input go in one request. Choice questions are sent as `choice` (criteria: option → description),
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

`extra_body`: server-specific request fields merged into every request (OpenRouter's `provider` routing, `user`); the
fields solvi sets (model, state, questions) are refused, and extra_body enters the fingerprint. Per decision
`extra["systemone"]` records the endpoint, the model name (and `served_by` when the service names another), the request's
`ms` and, when the service reports them, its `usage` and `cost` — for the whole request, which answers `questions`
questions at once. The API key is sent in the Authorization header only, never recorded.

A hosted model is not replayed (`deterministic=False`, the default): replay checks the recorded output instead of calling
the service again; `deterministic=True` for a local server whose output is reproducible. The trace records the endpoint
and the model name — not the weights behind them, which the service can change: calibrate again when it does."""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request

import numpy as np

from .decide import DecideModel, _unknown_caps

EPS = 1e-6
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


def _logit(p):
    p = min(max(float(p), EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _noul(it):
    return tuple(o.lower() for o in it.options) in (("yes", "no"), ("true", "false"))


def _not_stated(it):
    """Does this question allow "not stated" as an answer of its own (an option named so is just an option)?"""
    return bool(getattr(it, "unknown", False)) and NOT_STATED not in it.options


class SystemOneScorer:
    """A scorer for DecideModel over `POST {base_url}/v1/systemone` (standard library HTTP, no dependencies)."""

    tag = "systemone"

    def __init__(self, base_url, model, api_key=None, timeout=30.0, opener=None, *, extra_body=None):
        if not str(base_url).lower().startswith(("http://", "https://")):
            raise ValueError(f"a System One service is an http(s):// URL, not {str(base_url)[:40]!r}")  # no file: / ftp:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.extra_body = _extra_body(extra_body)
        self.model_id = f"systemone:{model}"
        self.requests = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.cost = 0.0                                  # the sum of the replies' usage.cost

    def __repr__(self):                                  # never the key
        return f"SystemOneScorer({self.url!r}, {self.model!r})"

    def fingerprint(self):
        fp = f"systemone|{self.url}|{self.model}"
        if self.extra_body:                              # provider pinning ... changes what answers
            blob = json.dumps(self.extra_body, sort_keys=True, ensure_ascii=False)
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

    @classmethod
    def question(cls, it):
        """An Item that is one System One question → that question (a multi-label one is several: see questions)."""
        qs = cls.questions(it)
        if len(qs) != 1:
            raise ValueError("a multi-label question is one noul per option: use SystemOneScorer.questions")
        return qs[""]

    @staticmethod
    def answer_logits(it, ans):
        """An answer to a single (not multi-label) question → log-probabilities in the item's option order."""
        if ans.get("type") == "noul" or "noul" in ans:
            p = min(max(float(ans["noul"]), EPS), 1 - EPS)
            yes_first = it.options[0].lower() in ("yes", "true")
            return np.log(np.array([p, 1 - p] if yes_first else [1 - p, p]))
        probs = ans.get("probabilities") or {}
        missing = [o for o in it.options if o not in probs]
        if missing:
            raise ValueError(f"the service returned no probability for {missing}")
        return np.log(np.clip(np.array([float(probs[o]) for o in it.options]), EPS, None))

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

    def body(self, text, questions):
        b = json.loads(json.dumps(self.extra_body)) if self.extra_body else {}      # a fresh copy per request
        b.update({"state": text, "model": self.model, "questions": questions})
        return b

    def _info(self, resp, ms, n):
        """What a decision records of its request (extra["systemone"])."""
        info = {"endpoint": self.url, "model": self.model, "ms": round(ms, 3), "questions": n}
        if resp.get("model") and resp.get("model") != self.model:
            info["served_by"] = str(resp["model"])
        u = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
        tokens = {k: u[k] for k in self.usage if isinstance(u.get(k), int) and not isinstance(u.get(k), bool)}
        if tokens:
            info["usage"] = tokens
        c = u.get("cost")
        if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c):
            info["cost"] = float(c)
        return info

    def logits(self, items):
        by_text = {}
        for i, it in enumerate(items):
            by_text.setdefault(it.text, []).append(i)
        out = [None] * len(items)
        for text, idx in by_text.items():
            names = {f"q{k}": i for k, i in enumerate(idx)}
            qs = {n: self.questions(items[i]) for n, i in names.items()}
            t0 = time.perf_counter()
            resp = self._post(self.body(text, {n + s: q for n, sub in qs.items() for s, q in sub.items()}))
            ms = (time.perf_counter() - t0) * 1000
            info = self._info(resp, ms, len(idx))
            for k in info.get("usage", {}):
                self.usage[k] += info["usage"][k]
            self.cost += info.get("cost", 0.0)
            answers = resp.get("answers") or {}
            for n, i in names.items():
                gone = [n + s for s in qs[n] if n + s not in answers]
                if gone:
                    raise ValueError(f"the service gave no answer to question {gone[0]!r}")
                out[i] = {**self.read(items[i], {s: answers[n + s] for s in qs[n]}), "info": {"systemone": dict(info)}}
        return out


def systemone(base_url, model, api_key=None, timeout=30.0, opener=None, *, extra_body=None, deterministic=False):
    """A DecideModel over a System One endpoint (see the module docs).

    extra_body: request fields merged into every request's JSON, e.g. OpenRouter's provider routing and `user`; a field
    solvi sets itself (model, state, questions) is refused with ValueError, never overridden; it enters the fingerprint.
    Pinning one OpenRouter provider, with no fallback to another:

        systemone("https://openrouter.ai/api", "<model>", api_key=KEY,
                  extra_body={"provider": {"only": ["<provider>"], "allow_fallbacks": False}})

    deterministic: False (default) — replay checks the recorded output instead of calling the service again; True for a
    local server whose output is reproducible (replay re-runs it and compares). opener: a replacement for urllib's
    urlopen (tests, proxies)."""
    sc = SystemOneScorer(base_url, model, api_key, timeout, opener, extra_body=extra_body)
    m = DecideModel(sc, meta={"format": "systemone", "temperature": 1.0}, model_id=sc.model_id, backend="systemone")
    # "not stated" is asked in words (an option of its own, as solvi.llm does); set on the capabilities directly so that
    # the model's fingerprint stays what it was
    m.caps["unknown"] = _unknown_caps({"label": NOT_STATED}, m.caps["columns"])
    m.deterministic = bool(deterministic)
    return m
