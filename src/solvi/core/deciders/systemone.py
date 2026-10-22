"""Any decision model that speaks the System One HTTP API (`POST /v1/systemone`: Jev, and open servers such as Kev,
Jeeves, Von, Laya-serve, Intern-Decision) as a solvi decider — the model proposes, solvi's checks, rules and thresholds
decide.

    from solvi.core.deciders.systemone import systemone
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
  competes with the options' in one softmax, as with solvi.core.deciders.llm and the checkpoints that have a "not stated" output: the
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
answers `questions` questions at once — the service's own `confidence` for the question when the reply has one
(Jev's Choice / Score: a number distinct from the probability, https://docs.typesafe.ai/confidence; per option for a
multi-label question), `prob_floor` when the floor used was not 1e-6 (systemone(prob_floor=...)) — and the question's
`reasoning` when the service returns it (Jeeves with
`return_reasoning`: the text cut to REASONING_CHARS characters, for the audit; the answer is read from the
probabilities, never from that text). The API key is sent in the Authorization header only, never recorded. A service
that does not answer (network errors, timeouts, 429, 5xx: `retries` more attempts with backoff), refuses the request
(another 4xx: its error text) or gives a reply that breaks the contract escalates the decision — never a guess, never
an exception; a failed request is not cached.

A hosted model is not replayed (`deterministic=False`, the default): replay checks the recorded output instead of calling
the service again; `deterministic=True` for a local server whose output is reproducible. The trace records the endpoint
and the model name — not the weights behind them, which the service can change: calibrate again when it does.

Calibrating on the service's confidence instead of solvi's (which is read from the probabilities): a guarantee reads
any recorded detail of the answer's decision with signal="extra:<path>", and the labelled examples decide which signal
separates right from wrong answers better —

    system.guarantee("route", examples, max_risk=0.05, signal="extra:systemone.confidence")

An answer whose reply carried no confidence cannot be read on that signal and abstains (with the reason)."""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from .. import catalog as core
from . import DecideModel, _unknown_caps
from ._remote import NoAnswer, Refused, RemoteClient, RemoteError
from ._remote import tokens as remote_tokens

EPS = 1e-6
REASONING_CHARS = 1000          # the most characters of a question's reasoning text kept in a decision's extra
NOT_STATED = core.NOT_STATED                  # one definition, in solvi.core
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


def _logit(p, floor=EPS):
    p = min(max(float(p), floor), 1 - floor)
    return math.log(p / (1 - p))


def _grid_floor(answers):
    """The probability floor of one reply under prob_floor="auto": half the step of the grid the reply's interior
    probabilities (0 < p < 1) lie on — all of them multiples of 10^-d within 1e-9, the coarsest such d of 2..4: a
    service that rounds to 0.01 gets 0.005, and so does one that rounds to 0.1 — else EPS, the floor a reply that is
    not on a grid always had. Exact 0s and 1s say nothing about a grid: a reply with no interior probability (all 0 /
    1) keeps EPS, so a service's confident 0 / 1 answer reads as before. Values that are not probabilities are left
    out here (they escalate when the answer is read)."""
    ps = []
    for a in answers.values() if isinstance(answers, dict) else ():
        if not isinstance(a, dict):
            continue
        vs = [a.get("noul")] + (list(a["probabilities"].values()) if isinstance(a.get("probabilities"), dict) else [])
        ps += [float(v) for v in vs if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               and 0 < v < 1]
    if not ps:
        return EPS
    for d in range(2, 5):
        step = 10.0 ** -d
        if all(abs(v - round(v / step) * step) <= 1e-9 for v in ps):
            return step / 2
    return EPS


def _prob_floor(v):
    """prob_floor= → "auto", or the floor as a float (None: EPS); ValueError for anything else."""
    if v == "auto" and isinstance(v, str):
        return "auto"
    if v is None:
        return EPS
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 0.5:
        raise ValueError(f'prob_floor must be "auto", None or a probability in (0, 0.5), not {v!r}')
    return float(v)


def _confidence(a):
    """A reply's own `confidence` for one answer (Jev's Choice / Score: a number distinct from the probability) → float,
    or None when it is absent or not a finite number (it is recorded, never read as the answer)."""
    c = a.get("confidence") if isinstance(a, dict) else None
    if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c):
        return float(c)
    return None


def _noul(it):
    return tuple(o.lower() for o in it.options) in (("yes", "no"), ("true", "false"))


def _not_stated(it):
    """Does this question allow "not stated" as an answer of its own (an option named so is just an option)?"""
    return bool(getattr(it, "unknown", False)) and NOT_STATED not in it.options


class SystemOneError(RemoteError):
    """The service refused the request itself (a wrong key, model or URL: HTTP 401, 403, 404 and other client errors
    except 400 / 413 / 422) — raised, as solvi.core.deciders.llm does, rather than escalated. The message has the endpoint, never the
    key."""


class _Failed(Exception):
    """A reply that is not a JSON object: the reason, for the escalation."""


class SystemOneScorer(RemoteClient):
    """A scorer for DecideModel over `POST {base_url}/v1/systemone` (standard library HTTP, no dependencies; the shared
    client of solvi.core.deciders._remote)."""

    tag = "systemone"
    service = "the System One service"
    error = SystemOneError

    def __init__(self, base_url, model, api_key=None, *, timeout=30.0, opener=None, extra_body=None, retries=2,
                 backoff=1.0, sleep=None, prob_floor="auto", max_options=None):
        super().__init__(base_url, model, api_key, path="/v1/systemone", timeout=timeout, retries=retries,
                         backoff=backoff, opener=opener, sleep=sleep)
        self.extra_body = _extra_body(extra_body)
        self.prob_floor = _prob_floor(prob_floor)       # "auto" or a float
        if max_options is not None and (isinstance(max_options, bool) or not isinstance(max_options, int)
                                        or max_options < 2):
            raise ValueError(f"max_options must be a number of options of at least 2, not {max_options!r}")
        self.max_options = max_options
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
        if self.prob_floor != "auto":                    # the default ("auto") hashes as before: stored records stay valid
            fp += f"|floor:{self.prob_floor!r}"
        return fp

    def floor(self, answers):
        """The probability floor for one reply's answers: the explicit prob_floor, or (auto) half its grid step."""
        return _grid_floor(answers) if self.prob_floor == "auto" else self.prob_floor

    def check_item(self, name, it):
        """At build time (DecideModel.decision): a question with more options than the service takes (max_options=)
        is refused here, rather than by every request."""
        if self.max_options is None:
            return
        for q in self.questions(it).values():
            n = len(q.get("criteria") or ())
            if n > self.max_options:
                ns = " (\"not stated\" included)" if NOT_STATED in q["criteria"] else ""
                raise ValueError(f"{name}: {n} options{ns}, but {self.model!r} at {self.endpoint} takes at most "
                                 f"{self.max_options} in a choice (max_options={self.max_options}): split the label set "
                                 "into groups and ask a question per group (e.g. a coarse question, then one per group)")

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
    def answer_logits(it, ans, floor=EPS):
        """An answer to a single (not multi-label) question → log-probabilities in the item's option order (each
        probability clipped at `floor` first: not renormalised — the logits go through a softmax)."""
        if ans.get("type") == "noul" or "noul" in ans:
            p = min(max(_prob(ans["noul"], "noul"), floor), 1 - floor)
            yes_first = it.options[0].lower() in ("yes", "true")
            return np.log(np.array([p, 1 - p] if yes_first else [1 - p, p]))
        probs = ans.get("probabilities") or {}
        missing = [o for o in it.options if o not in probs]
        if missing:
            raise ValueError(f"the service returned no probability for {missing}")
        return np.log(np.clip(np.array([_prob(probs[o], f"the probability of {o!r}") for o in it.options]), floor, None))

    @classmethod
    def read(cls, it, answers, floor=EPS):
        """The answers to an Item's questions {name suffix: answer} → the scorer's output {"logits", "unknown"?};
        probabilities are clipped at `floor` (EPS: as before 1.4)."""
        ns = _not_stated(it)
        if it.multi:
            z = np.array([_logit(answers[f"__{j}"]["noul"], floor) for j in range(len(it.options))])
            return {"logits": z, **({"unknown": _logit(answers["__ns"]["noul"], floor)} if ns else {})}
        ans = answers[""]
        z = cls.answer_logits(it, ans, floor)
        if not ns:
            return {"logits": z}
        p = (ans.get("probabilities") or {}).get(NOT_STATED)
        if p is None:
            raise ValueError(f"the service returned no probability for {NOT_STATED!r}")
        return {"logits": z, "unknown": math.log(max(float(p), floor))}

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

    @staticmethod
    def _confidences(it, qs, answers):
        """The service's own `confidence` for an Item's questions → a float for a single question, {option | "not
        stated": float} for a multi-label one; None when the reply carries none (Sage-like replies, older servers)."""
        out = {}
        for name, suffix in qs:
            c = _confidence(answers.get(name))
            if c is not None:
                out["" if not suffix else (NOT_STATED if suffix == "__ns" else it.options[int(suffix[2:])])] = c
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
            floor = self.floor(answers)
            if floor != EPS:
                info["prob_floor"] = floor
            for n, i in names.items():
                try:
                    gone = [n + s for s in qs[n] if not isinstance(answers.get(n + s), dict)]
                    if gone:
                        raise ValueError(f"the service gave no answer to question {gone[0]!r}")
                    rec = dict(info)
                    why = self._reasoning(items[i], [(n + s, s) for s in qs[n]], resp)
                    if why is not None:
                        rec["reasoning"] = why
                    conf = self._confidences(items[i], [(n + s, s) for s in qs[n]], answers)
                    if conf is not None:
                        rec["confidence"] = conf
                    out[i] = {**self.read(items[i], {s: answers[n + s] for s in qs[n]}, floor),
                              "info": {"systemone": rec}}
                except (ValueError, TypeError, KeyError) as e:     # a reply that breaks the contract: never a guess
                    out[i] = {"logits": np.zeros(len(items[i].options)), "escalate": f"invalid System One output — {e}",
                              "info": {"systemone": dict(info)}}
        return out


def systemone(base_url, model, api_key=None, *, timeout=30.0, opener=None, extra_body=None, deterministic=False,
              retries=2, backoff=1.0, sleep=None, max_len=None, prob_floor="auto", max_options=None):
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
    404 and any other 4xx — a wrong key, model or URL — raise SystemOneError, as solvi.core.deciders.llm raises LLMError (0.7
    escalated every decision instead, which read like a model that is never sure). Everything after `api_key` is
    keyword-only. A reply that breaks the contract escalates too ("invalid System One output — ..."). opener: a replacement for urllib's urlopen (tests,
    proxies); sleep: for the backoff (tests). max_len: the tokens one request reads under long="retrieve" (words and
    punctuation × 1.3, the question included; default None: 512, as for a local decider) — as llm(max_len=...).

    prob_floor: the smallest probability taken from a reply before its log (a 0 has no log). "auto" (default): half the
    step of the grid the reply's probabilities strictly between 0 and 1 lie on — all of them multiples of 0.01 (a
    service that rounds, as Jev does; or of 0.1) → 0.005; on no grid of 10^-2 … 10^-4 → 1e-6, as before 1.4, bit for
    bit. A reply whose probabilities are all exactly 0 or 1 shows no grid and keeps 1e-6 (a rounding service's exact 0
    on the right answer in such a reply is still read at 1e-6: set a float floor for that service). A float
    sets it for every reply; None keeps 1e-6 everywhere. Why: with a 1e-6 floor one rounded-to-0 probability on the
    right answer weighs log(1e-6) ≈ −13.8 and drags fit / teach / calibration — a third-party measurement on Jev
    (https://github.com/scienthoon/jev-ood-calibration, their numbers) refit the choice temperature to 3.29 with a 1e-6
    floor and to 1.30 with 0.005. A floor other than "auto" enters the fingerprint ("auto" hashes as before 1.4, so
    stored records stay valid); a decision records the floor used in extra["systemone"]["prob_floor"] when it is not
    1e-6. The logits are not renormalised after clipping: they go through a softmax.

    max_options: the most options the service takes in one choice question (Jev 255, Solar Decide 26, by their docs;
    "not stated", when the question allows it, counts as one); a question with more is refused when it is built
    (ValueError naming the limit) instead of escalating on every request. Default None: no limit is checked."""
    sc = SystemOneScorer(base_url, model, api_key, timeout=timeout, opener=opener, extra_body=extra_body,
                         retries=retries, backoff=backoff, sleep=sleep, prob_floor=prob_floor, max_options=max_options)
    meta = {"format": "systemone", "temperature": 1.0}
    if max_len is not None:                           # absent: 512, and the meta hashes as before
        if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 64:
            raise ValueError(f"max_len must be a number of tokens of at least 64, not {max_len!r}")
        meta["max_len"] = max_len
    m = DecideModel(sc, meta=meta, model_id=sc.model_id, backend="systemone")
    # "not stated" is asked in words (an option of its own, as solvi.core.deciders.llm does); set on the capabilities directly so that
    # the model's fingerprint stays what it was
    m.caps["unknown"] = _unknown_caps({"label": NOT_STATED}, m.caps["columns"])
    m.deterministic = bool(deterministic)
    return m


__all__ = ["REASONING_CHARS", "systemone", "SystemOneError", "SystemOneScorer"]
