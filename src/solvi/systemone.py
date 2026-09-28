"""Any decision model that speaks the System One HTTP API (`POST /v1/systemone`: Jev, and open servers such as Kev, Von,
Laya-serve, Intern-Decision) as a solvi decider — the model proposes, solvi's checks, rules and thresholds decide.

    from solvi.systemone import systemone
    model = systemone("http://127.0.0.1:8009", "kev-latest")            # api_key= for a hosted service
    part = model.decision("team", "Which team should handle this?", "email", {"billing": "Charges", "shipping": "Delivery"})

The questions of one input go in one request. Choice questions are sent as `choice` (criteria: option → description),
yes/no questions as `noul`, scores as `choice` over their levels (the probability of each level is what solvi needs).
The returned probabilities become the decider's logits (log p), so everything built on a DecideModel works unchanged:
act_guard / conformal / calibrate_for on your labelled examples (the service gives no act signal: solvi uses the
confidence), fit / teach / adapt, the audit and the trace. Multi-label questions, spans, evidence and "not stated" are not
part of the API. The trace records the endpoint and the model name — not the weights behind them, which the service can
change: calibrate again when it does."""
from __future__ import annotations

import json
import urllib.request

import numpy as np

from .decide import DecideModel

EPS = 1e-6


class SystemOneScorer:
    """A scorer for DecideModel over `POST {base_url}/v1/systemone` (standard library HTTP, no dependencies)."""

    tag = "systemone"

    def __init__(self, base_url, model, api_key=None, timeout=30.0, opener=None):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.model_id = f"systemone:{model}"
        self.requests = 0

    def fingerprint(self):
        return f"systemone|{self.url}|{self.model}"

    @staticmethod
    def question(it):
        """An Item → a System One question: noul for yes / no, choice otherwise (criteria in the item's option order)."""
        if it.multi:
            raise ValueError("the System One API has no multi-label questions")
        if it.pointer:
            raise ValueError("the System One API gives no spans or evidence quotes")
        if tuple(o.lower() for o in it.options) in (("yes", "no"), ("true", "false")):
            return {"type": "noul", "instructions": it.task}
        desc = it.descriptions or (None,) * len(it.options)
        return {"type": "choice", "instructions": it.task, "criteria": {o: (d or None) for o, d in zip(it.options, desc)}}

    def _post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json",
                                              **({"authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with self.opener(req, timeout=self.timeout) as r:
            self.requests += 1
            return json.loads(r.read().decode())

    @staticmethod
    def answer_logits(it, ans):
        """An answer → log-probabilities in the item's option order."""
        if ans.get("type") == "noul" or "noul" in ans:
            p = min(max(float(ans["noul"]), EPS), 1 - EPS)
            yes_first = it.options[0].lower() in ("yes", "true")
            return np.log(np.array([p, 1 - p] if yes_first else [1 - p, p]))
        probs = ans.get("probabilities") or {}
        missing = [o for o in it.options if o not in probs]
        if missing:
            raise ValueError(f"the service returned no probability for {missing}")
        return np.log(np.clip(np.array([float(probs[o]) for o in it.options]), EPS, None))

    def logits(self, items):
        by_text = {}
        for i, it in enumerate(items):
            by_text.setdefault(it.text, []).append(i)
        out = [None] * len(items)
        for text, idx in by_text.items():
            names = {f"q{k}": i for k, i in enumerate(idx)}
            resp = self._post({"state": text, "model": self.model,
                               "questions": {n: self.question(items[i]) for n, i in names.items()}})
            answers = resp.get("answers") or {}
            for n, i in names.items():
                if n not in answers:
                    raise ValueError(f"the service gave no answer to question {n!r}")
                out[i] = self.answer_logits(items[i], answers[n])
        return out


def systemone(base_url, model, api_key=None, timeout=30.0, opener=None):
    """A DecideModel over a System One endpoint (see the module docs). `opener`: a replacement for urllib's urlopen
    (tests, proxies)."""
    sc = SystemOneScorer(base_url, model, api_key, timeout, opener)
    return DecideModel(sc, meta={"format": "systemone", "temperature": 1.0}, model_id=sc.model_id, backend="systemone")
