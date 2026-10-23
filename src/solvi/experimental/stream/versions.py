"""The closed learning loop: frozen versions, each with its own stream certificate, promoted under δ-spending
(experimental; stage 3b).

    from solvi.experimental.stream import HashedTextLearner, LearningLoop

    loop = LearningLoop(HashedTextLearner(), question="intent", max_risk=0.05, delta=0.05, secret=key,
                        storage=store)
    out = loop.step(text, oracle)          # oracle: () → the label (None: missing), called only when one is needed
    out["answer"], out["by"]               # "local" (the served version alone), "oracle", or "person" (no label)
    loop.versions()                        # [{"version", "state", "certified", "threshold", ...}]
    loop.replay()                          # every decision and event rebuilt from the records and compared

Versions. Version 0 is "no model": it answers nothing alone, has no certificate and spends no δ (it never
certifies). The live learner (`stream.learner.Learner`) takes every accepted oracle label and is never served. When no
candidate exists and the live learner has taken ≥ max(freeze_min, growth × the served version's labels at its
freeze, the refreeze floor) labels (300, 2: kadi's constants), a snapshot is frozen as the candidate, with its own
`StreamPromise` drawing the next (v, δ_v) from the loop's one `DeltaBudget` (Σ δ_v ≤ δ over every version ever
frozen). Each decision is scored by the served version and, in shadow, by the candidate, before its label is read.

Calibration. A version's promise reads only samples drawn after its freeze: the samples of decision n go to the
served version's promise and the candidate's — each frozen before n — and only then does the label reach the live
learner; a version frozen at decision n first reads decision n + 1. Nothing is ever scored by a model that saw the
label. Samples are drawn by the calibration coin of stage 2 — HMAC(secret, "sample|question|seed|n"), no input
bytes — at the rate from the past (1 while the served version is not certified, else min(1, max(floor, budget /
the share answered alone))); a decision the served version does not answer alone goes to the oracle anyway and its
label trains the live learner, but it reaches a promise only when the coin drew it (escalation-only calibration
breaks the promise: the stage-2 control).

Promotion (the joint / LTT kind). Once the candidate is certified, on the candidate's own samples (all after its
freeze): ℓ_k = the share with the candidate's score ≥ its threshold, ℓ_s = the share with the served version's score
≥ the served threshold now — a paired comparison on the same samples. Promote when ℓ_k ≥ ℓ_s + margin (margin 0,
kadi) and ℓ_k > 0. Discard a candidate not certified after `stale` samples (1 500) or certified and not better after
`trial` samples (default: stale); the refreeze floor becomes `refreeze` × the live learner's labels (1.5). A
discarded version's δ_v stays spent. Every version's certificate holds together under the budget, so the rule only
affects utility, not validity.

Labels. Accepted by default: the trusted sources ("human", "outcome", "rule", "audit"; solvi.core.sources). A slow
path's answer ("s2", or "verified") is refused (UntrustedLabel, before anything changes) unless `s2_labels=True`; with
it the label is taken and recorded as "s2". The loop is not a `Head` a System's `teach` reaches.

In the chain. Each decision is a record of kind "loop": "stream" (kept by a redaction) holds n, the served and
candidate versions, ρ, sampled, alone, by, label_source, the served score and threshold, the candidate's shadow
score, both promises' states and the live learner's labels; "loop" (not kept) holds the input, the would-be answers,
the label and the answer. Each state change is a record of kind "stream" ({"event": "freeze" | "promote" |
"discard" | "rollback" | the promises' "grid" / "certified" / "loosened", "question", "version", "v_index",
"delta_v", ...}); a freeze carries the frozen model's fingerprint, its decision and the labels it had seen. With
`versions_dir=`, each frozen model is pickled as `<versions_dir>/<version>.pkl`. `replay()` re-runs the protocol
from the records alone (the coin recomputed with the secret, the scores read from the records, the live learner
replaced by its label count) and compares every decision — which version served it and the certificates' states —
and every event; with `versions_dir` each pickle's fingerprint is checked and, with `rescore=True`, every recorded
score is recomputed by the frozen model that gave it.

With `system=` a promoted or rolled-back version is installed as the question's answer head
(`solvi.core.versions.serve_head`; version 0 removes it), so the System's fingerprint and every trace name it.

Not built here: the Dispatcher's `loop=` (the loop drives its own coin and oracle), a RiskDetector on the served
version, tickets (the oracle is a callable). Unmeasured: the gain is unproven (solvi's prior evidence is against
it); the stand (stage 4) measures it."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import pickle
import warnings
from pathlib import Path

from ._eprocess import DeltaBudget
from .certificate import StreamPromise, _sha
from .sampling import Rate, coin_message, coin_value

S2_SOURCES = ("s2", "verified")
_warned = False


def _num(x):
    """A score as recorded: None for no score or a non-finite one (the certificate reads both as -inf)."""
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _json(x):
    from ...core.store import plain
    return json.loads(json.dumps(x, default=plain))


class _Recorded:
    """A frozen model during replay: its recorded fingerprint, no scores (they are read from the records)."""

    def __init__(self, fp):
        self.fp = fp

    def score(self, x):
        raise RuntimeError("a replay reads the recorded scores")

    def fingerprint(self):
        return self.fp


class _Counting:
    """The live learner during replay: only its label count matters to the protocol."""

    def __init__(self):
        self._seen = 0

    def update(self, x, y):
        self._seen += 1

    def seen(self):
        return self._seen

    def snapshot(self):
        raise RuntimeError("replay takes the recorded fingerprints")


class LearningLoop:
    """See the module docstring. learner: the live Learner; question: the question's name (in the coin and the
    records); max_risk= / max_error=, delta: each version's StreamPromise (promise_options: its other keywords — grid,
    grid_from, bets ...); freeze_min, growth, stale, trial, margin, refreeze: the protocol's constants; s2_labels: take
    slow-path labels; source: the default source of the oracle's labels; secret, seed: the coin's key and run;
    sampling: {"floor", "budget", "window"} of the rate; storage: a TraceStorage for the records (None: kept in
    memory for replay); versions_dir: where frozen models are pickled; system, feature: install the served version
    as the question's answer head (feature: the text fact a text model reads)."""

    def __init__(self, learner, *, question, max_risk=None, max_error=None, delta=0.05, freeze_min=300, growth=2.0,
                 stale=1500, trial=None, margin=0.0, refreeze=1.5, s2_labels=False, source="human", secret=None,
                 seed=0, sampling=None, promise_options=None, storage=None, versions_dir=None, system=None,
                 feature="text"):
        global _warned
        self._kw = dict(question=question, max_risk=max_risk, max_error=max_error, delta=delta, freeze_min=freeze_min,
                        growth=growth, stale=stale, trial=trial, margin=margin, refreeze=refreeze,
                        s2_labels=s2_labels, source=source, secret=secret, seed=seed, sampling=sampling,
                        promise_options=promise_options, feature=feature)
        self._pkw = {"max_risk": max_risk, "max_error": max_error, **(promise_options or {})}
        for k in ("delta", "budget"):
            if k in self._pkw:
                raise ValueError(f"{k} is the loop's own (one budget over every version)")
        StreamPromise(**self._pkw, budget=DeltaBudget(delta))          # the promise's arguments, checked once
        if int(freeze_min) < 1 or float(growth) < 1.0 or int(stale) < 1 or float(refreeze) < 1.0:
            raise ValueError("freeze_min ≥ 1, growth ≥ 1, stale ≥ 1, refreeze ≥ 1")
        if float(margin) < 0.0:
            raise ValueError("margin is a share ≥ 0")
        self.live, self.question = learner, str(question)
        self.freeze_min, self.growth, self.stale = int(freeze_min), float(growth), int(stale)
        self.trial = self.stale if trial is None else int(trial)
        self.margin, self.refreeze = float(margin), float(refreeze)
        self.s2_labels, self.source, self.seed = bool(s2_labels), str(source), int(seed)
        if isinstance(secret, str):
            secret = secret.encode()
        self._secret = bytes(secret) if secret else None
        if self._secret is None and not _warned:
            _warned = True
            from ...core.dispatch import PredictableDrawWarning
            warnings.warn("LearningLoop: the calibration coin is a public hash of the question, seed and decision "
                          "number; give secret= to key it", PredictableDrawWarning, stacklevel=2)
        self.budget = DeltaBudget(delta)
        self.rate = Rate(**(sampling or {}))
        self.storage, self.system, self.feature = storage, system, feature
        self.versions_dir = None if versions_dir is None else Path(versions_dir)
        if self.versions_dir is not None:
            self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.served, self.candidate = 0, None
        self._vs = {0: {"version": 0, "state": "served", "model": None, "promise": None, "frozen_at": 0, "seen": 0,
                        "fingerprint": None, "samples": [], "fed": []}}
        self.refreeze_floor = 0.0
        self.log = []                                      # (kind, record) when there is no storage
        self._ev_seen = {}
        self._fps = None                                   # replay: the recorded fingerprints by version

    # --- the per-decision step
    def step(self, x, oracle=None, *, source=None) -> dict:
        """One input → {"n", "answer", "by", "served", "candidate", "sampled", "label", "label_source", "events"}.
        oracle: () → the label (None: missing), called only when the decision needs one (not answered alone, or
        drawn by the coin); source: its source (default the loop's)."""
        s, c = self._vs[self.served], self._vs.get(self.candidate)
        would_s, score_s = s["model"].score(x) if s["model"] is not None else (None, None)
        would_k, score_k = c["model"].score(x) if c is not None else (None, None)
        if oracle is None:
            def oracle():
                raise ValueError("this decision needs a label: step(x, oracle)")
        return self._advance(x, would_s, score_s, would_k, score_k, oracle, self.source if source is None else source)

    def _coin(self, n) -> float:
        msg = coin_message(self.question, self.seed, n).encode()
        h = hmac.new(self._secret, msg, hashlib.sha256).hexdigest() if self._secret else hashlib.sha256(msg).hexdigest()
        return coin_value(h)

    def _threshold(self, v):
        p = self._vs[v]["promise"]
        return math.inf if p is None else p.threshold

    def _check(self, src):
        if src in S2_SOURCES:
            if not self.s2_labels:
                from ...core.sources import UntrustedLabel
                raise UntrustedLabel(f"label source {src!r}: the learning loop takes slow-path labels only with "
                                     "LearningLoop(s2_labels=True) (opt-in; recorded as \"s2\")")
            return "s2"
        from ...core.sources import check_source
        return check_source(src)

    def _advance(self, x, would_s, score_s, would_k, score_k, label_of, src):
        n = self.n + 1
        sv, c = self.served, self.candidate
        thr_s = self._threshold(sv)
        rho = self.rate.rate(full=math.isinf(thr_s))
        sampled = self._coin(n) < rho
        alone = score_s is not None and math.isfinite(thr_s) and score_s >= thr_s
        label, lsrc = None, None
        if sampled or not alone:
            label = label_of()
            if label is not None:
                lsrc = self._check(src)                    # a refused source: nothing has changed yet
        # --- from here the state changes
        self.n = n
        for v, score in ((sv, score_s), (c, score_k)):
            if v is not None and self._vs[v]["promise"] is not None:
                self._vs[v]["promise"].observe(score)
        if sampled:
            for v, score, would in ((sv, score_s, would_s), (c, score_k, would_k)):
                if v is None or self._vs[v]["promise"] is None:
                    continue
                ver = self._vs[v]
                if not n > ver["frozen_at"]:               # cannot happen: a version is frozen at the end of a step
                    raise RuntimeError(f"decision {n} would calibrate version {v} frozen at {ver['frozen_at']}")
                ver["promise"].learn(score, would, label)
                ver["fed"].append(n)
            if c is not None:
                self._vs[c]["samples"].append([score_k, score_s])
        if label is not None:
            self.live.update(x, label)
        self.rate.push(alone)
        if alone and not sampled:
            answer, by = would_s, "local"
        elif label is not None:
            answer, by = label, "oracle"
        else:
            answer, by = None, "person"
        events = self._promise_events()
        events += self._protocol(n)
        events += self._promise_events()
        st = lambda v: None if v is None or self._vs[v]["promise"] is None else self._vs[v]["promise"].state()  # noqa: E731
        rec = {"n": n, "served": sv, "candidate": c, "rho": rho, "sampled": sampled, "alone": alone, "by": by,
               "label_source": lsrc, "missing": (sampled or not alone) and label is None, "score": _num(score_s),
               "threshold": None if math.isinf(thr_s) else thr_s, "shadow": _num(score_k), "served_state": st(sv),
               "candidate_state": st(c), "seen": self.live.seen()}
        payload = {"input": x, "would": would_s, "shadow_would": would_k, "label": label, "answer": answer}
        self._write(rec, payload, events)
        return {"n": n, "answer": answer, "by": by, "served": sv, "candidate": c, "sampled": sampled, "label": label,
                "label_source": lsrc, "events": [e["event"] for e in events]}

    # --- the version protocol
    def _shares(self, ver):
        thr_k, thr_s = ver["promise"].threshold, self._threshold(self.served)
        m = len(ver["samples"])
        if not m:
            return 0.0, 0.0
        lk = sum(1 for sk, _ in ver["samples"] if sk is not None and sk >= thr_k) / m
        ls = sum(1 for _, ss in ver["samples"] if ss is not None and ss >= thr_s) / m
        return lk, ls

    def _event(self, event, v, n, **extra):
        p = self._vs[v]["promise"]
        st = None if p is None else p.state()
        return {"event": event, "version": v, "v_index": None if p is None else p.v,
                "delta_v": None if p is None else p.delta_v, "threshold": None if st is None else st["threshold"],
                "n": None if p is None else p.n, "state_fp": None if st is None else _sha(st), "decision": n, **extra}

    def _protocol(self, n) -> list:
        ev = []
        c = self.candidate
        if c is not None:
            ver = self._vs[c]
            k = len(ver["samples"])
            if not math.isinf(ver["promise"].threshold):
                lk, ls = self._shares(ver)
                if lk > 0.0 and lk >= ls + self.margin:
                    ev.append(self._event("promote", c, n, **{"from": self.served, "to": c, "l_k": lk, "l_s": ls,
                                                             "samples": k, "why": None}))
                    self._serve(c)
                elif k >= self.trial:
                    ev.append(self._discard(c, n, f"certified, not better after {k} samples (l_k {lk:.4g}, "
                                                  f"l_s {ls:.4g})"))
            elif k >= self.stale:
                ev.append(self._discard(c, n, f"not certified after {k} samples"))
        need = max(self.freeze_min, self.growth * self._vs[self.served]["seen"], self.refreeze_floor)
        if self.candidate is None and self.live.seen() >= need:
            ev.append(self._freeze(n))
        return ev

    def _discard(self, c, n, why):
        self._vs[c]["state"] = "discarded"
        self._vs[c]["model"] = None                        # never served: its pickle (versions_dir) remains
        self.candidate = None
        self.refreeze_floor = self.refreeze * self.live.seen()
        return self._event("discard", c, n, samples=len(self._vs[c]["samples"]), why=why)

    def _serve(self, v):
        old = self.served
        if old != v:
            self._vs[old]["state"] = "retired"
        self._vs[v]["state"] = "served"
        self.served = v
        if self.candidate == v:
            self.candidate = None
        if self.system is not None:
            model = self._vs[v]["model"]
            if model is None:
                self.system.heads.pop(self.question, None)
            else:
                from ...core.versions import serve_head
                serve_head(self.system, self.question, model.as_head(self.feature))

    def _freeze(self, n):
        k = max(self._vs) + 1
        model = self.live.snapshot() if self._fps is None else _Recorded(self._fps.get(k))
        p = StreamPromise(**self._pkw, budget=self.budget)
        self._ev_seen[k] = 0
        fp = model.fingerprint()
        self._vs[k] = {"version": k, "state": "candidate", "model": model, "promise": p, "frozen_at": n,
                       "seen": self.live.seen(), "fingerprint": fp, "samples": [], "fed": []}
        self.candidate = k
        if self.versions_dir is not None and self._fps is None:
            with open(self.versions_dir / f"{k}.pkl", "wb") as f:
                pickle.dump(model, f)
        return self._event("freeze", k, n, frozen_at=n, seen=self.live.seen(), fingerprint=fp, why=None)

    def _promise_events(self) -> list:
        out = []
        for v, ver in self._vs.items():
            p = ver["promise"]
            if p is None:
                continue
            new = p.events[self._ev_seen.get(v, 0):]
            self._ev_seen[v] = len(p.events)
            out += [{**e, "version": v, "decision": self.n} for e in new]
        return out

    # --- a person's call
    def rollback(self, v, *, by=None, why=None) -> dict:
        """Serve an earlier version again (its certificate and threshold as it left them; version 0: nothing alone).
        A candidate in trial is discarded (its δ_v stays spent): its paired comparison was against the version that
        served → the "rollback" event."""
        if v not in self._vs or self._vs[v]["state"] not in ("retired", "served") or v == self.served:
            raise ValueError(f"version {v} is not a retired version (served: {self.served}; "
                             f"{[x for x, ver in self._vs.items() if ver['state'] == 'retired']} retired)")
        ev = []
        if self.candidate is not None:
            ev.append(self._discard(self.candidate, self.n, "rollback"))
        frm = self.served
        self._serve(v)
        ev.append(self._event("rollback", v, self.n, **{"from": frm, "to": v, "by": None if by is None else str(by),
                                                        "why": why}))
        self._write(None, None, ev)
        return ev[-1]

    # --- the records
    def _write(self, rec, payload, events):
        q = self.question
        evs = [_json({**e, "question": q}) for e in events]
        if self._fps is not None:                          # replay: kept for the comparison
            self.log += [("stream", e) for e in evs]
            if rec is not None:
                self.log.append(("loop", _json({"question": q, "stream": rec, "loop": payload})))
            return
        if self.storage is None:
            if rec is not None:
                self.log.append(("loop", _json({"question": q, "stream": rec, "loop": payload})))
            self.log += [("stream", e) for e in evs]
            return
        from ...core.store import FORMAT
        if rec is not None:
            self.storage._append({"v": FORMAT, "kind": "loop", "question": q, "stream": _json(rec),
                                  "loop": _json(payload)})
        for e in evs:
            self.storage._append({"v": FORMAT, "kind": "stream", "stream": e})

    def _items(self):
        if self.storage is None:
            return list(self.log)
        out = []
        for s in self.storage.iter(kind=None):
            d = s.data
            if d.get("kind") == "loop" and d.get("question") == self.question:
                out.append(("loop", {"question": d["question"], "stream": d["stream"], "loop": d.get("loop")}))
            elif d.get("kind") == "stream" and (d.get("stream") or {}).get("question") == self.question:
                out.append(("stream", d["stream"]))
        return out

    def replay(self, rescore=False) -> dict:
        """Re-run the protocol from the records and compare → {"ok", "mismatches": [(n, what, why)], "decisions",
        "versions"}. See the module docstring."""
        items = self._items()
        stored_ev = [e for k, e in items if k == "stream"]
        fps = {e["version"]: e.get("fingerprint") for e in stored_ev if e.get("event") == "freeze"}
        fresh = LearningLoop(_Counting(), **{**self._kw, "secret": self._secret})
        fresh._fps = fps
        bad, k = [], 0
        for kind, d in items:
            if kind == "stream":
                if d.get("event") == "rollback":
                    try:
                        fresh.rollback(d["to"], by=d.get("by"), why=d.get("why"))
                    except ValueError as e:
                        bad.append((d.get("decision"), "rollback", str(e)))
                continue
            k += 1
            st, lp = d["stream"], d.get("loop") or {}
            if st["n"] != fresh.n + 1:
                bad.append((st["n"], "order", f"decision {st['n']} after {fresh.n}"))
                break
            if st.get("served") != fresh.served or st.get("candidate") != fresh.candidate:
                bad.append((st["n"], "version", f"recorded served {st.get('served')} / candidate "
                                                f"{st.get('candidate')}, rebuilt {fresh.served} / {fresh.candidate}"))
            label = lp.get("label")
            fresh._advance(lp.get("input"), lp.get("would"), st.get("score"), lp.get("shadow_would"), st.get("shadow"),
                           lambda: label, st.get("label_source") or fresh.source)
            got = next(r for kk, r in reversed(fresh.log) if kk == "loop")["stream"]
            for f in sorted(set(st) | set(got)):
                if st.get(f) != got.get(f):
                    bad.append((st["n"], f, f"recorded {st.get(f)}, rebuilt {got.get(f)}"))
        rebuilt_ev = [e for kk, e in fresh.log if kk == "stream"]
        if rebuilt_ev != stored_ev:
            bad.append((None, "stream records", f"{len(stored_ev)} stored, {len(rebuilt_ev)} rebuilt, or they differ"))
        if self.versions_dir is not None:
            bad += self._check_pickles(fps, items, rescore)
        return {"ok": not bad, "mismatches": bad, "decisions": k, "versions": len(fps)}

    def _check_pickles(self, fps, items, rescore):
        bad, models = [], {}
        for v, fp in fps.items():
            p = self.versions_dir / f"{v}.pkl"
            if not p.exists():
                bad.append((None, "version file", f"{p} is missing"))
                continue
            with open(p, "rb") as f:
                m = pickle.load(f)  # noqa: S301 — the loop's own files (trusted as the calibration file is)
            if m.fingerprint() != fp:
                bad.append((None, "version file", f"version {v}: fingerprint {m.fingerprint()[:12]}, recorded "
                                                  f"{str(fp)[:12]}"))
            models[v] = m
        if rescore:
            for kind, d in items:
                if kind != "loop":
                    continue
                st, lp = d["stream"], d.get("loop") or {}
                for v, key in ((st.get("served"), "score"), (st.get("candidate"), "shadow")):
                    if v in models:
                        got = models[v].score(lp.get("input"))[1]
                        if _num(got) != st.get(key):
                            bad.append((st["n"], "rescore", f"version {v}: recorded {st.get(key)}, scores {got}"))
        return bad

    # --- what it says
    def versions(self) -> list:
        out = []
        for v, ver in sorted(self._vs.items()):
            p = ver["promise"]
            out.append({"version": v, "state": ver["state"], "certified": p is not None and not math.isinf(p.threshold),
                        "threshold": None if p is None or math.isinf(p.threshold) else p.threshold,
                        "n": None if p is None else p.n, "v_index": None if p is None else p.v,
                        "delta_v": None if p is None else p.delta_v, "frozen_at": ver["frozen_at"],
                        "seen": ver["seen"], "fingerprint": ver["fingerprint"]})
        return out

    def fed(self, v) -> list:
        """The decisions whose samples reached version v's promise, in order."""
        return list(self._vs[v]["fed"])

    @property
    def delta_spent(self) -> float:
        return self.budget.spent

    def __repr__(self):
        return (f"LearningLoop({self.question!r}, served {self.served}, candidate {self.candidate}, n={self.n}, "
                f"labels {self.live.seen()})")


__all__ = ["LearningLoop", "S2_SOURCES"]
