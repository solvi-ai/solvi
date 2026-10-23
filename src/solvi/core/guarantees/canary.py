"""Is the same model still answering? An identity canary for a remote decider (an LLM server, a System One service).

A hosted model can change behind the same name — a provider swaps the weights, a gateway routes to another provider or
a lower quantisation — and nothing in the request says so: the decider's fingerprint (`DecideModel.fingerprint`) is
the configuration solvi asked for (endpoint, model name, prompt template, settings), not the weights that answered. A
guarantee calibrated on one model then covers answers of another. The canary asks a small fixed set of probe questions
and keeps their answers and probabilities as a behavioural fingerprint of the model; asked again later, a different
fingerprint is a change.

    from solvi.core.guarantees.canary import IdentityCanary
    model = llm("https://openrouter.ai/api/v1", "openai/gpt-oss-20b", api_key=KEY)
    canary = IdentityCanary([("Refund my double charge", "Which team?", ["billing", "shipping"]), ...],
                            every=500).attach(model)
    canary.probe()                       # the reference, now (else: at the first scheduled probe)
    ...                                  # every 500 decisions of the model the probes are asked again
    canary.changed, canary.events        # what was found; decision.extra["canary"] on the decisions after a change

Probes: (text, task, options) or {"text", "task", "options", "descriptions"} — single-choice questions with clear
answers (borderline probes flip under a provider's own non-determinism and give false alarms). A probe's fingerprint is
its answer and the probability of every option, from the scorer's raw output (the model's own probabilities, before any
local adaptation or temperature, so fitting the part never looks like a new model), and the model name the server says
answered (`extra["llm"]["served_by"]`). Two probe rounds differ when at least `min_changed` probes answer differently,
the mean total-variation distance of their probabilities is above `tol`, or the server names another model. With
`confirm=True` (the default) a difference is asked once more and counts only when it repeats. A probe the server does
not answer (an escalation) is left out of the comparison; when fewer than half of the probes answer, the round gives no
verdict (`"verdict": "unanswered"`).

When: `every=N` probes after every N decisions the model makes (cache hits included); `canary.probe()` on demand;
`every=None` (the default) only on demand. Probes cost calls — len(probes) requests a round, twice that when a
difference is confirmed — so nothing is asked unless you attach a canary and ask or schedule it. The first round is
the reference; `accept()` takes the latest round as the new reference (after you recalibrate on the new model).

After a change every decision of the model carries `extra["canary"] = {"changed": True, "event", "reference",
"identity"}` — recorded in the trace like every decision's extra — until `accept()`. `on_change="flag"` (the default)
only marks: a `DriftMonitor` observing those decisions flags the signal "identity" (and a Dispatcher with that monitor
treats it as drift). `on_change="escalate"` also escalates them ("model identity changed ...; would have answered ...").
A decision that triggered a probe round carries `extra["canary"]["probed"]` with the round's number. `events` holds
every round: {"n", "at" (decisions seen), "reason", "verdict" ("reference", "same", "changed", "unanswered"),
"answers_changed", "tv", "served", "identity", "reference", "calls"}.

Replay never probes: `Trace.replay` runs under `no_probes()`, and a model's decisions made inside it are neither
counted nor marked. An LLM is not re-run by replay anyway (its recorded output is checked); a remote decider declared
deterministic is re-run, and under `on_change="escalate"` a recorded escalation then replays as a mismatch — the
canary's state is not part of the replayed input.

What it is not: a statistical test with a false-alarm rate. It compares answers of a fixed probe set, so its false
alarms come from the provider's own non-determinism on the probes (choose clear ones, keep `confirm=True`), and a
change the probes do not touch is missed. fpverify (MIT) does an anytime-valid sequential test on many probes; this
canary is the cheap first step, measured on fakes only."""
from __future__ import annotations

import contextlib
import contextvars
import threading

import numpy as np

from ..provenance import digest

_QUIET = contextvars.ContextVar("solvi_canary_quiet", default=False)
ON_CHANGE = ("flag", "escalate")


@contextlib.contextmanager
def no_probes():
    """Within this block no canary probes, counts or marks a decision (replay runs under it)."""
    tok = _QUIET.set(True)
    try:
        yield
    finally:
        _QUIET.reset(tok)


def _probe(p):
    if isinstance(p, dict):
        return str(p["text"]), str(p["task"]), list(p["options"]), p.get("descriptions")
    text, task, options, *rest = p
    return str(text), str(task), list(options), (rest[0] if rest else None)


def changed_of(decision):
    """The canary mark of a decision (a Decision, a Result, or a dict with "extra") when it says the model changed,
    else None."""
    ex = decision.get("extra") if isinstance(decision, dict) else getattr(decision, "extra", None)
    c = ex.get("canary") if isinstance(ex, dict) else None
    return c if isinstance(c, dict) and c.get("changed") else None


class IdentityCanary:
    """See the module docs. probes: the fixed probe questions (at least one). every: probe after every N decisions of
    the model (None: on demand only). on_change: "flag" or "escalate". tol: the largest mean total-variation distance of
    the probe probabilities that is still the same model. min_changed: this many probes with another answer is a
    change. confirm: ask a difference once more before calling it a change."""

    def __init__(self, probes, *, every=None, on_change="flag", tol=0.15, min_changed=1, confirm=True):
        self.probes = [_probe(p) for p in probes]
        if not self.probes:
            raise ValueError("an identity canary needs at least one probe")
        if every is not None and (isinstance(every, bool) or int(every) < 1):
            raise ValueError(f"every must be a number of decisions of at least 1, or None, not {every!r}")
        if on_change not in ON_CHANGE:
            raise ValueError(f"on_change must be one of {ON_CHANGE}, not {on_change!r}")
        if not 0 <= float(tol) <= 1 or int(min_changed) < 1:
            raise ValueError("tol must be in [0, 1] and min_changed at least 1")
        self.every = None if every is None else int(every)
        self.on_change, self.tol, self.min_changed, self.confirm = on_change, float(tol), int(min_changed), bool(confirm)
        self.model = None
        self.reference = None            # the reference round's probe results
        self.latest = None
        self.events: list[dict] = []
        self.seen = 0                    # decisions of the model counted since attach
        self.calls = 0                   # probe questions asked
        self._change = None              # the event of the change in force (until accept)
        self._lock = threading.RLock()

    # --- wiring
    def attach(self, model):
        """Watch this DecideModel (an llm(...) or systemone(...) decider) → the canary. One canary per model."""
        if not callable(getattr(model, "_score", None)):
            raise TypeError(f"an identity canary watches a DecideModel, not {type(model).__name__}")
        model.canary = self
        self.model = model
        return self

    @property
    def changed(self):
        """Is a change in force (found and not yet accepted)?"""
        return self._change is not None

    @property
    def identity(self):
        """The reference round's fingerprint (a hash of its answers and served model names), None before it."""
        return None if self.reference is None else _identity(self.reference)

    # --- probing
    def _ask(self):
        from ..deciders.adapt import _Spec
        m = self.model
        pairs = [(_Spec(task, options, desc), m.text(text)) for text, task, options, desc in self.probes]
        out, _ = m._score(pairs)
        self.calls += len(pairs)
        res = []
        for (sp, text), o in zip(pairs, out):
            z, _ = m._pick(sp, o, text)
            why = getattr(z, "escalate", None)
            if why:
                res.append({"answer": None, "unanswered": str(why)[:120]})
                continue
            z = np.asarray(z, float)
            p = np.exp(z - z.max())
            p = p / p.sum()
            info = o.get("info") if isinstance(o, dict) else None
            served = None
            for v in info.values() if isinstance(info, dict) else ():     # {"llm": {..., "served_by": name}}
                if isinstance(v, dict) and v.get("served_by"):
                    served = str(v["served_by"])
            res.append({"answer": sp.real[int(np.argmax(p))], "p": [round(float(x), 4) for x in p],
                        **({"served_by": served} if served else {})})
        return res

    def _compare(self, ref, cur):
        both = [(a, b) for a, b in zip(ref, cur) if a["answer"] is not None and b["answer"] is not None]
        if 2 * len(both) < len(self.probes):
            return None
        changed = sum(a["answer"] != b["answer"] for a, b in both)
        tv = float(np.mean([0.5 * np.abs(np.subtract(a["p"], b["p"])).sum() for a, b in both])) if both else 0.0
        served = sorted({b.get("served_by") for _, b in both if b.get("served_by")} -
                        {a.get("served_by") for a, _ in both if a.get("served_by")})
        return {"answers_changed": int(changed), "tv": round(tv, 4), "served": served,
                "differs": changed >= self.min_changed or tv > self.tol or bool(served)}

    def probe(self, reason="on demand"):
        """Ask the probes now → the round's event (see the module docs). The first answered round is the reference."""
        if self.model is None:
            raise ValueError("attach the canary to a model first: canary.attach(model)")
        with self._lock:
            calls0 = self.calls
            cur = self._ask()
            ev = {"n": len(self.events) + 1, "at": self.seen, "reason": reason}
            if self.reference is None:
                answered = sum(r["answer"] is not None for r in cur)
                if 2 * answered < len(self.probes):
                    ev.update(verdict="unanswered")
                else:
                    self.reference = cur
                    ev.update(verdict="reference")
            else:
                cmp = self._compare(self.reference, cur)
                if cmp is not None and cmp["differs"] and self.confirm:
                    again = self._ask()
                    cmp2 = self._compare(self.reference, again)
                    if cmp2 is None or not cmp2["differs"]:
                        cmp = {**(cmp2 or cmp), "differs": False, "unconfirmed": cmp}
                    else:
                        cur, cmp = again, cmp2
                if cmp is None:
                    ev.update(verdict="unanswered")
                else:
                    ev.update(verdict="changed" if cmp.pop("differs") else "same", **cmp)
            self.latest = cur
            ev.update(identity=_identity(cur), reference=self.identity, calls=self.calls - calls0)
            if ev["verdict"] == "changed" and self._change is None:
                self._change = ev
            self.events.append(ev)
            return ev

    def accept(self):
        """Take the latest round as the new reference (after recalibrating on the new model): the change is cleared."""
        with self._lock:
            if self.latest is not None and any(r["answer"] is not None for r in self.latest):
                self.reference = self.latest
            self._change = None
            self.events.append({"n": len(self.events) + 1, "at": self.seen, "reason": "accept", "verdict": "accepted",
                                "identity": self.identity, "reference": self.identity, "calls": 0})

    # --- the model's decisions (called by DecideModel._decision)
    def mark(self, d):
        """Count one decision of the model, probe when it is due, and mark the decision after a change."""
        if _QUIET.get():
            return d
        with self._lock:
            self.seen += 1
            ev = None
            if self.every is not None and self.seen % self.every == 0:
                ev = self.probe(reason=f"every {self.every} decisions")
            ch = self._change
        if ev is not None:
            d.extra.setdefault("canary", {})["probed"] = ev["n"]
        if ch is not None:
            d.extra.setdefault("canary", {}).update(changed=True, event=ch["n"], reference=ch["reference"],
                                                    identity=ch["identity"])
            if self.on_change == "escalate" and d.escalate is None:
                d.escalate = (f"model identity changed (canary event {ch['n']}: {ch.get('answers_changed', 0)} of "
                              f"{len(self.probes)} probes answer differently, mean probability shift {ch.get('tv', 0):.2f}"
                              + (f", served by {', '.join(ch['served'])}" if ch.get("served") else "")
                              + f"); would have answered {d.value!r}")
        return d

    def record(self):
        """The canary as data: settings, the reference identity, the events."""
        return {"probes": len(self.probes), "every": self.every, "on_change": self.on_change, "tol": self.tol,
                "min_changed": self.min_changed, "confirm": self.confirm, "identity": self.identity,
                "changed": self.changed, "seen": self.seen, "calls": self.calls, "events": list(self.events)}


def _identity(rounds):
    return digest("canary", [(r.get("answer"), r.get("served_by")) for r in rounds])[:16]


__all__ = ["IdentityCanary", "changed_of", "no_probes"]
