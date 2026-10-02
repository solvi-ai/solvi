"""A memory of corrected cases: the inputs people (or outcomes, or your rules) corrected, and at decision time the nearest
of them — a second signal next to the decider, with an abstain threshold, recorded in the trace.

    mem = team.memory()                          # a CorrectionMemory bound to the decision part `team`
    mem.add(email, "billing", source="human", by="ann", stored_id=res.stored_id)
    mem.learn_from(store)                        # or every trusted correction of the question in a TraceStorage
    mem.calibrate(risk=0.05)                     # the abstain threshold, leave-one-out over the stored corrections

    team(email=...)                              # extra["memory"]: its proposal, the cases it rests on, its fingerprint

What a case is: the decider's probabilities over the options for the input (from the raw logits at the checkpoint's
temperature, before any adapt / fit / teach, so a later fit does not move the stored cases), optionally the input's words
(`text=True`: a set of hashed words, no embedding model), the label (the decision's label: an option, "yes" / "no", a tuple
for multi-label, "<not stated>") and where it came from: source ("human", "outcome" or "rule" — never the system's own
answer: anything else is refused), by (who), time and stored_id (the correction's or the decision's id in a TraceStorage).

The proposal: the k nearest cases within `radius` (total-variation distance between the probability vectors, averaged with
the words' Jaccard distance when text=True), each weighted 1 − distance / radius; the label with the most weight is
proposed when it leads the others by at least `min_strength` (its weight minus theirs) and holds at least
`min_agreement` of the weight — otherwise the memory abstains and says why. Ties are broken by case id: the same memory
gives the same answer every time.

What it does with a proposal (mode):
  "check" (default)  it never answers: when the decider would answer alone and the memory proposes another label, the
                     decision escalates ("memory of corrections disagrees: …"); agreement is recorded. It can only make
                     more decisions escalate, so a guarantee from act_guard still holds.
  "answer"           as "check", and when the decider escalated by its own threshold (act, confidence, margin) and the
                     memory proposes a label, the memory answers with it; the trace says so (action "answered", the
                     escalation it replaced) and the part's act_guard promise is not claimed for that answer (the
                     memory's own promise, from calibrate, is recorded instead). The decision's confidence is then
                     the model's own probability of that answer — its `probs` still describe the model — so a
                     `min_confidence` meant for the model is not passed on the neighbours' word; their agreement
                     is in extra["memory"]. The answer is listed among the conformal candidates.
Inside a Cascade / Vote / Route the memory only checks.

Every decision records extra["memory"]: {"fp", "n", "mode", "proposal", "strength", "agreement", "abstain", "neighbours":
[{"id", "label", "distance", "weight", "source", "by", "time", "stored_id"}], "action"}; the memory's fingerprint is part of
the decision part's fingerprint, so a replay tells a decision made with another memory state."""
from __future__ import annotations

import json
import math
import re
import threading
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np

from .core import NOT_STATED_KEY, Unknown
from .provenance import ESCALATED, MEMORY, digest
from .storage import TRUSTED_SOURCES, UntrustedLabel, check_source
MODES = ("check", "answer")
_WORD = re.compile(r"\w{3,}", re.U)


def words(text):
    """The words of a text as a sorted tuple of 32-bit hashes (lower case, 3+ letters) — the cheap text representation."""
    return tuple(sorted({zlib.crc32(w.encode()) for w in _WORD.findall(str(text).lower())}))


def _key(label):
    return json.dumps(label, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class Case:
    """One corrected case (see the module docstring)."""
    id: str
    features: tuple
    label: object
    source: str
    by: str | None = None
    time: float | None = None
    stored_id: str | None = None
    words: tuple | None = None

    def to_dict(self):
        d = asdict(self)
        d["features"] = list(self.features)
        d["words"] = None if self.words is None else list(self.words)
        return d


@dataclass
class Proposal:
    """What the memory proposes for one input: a label (None: it abstains, `abstain` says why), how strongly (the label's
    weight minus the others'), the share of the weight it holds, and the cases it rests on."""
    label: object = None
    strength: float = 0.0
    agreement: float = 0.0
    neighbours: list = field(default_factory=list)
    abstain: str | None = None

    def to_dict(self):
        return {"proposal": self.label, "strength": round(self.strength, 6), "agreement": round(self.agreement, 6),
                "abstain": self.abstain, "neighbours": self.neighbours}


class CorrectionMemory:
    """Corrected cases of one decision part and their nearest neighbours (see the module docstring). Usually made by
    `part.memory(...)`, which also attaches it to the part."""

    def __init__(self, part, k=7, radius=0.15, min_strength=1.0, min_agreement=0.8, text=False, text_weight=0.5,
                 mode="check"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if part.spec.kind in ("span", "rank", "number"):
            raise ValueError(f"a memory of corrections needs a question with options; not for {part.spec.kind!r}")
        self.part = part
        self.k, self.radius = max(1, int(k)), float(radius)
        self.min_strength, self.min_agreement = float(min_strength), float(min_agreement)
        self.text, self.text_weight, self.mode = bool(text), float(text_weight), mode
        self.guarantee = None                     # what calibrate promises for the memory's own answers
        self.cases = []                           # replaced, never changed in place (a proposal reads a snapshot)
        self._ids = set()
        self._F = None                            # the features as a matrix (rebuilt when cases change)
        self._lock = threading.Lock()             # add / remove / load against a proposal's snapshot of cases + matrix
        self.weights = part.model.weights_fingerprint()

    # --- the cases
    def __len__(self):
        return len(self.cases)

    def features(self, text, learning=False):
        """The decider's probabilities over the options for a text (raw logits, checkpoint temperature), rounded.
        learning: the features are for a case to store — a model that gave no usable output raises (see decide._usable)."""
        sp = self.part.spec
        z, _ = self.part._read([text])[0]           # a long input: the window a decision reads
        if learning:
            from .decide import _usable
            _usable([z])
        z = np.asarray(z, float) / self.part.model._T(sp)
        if sp.multi:
            p = 1 / (1 + np.exp(-z))
        else:
            p = np.exp(z - z.max())
            p = p / p.sum()
        return tuple(round(float(x), 6) for x in p)

    def _label(self, correct):
        if correct is Unknown or correct == NOT_STATED_KEY:
            return NOT_STATED_KEY
        lab = self.part.spec.label(correct)
        return list(lab) if isinstance(lab, tuple) else lab

    def add(self, x, correct, source="human", by=None, time=None, stored_id=None):
        """Store one corrected case: an input (a text, a state, or Facts by name) and its right answer. source: "human",
        "outcome" or "rule" — anything else raises UntrustedLabel (the system's own answers are never labels). → the
        Case (an identical case already stored is not added twice)."""
        check_source(source)
        text = self.part._input_text(x)
        f = self.features(text, learning=True)
        lab = self._label(correct)
        ws = words(text) if self.text else None
        cid = digest("case", f, ws, lab, source, by, time, stored_id)
        case = Case(cid, f, lab, source, None if by is None else str(by), None if time is None else float(time),
                    None if stored_id is None else str(stored_id), ws)
        with self._lock:
            if cid not in self._ids:
                self.cases = self.cases + [case]
                self._ids.add(cid)
                self._F = None
        return case

    def learn_from(self, storage, question=None, system=None):
        """Add every trusted correction of `question` (default: the part's name) stored in a TraceStorage (its teach
        records: System.teach, save_correction). An input is the correction's state: the part's facts are read from it,
        or computed by `system` (System.facts_for) when the state holds only the inputs they are computed from. →
        {"added", "skipped": [(id, why)]} — a correction from an untrusted source or with an answer this decision
        cannot give is skipped, never added."""
        from .decide import Facts
        q = question or self.part.__name__
        added, skipped = 0, []
        for c in storage.corrections():
            if c["question"] != q:
                continue
            try:
                check_source(c.get("source", "human"))
                init = c["init"]
                if not all(f in init for f in self.part.facts):
                    if system is None:
                        raise ValueError(f"the correction's state does not give {self.part.facts}: pass system=")
                    init = system.facts_for(init)
                n = len(self.cases)
                self.add(Facts({f: init[f] for f in self.part.facts}), c["answer"], c.get("source", "human"),
                         c.get("by"), c["time"], c["id"])
                added += len(self.cases) - n
            except (UntrustedLabel, ValueError, KeyError) as e:
                skipped.append((c["id"], f"{type(e).__name__}: {e}"))
        return {"added": added, "skipped": skipped}

    def remove(self, ids):
        """Forget cases by id (e.g. a correction found to be wrong) → how many were removed."""
        ids = set(ids)
        with self._lock:
            n = len(self.cases)
            self.cases = [c for c in self.cases if c.id not in ids]
            self._ids = {c.id for c in self.cases}
            self._F = None
            return n - len(self.cases)

    # --- nearest neighbours
    def _snapshot(self):
        """(the cases, their features as a matrix), taken together: a case added meanwhile is in neither."""
        with self._lock:
            if self._F is None:
                self._F = np.array([c.features for c in self.cases], float) if self.cases else np.zeros((0, 0))
            return self.cases, self._F

    def _matrix(self):
        return self._snapshot()[1]

    def _distances(self, f, ws, cases=None, F=None):
        if cases is None:
            cases, F = self._snapshot()
        if F.shape[0] and F.shape[1] != len(f):     # loaded with strict=False from another question's file
            raise ValueError(f"the memory's cases have {F.shape[1]} features, this question gives {len(f)}: they were "
                             "stored for another question")
        d = 0.5 * np.abs(F - np.asarray(f, float)).sum(1)
        if self.part.spec.multi:
            d = d / max(1, F.shape[1]) * 2        # sigmoids: the mean difference per option, in [0, 1]
        if self.text and ws is not None:
            a = set(ws)
            jd = np.array([1.0 - (len(a & set(c.words or ())) / len(a | set(c.words or ())) if (a or c.words) else 0.0)
                           for c in cases])
            d = (1 - self.text_weight) * d + self.text_weight * jd
        return d

    def propose(self, x, exclude=None):
        """The memory's proposal for an input (a text, a state or Facts) → Proposal."""
        return self._propose(self.part._input_text(x), exclude)

    def _propose(self, text, exclude=None, f=None, ws=None, min_strength=None):
        """exclude: a case id or a set of them (leave-one-out); min_strength: in place of self.min_strength."""
        cases, F = self._snapshot()
        if not cases:
            return Proposal(abstain="the memory holds no corrected cases")
        f = self.features(text) if f is None else f
        if self.text and ws is None and text is not None:
            ws = words(text)
        d = self._distances(f, ws, cases, F)
        ex = set() if exclude is None else {exclude} if isinstance(exclude, str) else set(exclude)
        order = sorted((round(float(d[i]), 9), cases[i].id, i) for i in range(len(cases))
                       if cases[i].id not in ex and d[i] < self.radius)[: self.k]
        if not order:
            return Proposal(abstain=f"no corrected case within distance {self.radius:g}")
        floor = self.min_strength if min_strength is None else min_strength
        weight, near = {}, []
        for dist, _, i in order:
            c = cases[i]
            w = 1.0 - dist / self.radius
            weight[_key(c.label)] = weight.get(_key(c.label), 0.0) + w
            near.append({"id": c.id, "label": c.label, "distance": round(dist, 6), "weight": round(w, 6),
                         "source": c.source, "by": c.by, "time": c.time, "stored_id": c.stored_id})
        ranked = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
        top, wt = ranked[0]
        total = sum(weight.values())
        strength, agreement = wt - (total - wt), wt / total if total > 0 else 0.0
        label = json.loads(top)
        p = Proposal(label, float(strength), float(agreement), near)
        if agreement < self.min_agreement:
            p.abstain = ("similar cases disagree: " + ", ".join(f"{json.loads(k)!r} {v:.2f}" for k, v in ranked[:3])
                         + f" (agreement {agreement:.2f} < {self.min_agreement:g})")
        elif strength < floor:
            p.abstain = f"too little support: strength {strength:.2f} < {floor:g}"
        if p.abstain:
            p.label = None
        return p

    # --- the abstain threshold
    def calibrate(self, risk=0.05):
        """Choose min_strength by conformal risk control, leave-one-out over the stored cases: each case is proposed for by
        the others, and the lowest strength is taken at which P(the memory proposes AND is wrong) ≤ risk, for inputs like
        the stored corrections (inf: it never proposes). A case's twins — cases with the same features (and words) —
        are left out with it: a correction stored twice would otherwise vouch for itself. Recorded as the memory's
        promise; changes its fingerprint.
        → {"min_strength", "proposed" (share of the cases it would propose for), "error" (among them), "risk", "n",
        "guarantee", "radius", "nearest": {"min", "median", "max"} (each case's distance to its nearest other case)} and,
        when no case has another within the radius, "note": the memory then proposes for none of them, nothing can be
        learned about its proposals, and min_strength is inf (it does not propose) — compare `nearest` with `radius`
        (its values are None, and the note says so, when every stored case is a twin of every other)."""
        from .calibration import crc_threshold
        cases, _ = self._snapshot()
        if len(cases) < 2:
            raise ValueError("calibration needs at least two stored cases")
        twins = {}
        for c in cases:
            twins.setdefault((c.features, c.words), set()).add(c.id)
        sig, wrong = [], []
        for c in cases:                           # the live min_strength is not touched: decisions go on meanwhile
            p = self._propose(None, exclude=twins[(c.features, c.words)], f=c.features, ws=c.words,
                              min_strength=-math.inf)
            sig.append(p.strength if p.label is not None else -1e9)
            wrong.append(float(p.label is not None and _key(p.label) != _key(c.label)))
        real = [x for x in sig if x > -1e9]       # the strengths of the proposals (-1e9 marks "no proposal")
        if not real:
            t = math.inf                          # nothing was proposed: no proposal has been checked, so none is made
        else:
            t = crc_threshold(sig, wrong, risk)
            if t <= -1e9:                         # every proposal can stand: the floor is the weakest of them, not the mark
                t = min(real)
        self.min_strength = t
        s, w = np.array(sig), np.array(wrong)
        auto = (s >= t) & (s > -1e9)              # an abstention (no label) is never a proposal, whatever the floor
        self.guarantee = {"method": "crc-loo", "risk": risk, "n": len(cases),
                          "promise": f"P(the memory answers and is wrong) ≤ {risk:g} for inputs like the stored "
                                     "corrections (leave-one-out)"}
        out = {"min_strength": t, "proposed": float(auto.mean()),
               "error": float(w[auto].mean()) if auto.any() else 0.0, "risk": float((w * auto).mean()),
               "n": len(cases), "guarantee": self.guarantee["promise"], "radius": self.radius,
               "nearest": self._nearest(twins)}
        if not real and out["nearest"]["min"] is None:
            out["note"] = ("every stored case has the same features (one input corrected more than once?): each is left "
                           "out together with its twins, so no case is left to propose for another — nothing can be "
                           "learned about the memory's proposals and it will not propose (min_strength is inf)")
        elif not real:
            out["note"] = (f"no stored case has another within the radius {self.radius:g} (the nearest are "
                           f"{out['nearest']['min']:g} to {out['nearest']['max']:g} away): the memory proposes for none "
                           "of them and will not propose (min_strength is inf) — a larger radius?")
        return out

    def _nearest(self, twins):
        """Each stored case's distance to its nearest other case (its twins left out) → {"min", "median", "max"}; None
        values when every case is a twin of every other."""
        cases, F = self._snapshot()
        near = []
        for c in cases:
            d = self._distances(c.features, c.words, cases, F)
            other = [float(d[i]) for i, x in enumerate(cases) if x.id not in twins[(c.features, c.words)]]
            if other:
                near.append(min(other))
        if not near:
            return {"min": None, "median": None, "max": None}
        return {"min": round(min(near), 6), "median": round(float(np.median(near)), 6), "max": round(max(near), 6)}

    # --- identity and state
    def settings(self):
        return {"k": self.k, "radius": self.radius, "min_strength": self.min_strength, "min_agreement": self.min_agreement,
                "text": self.text, "text_weight": self.text_weight if self.text else None, "mode": self.mode,
                "guarantee": self.guarantee}

    def fingerprint(self):
        """A hash of everything that changes a proposal: the settings, the checkpoint, and every case."""
        return digest("CorrectionMemory", self.weights, self.settings(), sorted(c.id for c in self.cases))

    def to_dict(self):
        """The memory as JSON-able data: the checkpoint's hash, the question, the settings and every case (see
        load_dict)."""
        return {"weights": self.weights, "question": self.part.spec.describe(), "settings": self.settings(),
                "cases": [c.to_dict() for c in self.cases]}

    def load_dict(self, data, strict=True):
        """Replace the settings and cases with a to_dict() (strict: refuse cases from another checkpoint or of another
        question). A label this question cannot give, or cases whose features differ in length, are refused always."""
        if strict and data.get("weights") != self.weights:
            raise ValueError(f"the memory's cases were scored by checkpoint #{data.get('weights')}, this part's is "
                             f"#{self.weights}: build it again from the corrections (learn_from)")
        if strict:
            from .calibfile import _norm           # as it reads back from a file: tuples as lists
            mine, theirs = _norm(self.part.spec.describe()), _norm(data.get("question"))
            if theirs != mine:
                raise ValueError(f"the memory was made for another question ({theirs}); this part asks {mine}: its "
                                 "cases say nothing about this question — build it from this question's corrections "
                                 "(learn_from)")
        s = data.get("settings") or {}
        for k in ("k", "radius", "min_strength", "min_agreement", "text", "mode", "guarantee"):
            if k in s:
                setattr(self, k, s[k])
        if s.get("text_weight") is not None:
            self.text_weight = s["text_weight"]
        cases = [Case(c["id"], tuple(c["features"]), c["label"], c["source"], c.get("by"), c.get("time"),
                      c.get("stored_id"), None if c.get("words") is None else tuple(c["words"]))
                 for c in data.get("cases") or ()]
        for c in cases:
            check_source(c.source)
            try:
                self._label(tuple(c.label) if isinstance(c.label, list) else c.label)
            except ValueError as e:
                raise ValueError(f"case {c.id}: its label is not an answer of this question ({e}) — a memory of "
                                 "another question?") from None
        if len({len(c.features) for c in cases}) > 1:
            raise ValueError("the cases' features differ in length: they are not of one question")
        with self._lock:
            self.cases = cases
            self._ids = {c.id for c in cases}
            self._F = None
        return self

    def save(self, path):
        """Write the memory (to_dict) to a JSON file. → path."""
        from .schema import tag_floats
        with open(path, "w") as fh:
            json.dump(tag_floats(self.to_dict()), fh, ensure_ascii=False, indent=1, allow_nan=False)
        return path

    def load(self, path, strict=True):
        """Replace the settings and cases with a file written by save (strict: refuse a memory of another checkpoint
        or question, as load_dict). → self."""
        with open(path) as fh:
            from .schema import untag_floats
            return self.load_dict(untag_floats(json.load(fh)), strict)

    def __repr__(self):
        return f"CorrectionMemory({self.part.__name__!r}, {len(self.cases)} case(s), mode={self.mode!r})"

    # --- at decision time (called by DecisionPart._finish)
    def apply(self, d, text, alone=True):
        """Record the memory's proposal on a finished decision and act on it by the mode (see the module docstring).
        alone: the part decides by its own thresholds (False inside a combination: the memory only checks)."""
        sp = self.part.spec
        p = self._propose(text)
        rec = {"fp": self.fingerprint(), "n": len(self.cases), "mode": self.mode, **p.to_dict()}
        if p.label is None:
            rec["action"] = "abstained"
        else:
            try:
                mine = NOT_STATED_KEY if d.value is Unknown else self._label(d.value)
            except ValueError:
                mine = None
            same = mine is not None and _key(mine) == _key(p.label)
            if d.escalate is None:
                if same:
                    rec["action"] = "agrees"
                else:
                    d.escalate = (f"{MEMORY}: {len(p.neighbours)} similar corrected case(s) say {_shown(p.label)!r} "
                                  f"(strength {p.strength:.2f}); would have answered {d.value!r}")
                    rec["action"] = "escalated"
            elif self.mode == "answer" and alone and _threshold_escalation(d.escalate):
                rec["action"], rec["replaced"] = "answered", d.escalate
                rec["model_answer"] = _shown(mine)
                d.value = Unknown if p.label == NOT_STATED_KEY else sp.out(tuple(p.label) if sp.multi else p.label)
                d.confidence = _model_p(sp, d.probs, p.label)     # the model's own probability of the memory's answer
                d.escalate = None
                d.extra.pop("guarantee", None)
                rec["guarantee"] = (self.guarantee or {}).get("promise", "none: the memory's threshold is not calibrated")
            else:
                rec["action"] = "agrees (already escalated)" if same else "disagrees (already escalated)"
        d.extra["memory"] = rec
        return d


def _shown(label):
    return tuple(label) if isinstance(label, list) else label


def _model_p(sp, probs, label):
    """The model's probability of a label the memory answers with (multi-label: the least certain option's probability of
    being in / out of that set) — the confidence of a memory-given answer: the neighbours' agreement is not a
    probability the model's thresholds (min_confidence, a guard on the confidence) were made for."""
    if label == NOT_STATED_KEY:
        return float(probs.get(Unknown, 0.0))
    if sp.multi:
        return float(min((probs.get(o, 0.0) if o in label else 1.0 - probs.get(o, 0.0)) for o in sp.real)) if sp.real else 0.0
    return float(probs.get(label, 0.0))


def _threshold_escalation(reason):
    """Did the part escalate by its own threshold (the act signal, the confidence, the margin) — the only escalations a
    memory in mode "answer" may answer in place of?"""
    return bool(reason) and (reason.startswith(ESCALATED + ": act") or reason.startswith(("confidence ", "margin ")))


__all__ = ["Case", "CorrectionMemory", "MEMORY", "Proposal", "TRUSTED_SOURCES", "UntrustedLabel", "words"]
