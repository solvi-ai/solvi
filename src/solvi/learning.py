"""Learning from corrections, with gates and rollback (experimental): System.learning(...).

    loop = system.learning(store)              # nothing happens until you run it; System.teach now only stores
    rep = loop.run()                           # labels → a proposed update → gates → promoted or rejected, recorded
    print(rep)                                 # what changed, each gate's numbers
    loop.versions()                            # every promoted state; loop.rollback(2) restores one

Labels come only from outside the model, read from a TraceStorage: human corrections (System.teach, save_correction),
known outcomes (source="outcome") and your rules' rejections (source="rule"; with harvest_rules=True also the stored
decisions where a hard check forced another answer than the model proposed). A correction from any other source is
refused and listed in `labels()["rejected"]`; the stored decisions — the system's own answers — are never read as labels,
so self-training is impossible by construction, not by a setting.

Each label is put, by a hash of its stored id, into "train", "calibration" or "holdout" (default 50 / 20 / 30%): a
held-out label is never trained on, in this update or any later one.

The ladder, per question, by the number of training labels: fewer than `fit_below` (50) — the shift / scale of the decider
(DecisionPart.fit); up to `memory_below` (1000) — fit plus a memory of the corrected cases (solvi.memory, mode "check" by
default); beyond — the `adapter` hook when you give one (a callable (part, [(text, answer)]) → a JSON-able description; an
object with state(part) / restore(part, state) is rolled back too), else the memory.

The gates (every one must pass, else the update is undone and recorded as rejected):
  consistency  the new training labels agree with a memory of the labels already learned: at most `max_conflict` (20%)
               of them may be contradicted by close, agreeing earlier corrections (B9: wrong corrections hurt more than
               right ones help);
  heldout      on the held-out labels, asked through the whole system: the share answered alone and right, minus the
               share answered alone and wrong, must improve by at least `min_gain` (0.01), with at least `min_holdout`
               (5) held-out labels;
  honesty      the honesty numbers (solvi.honesty: confident errors, coverage at `risk`, quote support) on the held-out
               labels — and on your own honesty set when given (gates={"honesty": cases or a set file}) — must not get
               worse by more than `tolerance` (0.02);
  act_guard    a part calibrated with act_guard / calibrate_for is recalibrated on the calibration labels (at least
               `min_calibration`, 30), with the same risk: an old threshold says nothing about a changed signal; a
               part's conformal answer sets are recalibrated on them too (with fewer labels they are dropped, and the
               gate's record says so — they no longer hold for the changed probabilities);
  size         shadow run: the stored decisions whose input is held out for a learned question (the same split as the
               labels: per question, by content_key) — at most `shadow_limit`, 500 — are asked with the current and the
               candidate state and compared (solvi.diff.compare) on those questions and the unlearned ones; at most
               `max_change` (30%) of them may change — one update may not move more.

The candidate is built and gated on a shadow of the system: copies of the decision parts (their thresholds, memory,
and the model's adaptations; the checkpoint itself is shared) in a shallow copy of the catalog and the System. The live
parts are not touched until the update is promoted, so asks that run meanwhile (another thread, the server) see the
state in force, never an un-gated candidate. An `adapter` hook runs on the shadow part: its effect reaches the live part
through its state(part) / restore(part, state); a hook without them only keeps what it changes in objects the shadow
shares with the live part (the checkpoint), and such changes are not isolated. When a gated candidate cannot be carried
over to the live parts (a hook without state / restore changed the shadow part itself, so the promoted fingerprint is
not the candidate's), run() puts the live parts back as they were, records the update as not promoted (gate
"promotion") and raises RuntimeError.

A promoted update gets the next version number; every run that proposes an update is recorded in the changelog (a
TraceStorage, by default the same store: kind "update", hash-chained with the decisions) with its gates' results, the
fingerprints before and after, the labels it trained on and — for a promoted state — the state itself (adaptations,
thresholds, memory), so `rollback(version)` restores any promoted version, in this process or another. Each decision
records the fingerprint of the part that made it, which names the version (`loop.version_of(fp)`)."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import warnings
from dataclasses import asdict, dataclass, field

from .storage import FORMAT, TRUSTED_SOURCES, UntrustedLabel, check_source, plain

LADDER = {"fit_below": 50, "memory_below": 1000, "adapter": None,
          "memory": {"k": 7, "radius": 0.15, "min_strength": 1.0, "min_agreement": 0.8, "mode": "check"}}
GATES = {"min_gain": 0.01, "min_holdout": 5, "tolerance": 0.02, "risk": 0.10, "max_change": 0.30, "shadow_limit": 500,
         "max_conflict": 0.20, "min_calibration": 30, "honesty": None}


def _check_settings(ladder, gates, holdout, calibration):
    """System.learning's settings → ValueError naming the first that is wrong: an unknown ladder, memory or gate key
    (a typo would be ignored), a memory mode that does not exist, a gate rate outside [0, 1] (risk strictly inside),
    a negative count, or holdout / calibration shares that leave no label to train on."""
    from .calibration import check_rate
    from .memory import _settings
    unknown = set(ladder or ()) - set(LADDER)
    if unknown:
        raise ValueError(f"unknown ladder settings: {sorted(unknown)} (known: {sorted(LADDER)})")
    mem = (ladder or {}).get("memory") or {}
    if not isinstance(mem, dict) or set(mem) - set(LADDER["memory"]):
        raise ValueError(f"unknown memory settings: {sorted(set(mem) - set(LADDER['memory'])) if isinstance(mem, dict) else mem!r} "
                         f"(known: {sorted(LADDER['memory'])})")
    _settings(**mem)
    for k in ("fit_below", "memory_below"):
        v = (ladder or {}).get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            raise ValueError(f"ladder {k} must be a number of labels ≥ 0, not {v!r}")
    unknown = set(gates or ()) - set(GATES)
    if unknown:
        raise ValueError(f"unknown gate settings: {sorted(unknown)} (known: {sorted(GATES)})")
    for k, v in (gates or {}).items():
        if k == "risk":
            check_rate("gate risk", v)
        elif k in ("min_gain", "tolerance", "max_change", "max_conflict"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                raise ValueError(f"gate {k} must be a share in [0, 1], not {v!r}")
        elif k in ("min_holdout", "shadow_limit", "min_calibration"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f"gate {k} must be a count ≥ 0, not {v!r}")
    for name, v in (("holdout", holdout), ("calibration", calibration)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1:
            raise ValueError(f"{name} must be a share in [0, 1), not {v!r}")
    if holdout + calibration >= 1:
        raise ValueError(f"holdout ({holdout}) + calibration ({calibration}) leave no label to train on: their sum must "
                         "be below 1")


class ExperimentalWarning(UserWarning):
    """A feature whose API and behaviour may still change."""


OPEN_KINDS = ("span", "rank", "number")        # answers not from a closed list: the loop leaves them alone


def content_key(question, init):
    """What a label's split is decided by: the question and a hash of its input — never a stored id, whose hash covers
    measured timings and so differs from run to run."""
    from .runtime import vhash
    return f"{question}|{vhash(dict(init or {}))}"


def split_of(label_id, holdout=0.3, calibration=0.2):
    """A label's split by a hash of its key (content_key) — "holdout", "calibration" or "train" — the same in every run."""
    h = int(hashlib.sha256(("solvi-learning:" + str(label_id)).encode()).hexdigest()[:8], 16) / 2 ** 32
    return "holdout" if h < holdout else "calibration" if h < holdout + calibration else "train"


@dataclass
class Label:
    """A trusted label: a stored correction (or rule rejection) of one question."""
    id: str
    question: str
    init: dict
    answer: object                                # normalized to the question's answer type
    source: str
    by: str | None = None
    of: str | None = None
    time: float | None = None
    split: str = "train"


@dataclass
class UpdateReport:
    """What one run of the loop did: "none" (nothing new to learn), "promoted" or "rejected"."""
    action: str
    version: int | None = None                    # the promoted version (None when not promoted)
    current: int | None = None                    # the version in force after the run
    questions: dict = field(default_factory=dict)  # question → {"rung", "train", "new", "calibration", "holdout"}
    gates: dict = field(default_factory=dict)      # gate → {"ok", ...numbers}
    fp_before: str | None = None
    fp_after: str | None = None
    rejected_labels: list = field(default_factory=list)
    record_id: str | None = None

    @property
    def promoted(self):
        return self.action == "promoted"

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        head = {"none": "learning: nothing new to learn", "promoted": f"learning: update promoted as version {self.version}",
                "rejected": "learning: update rejected by the gates; the current state is kept"}[self.action]
        lines = [head + (f" (version in force: {self.current})" if self.current is not None else "")]
        for q, x in self.questions.items():
            lines.append(f"  {q}: {x['rung']} on {x['train']} training label(s) ({x['new']} new), "
                         f"{x['calibration']} for calibration, {x['holdout']} held out")
        for g, r in self.gates.items():
            lines.append(f"  {'✓' if r.get('ok') else '✗'} {g}: {r.get('why', '')}")
        if self.rejected_labels:
            lines.append(f"  {len(self.rejected_labels)} label(s) refused: "
                         + "; ".join(f"{i}: {w}" for i, w in self.rejected_labels[:5]))
        return "\n".join(lines)


class Learning:
    """The learning loop of a System (see the module docstring). Made by System.learning(...); experimental."""

    experimental = True

    def __init__(self, system, storage=None, parts=None, ladder=None, gates=None, changelog=None, holdout=0.3,
                 calibration=0.2, gate_teach=True, harvest_rules=False):
        from .decide import DecisionPart, decision_of
        from .storage import open_storage
        warnings.warn("System.learning is experimental: its API and gates may change", ExperimentalWarning, stacklevel=3)
        self.system = system
        self.storage = open_storage(storage, system) if storage is not None else system.storage
        if self.storage is None:
            raise ValueError("the learning loop reads labels from a TraceStorage: pass storage= or System(storage=...)")
        self.changelog = open_storage(changelog, system) if changelog is not None else self.storage
        if isinstance(parts, dict):
            chosen = dict(parts)
        else:
            names = list(system.questions) if parts is None else list(parts)
            chosen = {q: decision_of(system.catalog, q) for q in names}
            if parts is None:                          # closed-list decision parts only (see below; not combinations)
                chosen = {q: p for q, p in chosen.items()
                          if isinstance(p, DecisionPart) and p.spec.kind not in OPEN_KINDS}
        for q, p in chosen.items():
            if q not in system.questions:
                raise ValueError(f"{q!r} is not a question of this system")
            if not isinstance(p, DecisionPart):
                raise ValueError(f"question {q!r}: the learning loop updates a decision part (model.decision(...)); "
                                 f"{p!r} is not one (combinations are not supported yet)")
            if p.spec.kind in OPEN_KINDS:
                raise ValueError(f"question {q!r}: the learning loop learns closed-list questions (choice, multi-label, "
                                 f"score, yes/no); a {p.spec.kind} question is not one — in a simulation on real "
                                 "streams learning did not help spans and a memory of corrections hurt them")
        if not chosen:
            raise ValueError("no question of this system is answered by a decision part: nothing to learn")
        self.parts = chosen
        _check_settings(ladder, gates, holdout, calibration)
        self.ladder = {**LADDER, **(ladder or {})}
        self.ladder["memory"] = {**LADDER["memory"], **((ladder or {}).get("memory") or {})}
        self.gates = {**GATES, **(gates or {})}
        self.holdout, self.calibration = float(holdout), float(calibration)
        self.gate_teach, self.harvest_rules = bool(gate_teach), bool(harvest_rules)
        self._states = {}                              # version → in-process snapshot (keeps objects JSON cannot hold)
        system._learning = self

    def detach(self):
        """Stop gating System.teach (the loop's records stay in the changelog)."""
        if getattr(self.system, "_learning", None) is self:
            self.system._learning = None

    # --- labels
    def labels(self):
        """The trusted labels of the loop's questions → {"labels": [Label], "rejected": [(id, why)]}. Only corrections
        (teach records) from TRUSTED_SOURCES, with an answer the question and its decision can take; with
        harvest_rules=True also the stored decisions where a hard check forced another answer than the model's."""
        out, rejected, seen = [], [], set()
        for c in self.storage.corrections():
            q = c["question"]
            if q not in self.parts:
                continue
            try:
                check_source(c.get("source", "human"))
                ans = self._normalize(q, c["answer"])
            except (UntrustedLabel, ValueError) as e:
                rejected.append((c["id"], f"{type(e).__name__}: {e}"))
                continue
            out.append(Label(c["id"], q, dict(c["init"]), ans, c.get("source", "human"), c.get("by"), c.get("of"),
                             c.get("time"), split_of(content_key(q, c["init"]), self.holdout, self.calibration)))
            seen.add((c.get("of"), q))
        if self.harvest_rules:
            for s in self.storage.iter():
                resp = s.data.get("response") or {}
                for q in self.parts:
                    lab = self._rule_label(s, resp, q)
                    if lab is not None and (s.id, q) not in seen:
                        out.append(lab)
        return {"labels": out, "rejected": rejected}

    def _normalize(self, q, answer):
        from .core import NOT_STATED_KEY, Unknown
        at = self.system.questions[q].answer
        ans = Unknown if answer == NOT_STATED_KEY else at.normalize(answer)
        if ans is not Unknown:
            self.parts[q].spec.label(ans)              # the decision can give it (a bool question: yes / no)
        return ans

    def _rule_label(self, s, resp, q):
        """A stored decision where a hard check forced an answer that differs from the model's proposal → Label (source
        "rule") or None."""
        res = (resp.get("results") or {}).get(q) or {}
        if res.get("guard") != "hard_check" or res.get("status") != "forced":
            return None
        part = self.parts[q]
        prop = None
        for r in (resp.get("trace") or {}).get("records") or ():
            if r.get("name") in ("answer:" + q, part.__name__) and r.get("model") is not None:
                prop = r.get("value")
        try:
            ans = self._normalize(q, res.get("answer"))
            if prop is None or part.spec.label(self._normalize(q, prop)) == part.spec.label(ans):
                return None
        except (ValueError, TypeError):
            return None
        init = (resp.get("trace") or {}).get("init") or {}
        lid = f"{s.id}:{q}"
        return Label(lid, q, dict(init), ans, "rule", None, s.id, s.time,
                     split_of(content_key(q, init), self.holdout, self.calibration))

    def _text(self, q, init):
        part = self.parts[q]
        vals = init if all(f in init for f in part.facts) else self.system.facts_for(init)
        return part.text_of(vals)

    # --- state: snapshots, fingerprints, versions
    def fingerprint(self):
        """The fingerprint of the loop's current state: every learned part's fingerprint."""
        return self._fingerprint_of(self.parts)

    @staticmethod
    def _fingerprint_of(parts):
        from .provenance import digest
        return digest("learning", sorted((q, p.fingerprint()) for q, p in parts.items()))

    def _snapshot(self, parts=None):
        from .decide import Adaptation
        out = {}
        for q, p in (parts or self.parts).items():
            a = p.adaptation
            ad = None if a is None else {**Adaptation.params(a), "examples": [[list(z), list(y) if isinstance(y, tuple)
                                                                               else y] for z, y in a.examples]}
            mem = p.correction_memory
            out[q] = {"adaptation": ad, "escalate_below": p.escalate_below, "act_threshold": p.act_threshold,
                      "guarantee": copy.deepcopy(p.guarantee), "conformal": copy.deepcopy(p.conformal_set),
                      "groups": None if p.groups is None else "per-group thresholds (kept in this process only)",
                      "memory": None if mem is None else mem.to_dict(), "adapter": self._adapter_state(p)}
        return out

    def _adapter_state(self, p):
        ad = self.ladder.get("adapter")
        return ad.state(p) if ad is not None and callable(getattr(ad, "state", None)) else None

    def _live(self):
        """What JSON cannot hold (per-group thresholds, the memory object) for an in-process restore."""
        return {q: {"groups": p.groups, "memory": p.correction_memory} for q, p in self.parts.items()}

    # --- the shadow a candidate is built on
    def _shadow(self):
        """→ (a shadow System, {question: shadow part}): every decision part of the catalog whose model is behind a learned
        part is copied onto a copy of that model (its own adaptations; the scorer, cache and checkpoint are shared), with
        its own thresholds; the catalog and the System are shallow copies whose parts point at the copies."""
        import dataclasses

        from .core import _group_func
        from .decide import DecisionPart
        models = {}
        for p in self.parts.values():
            if id(p.model) not in models:
                m = copy.copy(p.model)
                if isinstance(getattr(p.model, "adaptations", None), dict):
                    m.adaptations = copy.deepcopy(p.model.adaptations)
                models[id(p.model)] = m
        made = {}

        def sub(obj):
            if not isinstance(obj, DecisionPart) or id(obj.model) not in models:
                return obj
            if id(obj) not in made:
                made[id(obj)] = _shadow_part(obj, models[id(obj.model)])
            return made[id(obj)]

        def part(p):
            if p.alternatives is not None:
                alts = [part(a) for a in p.alternatives]
                if all(a is b for a, b in zip(alts, p.alternatives)):
                    return p
                g = dataclasses.replace(p, alternatives=alts)
                g.func = _group_func(g)
                return g
            f, m = sub(p.func), sub(p.model)
            return p if f is p.func and m is p.model else dataclasses.replace(p, func=f, model=m)

        cat = copy.copy(self.system.catalog)
        cat.parts = {k: part(p) for k, p in cat.parts.items()}
        cat.rules = {k: part(p) for k, p in cat.rules.items()}
        cat.__dict__.pop("_fp_cache", None)
        system = copy.copy(self.system)
        system.catalog = cat
        system._learning = None
        return system, {q: sub(p) for q, p in self.parts.items()}

    def _promote(self, cand, fp):
        """Carry a gated candidate's state (adaptations, thresholds, conformal sets, memory, adapter state) from the
        shadow parts to the live ones."""
        live = {}
        for q, sp in cand.items():
            mem = sp.correction_memory
            live[q] = {"groups": sp.groups, "memory": mem if mem is not None and mem.part is self.parts[q] else None}
        self._restore(self._snapshot(cand), live)
        if self.fingerprint() != fp:
            raise RuntimeError(f"promoting the candidate did not give its fingerprint (#{fp}, now #{self.fingerprint()}): "
                               "a part changed in a way the loop cannot carry over")

    def _restore(self, state, live=None):
        from .decide import Adaptation
        from .memory import CorrectionMemory
        for q, p in self.parts.items():
            st = state.get(q)
            if st is None:
                continue
            key = p.spec.key
            ad = st.get("adaptation")
            if ad is None:
                p.model.adaptations.pop(key, None)
            else:
                ad = dict(ad)
                ex = [(list(z), tuple(y) if isinstance(y, list) else y) for z, y in ad.pop("examples", [])]
                p.model.adaptations[key] = Adaptation(**ad, examples=ex)
            p.escalate_below, p.act_threshold = st.get("escalate_below"), st.get("act_threshold")
            p.guarantee, p.conformal_set = copy.deepcopy(st.get("guarantee")), copy.deepcopy(st.get("conformal"))
            lv = (live or {}).get(q) or {}
            p.groups = lv.get("groups")
            extra = [n for n in (p.groups["by"].names if p.groups else []) if n not in p.facts]
            p.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                 for f in p.facts + extra])
            if st.get("memory") is None:
                p.correction_memory = None
            elif lv.get("memory") is not None and lv["memory"].to_dict() == st["memory"]:
                p.correction_memory = lv["memory"]
            else:
                p.correction_memory = CorrectionMemory(p).load_dict(st["memory"])
            ad_hook = self.ladder.get("adapter")
            if st.get("adapter") is not None and callable(getattr(ad_hook, "restore", None)):
                ad_hook.restore(p, st["adapter"])

    def history(self):
        """Every record the loop wrote to its changelog, in order (dicts: "action", "version", "promoted", "gates", ...)."""
        return [s.data for s in self.changelog.iter("update")]

    def versions(self):
        """The promoted versions → [{"version", "fp", "time", "action", "id"}] (a rollback is listed as the version it
        restored)."""
        return [{"version": d["version"], "fp": d["fp_after"], "time": d["time"], "action": d["action"], "id": d["id"]}
                for d in self.history() if d.get("version") is not None and d.get("promoted")]

    @property
    def current(self):
        """The version whose state is in force (the latest one with this fingerprint), or None when the live state is not
        a recorded version (before the first run, or after changing the parts by hand)."""
        fp = self.fingerprint()
        hit = [v["version"] for v in self.versions() if v["fp"] == fp]
        return hit[-1] if hit else None

    def version_of(self, part_fp):
        """The versions in which some learned part had this fingerprint (as recorded in a decision's trace)."""
        return sorted({d["version"] for d in self.history() if d.get("promoted") and d.get("version") is not None
                       and part_fp in (d.get("parts") or {}).values()})

    def _record(self, body):
        body = {"v": FORMAT, "kind": "update", "experimental": True, **body}
        return self.changelog._append(json.loads(json.dumps(plain(body), default=str)))

    def _next_version(self):
        vs = [v["version"] for v in self.versions()]
        return (max(vs) + 1) if vs else 0

    def _parts_fp(self):
        return {q: p.fingerprint() for q, p in self.parts.items()}

    def _baseline(self):
        """Record the live state as a version when it is not one yet."""
        if self.current is not None:
            return self.current
        v = self._next_version()
        self._states[v] = (self._snapshot(), self._live())
        self._record({"action": "baseline", "version": v, "promoted": True, "fp_after": self.fingerprint(),
                      "parts": self._parts_fp(), "state": self._states[v][0], "labels": {}})
        return v

    def rollback(self, version):
        """Restore a promoted version's state (adaptations, thresholds, memory) and record the rollback. → the version."""
        rec = next((d for d in reversed(self.history()) if d.get("version") == version and d.get("state") is not None),
                   None)
        if rec is None:
            raise KeyError(f"no promoted version {version} in the changelog")
        state, live = self._states.get(version, (rec["state"], None))
        self._restore(state, live)
        if self.fingerprint() != rec["fp_after"]:
            raise RuntimeError(f"restoring version {version} did not give its fingerprint (#{rec['fp_after']}, now "
                               f"#{self.fingerprint()}): a part changed in a way the loop cannot restore")
        self._record({"action": "rollback", "version": version, "promoted": True, "fp_after": self.fingerprint(),
                      "parts": self._parts_fp(), "state": None, "labels": rec.get("labels") or {}, "to": rec["id"]})
        return version

    def _trained(self, version):
        """The training label ids of a version, per question."""
        for d in reversed(self.history()):
            if d.get("version") == version and d.get("promoted"):
                return {q: set(ids) for q, ids in (d.get("labels") or {}).items()}
        return {}

    # --- one cycle
    def run(self):
        """Collect the labels, propose an update by the ladder, run the gates, promote or undo it, record it. →
        UpdateReport. The system's answers do not change unless the update is promoted."""
        got = self.labels()
        labels = got["labels"]
        by = {q: {"train": [], "calibration": [], "holdout": []} for q in self.parts}
        for lab in labels:
            by[lab.question][lab.split].append(lab)
        cur = self.current
        trained = self._trained(cur) if cur is not None else {}
        plan = {}
        for q, s in by.items():
            ids = {lab.id for lab in s["train"]}
            new = ids - trained.get(q, set())
            if s["train"] and new:
                plan[q] = {"rung": self._rung(len(s["train"])), "train": len(s["train"]), "new": len(new),
                           "calibration": len(s["calibration"]), "holdout": len(s["holdout"])}
        if not plan:
            return UpdateReport("none", current=cur, rejected_labels=got["rejected"], fp_before=self.fingerprint())
        cur = self._baseline()
        fp_before = self.fingerprint()
        holdout = [lab for q in self.parts for lab in by[q]["holdout"]]
        shadow = self._shadow_set()
        before = self._evaluate(holdout, shadow)
        system, cand = self._shadow()               # the candidate is built on copies: the live parts stay as they are
        gates = {}
        gates["consistency"] = self._consistency(plan, by, trained)
        applied = {q: self._apply(q, x["rung"], by[q]["train"], cand[q]) for q, x in plan.items()}
        for q in plan:
            plan[q]["applied"] = applied[q]
        gates["act_guard"] = self._recalibrate(plan, by, cand)
        after = self._evaluate(holdout, shadow, system)
        gates["heldout"] = self._gate_heldout(before, after, len(holdout))
        gates["honesty"] = self._gate_honesty(before, after)
        gates["size"] = self._gate_size(before, after)
        ok = all(g["ok"] for g in gates.values())
        fp_after = self._fingerprint_of(cand)
        train_ids = {q: sorted(lab.id for lab in by[q]["train"]) for q in self.parts if by[q]["train"]}
        if ok:
            back = (self._snapshot(), self._live())
            try:
                self._promote(cand, fp_after)
            except Exception as e:                  # half promoted: put the live parts back, record it, then say so
                self._restore(*back)
                same = self.fingerprint() == fp_before
                gates["promotion"] = {"ok": False, "restored": same, "why": f"{type(e).__name__}: {e}"}
                rec = self._record({"action": "update", "version": None, "promoted": False, "parent": cur,
                                    "fp_before": fp_before, "fp_after": fp_after, "questions": plan, "labels": train_ids,
                                    "gates": gates, "state": None})
                raise RuntimeError(
                    f"{e} — " + (f"the live parts were put back as they were (version {cur}) and the update is recorded "
                                 f"as not promoted (#{rec['id']})" if same else
                                 f"and the live parts could not be put back (now #{self.fingerprint()}, before "
                                 f"#{fp_before}): restore them yourself; recorded as not promoted (#{rec['id']})")
                ) from e
            version = self._next_version()
            self._states[version] = (self._snapshot(), self._live())
            rec = self._record({"action": "update", "version": version, "promoted": True, "parent": cur,
                                "fp_before": fp_before, "fp_after": fp_after, "parts": self._parts_fp(),
                                "questions": plan, "labels": {q: train_ids.get(q, []) for q in self.parts},
                                "gates": gates, "state": self._states[version][0]})
            return UpdateReport("promoted", version, version, plan, gates, fp_before, fp_after, got["rejected"], rec["id"])
        rec = self._record({"action": "update", "version": None, "promoted": False, "parent": cur, "fp_before": fp_before,
                            "fp_after": fp_after, "questions": plan, "labels": train_ids, "gates": gates, "state": None})
        return UpdateReport("rejected", None, cur, plan, gates, fp_before, fp_after, got["rejected"], rec["id"])

    def _rung(self, n):
        if n < self.ladder["fit_below"]:
            return "fit"
        if n < self.ladder["memory_below"] or self.ladder.get("adapter") is None:
            return "memory"
        return "adapter"

    def _examples(self, q, labels):
        return [(self._text(q, lab.init), lab.answer) for lab in labels]

    def _apply(self, q, rung, train, part=None):
        """Apply one question's update to `part` (the shadow part of the candidate; default: the live part) → a JSON-able
        description."""
        from .core import Unknown
        from .memory import CorrectionMemory
        part = self.parts[q] if part is None else part
        ex = self._examples(q, train)
        fit_ex = [(t, y) for t, y in ex if y is not Unknown]
        out = {}
        if fit_ex:
            a = part.fit(fit_ex)
            out["fit"] = {"n": len(fit_ex), "scale": a.scale, "temperature": a.temperature}
        if rung in ("memory", "adapter"):
            mem = CorrectionMemory(part, **self.ladder["memory"])
            for (t, y), lab in zip(ex, train):
                mem.add(t, y, source=lab.source, by=lab.by, time=lab.time, stored_id=lab.id)
            part.memory(mem)
            out["memory"] = {"cases": len(mem), "fp": mem.fingerprint(), "mode": mem.mode}
        if rung == "adapter":
            out["adapter"] = self.ladder["adapter"](part, ex)
        return out

    def _consistency(self, plan, by, trained):
        """New training labels against a memory of the labels already learned (B9)."""
        from .memory import CorrectionMemory
        per, n_new, n_bad = {}, 0, 0
        for q in plan:
            old = [lab for lab in by[q]["train"] if lab.id in trained.get(q, set())]
            new = [lab for lab in by[q]["train"] if lab.id not in trained.get(q, set())]
            n_new += len(new)
            if not old:
                per[q] = {"new": len(new), "conflicts": 0, "checked_against": 0}
                continue
            mem = CorrectionMemory(self.parts[q], **{**self.ladder["memory"], "mode": "check"})
            for (t, y), lab in zip(self._examples(q, old), old):
                mem.add(t, y, source=lab.source, stored_id=lab.id)
            bad = []
            for (t, y), lab in zip(self._examples(q, new), new):
                p = mem.propose(t)
                if p.label is not None and p.label != mem._label(y):
                    bad.append({"id": lab.id, "label": plain(y), "memory": p.label, "strength": round(p.strength, 3)})
            n_bad += len(bad)
            per[q] = {"new": len(new), "conflicts": len(bad), "checked_against": len(old), "examples": bad[:5]}
        share = n_bad / n_new if n_new else 0.0
        mx = self.gates["max_conflict"]
        return {"ok": share <= mx, "share": share, "max": mx, "questions": per,
                "why": f"{n_bad} of {n_new} new label(s) contradicted by earlier corrections ({share:.0%}, max {mx:.0%})"}

    def _recalibrate(self, plan, by, parts=None):
        """Parts calibrated with act_guard / calibrate_for: calibrate again on the calibration labels. Conformal answer
        sets: recalibrated on them too, or dropped (recorded) when there are too few."""
        per, ok = {}, True
        parts = parts or self.parts
        for q in plan:
            part = parts[q]
            cal = [(t, y) for t, y in self._examples(q, by[q]["calibration"])]
            fine, per[q] = self._recalibrate_guard(part, cal)
            ok = ok and fine
            if part.conformal_set is not None:
                per[q]["conformal"] = self._recalibrate_conformal(part, cal)
        def said(x):                                 # a failure's message (the recalibrated error rate is a number)
            return x["error"] if isinstance(x.get("error"), str) else None
        why = "; ".join(f"{q}: " + (said(x) or x.get("skipped") or f"threshold {x['threshold']:.3f} on {x['n']}")
                        + ("" if "conformal" not in x else "; conformal sets " + ("dropped" if x["conformal"].get("dropped")
                                                                                  else "recalibrated"))
                        for q, x in per.items())
        return {"ok": ok, "questions": per, "why": why}

    def _recalibrate_guard(self, part, cal):
        """→ (ok, the record)."""
        g = part.guarantee
        if g is None:
            return True, {"skipped": "no act_guard / calibrate_for on this part"}
        if part.groups is not None:
            return False, {"error": "per-group thresholds are not recalibrated by the loop yet"}
        if len(cal) < self.gates["min_calibration"]:
            return False, {"error": f"{len(cal)} calibration label(s) < {self.gates['min_calibration']}: the old "
                                    "threshold no longer holds for the changed signal"}
        sig = g.get("signal", "auto")
        if g.get("method") == "crc":
            r = part.act_guard(cal, risk=g["risk"], signal=sig)
        else:
            r = part.calibrate_for(cal, error=g["error"], signal=sig, method=g["method"], delta=g.get("delta", 0.10))
        return True, {k: r.get(k) for k in ("signal", "threshold", "answered", "error", "risk", "n", "guarantee")}

    def _recalibrate_conformal(self, part, cal):
        """The conformal set was calibrated on the old probabilities: recalibrate it at the same coverage on the
        calibration labels, or drop it (a set that is claimed must hold for the changed signal)."""
        cov = part.conformal_set["coverage"]
        if len(cal) < self.gates["min_calibration"]:
            part.conformal_set = None
            return {"dropped": True, "coverage": cov,
                    "why": f"{len(cal)} calibration label(s) < {self.gates['min_calibration']}"}
        try:
            r = part.conformal(cal, coverage=cov)
        except ValueError as e:
            part.conformal_set = None
            return {"dropped": True, "coverage": cov, "why": str(e)}
        return {"dropped": False, **r}

    # --- evaluation
    def _shadow_set(self):
        """The stored decisions for the size gate → [(stored, {learned questions its input is held out for})]: the split is
        the labels' (content_key(question, input)), so an input trained on for a question is never compared on it."""
        rows = []
        for s in self.storage.iter():
            init = ((s.data.get("response") or {}).get("trace") or {}).get("init") or {}
            held = {q for q in self.parts if split_of(content_key(q, init), self.holdout, self.calibration) == "holdout"}
            if held:
                rows.append((s, held))
        return rows[-self.gates["shadow_limit"]:] if self.gates["shadow_limit"] else []

    def _honesty_cases(self):
        h = self.gates.get("honesty")
        if h is None:
            return None
        if isinstance(h, (list, tuple)):
            return list(h)
        from .honesty import load_set
        return load_set(h)["cases"]

    def _evaluate(self, holdout, shadow, system=None):
        """The held-out labels, the honesty set and the shadow rows asked through `system` (the candidate's shadow; default:
        the live system)."""
        from .honesty import run
        system = self.system if system is None else system
        cases = [{"name": lab.id, "state": dict(lab.init), "gold": {lab.question: plain(lab.answer)}} for lab in holdout]
        out = {"heldout": run(system, cases, store=False) if cases else []}
        hc = self._honesty_cases()
        out["honesty_set"] = None if hc is None else run(system, hc, store=False)
        resp = {}
        for s, held in shadow:
            try:
                old = s.response(self.system)
                names = [q for q in old.results if q in self.system.questions and (q in held or q not in self.parts)]
                if not names:
                    continue
                resp[s.id] = system.ask(dict(old.trace.init), names, store=False)
            except Exception as e:  # noqa: BLE001
                resp[s.id] = e
        out["shadow"] = resp
        return out

    def _gate_heldout(self, before, after, n):
        from .honesty import metrics
        a, b = metrics(before["heldout"], self.gates["risk"]), metrics(after["heldout"], self.gates["risk"])
        if n < self.gates["min_holdout"]:
            return {"ok": False, "n": n, "why": f"{n} held-out label(s) < {self.gates['min_holdout']}: an improvement "
                                                  "cannot be shown yet"}
        score = lambda m: (m["correct"] - m["confident_errors"]) / m["n"]    # noqa: E731
        gain = score(b) - score(a)
        return {"ok": gain >= self.gates["min_gain"], "n": n, "gain": gain, "min_gain": self.gates["min_gain"],
                "before": {"correct": a["correct"], "wrong": a["confident_errors"], "escalated": a["escalated"]},
                "after": {"correct": b["correct"], "wrong": b["confident_errors"], "escalated": b["escalated"]},
                "why": f"right − wrong answered alone on {n} held-out label(s): {score(a):+.3f} → {score(b):+.3f} "
                       f"(gain {gain:+.3f}, needed {self.gates['min_gain']:+.3f})"}

    def _gate_honesty(self, before, after):
        from .honesty import compare, metrics
        tol, risk = self.gates["tolerance"], self.gates["risk"]
        out, regs = {}, []
        for name in ("heldout", "honesty_set"):
            if before[name] is None or not before[name]:
                continue
            a, b = metrics(before[name], risk), metrics(after[name], risk)
            r = compare(b, a, tol)
            out[name] = {"before": {k: a[k] for k in ("confident_error_rate", "coverage_at_risk", "quote_support_proxy")},
                         "after": {k: b[k] for k in ("confident_error_rate", "coverage_at_risk", "quote_support_proxy")},
                         "regressions": r}
            regs += [f"{name}: {x}" for x in r]
        return {"ok": not regs, "tolerance": tol, **out,
                "why": "no honesty number got worse" if not regs else "; ".join(regs)}

    def _gate_size(self, before, after):
        from .diff import DiffReport, compare
        rep = DiffReport()
        for sid, old in before["shadow"].items():
            new = after["shadow"].get(sid)
            rep.total += 1
            if isinstance(old, Exception) or isinstance(new, Exception):
                rep.errors.append({"id": sid, "seq": None, "error": repr(old if isinstance(old, Exception) else new)})
                continue
            ch = compare(old, new, None)
            if ch:
                rep.changed.append({"id": sid, "seq": None, "time": None, "questions": ch})
        share = len(rep.changed) / rep.total if rep.total else 0.0
        mx = self.gates["max_change"]
        return {"ok": share <= mx and not rep.errors, "total": rep.total, "changed": len(rep.changed), "share": share,
                "max": mx, "errors": len(rep.errors), "by_question": rep.by_question(),
                "why": f"{len(rep.changed)} of {rep.total} stored decision(s) change in shadow ({share:.0%}, max {mx:.0%})"
                       + (f"; {len(rep.errors)} could not be re-run" if rep.errors else "")}


def _shadow_part(p, model):
    """A copy of a decision part on `model` (a copy of its model): its own thresholds, guarantee and conformal set; the
    correction memory is shared until the candidate replaces it (it is only read)."""
    sp = copy.copy(p)
    sp.model = model
    sp.__solvi_model__ = sp.__solvi_decision__ = sp
    sp.guarantee, sp.conformal_set = copy.deepcopy(p.guarantee), copy.deepcopy(p.conformal_set)
    return sp


__all__ = ["ExperimentalWarning", "Label", "Learning", "TRUSTED_SOURCES", "UpdateReport", "split_of"]
