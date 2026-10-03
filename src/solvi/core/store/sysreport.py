"""The system report: what a System did over a stored period, for its owner — read from the store alone (no model, no
catalog, nothing re-run).

    rep = system.report(since="2026-09-01", until="2026-10-01")       # the System's own storage
    rep = system_report(SQLiteStorage("decisions.db"))                 # any store, offline
    print(rep)                                                         # plain text
    rep.to_dict()                                                      # the same as data

    solvi report decisions.db --overview [--since ISO] [--until ISO] [--json]

What it shows, per question, from the records the library already writes:

- who answered: the decisions answered alone, by what (a rule, a model decision with the model's id, a fitted head, a
  hard check that forced the answer) and how many were handed over (abstained), by the safeguard that held them back;
  with `solvi.core.dispatch`, who gave the final answer (System 1, the slow path, a person), by which action and slice;
- what it cost: the time of every decision, the model calls and tokens the traces record (dollars where the
  dispatcher or a generator built with `price=` recorded them, or with `price=`), the dispatcher's spend split between
  System 1 and the slow path, and the refinement loops (solvi.core.slow.refine): how many, how they ended (accepted, escalated,
  stopped by their budget, over it), their rounds and what their proposals and rounds cost;
- the promise in force against what the labels show: every guarantee the decisions were gated by (its method, level
  and text) and every calibrated dispatch policy (who answers each slice), next to the error found on the decisions
  that have a label — the corrections in the store (`System.teach`, `save_correction`). Labels from a person, an outcome
  or a rule are measured; "verified" labels (System 2's own vouched answers) are counted but not measured, since they
  are the system's own answers. Labels that are not a random sample (people correct what looks wrong) overstate the
  error: the report says how many decisions are labelled;
- drift: the flags the decisions recorded (an open-set gate's change point, the dispatcher's drift flag) and a
  `DriftMonitor` run over the period's decisions in order (`drift_window=`; None: not run), with the first decision at
  which it flagged and why.

A correction labels the decision it names (`of=`), else the latest decision before it on the same input. Corrections
recorded after the period still label its decisions."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from . import VERIFIED, _cj, _when, akey, view

KIND = {"computed": "rule", "decided": "model", "learned": "learned head", "proposed": "model proposal",
        "quoted": "quote", "given": "given"}
GUARD = {"low_confidence": "below the guarantee or threshold", "hard_check": "a hard check", "missing_facts":
         "facts missing", "rule_abstained": "the rule abstained", "outside_options": "outside the options"}


def _iso(t):
    return datetime.datetime.fromtimestamp(t).isoformat(sep=" ", timespec="seconds") if t is not None else None


def _usages(records):
    from ..costs import _usages as one
    out = []
    for r in records or ():
        out += one(r.get("extra") if isinstance(r, dict) else None)
    return out


def _cost(calls, ms, price):
    from ..costs import price_of
    return {"decisions": 0, "ms": ms, "calls": len(calls), "input_tokens": sum(int(u.get("input_tokens", 0)) for _, u in calls),
            "output_tokens": sum(int(u.get("output_tokens", 0)) for _, u in calls),
            "usd": price_of(price, calls, recorded=True)}


def _add_cost(a, b):
    out = dict(a)
    for k in ("decisions", "ms", "calls", "input_tokens", "output_tokens"):
        out[k] = a.get(k, 0) + b.get(k, 0)
    out["usd"] = None if a.get("usd") is None or b.get("usd") is None else a["usd"] + b["usd"]
    return out


def _p_above(wrong, n, level):
    """P(at least `wrong` wrong of `n` if the true error were `level`): small → the labels say the promise is broken."""
    from ..calibration import _binom_cdf
    if n == 0 or wrong == 0:
        return 1.0
    return max(0.0, 1.0 - _binom_cdf(wrong - 1, n, level))


def _verdict(measured, level, wrong, n):
    if n == 0 or measured is None:
        return "no labelled decisions"
    if measured <= level:
        return "within the promise"
    p = _p_above(wrong, n, level)
    return f"above the promise (p = {p:.2g} if the true rate were {level:g})" if p < 0.05 else \
        f"above the promised level on these labels, not significantly (p = {p:.2g})"


def _labels_block(rows, level=None, measure="error"):
    """rows: [(alone: bool, wrong: bool | None)] of labelled decisions → counts and, against a promise, the verdict."""
    n = len(rows)
    alone = [w for a, w in rows if a]
    wrong = sum(bool(w) for w in alone)
    out = {"labelled": n, "alone": len(alone), "wrong_alone": wrong,
           "error": wrong / len(alone) if alone else None, "risk": wrong / n if n else None}
    if level is not None:
        m = out[measure]
        out.update(measure=measure, level=level, verdict=_verdict(m, level, wrong, len(alone) if measure == "error" else n))
        out["p_value"] = _p_above(wrong, len(alone) if measure == "error" else n, level) if n else None
    return out


@dataclass
class SystemReport:
    """What a System did over a period (see the module docs). period: first and last decision, the filters; system1: per
    question who answered, the promises, the labels, drift; dispatch: per dispatched question the same for the final
    answers; cost: System 1's decisions and the dispatcher's spend; corrections: the labels in the store."""
    period: dict
    system1: dict
    dispatch: dict
    cost: dict
    corrections: dict
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {"kind": "system", "period": self.period, "system1": self.system1, "dispatch": self.dispatch,
                "cost": self.cost, "corrections": self.corrections, "notes": list(self.notes)}

    def __str__(self):
        return render(self.to_dict())


# ------------------------------------------------------------------------------------------------ building
def system_report(store, since=None, until=None, *, question=None, price=None, drift_window=100, monitor=None):
    """The report of the stored period [since, until) (seconds, a datetime, a date or an ISO string) → SystemReport.
    question: only this question. price: dollars per million (input, output) tokens, or a function (model, usage) →
    dollars, for the model calls System 1's traces record (the dispatcher records its own dollars). drift_window: the
    window of the DriftMonitor run over each question's decisions (None: not run); monitor: a function () → a fresh
    DriftMonitor to run instead."""
    from ..schema import untag_floats
    t0, t1 = _when(since), _when(until)

    def within(t):
        return (t0 is None or t >= t0) and (t1 is None or t < t1)
    asks, disp, policies, teach, erased, loops = [], [], {}, [], 0, []
    for s in store.iter(None, redacted=True):
        d = s.data
        if s.kind == "teach":
            teach.append(s)
        elif s.kind == "policy":
            policies[d.get("config")] = d
        elif not within(s.time):
            continue
        elif d.get("redacted") and s.kind == "ask":
            erased += 1
        elif s.kind == "ask":
            asks.append(s)
        elif s.kind == "dispatch" and (question is None or d.get("question") == question):
            disp.append(s)
        elif s.kind == "refine" and (question is None or d.get("question") == question):
            loops.append(s)
    # the labels: a correction names its decision (of=), else the latest decision before it on the same input
    by_init, ids = {}, {}
    for s in asks:
        ids[s.id] = s
        init = ((view(s.data) or {}).get("trace") or {}).get("init")
        by_init.setdefault(_cj(untag_floats(init)), []).append(s)
    for s in disp:
        ids[s.id] = s
        init = (((s.data.get("s1") or {}).get("trace") or {}).get("init"))
        by_init.setdefault(_cj(untag_floats(init)), []).append(s)
    labels, sources, unmatched, in_period = {}, {}, 0, 0
    for c in teach:
        d = c.data
        src = d.get("source", "human")
        sources[src] = sources.get(src, 0) + 1
        in_period += within(c.time)
        q = d.get("teach")
        target = ids.get(d.get("of")) if d.get("of") is not None else None
        if target is None and d.get("of") is None:
            cands = [x for x in by_init.get(_cj(untag_floats(d.get("init"))), ()) if x.seq < c.seq]
            target = cands[-1] if cands else None
        if target is None:
            unmatched += 1
            continue
        key = (target.id, q)
        if src == VERIFIED and key in labels and labels[key][0] != VERIFIED:
            continue                              # a label from outside the system stays over a vouched System 2 answer
        labels[key] = (src, akey(store._answer_key(q, untag_floats(d.get("answer")))))
    s1, cost1 = _system1(asks, labels, question, price, drift_window, monitor)
    dp, cost2 = _dispatch(disp, labels, policies)
    times = [s.time for s in asks + disp]
    catalogs, models, experimental = {}, {}, {}
    for s in asks:
        for name in ((s.data.get("meta") or {}) if isinstance(s.data.get("meta"), dict) else {}).get("experimental") or ():
            experimental[name] = experimental.get(name, 0) + 1
        c = s.data.get("catalog") or "not recorded"
        e = catalogs.setdefault(c, {"decisions": 0, "first": _iso(s.time)})
        e["decisions"] += 1
        e["last"] = _iso(s.time)
        for m in s.data.get("models") or ():
            k = f"{m.get('type')} {m.get('id')} #{str(m.get('fp'))[:8]}"
            models[k] = models.get(k, 0) + 1
    notes = []
    if labels and len(labels) < len(asks) + len(disp):
        notes.append("labels cover part of the decisions: unless they were drawn at random, the measured error is "
                     "not the error of the whole stream (corrections are usually made where the answer looked wrong)")
    if asks and disp:
        notes.append("the store holds System 1's own decisions and the dispatcher's: System 1's section counts every "
                     "System 1 answer, the dispatcher's section the final answers")
    return SystemReport(
        period={"since": _iso(min(times)) if times else None, "until": _iso(max(times)) if times else None,
                "filters": {k: str(v) for k, v in (("since", since), ("until", until), ("question", question)) if v is not None},
                "decisions": len(asks), "dispatched": len(disp), "erased": erased, "catalogs": catalogs,
                "models": models, **({"experimental": experimental} if experimental else {})},
        system1=s1, dispatch=dp, cost={"system1": cost1, "dispatch": cost2, **({"refine": _refine(loops, price)} if loops
                                                                              else {})},
        corrections={"total": len(teach), "in_period": in_period, "by_source": sources,
                     "labelled_decisions": len(labels), "unmatched": unmatched},
        notes=notes)


def _who(res, q, recs):
    """Who gave an answer, in words: a rule (its name), a model (its id), a learned head, a hard check."""
    if res.get("status") == "forced":
        return f"hard check: {res.get('source') or '?'}"
    kind = KIND.get(res.get("provenance"), res.get("provenance") or "unknown")
    rec = next((r for r in recs if r.get("name") == f"answer:{q}"), None)
    m = (rec or {}).get("model") or {}
    return f"{kind}: {m.get('id') or res.get('source') or '?'}"


def _system1(asks, labels, question, price, drift_window, monitor):
    per_q, cost = {}, _cost([], 0.0, price)
    obs = {}
    for s in asks:
        d = s.data
        resp = view(d) or {}                          # a compact record: its kept steps hold every model call
        recs = (resp.get("trace") or {}).get("records") or []
        c = _cost(_usages(recs), float(resp.get("ms") or 0.0), price)
        c["decisions"] = 1
        cost = _add_cost(cost, c)
        results = resp.get("results") or {}
        for q, (ans, conf, status) in (d.get("answers") or {}).items():
            if question is not None and q != question:
                continue
            p = per_q.setdefault(q, {"asked": 0, "alone": 0, "forced": 0, "abstained": 0, "answered_by": {},
                                     "abstained_by": {}, "promises": {}, "unguarded_alone": 0, "_labels": [],
                                     "recorded_drift": []})
            p["asked"] += 1
            r = results.get(q) or {}
            if status == "abstain":
                p["abstained"] += 1
                g = (d.get("guards") or {}).get(q) or r.get("guard") or "abstained"
                p["abstained_by"][g] = p["abstained_by"].get(g, 0) + 1
            else:
                p["forced" if status == "forced" else "alone"] += 1
                w = _who(r, q, recs)
                p["answered_by"][w] = p["answered_by"].get(w, 0) + 1
            g = (r.get("extra") or {}).get("guarantee")
            fp = None
            if isinstance(g, dict) and g.get("promise"):
                fp = g.get("fingerprint") or _cj([g.get("method"), g.get("promise")])
                pr = p["promises"].get(fp)
                if pr is None:
                    measure = "risk" if "risk" in g else "error"
                    pr = p["promises"][fp] = {"fingerprint": fp, "method": g.get("method"), "measure": measure,
                                              "level": g.get(measure), "delta": g.get("delta"), "promise": g.get("promise"),
                                              "calibrated_on": g.get("n"), "decisions": 0, "alone": 0, "held_back": 0,
                                              "_labels": []}
                pr["decisions"] += 1
                pr["alone" if g.get("answered") else "held_back"] += 1
                st = g.get("state") or {}
                if st.get("flag_at") is not None and not any(x.get("flag_at") == st["flag_at"] for x in p["recorded_drift"]):
                    p["recorded_drift"].append({"by": f"guarantee {g.get('method')}", "flag_at": st["flag_at"],
                                                "change_at": st.get("change_at"), "why": st.get("why"),
                                                "first_seen": {"id": s.id, "time": _iso(s.time)}})
            elif status == "ok":
                p["unguarded_alone"] += 1
            lab = labels.get((s.id, q))
            if lab is not None and lab[0] != VERIFIED:
                row = (status != "abstain", None if status == "abstain" else _cj(ans) != lab[1])
                p["_labels"].append((lab[0], row))
                if fp is not None:
                    p["promises"][fp]["_labels"].append(row)
            elif lab is not None:
                p.setdefault("verified_labels", 0)
                p["verified_labels"] += 1
            act = next(((r2.get("extra") or {}).get("act") for r2 in recs if r2.get("name") == f"answer:{q}"), None)
            obs.setdefault(q, []).append((s, {"answer": _cj(ans), "confidence": float(conf if conf is not None else 0.0),
                                              "alone": status == "ok", "act": act},
                                          None if lab is None or lab[0] == VERIFIED else lab[1]))
    for q, p in per_q.items():
        got = p.pop("_labels")
        by_src = {}
        for src, _ in got:
            by_src[src] = by_src.get(src, 0) + 1
        p["labels"] = _labels_block([r for _, r in got])
        p["labels"]["verified_not_measured"] = p.pop("verified_labels", 0)
        p["labels"]["by_source"] = by_src
        p["promises"] = list(p["promises"].values())
        for pr in p["promises"]:
            pr["labels"] = _labels_block(pr.pop("_labels"), pr["level"], pr["measure"]) if pr["level"] is not None \
                else _labels_block(pr.pop("_labels"))
        p["alone_share"] = p["alone"] / p["asked"] if p["asked"] else 0.0
        p["drift"] = _drift(obs.get(q, []), drift_window, monitor)
    return per_q, cost


def _drift(rows, window, monitor):
    """A DriftMonitor over the decisions in order → the first flag (decision number in the period, id, time, why)."""
    if monitor is None and window is None:
        return {"tested": False, "why": "not run (drift_window=None)"}
    from ..guarantees.drift import DriftMonitor
    mon = monitor() if monitor is not None else DriftMonitor(window=int(window))
    need = mon.window + mon.min_n
    if len(rows) < need:
        return {"tested": False, "why": f"{len(rows)} decisions, fewer than the {need} a window of {mon.window} needs "
                                        f"(the reference, then at least {mon.min_n} to compare)"}
    import warnings
    out = {"tested": True, "window": mon.window, "alpha": mon.alpha, "decisions": len(rows), "flagged": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (s, o, lab) in enumerate(rows, start=1):
            rep = mon.observe(o, label=lab)
            if rep.get("drift"):
                out.update(flagged=True, at=i, id=s.id, time=_iso(s.time), signals=list(rep["flags"]), why=list(rep["why"]))
                break
    return out


def _refine(loops, price):
    """The refinement loops of the period (records of kind "refine"): how they ended, their rounds, the proposals' model
    calls (outside the rounds' traces, which System 1's cost counts) and the whole loops' cost as recorded."""
    from ..costs import _usages
    out = {"loops": len(loops), "accepted": 0, "escalated": 0, "stopped_by_budget": 0, "over_budget": 0, "rounds": 0}
    calls, total = [], _cost([], 0.0, None)
    total["usd"] = 0.0
    for s in loops:
        d = s.data
        out["accepted" if d.get("accepted") else "escalated"] += 1
        out["stopped_by_budget"] += bool(d.get("stopped"))
        out["over_budget"] += bool(d.get("over_budget"))
        out["rounds"] += len(d.get("rounds") or ())
        for r in d.get("rounds") or ():
            g = r.get("generated")
            calls += _usages({"generated": g}) if g is not None else []
        c = d.get("cost") or {}
        total = _add_cost(total, {"decisions": 1, "ms": float(c.get("ms", 0.0)), "calls": int(c.get("calls", 0)),
                                  "input_tokens": int(c.get("input_tokens", 0)),
                                  "output_tokens": int(c.get("output_tokens", 0)), "usd": c.get("usd")})
    prop = _cost(calls, 0.0, price)
    prop["decisions"] = len(loops)
    out["proposals"], out["total"] = prop, total
    return out


def _dispatch(disp, labels, policies):
    per_q, total = {}, {"system1": _cost([], 0.0, None), "slow_path": _cost([], 0.0, None)}
    total["system1"]["usd"] = total["slow_path"]["usd"] = 0.0
    for s in disp:
        d = s.data
        q = d.get("question")
        p = per_q.setdefault(q, {"decisions": 0, "by": {"s1": 0, "s2": 0, "human": 0}, "actions": {}, "slices": {},
                                 "over_budget": 0, "disagreements": 0, "drift_flagged": 0, "first_drift": None,
                                 "policies": {}, "uncalibrated": 0, "_labels": {"all": [], "s1": [], "s2": []},
                                 "_slices": {}})
        p["decisions"] += 1
        p["by"][d.get("by")] = p["by"].get(d.get("by"), 0) + 1
        p["actions"][d.get("action")] = p["actions"].get(d.get("action"), 0) + 1
        if d.get("slice"):
            p["slices"][d["slice"]] = p["slices"].get(d["slice"], 0) + 1
        p["over_budget"] += bool(d.get("over_budget"))
        p["disagreements"] += d.get("disagreement") is not None
        if d.get("drift"):
            p["drift_flagged"] += 1
            if p["first_drift"] is None:
                p["first_drift"] = {"id": s.id, "time": _iso(s.time), "n": d.get("n"), "why": d.get("drift")}
        for k, name in (("s1", "system1"), ("s2", "slow_path")):
            c = (d.get("cost") or {}).get(k) or {}
            total[name] = _add_cost(total[name], {"decisions": 1 if k == "s2" and d.get("s2") else int(k == "s1"),
                                                  "ms": float(c.get("ms", 0.0)), "calls": int(c.get("calls", 0)),
                                                  "input_tokens": int(c.get("input_tokens", 0)),
                                                  "output_tokens": int(c.get("output_tokens", 0)), "usd": c.get("usd")})
        pol = policies.get(d.get("config"))
        if pol is not None:
            pk = pol.get("policy", {}).get("fingerprint")
            if pk not in p["policies"]:
                pp = pol["policy"]
                measure = "risk" if pp.get("method") == "crc" else "error"
                p["policies"][pk] = {"fingerprint": pk, "method": pp.get("method"), "measure": measure,
                                     "level": pp.get("level"), "delta": pp.get("delta"), "promise": pp.get("promise"),
                                     "calibrated_on": pp.get("n"), "decisions": 0,
                                     "slices": {k: {x: v.get(x) for x in ("answer", "threshold", "n", "accuracy", "answered",
                                                                           "wrong", "why") if v.get(x) is not None}
                                                for k, v in (pp.get("slices") or {}).items()}, "_labels": []}
            p["policies"][pk]["decisions"] += 1
        else:
            p["uncalibrated"] += 1
        lab = labels.get((s.id, q))
        if lab is not None and lab[0] != VERIFIED:
            alone = d.get("by") in ("s1", "s2")
            row = (alone, (_cj(d.get("answer")) != lab[1]) if alone else None)
            p["_labels"]["all"].append(row)
            if alone:
                p["_labels"][d["by"]].append(row)
            p["_slices"].setdefault(d.get("slice") or "none (System 1 alone)", []).append(row)
            if pol is not None:
                p["policies"][pol["policy"].get("fingerprint")]["_labels"].append(row)
    for p in per_q.values():
        L = p.pop("_labels")
        p["labels"] = {"all": _labels_block(L["all"]), "by_s1": _labels_block(L["s1"]), "by_s2": _labels_block(L["s2"]),
                       "by_slice": {k: _labels_block(v) for k, v in p.pop("_slices").items()}}
        p["policies"] = list(p["policies"].values())
        for pr in p["policies"]:
            pr["labels"] = _labels_block(pr.pop("_labels"), pr["level"], pr["measure"])
    return per_q, total


# ------------------------------------------------------------------------------------------------ text
def _pct(a, b):
    return f"{a / b:.1%}" if b else "—"


def _money(c):
    usd = c.get("usd")
    return "dollars not recorded" if usd is None else f"${usd:.4f}"


def _cost_line(c, what="decisions"):
    ms = c.get("ms", 0.0)
    n = c.get("decisions", 0)
    s = f"{n} {what}, {ms / 1000:.2f} s in total" + (f" ({ms / n:.1f} ms mean)" if n else "")
    if c.get("calls"):
        s += (f"; {c['calls']} model call(s), {c['input_tokens']:,} input + {c['output_tokens']:,} output tokens, "
              + _money(c))
    else:
        s += "; no model calls recorded"
    return s


def _labels_line(L):
    if not L["labelled"]:
        return "no labelled decisions"
    s = f"{L['labelled']} labelled; answered alone and labelled {L['alone']}, wrong {L['wrong_alone']}"
    if L["error"] is not None:
        s += f" — error among them {L['error']:.2%}"
    if L["risk"] is not None:
        s += f", alone and wrong {L['risk']:.2%} of the labelled"
    return s


def render(d):
    """A system report's data → plain text."""
    P = d["period"]
    L = ["System report", "============="]
    head = f"{P['decisions']} System 1 decision(s)" + (f", {P['dispatched']} dispatched decision(s)" if P["dispatched"] else "")
    if P["since"]:
        head += f", {P['since']} to {P['until']}"
    if P["erased"]:
        head += f" ({P['erased']} erased and not counted)"
    if P["filters"]:
        head += " — " + ", ".join(f"{k} {v}" for k, v in P["filters"].items())
    L.append(head + ".")
    if len(P["catalogs"]) > 1:
        L.append("The catalog changed over the period: " + "; ".join(
            f"{fp} — {e['decisions']} decision(s), {e['first']} to {e['last']}" for fp, e in P["catalogs"].items()) + ".")
    if P["models"]:
        L.append("Models that ran: " + "; ".join(f"{k} ({n} decisions)" for k, n in P["models"].items()) + ".")
    if P.get("experimental"):
        L.append("Made with experimental pieces (solvi.experimental: the API may change): " + "; ".join(
            f"{k} ({n} decisions)" for k, n in sorted(P["experimental"].items())) + ".")
    L.append("")
    C = d["corrections"]
    L.append(f"Labels: {C['total']} correction(s) in the store"
             + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(C["by_source"].items())) + ")" if C["by_source"] else "")
             + f"; {C['labelled_decisions']} label a decision of this period"
             + (f"; {C['unmatched']} name no decision of it" if C["unmatched"] else "") + ".")
    for q, p in d["system1"].items():
        L += ["", f"Question {q} — System 1, asked {p['asked']} time(s)", "-" * 40]
        L.append(f"  answered alone      {p['alone']} ({_pct(p['alone'], p['asked'])})")
        if p["forced"]:
            L.append(f"  forced by a check   {p['forced']} ({_pct(p['forced'], p['asked'])})")
        if p["answered_by"]:
            L.append("  who answered        " + "; ".join(f"{w} {n}" for w, n in
                                                         sorted(p["answered_by"].items(), key=lambda t: -t[1])))
        L.append(f"  handed over         {p['abstained']} ({_pct(p['abstained'], p['asked'])})"
                 + (" — " + ", ".join(f"{GUARD.get(g, g)} {n}" for g, n in sorted(p["abstained_by"].items(), key=lambda t: -t[1]))
                    if p["abstained_by"] else ""))
        if p["promises"]:
            for pr in p["promises"]:
                L.append(f"  promise ({pr['method']}, {pr['decisions']} decisions, {pr['alone']} let through): {pr['promise']}")
                lb = pr["labels"]
                L.append("    labels: " + _labels_line(lb)
                         + (f" — promised {lb['measure']} ≤ {lb['level']:g}: {lb['verdict']}" if lb["labelled"] and "level" in lb else ""))
        else:
            L.append("  promise: none — no guarantee on this question (System.guarantee)")
        if p["unguarded_alone"] and p["promises"]:
            L.append(f"  answered alone without a guarantee: {p['unguarded_alone']}")
        lb = p["labels"]
        L.append("  labels (all): " + _labels_line(lb)
                 + (f"; {lb['verified_not_measured']} verified label(s) not measured" if lb["verified_not_measured"] else ""))
        for x in p["recorded_drift"]:
            L.append(f"  drift recorded by the {x['by']}: flagged at its decision {x['flag_at']} (change from "
                     f"{x.get('change_at')}): {x['why']}")
        dr = p["drift"]
        if dr["tested"]:
            L.append(f"  drift (DriftMonitor, window {dr['window']}, over {dr['decisions']} decisions): "
                     + (f"flagged at decision {dr['at']} of the period ({dr['id']}, {dr['time']}): {'; '.join(dr['why'])}"
                        if dr["flagged"] else "no flag"))
        else:
            L.append(f"  drift (DriftMonitor): not tested — {dr['why']}")
    for q, p in d["dispatch"].items():
        L += ["", f"Question {q} — dispatcher, {p['decisions']} decision(s)", "-" * 40]
        b = p["by"]
        L.append(f"  final answer by     System 1 {b.get('s1', 0)} ({_pct(b.get('s1', 0), p['decisions'])}), slow path "
                 f"{b.get('s2', 0)} ({_pct(b.get('s2', 0), p['decisions'])}), a person {b.get('human', 0)} "
                 f"({_pct(b.get('human', 0), p['decisions'])})")
        L.append("  actions             " + ", ".join(f"{k} {v}" for k, v in p["actions"].items())
                 + (" — slices handed over: " + ", ".join(f"{k} {v}" for k, v in p["slices"].items()) if p["slices"] else ""))
        if p["disagreements"] or p["over_budget"]:
            L.append(f"  checks that disagreed {p['disagreements']}; slow-path runs over the budget per decision "
                     f"{p['over_budget']}")
        for pr in p["policies"]:
            L.append(f"  promise ({pr['method']}, calibrated on {pr['calibrated_on']}, {pr['decisions']} decisions): "
                     f"{pr['promise']}")
            for k, v in pr["slices"].items():
                thr = v.get("threshold")
                L.append(f"    slice {k}: {v.get('answer')}" + (f" at ≥ {thr:.4g}" if isinstance(thr, (int, float)) else "")
                         + f" (calibrated on {v.get('n')})")
            lb = pr["labels"]
            L.append("    labels: " + _labels_line(lb)
                     + (f" — promised {lb['measure']} ≤ {lb['level']:g}: {lb['verdict']}" if lb["labelled"] else ""))
        if p["uncalibrated"]:
            L.append(f"  {p['uncalibrated']} decision(s) under no calibrated policy (Dispatcher.calibrate): the answers "
                     "given alone carry System 1's guarantee and the slow path's own checks")
        lab = p["labels"]
        L.append("  labels (final answers): " + _labels_line(lab["all"]))
        for who, key in (("System 1", "by_s1"), ("slow path", "by_s2")):
            if lab[key]["labelled"]:
                L.append(f"    by {who}: " + _labels_line(lab[key]))
        if p["first_drift"]:
            f = p["first_drift"]
            L.append(f"  drift flag up on {p['drift_flagged']} decision(s), first at decision {f['n']} ({f['id']}): {f['why']}")
    c = d["cost"]
    L += ["", "Cost", "----"]
    if c["system1"]["decisions"] or not d["dispatch"]:
        L.append("  System 1: " + _cost_line(c["system1"]))
    if d["dispatch"]:
        L.append("  dispatcher, System 1: " + _cost_line(c["dispatch"]["system1"]))
        L.append("  dispatcher, slow path: " + _cost_line(c["dispatch"]["slow_path"], "runs"))
    if c.get("refine"):
        r = c["refine"]
        L.append(f"  refinement loops: {r['loops']} ({r['accepted']} accepted, {r['escalated']} escalated, "
                 f"{r['stopped_by_budget']} stopped by their budget, {r['over_budget']} over it), {r['rounds']} rounds")
        L.append("    whole loops: " + _cost_line(r["total"], "loops"))
        p = r["proposals"]
        L.append(f"    of which the proposals: {p['calls']} model call(s), {p['input_tokens']:,} input + "
                 f"{p['output_tokens']:,} output tokens, " + _money(p))
    for n in d["notes"]:
        L += ["", "Note: " + n]
    return "\n".join(L) + "\n"


__all__ = ["SystemReport", "render", "system_report"]
