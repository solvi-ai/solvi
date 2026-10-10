"""The change pipeline (solvi.experimental.changes) on a credit policy: two document versions, 20 corrections by a
person, one rule that must be refused, one threshold change that washes out the hand-overs — every one a candidate through the
same stages and one hash-chained journal.

    benchmarks/tasks/fetch.sh credit && python benchmarks/tasks/prepare.py      # the German Credit applications, once
    python benchmarks/changes_credit.py                                          # about a minute or two; no LLM

The policy is benchmarks/tasks/credit/POLICY_v1_v2.md (version 1, version 2 and a rule that must be rejected); the
catalogs are written by hand here, each rule citing its clause (`clause=` / `cite=`, the ids from Spec(text)). The
corrections are benchmarks/tasks/credit/corrections.json: 20 applications a person says were decided wrongly, each with
an exception rule (a hard check on equalities of declared fields) that is either exact (it changes only that
application) or overreaching (it changes others too), narrowing or widening. The person's rulings on held changes are
simulated: every held change is approved.

The stages: KnownFailures, Static (sensitive fields of about.json; every citation found in the policy or the
correction it quotes, inside its clause), Consistency (a correction that makes any matched decision more permissive
widens: held for a person), Impact (v2: the stored history re-decided — decision_diff with the clauses of each change;
a correction: every decided application re-decided, `expected["only"]` = the corrected one), RefusalRegression (not
switchable: the target here answers, so every change is checked; the held should-abstain set is the applications the
policy in force hands to a person — decision "review" — under version 2, without the 20 corrected ones; declared once
version 2 is in force, because a document version redefines what goes to a person), Approval (documents: gate +
person; corrections: gate, a person for a widening one).

The threshold change: the system's own tuning "points = 3 → approve" (the review band closed: fewer hand-overs). It
answers every application of the should-abstain set and the refusal-regression stage refuses it; impact alone would
not (no `only`, no share limit).

Printed: decisions equal to the reference (reference.py) under v1 and v2; the history decisions v2 changes and how
many trace to a clause; each correction's verdict against the one corrections.json expects; extraneous changes of the
admitted corrections; the `young` rule's refusal; replay of every version; the journal's verify. The exit status is 1
when any count differs from the expected one (144 of 144 traced; 8 refused, 7 held → promoted, 5 admitted; 0
extraneous; young refused by static; the threshold change refused by refusal_regression, answering all 137 of 137
hand-overs; one refusal-regression record per proposal; 14 versions replay; the journal verifies). Without the applications it prints how
to fetch them and exits 0."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
sys.path.insert(0, str(TASKS))
sys.path.insert(0, str(TASKS / "credit"))
warnings.simplefilter("ignore")

from common.llm import DATA, read_jsonl  # noqa: E402
from reference import decide  # noqa: E402

from solvi import Answer, Catalog, Question, System  # noqa: E402
from solvi.core.catalog import find_quote  # noqa: E402
from solvi.core.store import JSONLStorage  # noqa: E402
from solvi.experimental.changes import (Approval, Change, Consistency, Impact, KnownFailures, Pipeline,  # noqa: E402
                                        RefusalRegression, Static, is_refusal)
from solvi.core.versions import Spec, decision_diff  # noqa: E402

PREPARED = DATA / "credit" / "prepared"
POLICY = (TASKS / "credit" / "POLICY_v1_v2.md").read_text(encoding="utf-8")
SPEC = Spec(POLICY, name="credit policy (POLICY_v1_v2.md)")
CORRECTIONS = json.loads((TASKS / "credit" / "corrections.json").read_text(encoding="utf-8"))
DECLARED = ["checking_account", "duration_months", "credit_history", "purpose", "credit_amount_dm", "savings",
            "employed_since", "other_debtors", "property", "other_installment_plans"]
QUESTIONS = [Question("decision", "Approve the loan, refuse it, or send it to a person?",
                      Answer.choice(["approve", "refuse", "review"]))]
ORDER = {"refuse": 0, "review": 1, "approve": 2}             # more permissive →
PERSON = "person (simulated: every held change is approved)"
EXPECTED = {"v2_changed": 144, "v2_traced": 144, "refused by impact": 8,
            "held for the person, then promoted after the ruling": 7, "admitted (promoted)": 5, "extraneous": 0,
            "versions": 14, "should_abstain": 137}


def clause(quote):
    """The id of the clause of the policy that holds `quote` (Spec(text) numbers them c1, c2, ...)."""
    for cid, c in SPEC.clauses.items():
        if find_quote(quote, c.text, whole=False) is not None:
            return cid
    raise ValueError(f"no clause of the policy holds {quote!r}")


def cite(quote):
    return {"clause": clause(quote), "cite": quote}


# ───────────────────────────── the rules, one function each, as the policy states them
def checking_points(checking_account):
    return {"below 0 DM": 3, "0 to 200 DM": 2, "no checking account": -1}.get(checking_account, 0)  # solvi: ok


def duration_points(duration_months):
    return 2 if duration_months > 36 else 1 if duration_months > 24 else 0


def amount_points(credit_amount_dm):
    return 2 if credit_amount_dm > 8000 else 0


def savings_points(savings):
    return 1 if savings == "below 100 DM" else 0


def employment_points(employed_since):
    return 1 if employed_since in ("unemployed", "less than 1 year") else 0


def history_points(credit_history):
    return 2 if credit_history in ("all credits at this bank paid back duly",
                                   "no credits taken, or all paid back duly") else 0


def other_plans_points(other_installment_plans):
    return 1 if other_installment_plans in ("bank", "stores") else 0


def property_points(property):                                  # noqa: A002 — the field's name
    return 1 if property == "unknown or none" else 0


def guarantor_points(other_debtors):
    return -1 if other_debtors == "guarantor" else 0


def purpose_points(purpose):
    return -1 if purpose == "car (used)" else 1 if purpose in ("education", "repairs") else 0


def young_points(age):
    return 1 if age <= 25 else 0


def not_too_large(credit_amount_dm, duration_months) -> bool:
    return not (credit_amount_dm > 15000 and duration_months > 48)


RULES = {
    checking_points: ['checking_account is "below 0 DM"', 'checking_account is "0 to 200 DM"',
                      'checking_account is "no checking account"'],
    duration_points: ["duration_months > 36", "24 < duration_months ≤ 36"],
    amount_points: ["credit_amount_dm > 8000"],
    savings_points: ['savings is "below 100 DM"'],
    employment_points: ['employed_since is "unemployed" or "less than 1 year"'],
    history_points: ['credit_history is "all credits at this bank paid back duly" or "no credits taken, or all paid '
                     'back duly"'],
    other_plans_points: ['other_installment_plans is "bank" or "stores"'],
    property_points: ['property is "unknown or none"'],
    guarantor_points: ['new rule `guarantor`: other_debtors is "guarantor" → −1;'],
    purpose_points: ['new rule `purpose`: purpose is "car (used)" → −1; purpose is "education" or "repairs" → +1;'],
    young_points: ["Proposed rule `young`: age ≤ 25 → +1."],
}
COMMON = [checking_points, duration_points, amount_points, savings_points, employment_points, history_points,
          other_plans_points]


def points_v1(checking_points, duration_points, amount_points, savings_points, employment_points, history_points,
              other_plans_points, property_points):
    return (checking_points + duration_points + amount_points + savings_points + employment_points + history_points
            + other_plans_points + property_points)


def points_v2(checking_points, duration_points, amount_points, savings_points, employment_points, history_points,
              other_plans_points, guarantor_points, purpose_points):
    return (checking_points + duration_points + amount_points + savings_points + employment_points + history_points
            + other_plans_points + guarantor_points + purpose_points)


def points_v2_young(checking_points, duration_points, amount_points, savings_points, employment_points, history_points,
                    other_plans_points, guarantor_points, purpose_points, young_points):
    return points_v2(checking_points, duration_points, amount_points, savings_points, employment_points,
                     history_points, other_plans_points, guarantor_points, purpose_points) + young_points


points_v1.__name__ = points_v2.__name__ = points_v2_young.__name__ = "points"     # the fact they set


def decision_v1(points):
    return "refuse" if points >= 5 else "review" if points == 4 else "approve"


def decision_v2(points):
    return "refuse" if points >= 4 else "review" if points == 3 else "approve"


def decision_v2_no_review(points):
    return "refuse" if points >= 4 else "approve"


decision_v2_no_review.__name__ = "decision"


def exception_rule(cor):
    """A person's exception as a hard check: False (the condition holds) forces the corrected decision. Written as
    source so that its arguments are the fields it reads (and its code is fingerprinted like any part's)."""
    import linecache
    name = f"exception_{cor['cid']}"
    cond = cor["condition"]
    src = (f"def {name}({', '.join(cond)}) -> bool:\n"
           f"    return not ({' and '.join(f'{k} == {v!r}' for k, v in cond.items())})\n")
    filename = f"<changes_credit {name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns = {}
    exec(compile(src, filename, "exec"), ns)                    # noqa: S102 — our own fixed text
    return ns[name]


def build(version, exceptions=(), young=False, sensitive=(), review=True):
    """The System of one hand-written catalog: version 1 or 2, the admitted exceptions (and the candidate's), the
    `young` proposal."""
    cat = Catalog(sensitive=sensitive)
    for cor in exceptions:                                      # the more specific rule forces first
        cat.check(hard=True, then={"decision": cor["corrected"]}, clause=f"person:{cor['cid']}",
                  cite=cor["note"])(exception_rule(cor))
    cat.check(hard=True, then={"decision": "refuse"}, **cite("Hard rule `too_large`: credit_amount_dm > 15000 and "
                                                             "duration_months > 48 → refuse"))(not_too_large)
    rules = COMMON + ([property_points] if version == 1 else [guarantor_points, purpose_points])
    if young:
        rules = rules + [young_points]
    for f in rules:
        qs = RULES[f]
        cat.fn(clause=[clause(q) for q in qs], cite=qs)(f)
    points = points_v1 if version == 1 else points_v2_young if young else points_v2
    pts = (["An application is scored in points; the points decide."] +
           (["rule `property` is removed;"] if version == 2 else []))
    cat.fn(clause=[clause(q) for q in pts], cite=pts)(points)
    dq = "points ≥ 5 → **refuse**" if version == 1 else "the thresholds move: points ≥ 4 → refuse; points = 3 → review;"
    cat.rule("decision", **cite(dq))(decision_v1 if version == 1 else decision_v2 if review else decision_v2_no_review)
    return System(cat, QUESTIONS)


def app_input(a):
    """What the policy may see: the id and the declared fields (no sensitive field, no outcome)."""
    return {"id": a["id"], **{k: a[k] for k in DECLARED}}


class Versions:
    """The pipeline's target: each promoted catalog, by version; rollback puts the one before back in force."""
    artifact = "update"

    def __init__(self):
        self.systems, self.labels, self.in_force = {}, {}, None

    def promote(self, change, measured):
        n = len(self.systems) + 1
        self.systems[n], self.labels[n] = change.payload["system"], change.payload["label"]
        before, self.in_force = self.in_force, n
        return {"version": n, "before": before, "label": change.payload["label"]}

    def rollback(self, ref, by=None, why=None):
        self.in_force = ref["before"]

    def answerer(self, change, ctx):
        """The target answers (so every change to it passes the refusal-regression stage): the version in force, or
        the candidate's system."""
        s = change.payload["system"] if change is not None else self.systems.get(self.in_force)
        return None if s is None else (lambda x: s.ask(x, store=False))


def handed_over(response):
    """The policy hands an application to a person with "review" (or abstains)."""
    return is_refusal(response, "decision") or response["decision"].answer == "review"


def main():
    try:
        apps = {s: read_jsonl(PREPARED / f"applications_{s}.jsonl") for s in ("history", "new")}
        sensitive = set(json.loads((PREPARED / "about.json").read_text())["sensitive"])
    except OSError:
        print(f"the German Credit applications are not in {PREPARED}: fetch and prepare them first —\n"
              "    benchmarks/tasks/fetch.sh credit && python benchmarks/tasks/prepare.py\n"
              "(or set STAND_DATA to where they are). Nothing was run.")
        return 0
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="changes_credit_"))
    problems = []

    def expect(what, got, want):
        ok = got == want
        print(f"  {what}: {got}" + ("" if ok else f"   ✗ expected {want}"))
        if not ok:
            problems.append(f"{what}: {got} (expected {want})")

    try:
        texts = POLICY + "\n\n" + "\n\n".join(c["note"] for c in CORRECTIONS["corrections"])
        store = JSONLStorage(str(tmp / "decisions.jsonl"))
        target = Versions()
        state = {"current": None, "inputs": {}, "last": None}

        def classify(change, ctx):
            cor = change.payload.get("correction")
            if cor is None:
                return {"direction": "a document version"}
            matched = [x for x in state["inputs"].values() if all(x[k] == v for k, v in cor["condition"].items())]
            now = [state["current"].ask(x, store=False)["decision"].answer for x in matched]
            wid = any(ORDER[cor["corrected"]] > ORDER[d] for d in now)
            nar = any(ORDER[cor["corrected"]] < ORDER[d] for d in now)
            return {"direction": "widens" if wid else "narrows" if nar else "none", "matched": len(matched)}

        def impact(change, ctx):
            if state["current"] is None:
                return {"changed": 0, "total": 0, "note": "nothing decided yet"}
            if change.payload.get("correction") is None:          # a document version: the stored decisions
                dd = decision_diff(state["current"], change.payload["system"], store=store)
            else:                                                 # an exception: every decided application
                dd = decision_diff(state["current"], change.payload["system"], inputs=list(state["inputs"].values()),
                                   only=(change.expected or {}).get("only"))
            state["last"] = dd
            return dd

        pipe = Pipeline([KnownFailures(), Static(sensitive=sensitive, text=texts, spec=SPEC),
                         Consistency(classify=classify), Impact(impact), RefusalRegression(refused=handed_over),
                         Approval({("update", "document"): "gate+person", ("update", "person"): "gate"})],
                        str(tmp / "changes.jsonl"), targets=[target], refusals=lambda change, ctx: state["refusals"])
        state["refusals"] = []

        def propose(system, label, source, reason, expected=None, correction=None):
            payload = {"fingerprint": system.fingerprint()["catalog"], "system": system, "label": label,
                       "correction": correction}
            res = pipe.propose(Change(source, artifact="update", payload=payload, proposer="author" if source ==
                                      "document" else "person", reason=reason, expected=expected, change_type="rule"))
            first = res.status
            if res.status == "held":
                res = pipe.rule(res.change_id, "approve", by=PERSON)
            if res.promoted:
                state["current"] = system
            return first, res

        # version 1, then the history decided by it
        v1 = build(1, sensitive=sensitive)
        first, res = propose(v1, "v1", "document", "credit policy version 1")
        print(f"v1: {first} → {res.status}")
        if not res.promoted:
            problems.append(f"v1 not promoted: {res.stage}: {res.reason}")
            raise SystemExit
        v1.storage, store.catalog = store, v1
        for a in apps["history"]:
            x = app_input(a)
            v1.ask(x)
            state["inputs"][a["id"]] = x
        v1.storage = None

        # version 2: the stored history re-decided, each change traced to the clauses of the rules that caused it
        v2 = build(2, sensitive=sensitive)
        first, res = propose(v2, "v2", "document", "credit policy version 2 (the change under audit)")
        dd = state["last"]
        traced = sum(any(c["clauses"] for c in x["causes"]) for x in dd.moved)
        print(f"v2: {first} → {res.status}; history decisions it changes:")
        expect("changed", len(dd.moved), EXPECTED["v2_changed"])
        expect("traced to a clause", traced, EXPECTED["v2_traced"])
        for line in str(dd).splitlines()[1:8]:
            print("   ", line)
        if not res.promoted:
            problems.append(f"v2 not promoted: {res.stage}: {res.reason}")
            raise SystemExit
        for a in apps["new"]:
            x = app_input(a)
            v2.ask(x, store=False)
            state["inputs"][a["id"]] = x
        corrected = {c["application"] for c in CORRECTIONS["corrections"]}
        state["refusals"] = [x for i, x in state["inputs"].items() if i not in corrected and handed_over(v2.ask(x, store=False))]
        print(f"should-abstain set (handed to a person under v2, the corrected applications left out): "
              f"{len(state['refusals'])} applications")
        expect("should-abstain applications", len(state["refusals"]), EXPECTED["should_abstain"])

        # the person's corrections, one by one
        print("corrections (expected → got):")
        admitted, verdicts, extraneous = [], {}, 0
        for cor in CORRECTIONS["corrections"]:
            cand = build(2, admitted + [cor], sensitive=sensitive)
            first, res = propose(cand, f"v2+{cor['cid']}", "person", cor["note"],
                                 expected={"only": [cor["application"]], "to": cor["corrected"]}, correction=cor)
            got = ("refused by impact" if res.status == "rejected" and res.stage == "impact" else
                   "held for the person, then promoted after the ruling" if first == "held" and res.promoted else
                   "admitted (promoted)" if res.promoted else f"{res.status} by {res.stage}: {res.reason[:80]}")
            verdicts[got] = verdicts.get(got, 0) + 1
            if res.promoted:
                admitted.append(cor)
                imp = next(v for v in res.verdicts if v.stage == "impact").measured
                extraneous += len(imp.get("extraneous") or [])
                now = cand.ask(state["inputs"][cor["application"]], store=False)["decision"].answer
                if now != cor["corrected"]:
                    problems.append(f"{cor['cid']}: the corrected application is {now!r}, not {cor['corrected']!r}")
            mark = "" if got == cor["expected_verdict"] else "   ✗"
            if mark:
                problems.append(f"{cor['cid']}: {got} (expected {cor['expected_verdict']})")
            print(f"  {cor['cid']} {cor['kind']}/{cor['scope']}: {cor['expected_verdict']} → {got}{mark}"
                  + (f" ({res.reason[:90]})" if res.status == "rejected" else ""))
        for k in ("refused by impact", "held for the person, then promoted after the ruling", "admitted (promoted)"):
            expect(k, verdicts.get(k, 0), EXPECTED[k])
        expect("extraneous changes of the admitted corrections", extraneous, EXPECTED["extraneous"])

        # the rule that must be refused
        young = build(2, admitted, young=True, sensitive=sensitive)
        first, res = propose(young, "young", "person", "proposed rule `young`: age ≤ 25 → +1")
        refused = res.status == "rejected" and res.stage == "static" and "reads_sensitive" in res.reason
        print(f"young: {res.status} by {res.stage}: {res.reason[:120]}")
        if not refused:
            problems.append(f"young: {res.status} by {res.stage}")

        # a threshold change that washes the hand-overs out: refused by the refusal-regression stage
        closed = build(2, admitted, sensitive=sensitive, review=False)
        payload = {"fingerprint": closed.fingerprint()["catalog"], "system": closed, "label": "no review band",
                   "correction": None}
        res = pipe.propose(Change("outcome", artifact="update", payload=payload, proposer="self-tuning",
                                  reason="points = 3 → approve: fewer hand-overs", change_type="rule"))
        rr = next((v for v in res.verdicts if v.stage == "refusal_regression"), None)
        print(f"threshold change (review band closed): {res.status} by {res.stage}: {res.reason[:160]}")
        expect("threshold change refused by", res.stage if res.status == "rejected" else res.status, "refusal_regression")
        if rr is not None:
            m = rr.measured
            print(f"  refused before {m['refused_before']} → after {m['refused_after']} of {m['n']} (lost {m['lost']})")
            expect("hand-overs it answers", f"{m['lost']} of {m['refused_before']}",
                   f"{EXPECTED['should_abstain']} of {EXPECTED['should_abstain']}")
        n_rr = len([d for d in pipe.history("verdict") if d["stage"] == "refusal_regression"])
        n_prop = len(pipe.history("propose"))
        print(f"refusal-regression records: {n_rr} for {n_prop} proposals (the static stage refused young before it)")
        expect("refusal-regression records for the proposals that reached it", n_rr, n_prop - 1 - EXPECTED["refused by impact"])

        # the decisions against the reference, and every version replayed
        allapps = apps["history"] + apps["new"]
        print("decisions equal to the reference (reference.py):")
        for v, s in ((1, v1), (2, v2)):
            same = sum(s.ask(app_input(a), store=False)["decision"].answer == decide(a, v)[0] for a in allapps)
            expect(f"v{v}", f"{same} of {len(allapps)}", f"{len(allapps)} of {len(allapps)}")
        print("replay of every version (each decides every application into its own store, then replays it):")
        failed = 0
        xs = list(state["inputs"].values())
        for v in pipe.versions():
            n = v["ref"]["version"]
            s = target.systems[n]
            vs = JSONLStorage(str(tmp / f"v{n}.jsonl"))
            s.storage, vs.catalog = vs, s
            for x in xs:
                s.ask(x)
            s.storage = None
            bad = vs.replay_all(s)
            failed += len(bad) + (not vs.verify()["ok"])
            vs.close()
        expect("versions replayed", len(pipe.versions()), EXPECTED["versions"])
        expect("replay failures", failed, 0)
        journal = pipe.journal.verify()["ok"]
        expect("journal verifies", journal, True)
        print(f"journal: {len(pipe.history())} records; {time.time() - t0:.0f} s")
    except SystemExit:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if problems:
        print("DIFFERS from the expected counts:\n  " + "\n  ".join(problems))
        return 1
    print("all counts as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
