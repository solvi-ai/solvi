"""Credit with solvi: the policy of POLICY.md as a catalog of rules, every decision in a hash-chained store, and the
eight checks of POLICY.md carried out with what the library provides. No model, no LLM.

The story: version 1 decides the 600 history applications (stored) → version 2 is proposed and diffed against the store
→ version 2 is deployed and decides the 400 new applications in the same store, version 1 running next to it as a
shadow → a stored amount is edited in a copy of the store → old records are replayed under the new rules → a rule that
reads a sensitive field is proposed and refused before it decides anything.

How each check is done:
  1 correctness  one function per rule (its argument is the field it reads), a hard check, a rule for the decision;
  2 record       System(storage=SQLiteStorage(...)): input, catalog fingerprint and every rule's points, also when the
                 hard check decides (early_exit=False);
  3 replay       store.replay_all(v1, catalog_fp=...);
  4 diff         solvi.core.store.diff.diff(store, v2): the changed decisions and, per decision, the steps that changed it;
  5 tamper       store.verify() names the edited record;
  6 moved rules  replay_all under v2 tells "catalog changed" from "data damaged";
  7 sensitive    the input is a pydantic model without the sensitive fields; solvi.check.lint reports a rule that reads
                 a field the model does not declare (input_not_declared), and deploy() refuses such a catalog;
  8 cost         score.py; Shadow keeps v1's answer next to v2's.

    uv run python credit/solution.py [--out runs/solvi.jsonl]"""
import argparse
import inspect
import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.llm import read_jsonl, write_jsonl  # noqa: E402
from score import D, score  # noqa: E402

from solvi import Catalog, Question, Shadow, SQLiteStorage, System  # noqa: E402
from solvi.check import lint  # noqa: E402
from solvi.core.store.diff import diff  # noqa: E402


class Application(BaseModel):
    """What the policy may see. The sensitive fields (about.json) and the outcome are not declared: pydantic drops them
    before solvi gets the application, so they are neither given facts nor stored."""
    id: str
    checking_account: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount_dm: int
    savings: str
    employed_since: str
    other_debtors: str
    property: str
    other_installment_plans: str


# One function per rule: the argument is the field it reads, the name is the fact it sets (its points).
def checking_points(checking_account: str) -> int:
    return {"below 0 DM": 3, "0 to 200 DM": 2, "no checking account": -1}.get(checking_account, 0)


def duration_points(duration_months: int) -> int:
    return 2 if duration_months > 36 else 1 if duration_months > 24 else 0


def amount_points(credit_amount_dm: int) -> int:
    return 2 if credit_amount_dm > 8000 else 0


def savings_points(savings: str) -> int:
    return 1 if savings == "below 100 DM" else 0


def employment_points(employed_since: str) -> int:
    return 1 if employed_since in ("unemployed", "less than 1 year") else 0


def history_points(credit_history: str) -> int:
    return 2 if credit_history in ("all credits at this bank paid back duly", "no credits taken, or all paid back duly") else 0


def other_plans_points(other_installment_plans: str) -> int:
    return 1 if other_installment_plans in ("bank", "stores") else 0


def property_points(property: str) -> int:                       # version 1 only
    return 1 if property == "unknown or none" else 0


def guarantor_points(other_debtors: str) -> int:                 # new in version 2
    return -1 if other_debtors == "guarantor" else 0


def purpose_points(purpose: str) -> int:                         # new in version 2
    return -1 if purpose == "car (used)" else 1 if purpose in ("education", "repairs") else 0


def young_points(age: int) -> int:                               # the proposal that must be refused: age is sensitive
    return 1 if age <= 25 else 0


def not_too_large(credit_amount_dm: int, duration_months: int) -> bool:   # the hard rule too_large
    return not (credit_amount_dm > 15000 and duration_months > 48)


def total(names):
    """The fact `points`: the sum of the named rules' points (a part reads the facts its signature names)."""
    def points(*a, **k) -> int:
        return sum(a) + sum(k.values())
    points.__signature__ = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
                                              for n in names], return_annotation=int)
    return points


COMMON = [checking_points, duration_points, amount_points, savings_points, employment_points, history_points, other_plans_points]
RULES = {1: COMMON + [property_points], 2: COMMON + [guarantor_points, purpose_points]}
REFUSE_AT = {1: 5, 2: 4}


def build(rules, refuse_at=None, young=False, storage=None):
    """The System of one policy version: the rule set (1 or 2) and the threshold are separate, so that a change of one
    alone can be diffed; young=True adds the proposed rule."""
    cat = Catalog()
    cat.check(hard=True, then={"decision": "refuse"})(not_too_large)
    rules_used = RULES[rules] + ([young_points] if young else [])
    for f in rules_used:
        cat.fn(f)
    cat.fn(total([f.__name__ for f in rules_used]))
    refuse_at = refuse_at or REFUSE_AT[rules]

    @cat.rule("decision")
    def decision(points: int) -> Literal["approve", "review", "refuse"]:
        return "refuse" if points >= refuse_at else "review" if points == refuse_at - 1 else "approve"

    question = Question("decision", "Approve the loan, refuse it, or send it to a person?", requires=["not_too_large"])
    return System(cat, [question], input_model=Application, storage=storage, early_exit=False)


def deploy(rules, **kw):
    """A version goes live only when `solvi check` finds no rule reading a field the input does not declare."""
    system = build(rules, **kw)
    refused = [f.message for f in lint(system).findings if f.code == "input_not_declared"]
    if refused:
        raise PermissionError("catalog refused before use: " + "; ".join(refused))
    return system


def edited_copy(src, dst):
    """The attack (not the solution): a copy of the store in which the amount of stored decision #7 is raised by 1 DM."""
    shutil.copy(src, dst)
    con = sqlite3.connect(dst)
    (body,) = con.execute("select body from records where seq = 7").fetchone()
    amount = json.loads(body)["response"]["trace"]["init"]["credit_amount_dm"]
    con.execute("update records set body = ? where seq = 7",
                (body.replace(f'"credit_amount_dm": {amount}', f'"credit_amount_dm": {amount + 1}'),))
    con.commit()
    con.close()
    return SQLiteStorage(dst)


def main(out):
    runs = Path(out).parent
    runs.mkdir(parents=True, exist_ok=True)
    for f in runs.glob("solvi_*.db*"):
        f.unlink()
    history, new = read_jsonl(D / "applications_history.jsonl"), read_jsonl(D / "applications_new.jsonl")
    checks = {}

    store = SQLiteStorage(runs / "solvi_decisions.db")                       # version 1 decides the history
    v1 = deploy(1, storage=store)
    for a in history:
        v1.ask(Application.model_validate(a))
    fp1 = v1.fingerprint()["catalog"]
    stored = store.query(catalog_fp=fp1)
    every_rule = sum(all(f.__name__ in s.response().values for f in RULES[1]) for s in stored)
    checks["2 record"] = f"{len(stored)} stored; {every_rule} hold the input, the catalog fingerprint and every rule's points"

    rep = diff(store, build(2))                                              # version 2 proposed: what would it change?
    alone = {k: {c["id"] for c in diff(store, s).changed}
             for k, s in (("threshold", build(1, refuse_at=REFUSE_AT[2])), ("rules", build(2, refuse_at=REFUSE_AT[1])))}
    by_cause = Counter("+".join(sorted({c["name"] for c in ch["questions"]["decision"]["causes"]})) for ch in rep.changed)
    checks["4 diff"] = {"changed": len(rep.changed), "steps that changed each decision (diff)": dict(by_cause.most_common()),
                        "changed by the threshold alone / the rules alone": [len(alone["threshold"]), len(alone["rules"])]}
    changes = {c["id"]: c["questions"]["decision"]["new"]["answer"] for c in rep.changed}
    hist_v1 = {s.response().trace.init["id"]: s.answers["decision"] for s in stored}
    hist_v2 = {s.response().trace.init["id"]: changes.get(s.id, s.answers["decision"]) for s in stored}

    v2 = deploy(2, storage=store)                                            # version 2 deployed, v1 as its shadow
    shadow = Shadow(v2, build(1), storage=SQLiteStorage(runs / "solvi_shadow.db"))
    for a in new:
        shadow.ask(Application.model_validate(a))
    new_v2 = {s.response().trace.init["id"]: s.answers["decision"] for s in store.query(catalog_fp=v2.fingerprint()["catalog"])}
    new_v1 = {s.response().trace.init["id"]: s.answers["decision"] for s in shadow.storage.iter()}
    checks["8 cost: shadow"] = shadow.summary().splitlines()

    failed = store.replay_all(build(1), catalog_fp=fp1)
    checks["3 replay"] = f"{len(stored) - len(failed)} of {len(stored)} replay"

    edited = edited_copy(runs / "solvi_decisions.db", runs / "solvi_edited.db")
    checks["5 tamper"] = {"store.verify()": edited.verify()["problems"]}

    intact, damaged = store.replay_all(v2, catalog_fp=fp1), edited.replay_all(v2, catalog_fp=fp1)
    checks["6 moved rules"] = {"v1 records under v2": dict(Counter(r["summary"] for r in intact)),
                               "the same, store with the edit": dict(Counter(r["summary"] for r in damaged))}

    try:
        deploy(2, young=True)
        checks["7 sensitive field"] = "NOT refused"
    except PermissionError as e:
        checks["7 sensitive field"] = str(e)

    write_jsonl(out, [{"id": a["id"], "version": v, "decision": got[a["id"]]}
                      for a in history + new for v, got in ((1, {**hist_v1, **new_v1}), (2, {**hist_v2, **new_v2}))])
    checks["1 correctness, 8 cost"] = score(out)
    print(json.dumps(checks, indent=1, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(HERE / "runs" / "solvi.jsonl"))
    main(p.parse_args().out)
