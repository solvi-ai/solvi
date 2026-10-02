"""Propose → check → re-ask with the reasons → escalate: a loop around a System whose checks judge a proposal.

    from solvi.refine import Fail, refine

    @cat.check(hard=True, then={"accept": "no"})
    def nobody_busy(facts, slot) -> bool:
        busy = [f"{p} is busy {b}" for p, b in clashes(facts, slot)]
        return Fail(*busy) if busy else True          # False, with the reasons the proposer is told

    run = refine(system, {"problem": text}, "accept", propose=writer.proposer(messages), into="slot", rounds=3)
    run.accepted, run.proposal            # the first proposal the checks accepted, or None
    run.escalation                        # why a person gets it: not accepted after 3 rounds, with the last reasons
    run.rounds                            # every round: the proposal, its response (stored, replayable), the reasons
    run.replay(system)                    # every round's trace, and that the loop did what its record says

A check gives its reason by returning `Fail("...")` (one or more reasons) instead of `False`: it is False everywhere a
bool is read — rules, hard checks, plain Python — and the reasons are recorded with the check (`record.extra["reasons"]`),
added to the answer's `why` when the check decides ("hard check nobody_busy is false: Harold is busy 13:30 - 15:30"),
shown in the audit and compared on replay. A check that returns `False` keeps working: its reason is its docstring's
first line, else "<name> is false".

One round: the proposer is called with the state and the earlier rounds and returns a proposal (a value, or a
solvi.generate.Generated — its record of the request is kept in the round); the proposal is given to the System under
`into`, and the System is asked. The round is accepted when `accept` says so — "checks" (default): every hard check that
governs the question was evaluated and passed; an answer or a list of answers ("yes"); or a function of the Response.
Otherwise its reasons — what each failed hard check governing the question says, in catalog order, or, when the
question could not be decided at all, the errors of the parts that failed ("spec: ValueError: no slot in the plan") —
become the round's feedback (`feedback(round)` may reword them), and the next round's proposer sees them. The loop stops
at the first accepted round or after `rounds`, and then escalates. A reply the generator rejects (solvi.llm
InvalidOutput: not JSON, outside the schema, a quote not in the text) is a round too: its reason is fed back. A proposer
that fails otherwise (the server does not answer) ends the loop with an escalation.

Without a proposer the System generates itself (a part made by `Generator.part`): each round gives the feedback of the
earlier rounds as the fact `feedback_into` (a list of reasons), and the part reads it.

Not done here: no search over alternatives (a loop re-asks one proposer; it does not enumerate), no judgement of which
of two accepted proposals is better, and no guarantee that a re-ask converges — measure the rounds on your own
data: a model can trade one violation for another."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .core import Claim


class Fail(Claim):
    """A check's False with its reasons: `return Fail("Harold is busy 13:30 - 15:30")`. Falsy (`bool(Fail(...))` is
    False), so the check reads as False everywhere; the reasons are recorded with the check. `Fail()` with no reason is a
    plain False."""

    def __init__(self, *reasons):
        rs = []
        for x in reasons:
            rs += [str(y) for y in x] if isinstance(x, (list, tuple)) else [str(x)]
        rs = [r for r in rs if r.strip()]
        super().__init__(False, extra={"reasons": rs} if rs else {})

    def __bool__(self):
        return False

    @property
    def reasons(self):
        return list(self.extra.get("reasons", []))

    def __repr__(self):
        return "Fail(" + ", ".join(repr(r) for r in self.reasons) + ")"


@dataclass
class Failed:
    """A check that is False in a response: its name, whether it is hard, and its reasons."""
    check: str
    hard: bool
    reasons: list
    governs: bool = False            # a hard check that forces or blocks the question asked (set by refine)

    def to_dict(self):
        return {"check": self.check, "hard": self.hard, "reasons": list(self.reasons), "governs": self.governs}


def reasons_of(record, part=None):
    """A check's record → its reasons: what Fail recorded, else the check's docstring's first line, else "<name> is
    false"."""
    rs = (record.extra or {}).get("reasons") if isinstance(record.extra, dict) else None
    if rs:
        return list(rs)
    doc = (getattr(part, "doc", "") or "").strip().splitlines()
    return [doc[0]] if doc else [f"{record.name} is false"]


def failed_checks(res, question=None):
    """The checks that are False in a response, in catalog order → [Failed]. question: only the checks in that
    question's flow."""
    cat = res.catalog
    names = set(res.flow.per_question.get(question, [])) if question is not None else None
    order = {n: i for i, n in enumerate(cat.parts)} if cat is not None else {}
    out = []
    for r in res.trace.records:
        if r.kind != "check" or r.value is not False or (names is not None and r.name not in names):
            continue
        part = cat.parts.get(r.name) if cat is not None else None
        out.append((order.get(r.name, len(order)), Failed(r.name, bool(getattr(part, "hard", False)), reasons_of(r, part))))
    return [f for _, f in sorted(out, key=lambda t: t[0])]


def causes(res, question):
    """Why a question could not be decided: the errors of the parts that failed by themselves ("spec: ValueError: no
    slot in the plan"), not the steps that only lacked their inputs; else the answer's reason. [] when it was decided."""
    r = res[question]
    if r.status != "abstain":
        return []
    out = [f"{x.name}: {x.error}" for x in res.trace.records if getattr(x, "error", None)
           and "missing inputs" not in str(x.error)]
    return out or [r.why]


def _governing(res, question, failed):
    """The failed hard checks that govern the question (they force or block its answer)."""
    from .system import governs
    q = res._system.questions[question] if res._system is not None else None
    out = []
    for f in failed:
        part = res.catalog.parts.get(f.check) if res.catalog is not None else None
        if f.hard and part is not None and (q is None or governs(part, q)):
            out.append(f)
    return out


def accepted(res, question, accept="checks"):
    """Is a response's proposal accepted? accept: "checks" — every hard check that governs the question was evaluated
    and passed (the answer itself may still abstain, say for low confidence); an answer or a list / tuple / set of
    answers — the question's answer is one of them; a function of the Response → bool."""
    if callable(accept):
        return bool(accept(res))
    r = res[question]
    if accept == "checks":
        return r.guard != "hard_check"
    ok = accept if isinstance(accept, (list, tuple, set, frozenset)) else [accept]
    return r.status != "abstain" and r.answer in ok


@dataclass
class Round:
    """One round of a refinement: what was proposed, what the System said, and what goes back to the proposer."""
    index: int
    proposal: Any = None                 # the value given to the System (None when the System generates itself)
    generated: Any = None                # the generator's record of the request (Generated.meta), when there was one
    response: Any = None                 # the System's Response (None when the proposer failed)
    accepted: bool = False
    failed: list = field(default_factory=list)       # [Failed]: every check False in the question's flow
    causes: list = field(default_factory=list)       # why the question could not be decided (or the proposal was invalid)
    feedback: Any = None                 # what the proposer is told next (a text or a list of reasons); None: no next round
    error: str | None = None             # the proposer's own failure (an invalid reply, no answer)
    said: str | None = None              # the proposer's reply as text (for the assistant's turn of a re-ask)

    @property
    def reasons(self):
        """What is wrong with the proposal: the reasons of the failed hard checks that govern the question, else the
        causes."""
        hard = [r for f in self.failed if f.hard and f.governs for r in f.reasons]
        return hard or list(self.causes)

    @property
    def stored_id(self):
        return getattr(self.response, "stored_id", None)

    def to_dict(self):
        return {"index": self.index, "proposal": _plain(self.proposal), "generated": self.generated,
                "response": self.response.to_dict() if self.response is not None else None, "accepted": self.accepted,
                "failed": [f.to_dict() for f in self.failed], "causes": list(self.causes),
                "feedback": self.feedback, "error": self.error, "said": self.said, "stored_id": self.stored_id}


@dataclass
class Refinement:
    """The record of one refine() call. `accepted`, `proposal` (the accepted one, or None), `result` (the question's
    Result in the last round), `escalation` (why a person gets it), `rounds`; `to_dict` / `from_dict`; `replay`."""
    question: str
    rounds: list
    accepted: bool
    escalation: str | None
    max_rounds: int
    into: str | None
    feedback_into: str | None
    accept: Any = "checks"
    feedback_fn: str = "reasons"

    @property
    def proposal(self):
        return next((r.proposal for r in self.rounds if r.accepted), None)

    @property
    def response(self):
        return next((r.response for r in reversed(self.rounds) if r.response is not None), None)

    @property
    def result(self):
        res = self.response
        return None if res is None else res[self.question]

    def to_dict(self):
        acc = self.accept if not callable(self.accept) else f"function {getattr(self.accept, '__name__', '?')}"
        return {"question": self.question, "accepted": self.accepted, "escalation": self.escalation,
                "max_rounds": self.max_rounds, "into": self.into, "feedback_into": self.feedback_into,
                "accept": list(acc) if isinstance(acc, (set, frozenset, tuple)) else acc, "feedback": self.feedback_fn,
                "rounds": [r.to_dict() for r in self.rounds]}

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d, catalog=None):
        """A stored refinement back (responses restored with `catalog`'s types, as Response.model_validate does)."""
        from .system import Response
        rounds = []
        for x in d["rounds"]:
            res = Response.model_validate(x["response"], catalog=catalog) if x.get("response") is not None else None
            fs = [Failed(f["check"], f["hard"], f["reasons"], f.get("governs", False)) for f in x.get("failed", [])]
            rd = Round(x["index"], x.get("proposal"), x.get("generated"), res, x["accepted"], fs, x.get("causes", []),
                       x.get("feedback"), x.get("error"), x.get("said"))
            rounds.append(rd)
        return cls(d["question"], rounds, d["accepted"], d.get("escalation"), d["max_rounds"], d.get("into"),
                   d.get("feedback_into"), d.get("accept", "checks"), d.get("feedback", "reasons"))

    def replay(self, system, accept=None, feedback=None, trust_models=False):
        """Re-check the whole loop: every round's trace replays under `system` (its own mismatches are listed), and the
        loop did what its record says — the round's acceptance follows from its response (accept: needed again when it
        was a function), the recorded failed checks and causes are those of the response, the feedback is what
        `feedback` gives for the round (with the default, the round's reasons; a custom one is checked only when passed
        again), each round's input carries the proposal recorded for it and the feedback of the rounds before it, nothing
        ran after an accepted round, and the escalation matches the outcome. → {"ok", "rounds", "mismatches": [(round,
        what, why)], "feedback": "checked" / "unchecked"}."""
        acc = self.accept if accept is None else accept
        if isinstance(acc, str) and acc.startswith("function "):
            raise ValueError(f"this refinement was accepted by {acc}: pass it again, replay(system, accept=...)")
        fb = feedback if feedback is not None else (None if self.feedback_fn != "reasons" else (lambda r: r.reasons))
        bad, reps = [], []
        given = []
        for i, rd in enumerate(self.rounds):
            if i > 0 and self.rounds[i - 1].accepted:
                bad.append((i, "loop", "a round ran after an accepted one"))
            res = rd.response
            if res is None:
                if not rd.error:
                    bad.append((i, "round", "no response and no error recorded"))
                given += _items(rd.feedback)
                continue
            rep = res.trace.replay(system, trust_models=trust_models)
            reps.append(rep)
            for m in rep["mismatches"]:
                bad.append((i, "trace", f"{m[1]}: {m[2]}"))
            init = res.trace.init
            if self.into is not None and rd.proposal is not None and _vh(init.get(self.into)) != _vh(rd.proposal):
                bad.append((i, "input", f"the input {self.into} is not the recorded proposal"))
            if self.feedback_into is not None and _vh(list(init.get(self.feedback_into) or [])) != _vh(given):
                bad.append((i, "input", f"the input {self.feedback_into} is not the feedback of the rounds before"))
            now = _round_of(res, self.question, acc, i)
            if now.accepted != rd.accepted:
                bad.append((i, "accepted", f"recorded {rd.accepted}, the response gives {now.accepted}"))
            if [(f.check, f.reasons) for f in now.failed] != [(f.check, list(f.reasons)) for f in rd.failed]:
                bad.append((i, "failed", "the recorded failed checks are not those of the response"))
            if list(now.causes) != list(rd.causes):
                bad.append((i, "causes", "the recorded causes are not those of the response"))
            if fb is not None and not rd.accepted and rd.feedback is not None and _vh(fb(now)) != _vh(rd.feedback):
                bad.append((i, "feedback", "the recorded feedback is not what the feedback function gives"))
            given += _items(rd.feedback)
        last = self.rounds[-1] if self.rounds else None
        if self.accepted != bool(last and last.accepted):
            bad.append((len(self.rounds), "outcome", "recorded accepted does not match the last round"))
        if not self.accepted and not self.escalation:
            bad.append((len(self.rounds), "outcome", "not accepted, and no escalation recorded"))
        if not self.accepted and last is not None and not last.error and len(self.rounds) < self.max_rounds:
            bad.append((len(self.rounds), "loop", f"stopped after {len(self.rounds)} of {self.max_rounds} rounds "
                                                   "without an accepted proposal or a proposer error"))
        return {"ok": not bad, "rounds": len(self.rounds), "mismatches": bad,
                "feedback": "checked" if fb is not None else "unchecked", "traces": reps}


def _items(feedback):
    if feedback is None:
        return []
    return [feedback] if isinstance(feedback, str) else list(feedback)


def _vh(v):
    from .runtime import vhash
    return vhash(_plain(v))


def _plain(v):
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _round_of(res, question, accept, index):
    """A response → its Round (acceptance, failed checks with whether they govern the question, causes)."""
    failed = failed_checks(res, question)
    gov = {f.check for f in _governing(res, question, failed)}
    for f in failed:
        f.governs = f.check in gov
    return Round(index, response=res, accepted=accepted(res, question, accept), failed=failed,
                 causes=causes(res, question))


def _said(proposal, generated):
    """The proposer's reply as text: the reply recorded by the generator, else the proposal itself."""
    if isinstance(generated, dict) and isinstance(generated.get("text"), str):
        return generated["text"]
    if proposal is None:
        return None
    if isinstance(proposal, str):
        return proposal
    return json.dumps(_plain(proposal), ensure_ascii=False, default=str)


def refine(system, state, question, propose=None, *, into="proposal", rounds=3, accept="checks", feedback=None,
           feedback_into=None, history=None, store=True):
    """Propose → check → re-ask with the reasons → escalate (see the module docs) → a Refinement.

    system: the System whose checks judge a proposal; state: the given facts; question: the question whose hard checks
    (or answer) decide acceptance. propose: (state, earlier rounds) → a proposal — a value or a Generated
    (`Generator.proposer(...)` builds one) — given to the System as the fact `into`; None: the System generates itself
    and reads the earlier rounds' feedback as the fact `feedback_into` (default "feedback"). rounds: proposals at most.
    accept: "checks" (default), an answer or answers, or a function of the Response. feedback: round → the text or
    the list of reasons the proposer is told (default: round.reasons). history: earlier rounds the proposer should
    see first (a Refinement's rounds, to continue it). store: whether a System with storage stores every round."""
    if int(rounds) < 1:
        raise ValueError("rounds must be at least 1")
    if propose is None and feedback_into is None:
        feedback_into = "feedback"
    if propose is not None and not callable(propose):
        raise TypeError("propose is a function (state, rounds) → a proposal")
    if question not in system.questions:
        raise ValueError(f"{question!r} is not a question of the system")
    from .llm import InvalidOutput                     # a rejected reply (solvi.generate); not imported with solvi
    prior = list(getattr(history, "rounds", history) or [])
    out, stop = [], None
    for i in range(int(rounds)):
        seen = prior + out
        st = dict(state)
        if feedback_into is not None:
            st[feedback_into] = [x for r in seen for x in _items(r.feedback)]
        rd = Round(i)
        if propose is not None:
            try:
                got = propose(dict(state), seen)
            except InvalidOutput as e:            # a reply the generator rejected: a round whose reason is fed back
                rd.error = rd_err = f"the reply was rejected: {e}"
                rd.causes, rd.said = [rd_err], getattr(e, "reply", None)
                out.append(rd)
                if i < int(rounds) - 1:
                    rd.feedback = feedback(rd) if feedback is not None else rd.reasons
                continue
            except Exception as e:  # noqa: BLE001 — the proposer could not propose: a person takes over
                rd.error = f"the proposer failed: {type(e).__name__}: {str(e)[:200]}"
                out.append(rd)
                stop = rd.error
                break
            if isinstance(got, Claim) and isinstance(getattr(got, "extra", None), dict) and "generated" in got.extra:
                rd.generated, rd.proposal = got.extra["generated"], got.value
            else:
                rd.proposal = got.value if isinstance(got, Claim) else got
            rd.said = _said(rd.proposal, rd.generated)
            st[into] = rd.proposal
        res = system.ask(st, store=store)
        now = _round_of(res, question, accept, i)
        rd.response, rd.accepted, rd.failed, rd.causes = res, now.accepted, now.failed, now.causes
        out.append(rd)
        if rd.accepted:
            break
        if i < int(rounds) - 1:
            rd.feedback = feedback(rd) if feedback is not None else rd.reasons
    ok = bool(out and out[-1].accepted)
    esc = None
    if not ok:
        last = out[-1]
        esc = stop or (f"not accepted after {len(out)} round(s)"
                       + (": " + "; ".join(last.reasons[:3]) if last.reasons else ""))
    fb_name = "reasons" if feedback is None else f"function {getattr(feedback, '__name__', '?')}"
    return Refinement(question, out, ok, esc, int(rounds), into if propose is not None else None, feedback_into, accept,
                      fb_name)


__all__ = ["accepted", "causes", "Fail", "Failed", "failed_checks", "refine", "Refinement", "Round"]
