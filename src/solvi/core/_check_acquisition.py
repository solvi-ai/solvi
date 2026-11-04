"""Recorded adaptive acquisition of explicitly priced soft checks."""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

from .runtime import MISSING, Record, _ExecutionContext, aexecute, execute
from .chain import append as _append

__all__ = []


def freeze_policies(system, policies, *, existing=False):
    if policies is None:
        return {}, {}
    if not isinstance(policies, dict) or set(policies) - set(system.questions):
        raise ValueError("check_policies must map declared questions to CheckPolicy objects")
    from .plan.check_policy import CheckPolicy
    frozen, fps = {}, {}
    for name, policy in policies.items():
        if type(policy) is not CheckPolicy:
            raise TypeError("check_policies requires CheckPolicy")
        if name in system._check_models and not existing:
            raise ValueError("configure either check_models or check_policies for a question")
        frozen[name] = policy.freeze()
        fps[name] = frozen[name].fingerprint()
        system._check_models[name] = frozen[name].model
        system._check_model_fps[name] = frozen[name].model.fingerprint()
    return frozen, fps


def requested(system, prepared, loss):
    selected = [q for q in prepared.questions if q.name in getattr(system, "_check_policies", {})]
    if not selected:
        if loss is not None:
            raise ValueError("check_loss requires a configured check policy for this request")
        return False
    if len(prepared.questions) != 1:
        raise ValueError("adaptive check acquisition requires one question per ask")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("check_loss must be a finite nonnegative request loss")
    if system.strategist is not None or prepared.order is not None or prepared.policy is not None or system.learn:
        raise ValueError("adaptive checks require deterministic declared planning without online learning")
    name = selected[0].name
    if name in system.calib or name in system.guards or name in system.shift_watch:
        raise ValueError("adaptive check policies require their own acquisition calibration; existing guards cannot be reused")
    return True


def _flow(system, prepared, selected):
    question = prepared.questions[0]
    sources = set(system._check_policies[question.name].sources)
    requires = [name for name in question.requires if name not in sources] + list(selected)
    contracts = getattr(system.catalog, "contracts", {})
    if question.name in contracts and not contracts[question.name].applies(prepared.state)[0]:
        return system._plan([], prepared.state.keys())
    return system._plan([dataclasses.replace(question, requires=requires)], prepared.state.keys())


def _prepare(system, prepared, early_exit, speculate=False):
    if not system._early(early_exit) or speculate:
        raise ValueError("adaptive checks require early_exit=True and speculate=False")
    q = prepared.questions[0]
    policy = system._check_policies[q.name]
    base = _flow(system, prepared, [])
    names = {step.part.name for step in base.steps}
    if names & set(policy.sources):
        raise ValueError("optional checks cannot be mandatory primary or hard-check dependencies")
    from ._check_evidence import _primary_closure
    for source in policy.sources:
        part = system.catalog.parts[source]
        if (_primary_closure(system, part.inputs) & set(policy.sources)) - {source}:
            raise ValueError("an optional check cannot depend on another optional check")
    available = [s for s in policy.sources if s in {step.part.name for step in prepared.flow.steps}]
    return q, policy, base, available


def _primary(system, question, flow, trace, values):
    records = {r.name: r for r in trace.records}
    hard = [(n, p) for n, p in system.catalog.parts.items() if p.kind == "check" and p.hard]
    return system._answer(question, flow, trace, values, records, hard, record_only=True)


def _initial(system, question, flow, context):
    records = {r.name: r for r in context.trace.records}
    hard = [(n, p) for n, p in system.catalog.parts.items() if p.kind == "check" and p.hard]
    return system._answer(question, flow, context.trace, context.vals, records, hard)


def _outcome(trace, name):
    record = next((r for r in trace.records if r.name == name and r.kind == "check"), None)
    if record is None:
        raise ValueError("purchased check has no execution record")
    if record.value is True:
        return "agree"
    if record.value is False:
        return "disagree"
    if record.value is MISSING and record.error:
        return "fail"
    raise ValueError("purchased check has no boolean outcome or recorded failure")


def _finish(system, p, context, loss, prior, available, decisions, selected, t0, store):
    q = p.questions[0]
    context.flow.skipped.update({name: "not purchased by check policy" for name in available if name not in selected})
    record = Record(0, "check_policy", "checks:" + q.name, {"request": context.trace.init_hash}, selected,
                    extra={"question": q.name, "policy": system._check_policy_fps[q.name], "loss": loss,
                           "prior": prior, "available": available, "decisions": decisions})
    _append(context.trace, record, len(context.trace.records))
    p.flow = context.flow
    return system._respond(p, context.trace, context.vals, t0, store, record_only=True)


def _action(policy, prior, outcomes, loss, available):
    action = policy.decide(prior, outcomes, loss, available=available)
    return action, {"kind": action.kind, "check": action.check, "value": action.value,
                    "p_answer": action.p_answer}


def run_sync(system, p, loss, t0, store, workers, early_exit):
    q, policy, base, available = _prepare(system, p, early_exit)
    context = _ExecutionContext()
    execute(system.catalog, base, p.state, workers=workers, costs=system.cost_book, known=p.known, context=context)
    primary = _initial(system, q, base, context)
    if primary.status != "ok":
        return _finish(system, p, context, loss, primary.confidence, available,
                       [{"kind": "primary_stopped"}], [], t0, store)
    outcomes, selected, decisions = {}, [], []
    for _ in range(len(policy.sources) + 1):
        action, entry = _action(policy, primary.confidence, outcomes, loss, available)
        decisions.append(entry)
        if action.kind != "check":
            return _finish(system, p, context, loss, primary.confidence, available, decisions, selected, t0, store)
        selected.append(action.check)
        flow = _flow(system, p, selected)
        execute(system.catalog, flow, p.state, workers=workers, costs=system.cost_book, known=p.known, context=context)
        outcomes[action.check] = _outcome(context.trace, action.check)
    raise RuntimeError("check policy did not stop")


async def run_async(system, p, loss, t0, store, timeout, early_exit, speculate):
    q, policy, base, available = _prepare(system, p, early_exit, speculate)
    context = _ExecutionContext()
    timeout = system.timeout if timeout is None else timeout
    await aexecute(system.catalog, base, p.state, costs=system.cost_book, known=p.known, context=context, timeout=timeout)
    primary = _initial(system, q, base, context)
    if primary.status != "ok":
        return _finish(system, p, context, loss, primary.confidence, available,
                       [{"kind": "primary_stopped"}], [], t0, store)
    outcomes, selected, decisions = {}, [], []
    for _ in range(len(policy.sources) + 1):
        action, entry = _action(policy, primary.confidence, outcomes, loss, available)
        decisions.append(entry)
        if action.kind != "check":
            return _finish(system, p, context, loss, primary.confidence, available, decisions, selected, t0, store)
        selected.append(action.check)
        await aexecute(system.catalog, _flow(system, p, selected), p.state, costs=system.cost_book,
                       known=p.known, context=context, timeout=timeout)
        outcomes[action.check] = _outcome(context.trace, action.check)
    raise RuntimeError("check policy did not stop")


def _record(trace, name):
    return next((r for r in trace.records if r.kind == "check_policy" and isinstance(r.extra, dict)
                 and r.extra.get("question") == name), None)


def recorded_flow(system, trace, names):
    records = [_record(trace, name) for name in names]
    if not any(records):
        return None
    if len(names) != 1 or records[0] is None:
        raise ValueError("adaptive trace requires one recorded question")
    p = SimpleNamespace(state=trace.init, questions=[system.questions[names[0]]])
    return _flow(system, p, records[0].value)


def verify(system, trace, record):
    name = record.extra["question"]
    policy = getattr(system, "_check_policies", {}).get(name)
    if policy is None or record.extra["policy"] != system._check_policy_fps[name]:
        raise ValueError("check policy changed or is unavailable")
    if record.inputs != {"request": trace.init_hash}:
        raise ValueError("check policy request binding changed")
    q = system.questions[name]
    p = SimpleNamespace(state=trace.init, questions=[q], flow=system._plan([q], trace.init.keys()))
    _, _, base, available = _prepare(system, p, True)
    prior = _primary(system, q, base, trace, {**trace.init, **{r.name: r.value for r in trace.records if r.value is not MISSING}})
    if prior.confidence != record.extra["prior"] or available != record.extra["available"]:
        raise ValueError("check policy prior or availability differs")
    if prior.status != "ok":
        if record.value != [] or record.extra["decisions"] != [{"kind": "primary_stopped"}]:
            raise ValueError("check policy must not override a stopped primary")
        return {"kind": "primary_stopped"}
    outcomes, selected, expected = {}, [], []
    for _ in range(len(policy.sources) + 1):
        action, entry = _action(policy, prior.confidence, outcomes, record.extra["loss"], available)
        expected.append(entry)
        if action.kind != "check":
            break
        selected.append(action.check)
        outcomes[action.check] = _outcome(trace, action.check)
    if selected != record.value or expected != record.extra["decisions"]:
        raise ValueError("check acquisition decisions differ from recorded evidence")
    actually_run = [r.name for r in trace.records if r.kind == "check" and r.name in policy.sources]
    if actually_run != selected:
        raise ValueError("unpurchased or reordered check execution")
    if any(r.step >= record.step for r in trace.records if r.kind == "check" and r.name in selected):
        raise ValueError("check policy result must follow its purchased checks")
    return expected[-1]


def apply(system, question, result, trace, flow):
    record = _record(trace, question.name)
    if record is None:
        return result
    action = verify(system, trace, record)
    expected_flow = recorded_flow(system, trace, [question.name])
    if (set(flow.per_question.get(question.name, ())) != set(expected_flow.per_question.get(question.name, ()))
            or flow.unresolved.get(question.name) != expected_flow.unresolved.get(question.name)):
        raise ValueError("acquisition flow differs from the recorded purchases and mandatory work")
    policy = system._check_policies[question.name]
    details = {"policy": record.extra["policy"], "loss": record.extra["loss"], "used": list(record.value),
               "spent": sum(policy.costs[k] for k in record.value), "action": action["kind"],
               "decisions": record.extra["decisions"], "decision_confidence": action.get("p_answer")}
    extra = {**(result.extra or {}), "check_acquisition": details}
    confidence = result.confidence
    if result.status == "ok" and action.get("p_answer") is not None:
        confidence = min(confidence, action["p_answer"])
    utility = confidence - record.extra["loss"] * (1 - confidence)
    if result.status == "ok" and (action["kind"] == "abstain" or utility <= 0):
        return dataclasses.replace(result, answer=None, status="abstain", guard="check_policy", confidence=confidence,
                                   why="check policy abstained: expected utility not positive", extra=extra)
    return dataclasses.replace(result, extra=extra, confidence=confidence)
