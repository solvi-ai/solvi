"""Explicit frozen check-likelihood models for scalar choice answers.

No fitting, independence inference or statistical certification at ask time.
"""
from __future__ import annotations

import dataclasses
import math

from .runtime import MISSING

__all__ = []


def freeze_models(questions, models):
    if models is None:
        return {}, {}
    if not isinstance(models, dict) or any(name not in questions for name in models):
        raise ValueError("check_models must map declared question names to fitted check combiners")
    if not models:
        return {}, {}
    from ..stat import CorrelationAwareCombiner, NaiveCombiner

    frozen, fingerprints = {}, {}
    for name, model in models.items():
        if type(model) not in (NaiveCombiner, CorrelationAwareCombiner):
            raise TypeError("check_models takes fitted NaiveCombiner or CorrelationAwareCombiner objects")
        snapshot = model.freeze()
        if not {"agree", "disagree"}.issubset(snapshot.outcomes):
            raise ValueError("check models require agree/disagree outcomes")
        frozen[name], fingerprints[name] = snapshot, snapshot.fingerprint()
    return frozen, fingerprints


def freeze_intervals(questions, models, intervals):
    """Copy opt-in conditional model interval settings; no coverage promise."""
    if intervals is None:
        return {}
    if not isinstance(intervals, dict) or any(name not in questions for name in intervals):
        raise ValueError("check_intervals must map declared question names to settings")
    defaults = {"level": .95, "n": 1000, "seed": 0, "mode": "report"}
    frozen = {}
    for name, settings in intervals.items():
        if name not in models:
            raise ValueError("check_intervals requires a configured check model")
        if not hasattr(models[name], "_log_concentrations"):
            raise ValueError("refit a legacy check model before requesting uncertainty")
        if not isinstance(settings, dict) or set(settings) - set(defaults):
            raise ValueError("check_intervals settings allow only level, n, seed, mode")
        config = {**defaults, **settings}
        level, n, seed = config["level"], config["n"], config["seed"]
        if (isinstance(level, bool) or not isinstance(level, (int, float))
                or not math.isfinite(level) or not 0 < level < 1):
            raise ValueError("check_intervals level must be finite and between zero and one")
        if type(n) is not int or n <= 0:
            raise ValueError("check_intervals n must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("check_intervals seed must be a nonnegative integer")
        if config["mode"] not in ("report", "lower"):
            raise ValueError("check_intervals mode must be report or lower")
        config["level"] = float(level)
        frozen[name] = config
    return frozen


def _primary_closure(system, inputs):
    pending, visited = list(inputs), set()
    while pending:
        fact = pending.pop()
        if fact in visited:
            continue
        visited.add(fact)
        part = system.catalog.parts.get(fact)
        if part is not None:
            pending.extend(i for producer in (part.alternatives or [part]) for i in producer.inputs)
    return visited


def _validate_sources(system, name, model, visited):
    if set(model.sources) & visited:
        raise ValueError("a modeled check must not already be read by the primary answer")
    for check in model.sources:
        part = system.catalog.parts.get(check)
        if part is None or part.kind != "check" or part.hard:
            raise ValueError("check_models sources must be registered soft checks")
        pending, seen = [part.shares_call_with], set()
        while pending:
            shared = pending.pop()
            if shared is None or shared in seen:
                continue
            if shared in visited or shared in (name, "answer:" + name):
                raise ValueError("a declared self-check cannot be independent answer evidence")
            seen.add(shared)
            upstream = system.catalog.parts.get(shared)
            if upstream is not None:
                pending.append(upstream.shares_call_with)
                for producer in upstream.alternatives or [upstream]:
                    # Catalog.alias transfers a value unchanged; its source is the same evidence.
                    if getattr(producer.func, "__qualname__", "").startswith("fact_alias."):
                        pending.extend(producer.inputs)


def validate(system):
    models = getattr(system, "_check_models", {})
    if not models:
        return
    if system.catalog.constraints or system._answer_refs():
        raise ValueError("check_models does not support joint constraints or answer-fact composition")
    for name, model in models.items():
        q = system.questions[name]
        if q.answer.kind not in ("choice", "yes_no"):
            raise ValueError("check_models requires a scalar choice or yes/no answer")
        if not set(model.sources).issubset(q.requires):
            raise ValueError("every modeled check must be explicitly listed in Question.requires")
        rule = system.catalog.rules.get(name)
        head = system.heads.get(name)
        inputs = rule.inputs if rule is not None else head.features if head is not None else ()
        _validate_sources(system, name, model, _primary_closure(system, inputs))


def apply(system, question, result, flow, records):
    model = getattr(system, "_check_models", {}).get(question.name)
    if model is None or result.status != "ok":
        return result
    primary = records.get("answer:" + question.name)
    if primary is not None:
        _validate_sources(system, question.name, model, _primary_closure(system, primary.inputs))
    outcomes = {}
    planned = set(flow.per_question.get(question.name, ()))
    for name in model.sources:
        record = records.get(name)
        if name not in planned or record is None or record.kind != "check":
            continue
        if record.value is True:
            outcomes[name] = "agree"
        elif record.value is False:
            outcomes[name] = "disagree"
        elif record.value is MISSING and record.error and "fail" in model.outcomes:
            outcomes[name] = "fail"
    if not outcomes:
        return result
    posterior = model.posterior(result.confidence, outcomes)
    evidence = {"model": system._check_model_fps[question.name], "prior": result.confidence,
                "posterior": posterior, "outcomes": outcomes, "domain": None}
    confidence = posterior
    config = getattr(system, "_check_intervals", {}).get(question.name)
    if config is not None:
        lower, upper = model.posterior_interval(
            result.confidence, outcomes, level=config["level"], n=config["n"], seed=config["seed"])
        evidence["uncertainty"] = {
            **config, "lower": lower, "upper": upper, "interpretation": "model_conditional",
            "fixed": ["pooled_prior", "prior", "correlation_weights"],
        }
        if config["mode"] == "lower":
            confidence = lower
    return dataclasses.replace(result, confidence=confidence,
                               why=result.why + "; fitted check likelihoods applied",
                               extra={**(result.extra or {}), "check_evidence": evidence})
