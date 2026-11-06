"""Bounded counterexample-guided learning; certificates cover only the declared domain.

The host owns the oracle and allowed observation inputs. Transfer inputs/labels
must stay outside both. A finite committee is an incomplete heuristic. Durable
sessions require ExperienceJournal; journal=None is explicitly volatile.
"""
from __future__ import annotations

import random
from typing import Any

from ._common import BudgetStop, WorkBudget, canonical, digest, identifier, natural, portable
from .domain_checks import DomainReceipt, check_program
from .programs import Program, synthesize


class _InterruptedRead(Exception):
    pass


class RefinementSession:
    """Repeated proposal, optional paid observation, finite check and refinement.

    Use step/run, then inspect result. A verified result is not a registry entry.
    Registry.admit_checked must independently authorize later checked application.
    Reopen with restore(journal=..., grammar=..., domain=..., oracle=..., options=...).
    Ambiguous reads require host oracle.reconcile(request_id); they are never retried.
    """

    def __init__(self, grammar, domain, oracle, options, budget, *, session_id,
                 initial=(), mode="active", seed=0, max_rounds=6, committee_size=3,
                 max_observations=3, journal=None):
        identifier(session_id, "session_id")
        if mode not in ("none", "fixed", "random", "active"):
            raise ValueError("invalid refinement mode")
        for name, value in (("seed", seed), ("max_observations", max_observations)):
            natural(value, name)
        natural(max_rounds, "max_rounds", positive=True)
        natural(committee_size, "committee_size", positive=True)
        if committee_size > 32 or max_rounds > 64:
            raise ValueError("refinement exceeds bounded caps")
        self.grammar, self.domain, self.oracle, self.journal = grammar, domain, oracle, journal
        self.options = tuple(options)
        if len(self.options) > 256 or len({o.name for o in self.options}) != len(self.options):
            raise ValueError("invalid observation option count or duplicate name")
        points = set()
        for option in self.options:
            point = option.inputs
            if set(point) != set(grammar.inputs):
                raise ValueError("observation schema mismatch")
            if any(type(point[n]) is not (int if t == "int" else bool) or
                   (type(point[n]) is int and abs(point[n]) > grammar.max_abs)
                   for n, t in zip(domain.input_names, domain.input_types)):
                raise ValueError("observation type or bounds mismatch")
            key = canonical(point)
            if key in points:
                raise ValueError("duplicate observation input")
            points.add(key)
        initial = list(initial)
        if len(set(initial)) != len(initial) or not set(initial) <= {o.name for o in self.options}:
            raise ValueError("invalid initial observation names")
        if grammar.fingerprint != domain.grammar_fingerprint:
            raise ValueError("domain grammar mismatch")
        if journal is not None and journal.records():
            raise ValueError("existing journal requires restore")
        self.config: dict[str, Any] = dict(session_id=session_id, grammar=grammar.fingerprint, domain=domain.fingerprint,
                           options=[dict(name=o.name, inputs=o.inputs, cost=o.cost) for o in self.options],
                           initial=initial, mode=mode, seed=seed, max_rounds=max_rounds,
                           committee_size=committee_size, max_observations=max_observations)
        self.budget = budget
        self.state: dict[str, Any] = dict(setup_paid=False, v=1, config=self.config, phase="initial", status="running", reason="",
                          examples=[], used=[], rounds=0, observations=0, candidate=None,
                          committee=[], pending=None, receipts={}, checks=[], rejected=[], result=None)
        self._failed = False
        self._bind()
        self._save()

    def _bind(self):
        # Preserve a caller sink; failures after a debit poison this writer.
        sink = self.budget.on_charge

        def debit(snapshot):
            try:
                if sink is not None:
                    sink(snapshot)
                if self.journal is not None:
                    self.journal.debit(snapshot)
            except BaseException:
                self._failed = True
                raise

        self.budget.on_charge = debit

    def _save(self):
        if self._failed:
            raise RuntimeError("failed durable writer requires restore")
        self.state["ledger"] = self.budget.snapshot()
        if self.journal is not None:
            try:
                self.journal.checkpoint(self.state)
            except BaseException:
                self._failed = True
                raise

    def snapshot(self):
        self._save()
        return portable(self.state)

    @classmethod
    def restore(cls, *, journal, grammar, domain, oracle, options):
        records = journal.records()
        state = None
        for event in records:
            if event["kind"] == "checkpoint":
                state = portable(event["payload"]["state"])
            elif event["kind"] == "debit" and state is not None:
                state["ledger"] = portable(event["payload"]["ledger"])
        if state is None or state.get("v") != 1:
            raise ValueError("missing refinement checkpoint")
        config = state["config"]
        supplied = [dict(name=o.name, inputs=o.inputs, cost=o.cost) for o in options]
        if (config["grammar"] != grammar.fingerprint or config["domain"] != domain.fingerprint
                or supplied != config["options"]):
            raise ValueError("refinement recovery contract changed")
        obj = object.__new__(cls)
        obj.grammar, obj.domain, obj.oracle, obj.journal = grammar, domain, oracle, journal
        obj.options, obj.config, obj.state = tuple(options), config, state
        obj.budget = WorkBudget.restore(state["ledger"])
        obj._failed = False
        obj._bind()
        return obj

    def _bound(self):
        return (self.oracle.source_id == self.domain.source_id and
                self.oracle.source_revision == self.domain.source_revision and
                getattr(self.oracle, "available", True) is True)

    def _receipt(self, value, pending):
        if (type(value) is not DomainReceipt or value.request_id != pending["request_id"] or
                value.source_id != self.domain.source_id or
                value.source_revision != self.domain.source_revision or
                value.input_digest != digest(pending["point"]) or
                type(value.event_id) is not str or not value.event_id or
                type(value.expected) is not (int if self.domain.result_type == "int" else bool) or
                (type(value.expected) is int and abs(value.expected) > self.grammar.max_abs)):
            raise ValueError("invalid refinement oracle receipt")
        if not self._bound():
            raise ValueError("oracle source changed during read")
        for old in self.state["receipts"].values():
            if old["event_id"] == value.event_id and old != value.to_dict():
                raise ValueError("oracle event identity collision")
        return value

    def _accept_pending(self, receipt):
        pending = self.state["pending"]
        receipt = self._receipt(receipt, pending)
        self.state["receipts"][pending["request_id"]] = receipt.to_dict()
        self.state["pending"] = None
        self._save()
        return receipt

    def _read(self, request_id, point, purpose):
        old = self.state["receipts"].get(request_id)
        if old is not None:
            return self._receipt(DomainReceipt(**old), dict(request_id=request_id, point=point))
        pending = dict(request_id=request_id, point=portable(point), purpose=purpose,
                       round=self.state["rounds"], candidate=self.state["candidate"],
                       grammar=self.grammar.fingerprint, domain=self.domain.fingerprint)
        if self.state["pending"] is not None:
            raise _InterruptedRead("unsettled external read")
        self.state["pending"] = pending
        self._save()  # debit already durable; intent precedes host read
        try:
            receipt = self.oracle.read(request_id, self.domain.source_revision, portable(point))
        except Exception as error:
            raise _InterruptedRead("external read outcome unknown") from error
        return self._accept_pending(receipt)

    def reconcile(self):
        """Settle an ambiguous read without reissuing it or resetting its debit."""
        if self._failed:
            raise RuntimeError("restore required after writer failure")
        pending = self.state["pending"]
        if pending is None:
            return True
        if not self._bound():
            self._finish("unknown", "source_changed")
            return False
        method = getattr(self.oracle, "reconcile", None)
        receipt = method(pending["request_id"]) if method is not None else None
        if receipt is None:
            self.state["status"], self.state["reason"] = "blocked", "unresolved_read"
            self._save()
            return False
        self._accept_pending(receipt)
        self.state["status"], self.state["reason"] = "running", ""
        self._save()
        return True

    def _add(self, point, receipt, origin):
        for row in self.state["examples"]:
            if canonical(row["inputs"]) == canonical(point):
                if type(row["expected"]) is not type(receipt.expected) or row["expected"] != receipt.expected:
                    raise ValueError("conflicting oracle labels")
                return False
        self.state["examples"].append(dict(inputs=portable(point), expected=receipt.expected,
                                          receipt=receipt.to_dict(), origin=origin))
        return True

    def _observe(self, option, origin):
        key = f'{self.config["session_id"]}:observation:{option.name}'
        if key not in self.state["receipts"]:
            self.budget.charge("refinement_observation", option.cost)
        self.state["phase"] = "read_observation"
        self.state["selection"] = dict(name=option.name, origin=origin)
        receipt = self._read(key, option.inputs, origin)
        self._add(option.inputs, receipt, origin)
        self.state["used"].append(option.name)
        if origin == "observation":
            self.state["observations"] += 1
        self.state["phase"] = "search"
        self._save()

    def _finish(self, status, reason):
        self.state["status"], self.state["reason"] = status, reason
        self.state["phase"] = "done"
        self.state["result"] = dict(status=status, reason=reason, program=self.state["candidate"],
                                    domain=self.domain.fingerprint, rounds=self.state["rounds"])
        self._save()

    def step(self):
        if self._failed:
            raise RuntimeError("failed writer requires restore")
        if self.state["phase"] == "done":
            return self.state["status"]
        if not self._bound():
            self._finish("unknown", "source_changed")
            return self.state["status"]
        if self.state["pending"] is not None and not self.reconcile():
            return self.state["status"]
        try:
            self._step()
        except _InterruptedRead:
            self.state["status"], self.state["reason"] = "blocked", "unresolved_read"
            self._save()
        except BudgetStop:
            if self._failed:
                raise
            self._finish("unknown", "budget")
        return self.state["status"]

    def _step(self):
        s, c = self.state, self.config
        if not s["setup_paid"]:
            self.budget.charge("refinement_setup", len(self.options) * (len(self.grammar.inputs) + 1) + 1)
            s["setup_paid"] = True
            self._save()
        if s["phase"] == "read_observation":
            selection = s["selection"]
            self._observe(next(o for o in self.options if o.name == selection["name"]), selection["origin"])
            return
        initial = [name for name in c["initial"] if name not in s["used"]]
        if initial:
            self._observe(next(o for o in self.options if o.name == initial[0]), "initial")
            return
        if s["phase"] in ("initial", "search"):
            if not s["examples"]:
                self._finish("unknown", "no_examples")
                return
            if s["rounds"] >= c["max_rounds"]:
                self._finish("unknown", "round_limit")
                return
            rows = [(row["inputs"], row["expected"]) for row in s["examples"]]
            committee = []
            size = c["committee_size"] if c["mode"] == "active" else 1
            for _ in range(size):
                search = synthesize(rows, self.grammar, self.budget,
                                    exclude=[p.fingerprint for p in committee])
                if search.candidate is None:
                    if search.reason == "budget_exhausted":
                        self._finish("unknown", "budget")
                        return
                    break
                committee.append(search.candidate)
            if not committee:
                self._finish("unknown", search.reason)
                return
            s["rounds"] += 1
            s["candidate"] = committee[0].to_dict()
            s["committee"] = [p.to_dict() for p in committee]
            s["phase"] = "observe"
            self._save()
        elif s["phase"] == "observe":
            used_points = {canonical(row["inputs"]) for row in s["examples"]}
            available = [o for o in self.options if o.name not in s["used"] and canonical(o.inputs) not in used_points]
            option = None
            if available and s["observations"] < c["max_observations"]:
                if c["mode"] == "fixed":
                    self.budget.charge("refinement_selection", len(available))
                    option = available[0]
                elif c["mode"] == "random":
                    self.budget.charge("refinement_selection", len(available))
                    option = random.Random(c["seed"] + s["observations"]).choice(available)
                elif c["mode"] == "active":
                    from .observations import select_observation
                    choice = select_observation([Program.from_dict(p) for p in s["committee"]],
                                                available, self.grammar, self.budget)
                    if choice.reason == "budget":
                        self._finish("unknown", "budget")
                        return
                    option = next((o for o in available if o.name == choice.name), None)
            if option is not None:
                self._observe(option, "observation")
            else:
                s["phase"] = "check"
                self._save()
        elif s["phase"] == "check":
            owner = self
            prefix = f'{c["session_id"]}:round:{s["rounds"]}'

            class VerificationBudget:
                index = 0

                @property
                def spent(self):
                    return owner.budget.spent

                def charge(self, kind, units=1):
                    if kind == "domain_oracle":
                        request = f"{prefix}:{self.index}"
                        self.index += 1
                        if request in s["receipts"]:
                            return
                    owner.budget.charge(kind, units)

            class OracleView:
                source_id = owner.domain.source_id
                source_revision = owner.domain.source_revision

                @property
                def available(self):
                    return owner._bound()

                def read(self, request_id, source_revision, point):
                    return owner._read(request_id, point, "verification")

            # check_program turns ordinary oracle failures into UNKNOWN. An
            # unresolved intent is retained; the next step can only reconcile.
            outcome = check_program(self.domain, Program.from_dict(s["candidate"]), self.grammar,
                                    OracleView(), VerificationBudget(), request_prefix=prefix)
            if s["pending"] is not None:
                s["status"], s["reason"] = "blocked", "unresolved_read"
                self._save()
                return
            s["checks"].append(outcome.to_dict())
            if outcome.status == "verified":
                self._finish("verified", "complete_verification_domain")
            elif outcome.reason == "counterexample":
                witness = dict(outcome.witness)
                receipt = outcome.receipts[-1]
                point = self.domain.point_dict(witness["index"])
                if not self._add(point, receipt, "counterexample"):
                    self._finish("unknown", "repeated_counterexample")
                    return
                s["rejected"].append(s["candidate"])
                s["candidate"], s["committee"], s["phase"] = None, [], "search"
                self._save()
            else:
                self._finish("unknown", outcome.reason)

    def run(self, max_steps=1000):
        natural(max_steps, "max_steps", positive=True)
        for _ in range(max_steps):
            if self.state["phase"] == "done" or self.step() == "blocked":
                break
        return portable(self.state["result"] or dict(status=self.state["status"], reason=self.state["reason"]))


__all__ = ["RefinementSession"]
