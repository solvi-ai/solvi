"""Finite decision-loss learning and explicitly supported one-step policy evaluation.

The small integer linear ranker is not a System One checkpoint. Outcome values
and logging propensities are host evidence, not inferred causal facts. Training
never grants permission: separate validation and explicit activation are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._common import WorkBudget, digest, identifier as _identifier, natural, portable


def identifier(value, name="identifier"):
    result = _identifier(value, name)
    if len(result) > 256:
        raise ValueError("identifier exceeds 256 characters")
    return result


def _integer(value):
    if type(value) is not int or abs(value) > 10**9:
        raise ValueError("bounded strict integer required")
    return value


def _bounded(values, maximum, budget, kind):
    result = []
    iterator = iter(values)
    for _ in range(maximum + 1):
        budget.charge(kind)
        try:
            value = next(iterator)
        except StopIteration:
            return tuple(result)
        if len(result) >= maximum:
            raise ValueError("finite collection exceeds limit")
        result.append(value)
    raise ValueError("finite collection exceeds limit")


def _preflight(value, budget, depth=0):
    # Charge and reject structure BEFORE canonical serialization allocates copies.
    budget.charge("decision_restore_structure")
    if depth > 32:
        raise ValueError("snapshot nesting exceeds bound")
    if type(value) is int:
        if value.bit_length() > 512:
            raise ValueError("snapshot integer exceeds bound")
        return
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        if len(value) > 1000000:
            raise ValueError("snapshot string exceeds bound")
        budget.charge("decision_restore_string", len(value) // 64)
        return
    if type(value) in (dict, list):
        if len(value) > 4096:
            raise ValueError("snapshot collection exceeds bound")
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError("snapshot string keys required")
                _preflight(key, budget, depth+1)
                _preflight(child, budget, depth+1)
        else:
            for child in value:
                _preflight(child, budget, depth+1)
        return
    raise ValueError("snapshot value is not portable")


def _ratio(value):
    if type(value) is int:
        return Fraction(_integer(value))
    if type(value) not in (tuple, list) or len(value) != 2:
        raise ValueError("rational requires [numerator, positive denominator]")
    numerator, denominator = map(_integer, value)
    if denominator <= 0:
        raise ValueError("positive denominator required")
    return Fraction(numerator, denominator)


def _wire(value):
    return None if value is None else [value.numerator, value.denominator]


@dataclass(frozen=True)
class DecisionObjective:
    reward_weight: int = 1
    cost_weight: int = 1

    def __post_init__(self):
        natural(self.reward_weight, "reward weight", positive=True)
        natural(self.cost_weight, "cost weight")
        _integer(self.reward_weight)
        _integer(self.cost_weight)

    def utility(self, reward, cost):
        return self.reward_weight * _integer(reward) - self.cost_weight * natural(cost, "action cost")


@dataclass(frozen=True)
class DecisionAction:
    name: str
    features: tuple[int, ...]
    cost: int = 0
    allowed: bool = True

    def __post_init__(self):
        identifier(self.name, "action")
        if type(self.features) not in (tuple, list) or not 1 <= len(self.features) <= 32:
            raise ValueError("1..32 finite features required")
        object.__setattr__(self, "features", tuple(_integer(v) for v in self.features))
        natural(self.cost, "action cost")
        _integer(self.cost)
        if type(self.allowed) is not bool:
            raise ValueError("strict feasibility flag required")

    def to_dict(self):
        return {"name": self.name, "features": list(self.features), "cost": self.cost, "allowed": self.allowed}


@dataclass(frozen=True)
class DecisionContext:
    context_id: str
    revision: str
    actions: tuple[DecisionAction, ...]

    def __post_init__(self):
        identifier(self.context_id, "context")
        identifier(self.revision, "revision")
        if type(self.actions) not in (tuple, list) or not 1 <= len(self.actions) <= 128:
            raise ValueError("1..128 finite actions required")
        object.__setattr__(self, "actions", tuple(self.actions))
        if any(type(a) is not DecisionAction for a in self.actions):
            raise ValueError("DecisionAction required")
        if len({a.name for a in self.actions}) != len(self.actions) or len({len(a.features) for a in self.actions}) != 1:
            raise ValueError("unique actions and one feature dimension required")

    def to_dict(self):
        return {"context_id": self.context_id, "revision": self.revision, "actions": [a.to_dict() for a in self.actions]}

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"context_id", "revision", "actions"}:
            raise ValueError("invalid context")
        return cls(value["context_id"], value["revision"], tuple(DecisionAction(**a) for a in value["actions"]))


@dataclass(frozen=True)
class OutcomeFeedback:
    event_id: str
    source: str
    context: DecisionContext
    outcomes: tuple[tuple[str, int], ...]
    split: str = "train"

    def __post_init__(self):
        identifier(self.event_id, "outcome event")
        identifier(self.source, "outcome source")
        if type(self.context) is not DecisionContext or self.split not in ("train", "validation", "evaluation"):
            raise ValueError("context and explicit feedback split required")
        if type(self.outcomes) not in (tuple, list) or len(self.outcomes) != len(self.context.actions):
            raise ValueError("full outcomes required; use LoggedFeedback for partial feedback")
        pairs = []
        for row in self.outcomes:
            if type(row) not in (tuple, list) or len(row) != 2:
                raise ValueError("action/reward pair required")
            pairs.append((identifier(row[0]), _integer(row[1])))
        if len({k for k, _ in pairs}) != len(pairs) or {k for k, _ in pairs} != {a.name for a in self.context.actions}:
            raise ValueError("exactly one outcome per available action required")
        object.__setattr__(self, "outcomes", tuple(pairs))

    def to_dict(self):
        return {"event_id": self.event_id, "source": self.source, "context": self.context.to_dict(),
                "outcomes": [list(p) for p in self.outcomes], "split": self.split}


@dataclass(frozen=True)
class DecisionPolicy:
    weights: tuple[int, ...]
    objective: DecisionObjective = DecisionObjective()

    def __post_init__(self):
        if type(self.weights) not in (tuple, list) or not 1 <= len(self.weights) <= 32:
            raise ValueError("bounded finite weights required")
        object.__setattr__(self, "weights", tuple(_integer(v) for v in self.weights))
        if type(self.objective) is not DecisionObjective:
            raise ValueError("DecisionObjective required")

    def predict(self, action, budget):
        if type(action) is not DecisionAction or len(action.features) != len(self.weights):
            raise ValueError("feature dimension differs")
        budget.charge("decision_predict", 1 + len(self.weights))
        return sum(w * x for w, x in zip(self.weights, action.features))

    def choose(self, context, budget):
        if type(context) is not DecisionContext:
            raise ValueError("DecisionContext required")
        scored = []
        for action in context.actions:
            budget.charge("decision_feasibility")
            if action.allowed:
                reward = self.predict(action, budget)
                # Internal predictions may exceed individual feature bounds.
                score = self.objective.reward_weight * reward - self.objective.cost_weight * action.cost
                scored.append((score, action.name))
        if not scored:
            return None
        budget.charge("decision_argmax", len(scored))
        return min(scored, key=lambda p: (-p[0], p[1]))[1]

    def to_dict(self):
        return {"weights": list(self.weights), "objective": {"reward_weight": self.objective.reward_weight,
                                                              "cost_weight": self.objective.cost_weight}}

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or set(data) != {"weights", "objective"}:
            raise ValueError("invalid decision policy")
        return cls(tuple(data["weights"]), DecisionObjective(**data["objective"]))

    @property
    def fingerprint(self):
        return digest(self.to_dict())


def decision_regret(policy, feedback, budget):
    """Actual finite decision loss, not prediction error. No feasible action -> None."""
    selected = policy.choose(feedback.context, budget)
    if selected is None:
        return None
    rewards = dict(feedback.outcomes)
    utilities = {}
    for action in feedback.context.actions:
        budget.charge("decision_outcome")
        if action.allowed:
            utilities[action.name] = policy.objective.utility(rewards[action.name], action.cost)
    return max(utilities.values()) - utilities[selected]


def _rows(values, split, budget):
    rows = _bounded(values, 4096, budget, "decision_feedback")
    if not rows or any(type(r) is not OutcomeFeedback or r.split != split for r in rows):
        raise ValueError("nonempty feedback with matching split required")
    if len({r.event_id for r in rows}) != len(rows) or len({r.context.context_id for r in rows}) != len(rows):
        raise ValueError("duplicate event/context in feedback")
    return rows


@dataclass(frozen=True)
class DecisionCandidate:
    policy: DecisionPolicy
    training_events: tuple[str, ...]
    training_contexts: tuple[str, ...]
    loss: str
    train_loss: int

    @property
    def fingerprint(self):
        return digest({"policy": self.policy.to_dict(), "events": list(self.training_events),
                       "contexts": list(self.training_contexts), "loss": self.loss, "train_loss": self.train_loss})


_DEFAULT_OBJECTIVE = DecisionObjective()


def fit_decision_policy(feedback, budget, *, objective=_DEFAULT_OBJECTIVE, loss="decision", epochs=4, steps=(2, 1)):
    """Paid deterministic coordinate descent for an integer linear reward ranker.

    decision minimizes empirical downstream regret; prediction minimizes squared
    reward error, using the same model class. Neither is an SPO+ implementation.
    There is no automatic validation, promotion or generalization guarantee.
    """
    if loss not in ("decision", "prediction"):
        raise ValueError("unknown training objective")
    natural(epochs, "epochs", positive=True)
    if epochs > 32 or type(steps) not in (tuple, list) or not 1 <= len(steps) <= 16:
        raise ValueError("bounded training schedule required")
    for step in steps:
        natural(step, "coordinate step", positive=True)
        if step > 10000:
            raise ValueError("coordinate step too large")
    rows = _rows(feedback, "train", budget)
    dimension = len(rows[0].context.actions[0].features)
    if any(len(r.context.actions[0].features) != dimension for r in rows):
        raise ValueError("training dimensions differ")

    def evaluate(weights):
        budget.charge("decision_train_candidate")
        policy = DecisionPolicy(tuple(weights), objective)
        total = 0
        for row in rows:
            if loss == "decision":
                regret = decision_regret(policy, row, budget)
                if regret is None:
                    raise ValueError("training context has no feasible action")
                total += regret
            else:
                rewards = dict(row.outcomes)
                for action in row.context.actions:
                    budget.charge("decision_train_loss")
                    if action.allowed:
                        error = policy.predict(action, budget) - rewards[action.name]
                        total += error * error
        return total

    weights = [0] * dimension
    current = evaluate(weights)
    for step in steps:
        for _epoch in range(epochs):
            improved = False
            for coordinate in range(dimension):
                options = []
                for direction in (-1, 1):
                    proposed = weights.copy()
                    proposed[coordinate] += direction * step
                    options.append((evaluate(proposed), tuple(proposed)))
                score, proposal = min(options)
                if score < current:
                    weights, current, improved = list(proposal), score, True
            if not improved:
                break
    policy = DecisionPolicy(tuple(weights), objective)
    return DecisionCandidate(policy, tuple(r.event_id for r in rows), tuple(r.context.context_id for r in rows), loss, current)


class DecisionAuthority:
    """Host-owned current checkpoint/epoch; persist outside untrusted snapshots.

    One authority belongs to one learner. It is not a remote authentication
    service. Losing this trust anchor cannot restore active admission safely.
    """

    def __init__(self):
        self._epoch = 0
        self._checkpoint = None
        self._claimed = False
        self._ledger: dict | None = None
        self._budget: WorkBudget | None = None

    def export_trusted_state(self):
        """Store this anchor in a trusted host channel, separately from payloads."""
        if self._budget is not None:
            self._budget.charge("decision_authority_export")
            self._ledger = self._budget.snapshot()
        return portable({"v": 1, "epoch": self._epoch, "checkpoint": self._checkpoint,
                         "claimed": self._claimed, "ledger": self._ledger})

    @classmethod
    def from_trusted_state(cls, state, *, budget):
        """The host, not a self-reported snapshot, authorizes this trust anchor."""
        entry_ledger: dict = budget.snapshot()
        _preflight(state, budget)
        s = portable(state)
        if type(s) is not dict or set(s) != {"v", "epoch", "checkpoint", "claimed", "ledger"} or type(s["v"]) is not int or s["v"] != 1:
            raise ValueError("invalid decision authority anchor")
        natural(s["epoch"])
        if type(s["claimed"]) is not bool or (s["checkpoint"] is not None and (type(s["checkpoint"]) is not str or len(s["checkpoint"]) != 64)):
            raise ValueError("invalid decision authority fields")
        if s["ledger"] is not None:
            WorkBudget.restore(s["ledger"])
            floor = s["ledger"]
            if entry_ledger["limit"] != floor["limit"] or entry_ledger["spent"] < floor["spent"] or any(entry_ledger["counts"].get(k, 0) < v for k, v in floor["counts"].items()):
                raise ValueError("authority import refunds ledger")
        result = cls()
        result._epoch, result._checkpoint = s["epoch"], s["checkpoint"]
        result._claimed, result._ledger = s["claimed"], budget.snapshot()
        result._budget = budget
        return result


class DecisionLearner:
    """Explicit finite validation gate. Admission is local empirical evidence only."""

    def __init__(self, budget, *, authority=None):
        if type(budget) is not WorkBudget:
            raise ValueError("WorkBudget required")
        self._authority = DecisionAuthority() if authority is None else authority
        if type(self.authority) is not DecisionAuthority or self.authority._claimed:
            raise ValueError("fresh DecisionAuthority required; use restore for transfer")
        self.authority._claimed = True
        self.authority._budget = budget
        self._epoch = self.authority._epoch
        self._budget = budget
        self._candidates = {}
        self._admissions = {}
        self._active = None
        self._revoked = set()

    @property
    def authority(self):
        return self._authority

    @property
    def budget(self):
        return self._budget

    def _guard(self):
        if self._epoch != self.authority._epoch:
            raise ValueError("stale decision learner")

    def _advance(self):
        self._guard()
        self.authority._epoch += 1
        self._epoch = self.authority._epoch
        self.authority._checkpoint = None

    def train(self, feedback, **options):
        self._guard()
        if len(self._candidates) >= 128:
            raise ValueError("candidate capacity reached")
        candidate = fit_decision_policy(feedback, self.budget, **options)
        self.budget.charge("decision_register")
        self._candidates[candidate.fingerprint] = candidate
        self._advance()
        return candidate

    def admit(self, candidate, validation, baseline):
        self._guard()
        if len(self._admissions) >= 128:
            raise ValueError("admission capacity reached")
        if type(candidate) is not DecisionCandidate or self._candidates.get(candidate.fingerprint) != candidate:
            raise ValueError("candidate must be trained by this learner")
        if type(baseline) is not DecisionPolicy or baseline.objective != candidate.policy.objective:
            raise ValueError("baseline objective differs")
        rows = _rows(validation, "validation", self.budget)
        if set(candidate.training_events) & {r.event_id for r in rows} or set(candidate.training_contexts) & {r.context.context_id for r in rows}:
            raise ValueError("validation overlaps training")
        a = b = 0
        for row in rows:
            ca, cb = decision_regret(candidate.policy, row, self.budget), decision_regret(baseline, row, self.budget)
            if ca is None or cb is None:
                raise ValueError("validation requires feasible actions")
            a, b = a + ca, b + cb
        record = {"candidate": candidate.fingerprint, "validation": [r.to_dict() for r in rows],
                  "baseline": baseline.to_dict(), "regret": a, "baseline_regret": b, "accepted": a <= b}
        self.budget.charge("decision_admission")
        receipt = digest(record)
        self._admissions[receipt] = portable(record)
        self._advance()
        return {"receipt": receipt, **portable(record)}

    def activate(self, receipt):
        self._guard()
        identifier(receipt, "admission receipt")
        self.budget.charge("decision_activation")
        row = self._admissions.get(receipt)
        if row is None or not row["accepted"] or row["candidate"] in self._revoked:
            raise ValueError("current accepted admission required")
        self._active = row["candidate"]
        self._advance()

    def revoke(self, candidate_id):
        self._guard()
        if candidate_id not in self._candidates:
            raise ValueError("unknown candidate")
        self.budget.charge("decision_revoke")
        self._revoked.add(candidate_id)
        self._advance()
        if self._active == candidate_id:
            self._active = None

    def choose(self, context):
        self._guard()
        self.budget.charge("decision_active")
        if self._active is None or self._active in self._revoked:
            return None
        return self._candidates[self._active].policy.choose(context, self.budget)

    def snapshot(self):
        self._guard()
        self.budget.charge("decision_snapshot", 1 + len(self._candidates) + len(self._admissions))
        candidates = {k: {"policy": v.policy.to_dict(), "training_events": list(v.training_events),
                          "training_contexts": list(v.training_contexts), "loss": v.loss, "train_loss": v.train_loss}
                      for k, v in self._candidates.items()}
        body = {"v": 1, "epoch": self._epoch, "candidates": candidates, "admissions": portable(self._admissions),
                "active": self._active, "revoked": sorted(self._revoked)}
        result = {**body, "digest": digest(body)}
        self._guard()
        self.authority._checkpoint = result["digest"]
        self.authority._ledger = self.budget.snapshot()
        return result

    @classmethod
    def restore(cls, snapshot, budget, *, authority):
        """Host supplies the current authority capability; a hash is not authority.

        Restore with the SAME restored ledger, not a fresh allocation. Validation
        outcomes are retained host assertions; no external oracle is called.
        """
        entry_ledger: dict = budget.snapshot()
        current_ledger = authority._budget.snapshot() if type(authority) is DecisionAuthority and authority._budget is not None else None
        budget.charge("decision_restore")
        _preflight(snapshot, budget)
        s = portable(snapshot)
        fields = {"v", "epoch", "candidates", "admissions", "active", "revoked", "digest"}
        if type(s) is not dict or set(s) != fields or type(s["v"]) is not int or s["v"] != 1:
            raise ValueError("invalid learner snapshot")
        if type(authority) is not DecisionAuthority or not authority._claimed or type(s["epoch"]) is not int or s["epoch"] != authority._epoch or authority._checkpoint != s["digest"]:
            raise ValueError("stale or untrusted decision checkpoint")
        if authority._budget is not None and authority._budget is not budget:
            raise ValueError("live host ledger identity required; refund rejected")
        anchor: dict | None = current_ledger if current_ledger is not None else authority._ledger
        if anchor is None or entry_ledger["limit"] != anchor["limit"] or entry_ledger["spent"] < anchor["spent"] or any(entry_ledger["counts"].get(k, 0) < v for k, v in anchor["counts"].items()):
            raise ValueError("restored ledger refunds work or changes budget")
        if digest({k: v for k, v in s.items() if k != "digest"}) != s["digest"]:
            raise ValueError("corrupt learner snapshot")
        if type(s["candidates"]) is not dict or type(s["admissions"]) is not dict or len(s["candidates"]) > 128 or len(s["admissions"]) > 128:
            raise ValueError("bounded learner snapshot required")
        result = cls(budget)
        for key, row in s["candidates"].items():
            budget.charge("decision_restore")
            if type(row) is not dict or set(row) != {"policy", "training_events", "training_contexts", "loss", "train_loss"}:
                raise ValueError("invalid candidate")
            if row["loss"] not in ("decision", "prediction") or type(row["training_events"]) is not list or type(row["training_contexts"]) is not list:
                raise ValueError("invalid training metadata")
            natural(row["train_loss"])
            for value in row["training_events"] + row["training_contexts"]:
                identifier(value)
            c = DecisionCandidate(DecisionPolicy.from_dict(row["policy"]), tuple(row["training_events"]),
                                  tuple(row["training_contexts"]), row["loss"], row["train_loss"])
            if c.fingerprint != key:
                raise ValueError("candidate identity differs")
            result._candidates[key] = c
        for receipt, row in s["admissions"].items():
            budget.charge("decision_restore")
            if digest(row) != receipt or row.get("candidate") not in result._candidates:
                raise ValueError("untrusted admission receipt")
            # Recompute validation rather than trusting declared scores/accepted.
            validation = [OutcomeFeedback(r["event_id"], r["source"], DecisionContext.from_dict(r["context"]),
                                          tuple(tuple(p) for p in r["outcomes"]), r["split"]) for r in row["validation"]]
            checked = result.admit(result._candidates[row["candidate"]], validation, DecisionPolicy.from_dict(row["baseline"]))
            if checked["receipt"] != receipt:
                raise ValueError("admission evidence differs")
        if type(s["revoked"]) is not list or len(set(s["revoked"])) != len(s["revoked"]) or not set(s["revoked"]) <= set(result._candidates):
            raise ValueError("invalid revocations")
        result._revoked = set(s["revoked"])
        active = s["active"]
        if active is not None:
            if active in result._revoked or not any(r["accepted"] and r["candidate"] == active for r in result._admissions.values()):
                raise ValueError("active policy lacks current admission")
            result._active = active
        # Transfer sole ownership only after a complete paid validation.
        if authority._epoch != s["epoch"] or authority._checkpoint != s["digest"]:
            raise ValueError("decision authority changed during restore")
        authority._epoch += 1
        authority._checkpoint = None
        result._authority = authority
        authority._budget = budget
        result._epoch = authority._epoch
        return result


@dataclass(frozen=True)
class LoggedFeedback:
    event_id: str
    context: DecisionContext
    action: str
    reward: int
    logging: tuple[tuple[str, tuple[int, int]], ...]
    source: str
    propensity_known: bool = True

    def __post_init__(self):
        identifier(self.event_id)
        identifier(self.source)
        _integer(self.reward)
        if type(self.context) is not DecisionContext or type(self.propensity_known) is not bool:
            raise ValueError("typed logged feedback required")
        if type(self.logging) not in (tuple, list) or len(self.logging) != len(self.context.actions):
            raise ValueError("complete logging distribution required")
        pairs = []
        for row in self.logging:
            if type(row) not in (tuple, list) or len(row) != 2:
                raise ValueError("logging probability pair required")
            p = _ratio(row[1])
            if not 0 <= p <= 1:
                raise ValueError("probability outside [0,1]")
            pairs.append((identifier(row[0]), (p.numerator, p.denominator)))
        probabilities = {a: Fraction(*p) for a, p in pairs}
        if len(probabilities) != len(pairs) or set(probabilities) != {a.name for a in self.context.actions} or sum(probabilities.values()) != 1:
            raise ValueError("normalized complete logging distribution required")
        if self.action not in probabilities or probabilities[self.action] <= 0:
            raise ValueError("logged action must have positive probability")
        if any(not a.allowed and probabilities[a.name] for a in self.context.actions):
            raise ValueError("logging distribution selects infeasible action")
        object.__setattr__(self, "logging", tuple(pairs))


@dataclass(frozen=True)
class PolicyValueEstimate:
    status: str
    ips: Fraction | None
    snips: Fraction | None
    dr: Fraction | None
    effective_sample_size: Fraction
    samples: int
    max_weight: Fraction
    reason: str

    def to_dict(self):
        return {"status": self.status, "ips": _wire(self.ips), "snips": _wire(self.snips), "dr": _wire(self.dr),
                "effective_sample_size": _wire(self.effective_sample_size), "samples": self.samples,
                "max_weight": _wire(self.max_weight), "reason": self.reason}


def evaluate_logged_policy(feedback, target, budget, *, objective=_DEFAULT_OBJECTIVE, reward_model=None, minimum_ess=2):
    """IPS/SNIPS/DR for host-supplied fixed context→action distributions.

    No clipping or invented propensities. Positivity is checked per recorded
    context; that is not evidence of population coverage or no confounding.
    DR requires an independently fitted/fixed reward model; using logged labels
    to fit it in this function is forbidden. Estimates are not confidence bounds.
    """
    natural(minimum_ess, "minimum ESS", positive=True)
    rows = _bounded(feedback, 4096, budget, "policy_evaluation")
    if not rows or any(type(r) is not LoggedFeedback for r in rows) or len({r.event_id for r in rows}) != len(rows):
        raise ValueError("nonempty uniquely identified logged feedback required")
    if type(target) is not dict or len(target) > 4096:
        raise ValueError("bounded target distribution map required")
    if reward_model is not None and type(reward_model) is not dict:
        raise ValueError("fixed reward model mapping required")
    total = normalizer = squared = dr_total = maximum = Fraction(0)
    for row in rows:
        budget.charge("policy_evaluation", 1 + len(row.context.actions))
        if not row.propensity_known:
            return PolicyValueEstimate("unidentified", None, None, None, Fraction(0), len(rows), maximum, "logging propensity is unknown")
        distribution = target.get(row.context.fingerprint)
        if type(distribution) is not dict or set(distribution) != {a.name for a in row.context.actions}:
            raise ValueError("exact target actions for every context fingerprint required")
        q = {k: _ratio(v) for k, v in distribution.items()}
        if any(not 0 <= v <= 1 for v in q.values()) or sum(q.values()) != 1:
            raise ValueError("normalized target distribution required")
        logger = {k: Fraction(*p) for k, p in row.logging}
        if any(q[a.name] and (not a.allowed or not logger[a.name]) for a in row.context.actions):
            return PolicyValueEstimate("unsupported", None, None, None, Fraction(0), len(rows), maximum, "target lacks logging support or is infeasible")
        weight = q[row.action] / logger[row.action]
        action = next(a for a in row.context.actions if a.name == row.action)
        reward = objective.utility(row.reward, action.cost)
        total += weight * reward
        normalizer += weight
        squared += weight * weight
        maximum = max(maximum, weight)
        if reward_model is not None:
            model = reward_model.get(row.context.fingerprint)
            if type(model) is not dict or set(model) != set(q):
                raise ValueError("fixed reward model must cover exact actions")
            predicted = {a.name: objective.utility(_integer(model[a.name]), a.cost) for a in row.context.actions}
            dr_total += sum(q[k] * predicted[k] for k in q) + weight * (reward - predicted[row.action])
    ess = normalizer * normalizer / squared if squared else Fraction(0)
    if not normalizer or ess < minimum_ess:
        return PolicyValueEstimate("insufficient", None, None, None, ess, len(rows), maximum, "effective sample size below declared minimum")
    return PolicyValueEstimate("estimated", total / len(rows), total / normalizer,
                               dr_total / len(rows) if reward_model is not None else None, ess, len(rows), maximum,
                               "conditional on supplied propensities, support, fixed target and valid outcome evidence; no causal or confidence guarantee")


__all__ = ["DecisionObjective", "DecisionAction", "DecisionContext", "OutcomeFeedback", "DecisionPolicy", "DecisionCandidate",
           "DecisionLearner", "DecisionAuthority", "fit_decision_policy", "decision_regret", "LoggedFeedback", "PolicyValueEstimate", "evaluate_logged_policy"]
