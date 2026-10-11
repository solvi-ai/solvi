"""Learning in a world described by code (experimental): System 2 plans on an exact model of the world — a copy of the
environment — and only plans the code verified are compiled into System 1 skills, each used alone only while its
outcomes keep a certificate. No label from a person anywhere: the world's code is the judge.

    from solvi.experimental.laws import Atom, Grammar
    from solvi.experimental.skills import Certificate, SkillLearner, as_s2

    g = Grammar([Atom(f"x={i}", lambda s, i=i: s["pos"][0] == i, feature="x") for i in range(9)]
                + [Atom("has_key", lambda s: s["has_key"], feature="key")])
    learner = SkillLearner(env, g, {"key": lambda s: s["has_key"], "door": lambda s: s["door_open"]},
                           horizon=12, budget=20_000, certificate=Certificate(promise=0.05, delta=0.1))
    for seed in range(200):                                  # a stream of fresh worlds
        ep = learner.run(seed=seed, steps=100)
        ep["s1_share"], ep["planner_calls"], ep["goals"]      # System 1 grows, System 2 is called less
    learner.skills.match(state, "door")                      # the most specific certified skill, or None
    learner.skills.to_knowledge(store, {"env": "grid"})      # "skill" items: plan, start, simulation hash, outcomes
    solvi.Agent(env, knowledge=km, s2=as_s2(learner))        # or: planning on the exact model as the agent's System 2

The exact model. `ExactModel(env, clone=None)`: `fork()` is a deep copy of the environment as it is now (`clone(env,
state)` for environments with a snapshot / restore of their own); `simulate(actions)` runs a sequence on a fork. The
real environment is never stepped by the planner — `fingerprint()` hashes it, and the learner checks it before and
after every planner and verify call.

The planner. `plan(model, state, goal, actions=, horizon=, budget=, cost=, beam=)` → `Plan(actions, predicted, cost,
searched, start, end, simulation)`: a breadth-first search on forks (uniform cost with `cost=`, a beam with `beam=`)
with a visited set on the state key; refused steps are not plan steps; deterministic; None when nothing within the
horizon reaches the goal or the budget of simulated steps is used up. `verify(model, state, plan)` re-simulates the plan
on a fresh fork: every step accepted, the goal at the end, and — from the plan's own start — every predicted state.

Skills. `Skill(condition, actions, goal, source, support, outcomes, certificate)`: "when these atoms hold, for this goal,
do this sequence". `SkillSet(grammar, certificate)`: `compile(goal, plan, state)` (the condition rule: the atoms true
at the plan's start; a second verified plan of the same goal and actions from another start keeps only the atoms both
starts share, unless that would cover a state where the skill failed), `match(state, goal)` (the most specific skill
whose condition and certificate hold), `observe(skill, ok, alone=)` (its outcome), retirement and re-certification,
`to_knowledge(store, scope)` ("skill" items, source "outcome", the plan, its start key, the verifying simulation's
hash and the outcomes as evidence; retired skills retired in the store; unverified skills never written).

The certificate. `Certificate(promise=0.05, delta=0.1, bet=0.9, scope="skill")`: an anytime-valid e-process per skill
(or pooled per goal / over the set) on its outcomes — certified once its wealth reaches 1/delta, which takes `need`
clean outcomes (50 at 5%, 25 at 10%, 12 at 20%) and then stays certified; an e-detector on the outcomes of alone
decisions restarts it (at 5% one failure alone does). Every use of a skill has an outcome: before certification
System 2 re-verifies the skill on the exact model from the current state (the outcome is the code's answer, and the
step is System 2's); after it the real episode's (did the goal hold when the skill's actions ran out). So outcomes do
not come only from where the system acted alone — a certificate fed only by those can certify what it never tested.

The loop. `SkillLearner(env, grammar, goals, planner=plan, horizon, budget, promise, delta)`: per step a running skill
takes its next action; else for the first open goal a certified skill runs alone (System 1); else the top matching
skill is re-verified (System 2, supervised); else System 2 plans, verifies, runs and compiles; else a reproducible
fallback. Records per decision (who decided: s1 / s2 / fallback, the skill, the outcome) and per episode (System 1's
share, goals reached, planner calls, simulated steps, alone runs and failures). Controls by argument: `learn=False`
(plan always), `verify_plans=False` with a planner of your own (compiled without the judge), `certificate=False` (no
certificate), `grammar=None` (cases keyed by the start state). `as_s2(learner)`: a SlowPath for `solvi.Agent(env,
knowledge=km, s2=...)` that answers "action" with the next step of a verified plan (one search, then its steps while
the state is the predicted one); the option must pass the agent's hard checks.

Measured: benchmarks/skills_toy.py, a deterministic gridworld (a key, a door, a gem behind it, an exit; fresh layouts
from a seed, now and then a boulder): see the toy's own docstring and tests/test_skills_toy.py (`pytest -m slow`).

Footprints with variables (1.1.1). `SkillLearner(..., lift="lattice", vocabulary=voc, verify_n=16)`: every newly
compiled skill is generalized on the exact copy. A footprint is a conjunction of atoms `(relation, *args)` whose
arguments may be variables (`Var("D", "dir")`); the world's `Vocabulary` says what a relation means (`holds`), what a
variable ranges over (`domain`), how a term is grounded (`ground`: an offset turned with a direction, a resource's
material), what a plan read from its start (`footprint`) and which lifts a plan admits (`lifts`: a constant replaced by
a variable everywhere it appears — in the atoms, the plan and the goal — or a relation widened: a cell → «a resource
within 2 steps in direction D», a station's cell → «a station nearby», an exact count → a lower bound, the resource's
tool → «has a tool for R»). `lattice` lists the candidates above the plan's own footprint, most general first and
bounded (every subset of the lifts × the old condition kept or dropped, and single atoms dropped). `generalize` accepts
a candidate only when the plan, instantiated for every verification case that matches it but not the original — for each
goal its variables can take, the footprint's first binding — reaches its goal on a fresh copy of that state: up to
`verify_n` cases (at least `verify_min`) from the stream's recorded states (a reservoir of copies taken at choice
points) alternating with the world's `perturb` edits. No model judges; a candidate fails at its first failed check, and
a drop counts only when it was exercised: each dropped atom false in at least `drop_witness` (2) passing cases. The most
general passing candidate becomes the skill's `footprint` (and its plan a `template`: «move D, do» instead of
«move_left, do»); the skill keeps the `original` footprint, the `verified` count and the lattice record; `replay(skill)`
re-runs its verification cases and compares the simulation hashes. A generalized skill matches when its goal template
unifies with the goal and its footprint finds a binding; it runs the template grounded by that binding; the certificate
is unchanged in its mechanics. `lift=False` (the default) keeps the condition rule alone, exactly as before.
Tested on a small grid (tests/test_skills_footprint.py): a plan learned for one direction is accepted turned in every
direction where the copy agrees, refused where the world disagrees (an interaction refused in one direction), and a
dropped tool atom is refuted by a state without the tool; measured once in an open world in 1.1.1 (the experimental page).

The search cost. Each episode records `searched` (simulated steps of the planner's searches), `verify_calls` and
`cost` = {candidates, tests, probes, simulated} — report it next to System 1's share.

What is not done:
- skills are admitted by the certificate on their outcomes, not by discriminating probes (laws have
  `LawSet.admit_by_probes`); a skill set does not abstain — two certified skills of one goal are both verified plans,
  and abstaining would need conflicting predicted end states, which skills do not carry;
- the certificate needs its clean outcomes per certified unit: a skill used once per episode needs about fifty episodes
  at 5% before it acts alone; pool by goal or over the set (then the promise is the pool's) or loosen the promise;
- the condition rule only widens by shared atoms; the lattice (lift="lattice") searches a bounded set of wider
  footprints, most general first, judged on the copy — it is not the smallest sufficient condition, its verification
  cases are a sample (a candidate wrong on a rare state can pass; the certificate is what catches it), and neither
  specializes a skill after a failure (the failure blocks a widening and feeds the certificate);
- footprints with variables need a vocabulary written for the world (relations, domains, lifts, what a plan read):
  the library brings the lattice and the judge, not the relations;
- the planner is a blind search: no heuristic, no learned model; a world too large to search within the budget falls
  back to a reproducible random step;
- the exact model is a copy of the environment: a world that is not code (or not copyable) needs a clone of its own,
  and nothing here checks that a clone is faithful beyond the plan's re-simulation and the real outcomes."""
from __future__ import annotations

from .. import warn_on_import
from ._certificate import Certificate, EProcess
from ._footprint import (Candidate, Footprint, Generalization, Lift, Var, Vocabulary, generalize, lattice, replay,
                         unify)
from ._learner import ExactPlanPath, SkillLearner, as_s2
from ._model import Check, ExactModel, Plan, plan, simulation_hash, state_key, verify
from ._skills import Skill, SkillSet

warn_on_import(__name__)

__all__ = ["Candidate", "Certificate", "Check", "EProcess", "ExactModel", "ExactPlanPath", "Footprint", "Generalization",
           "Lift", "Plan", "Skill", "SkillLearner", "SkillSet", "Var", "Vocabulary", "as_s2", "generalize", "lattice",
           "plan", "replay", "simulation_hash", "state_key", "unify", "verify"]
