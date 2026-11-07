# Checked cooperative finite search

`solvi.experimental.experience.cooperation` implements a finite minimization
problem with host-owned feasibility predicates and integer objective callbacks.
Assignments are immutable portable typed dictionaries indexed by string IDs.
The host supplies a revision identifying the semantics of all callbacks, and
must change it when those semantics change. Callbacks must be deterministic,
terminating and free of external effects; work units do not sandbox arbitrary
Python callbacks or limit their wall time.

`FiniteProblem(assignments, constraints, objective, revision=..., budget=budget)` defines the
host task. Construction pays to materialize an immutable truth/cost table.
Later mutation of callback closures has no effect on that frozen problem;
construct a new problem to change semantics. Its fingerprint includes the table.
Use the same ledger for construction and all subsequent operations. `CooperativeSearch(problem, budget)` accepts `Evidence`
claims containing an ID, source, revision, kind, target IDs, value and optional
active dependencies. `Residual` exposes remaining IDs, incumbent and bounds.
Source labels are provenance labels, not authenticated identities.

* `exclusion`: each target violates the host constraint named by `value`.
* `incumbent`: one feasible target has exactly the claimed integer cost.
* `lower_bound`: every feasible original assignment has cost at least `value`.
  This deliberately expensive independent finite check does not rely on solver
  claims or a circular argument about an already pruned residual.

`ConstraintSolver` checks successive surviving assignments and emits checked
counterexamples to feasibility. `ObjectiveSolver` sorts the current residual by
cost and proposes a feasible improving incumbent. Both implement the callable
`solver(residual, budget) -> Evidence | None` protocol. In `run`, the next solver
receives the updated residual after each acceptance: exclusions directly reduce
its objective evaluations and an incumbent changes its improvement threshold.
Custom solvers must debit their internal work through the supplied ledger.
These are cooperating constraint and objective methods, not external solver
integrations or a claim of a generally faster optimization algorithm.

`submit` independently rechecks every claim; contradictory certificates,
unknown targets, stale revisions, duplicate IDs, missing and cyclic dependencies
are rejected without changing accepted state. Rejected checking still costs
work. `revoke(id)` removes that claim and its transitive dependents; IDs cannot
be reused. Constraints are never relaxed by evidence.

`run(solvers, rounds=...)` is bounded by a finite number of calls and returns
`unknown`, including when a budget is exhausted. A feasible incumbent does not
imply optimality. `certify()` pays for `finite_reference(problem, budget)`, an
exhaustive independent search, and returns `optimal`, `infeasible`, or `unknown`
on exhaustion. The reference returns `(cost, assignment_id)` with deterministic
ID tie breaking.

`snapshot()` records the evidence journal and complete budget. To restore,
call `CooperativeSearch.restore(snapshot, host_problem, existing_host_budget, authority=search.authority)`. Replay validates the
schema, digest and problem fingerprint, uses the existing host ledger (rejecting refunded or inflated-limit ledgers), charges
replay and verification again, and fails closed on invalid evidence or exhausted
budget. Neither callbacks nor solver objects are deserialized. Callback semantics are frozen in the problem fingerprint; a digest is corruption detection,
not authentication. Solvers can be recreated after restore, with fresh source
names to avoid reusing evidence IDs. No snapshot can establish final optimality
without another paid `certify` call.

See `examples/36_cooperative_solvers.py` for a complete local example. The
implementation promises correctness on a finite declared domain, not a speed
advantage over enumeration; measurements must include all checker work.

Finite limits: 4096 assignments, 32 named constraints, 32 attributes per assignment,
4096 journal entries, 4096 targets/dependencies per claim, 256 characters per
evidence identifier and 4096 rounds. Non-sequence iterators for task mappings
and claim targets/dependencies are rejected before materialization. Residual
reads, snapshots, rounds and even empty-domain reference certificates cost work.
Exhaustion can make residual access raise BudgetStop; run returns residual=None
in its budget result. Replay receives a host ledger rather than constructing
a replacement ledger from untrusted snapshot data.

`CooperativeAuthority` is a trusted host capability holding the current epoch,
exact authorized checkpoint identity and a live reference to the shared budget.
Every accepted claim/retraction invalidates prior snapshots. Successful restore
transfers ownership and makes the old search instance stale; a snapshot by
itself cannot authorize replay. A restore compares the incoming budget against
the live ledger **before** charging restore, so spending in other users of the
shared ledger cannot be refunded by an old ledger clone. While the authority
is live, restore requires that same budget object.

For a fresh process, the host persists `authority.export_trusted_state()` in a
trusted monotonic channel, separately from untrusted snapshot data, and reloads
with `CooperativeAuthority.from_trusted_state(anchor)`. Persist the CURRENT
budget as well. The host must prevent rollback/duplication of that trust anchor;
loading a self-reported or old authority anchor is outside this security contract.
No cryptographic signature or external durable storage service is provided.
