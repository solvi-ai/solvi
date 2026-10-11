# `solvi.experimental.relations`

Relations over a state and rules found over them (experimental): packed-bit `Relations`, the relation algebra, a
bottom-up enumerator with observational equivalence (`enumerate_programs`, `synthesize`, `top_programs`), a `Library` of
accepted programs, compilation into plain set code or index arrays with a differential check (`compile_program`,
`check_program`), encoders (`encode`, `encode_paths`), `RelAlgProposer` for the laws' held-out test, `CompositionTable` (a composition
table of relations learned by counting, derivation over every bracketing, refusal of what it cannot derive),
`forward_chain` (its rules applied to the facts of a knowledge store to a fixed point) and `GuardedRules` /
`chain_rules` (Datalog-lite rules with unary guards and a negative stratum, learned by counting on labelled pairs).

## Composites with a library

`solvi.experimental.relations.library.find_rules(examples, targets, base=..., size=..., budget=..., library=(),
test=HeldOut(other_worlds, targets))` enumerates small programs over relations, admits only what passes the held-out
test, and builds the next rules on the admitted ones (`Library` entries as new terminals); `run(program, world,
library=)` runs a found rule on any world. The search is this package's enumerator and the programs are its tuples;
`library=` takes a `Library` (the collection of accepted programs) or a sequence of `Entry` (one accepted program:
program, name, examples, test), and admitted rules are `Entry` objects. `include_universal=False` leaves the universal
relation U out of the enumerator's terminals (opt-in; the default keeps it).

Operators: unary `T` (converse), `NOT` (bounded complement), `PLUS` (transitive closure), `CNT2` / `CNT3` (the nodes
with at least 2 / 3 successors, as a diagonal) and `CNT1` (at least one successor) — `CNT1` is opt-in: known to the
algebra, the enumerator and both compile backends, but not in the default `OPS` (pass it in `ops=`; `find_rules` uses
it through `counts=(1,)`); binary `OR`, `AND`, `MINUS`, `XOR`, `COMP`, `C2`, `C3`.

## Composition tables over kinship chains

`CompositionTable(majority=0.9, support=2)` learns which relation a chain of two relations composes to
(father ∘ sister → aunt) by exact counting, derives longer chains by composing over every bracketing, and refuses what
the table cannot derive.

    from solvi.experimental.relations import CompositionTable, base_rates, sized_admit

    examples = [(("father", "sister"), "aunt"), (("father", "sister"), "aunt"), (("aunt", "son"), "cousin"), ...]
    t = CompositionTable(majority=0.9, support=2).fit(examples)   # chains of length 2 vote; length 3 fill residuals
    d = t.derive(("father", "sister", "son"))
    d.relation, d.reason          # "cousin", "derived" — or None, "underivable" / "disagree" / "identity"
    d.candidates, d.cells         # bracketings per relation; the cells (left, right, relation, hits, support) used
    t.derive_paths([chain_a, chain_b])                            # one query reached by several chains
    t.to_knowledge(store, {"domain": "kinship"})                  # "rule" items with their support

- **Fit.** A chain (r1, r2) labelled r is a vote for the cell r1 ∘ r2 → r; a cell is admitted when its majority
  relation has a share ≥ `majority` and at least `support` votes. **Residual fill:** a chain (a, b, c) labelled r with
  a ∘ b = x in the table votes for x ∘ c → r, and with b ∘ c = y for a ∘ y → r; the residual votes are recomputed from
  the current table each round (a chain counts once per round) until the table stops changing (`rounds=` caps it).
  `fit` is deterministic and independent of the order of the examples; other chain lengths are counted in `ignored`.
- **Derive.** CYK over the chain: each span keeps the relations it composes to and how many bracketings give each; a
  bracketing that meets an empty cell gives nothing. `agree="all"` (default): answered only when every bracketing that
  gives a relation gives the same one; `agree="majority"`: the relation of strictly the most bracketings. Refusals:
  `"underivable"` (no bracketing composes), `"disagree"` (two relations), `"identity"` (only a learned identity).
  `explain(chain, relation=None)` returns the cells used.
- **Identity (opt-in).** `identity="self"`: a chain (a, b, c) labelled c votes a ∘ b → self where the pairs admit no
  cell; self composes as neutral and is never an answer. In domains where a cell depends on an attribute the chain
  does not carry (whose son's mother is "self" depends on the subject's gender), a learned identity adds wrong answers.
- **Admission hook.** `admit=(cell, relation, hits, support, tested) → bool` is an extra gate after the thresholds;
  `sized_admit(base_rates(examples), delta=0.1)` is the sized acceptance of `solvi.experimental.laws` with a base-rate
  null: P[Binomial(support, base_rate(r)) ≥ hits] ≤ δ / N, N the cells tested.

## Forward chaining over the rules in a knowledge store

`forward_chain(store, scope=..., rule_scope=...)` applies the composition rules kept in a `KnowledgeStore` (the cells
`CompositionTable.to_knowledge` writes) to the store's pair facts until nothing new follows, and writes every derived
fact through the store's gates.

    from solvi.core.knowledge import KnowledgeStore
    from solvi.experimental.relations import CompositionTable, explain, forward_chain, pair_fact

    store = KnowledgeStore()
    table.to_knowledge(store, {"domain": "kinship"})                       # the rules
    for x, y, rel in edges:                                                # the story: who is what of whom
        store.add("fact", pair_fact(x, y, rel), {"story": 7}, source="outcome")
    rep = forward_chain(store, scope={"story": 7}, rule_scope={"domain": "kinship"})
    rep.rounds, rep.facts_added, rep.disputes_opened, rep.hypotheses       # the fixed point and what it wrote
    explain(store, fact_id).cells                                          # the rule cells the derivation used

- **Facts are pairs.** `pair_fact(x, y, relation)` = `{"s": [x, y], "r": "relation", "o": relation}`: the store's key
  is the ordered pair and its value the relation, so two different relations of one pair meet on one key.
- **Chaining.** Semi-naive rounds: the facts new in one round are joined with every fact of the same scope; for
  (x r1 y), (y r2 z) and a rule r1 ∘ r2 → r the fact (x r z) is proposed with `derived_from` = the two facts and the
  rule, `inferred=True`, the source of the weaker premise and evidence `{"rule", "via", "depth"}`. Facts join only
  within one exact scope; `scope=None` chains every scope apart.
- **What the store does with it.** The same relation is one more justification; a different relation from an
  equal-rank derivation makes the pair disputed — the table's refusal by disagreement, kept as a question for a
  person (`store.disputes()`, `store.resolve`); a retracted or refuted premise takes its consequences along. Disputed
  facts are chained too, so both sides carry their consequences as hypotheses, and a person's answer settles them
  without a re-run. `rules="any"` also applies hypothesis rules: what they derive is a hypothesis until the rule is
  promoted. A justification that would rest on the fact it justifies (a cycle in the graph) is not written.
- **Guards.** `max_rounds=` stops early (`fixed_point=False`; a later call resumes); `max_facts=` (default |entities|²
  × |relations| per scope) raises `ChainLimit`. A second call writes nothing.
- **`explain(store, fact_id)`** → `Explanation`: every live justification of the fact, recursively down to the facts
  asserted without premises (`facts`), the rules (`rules`) and their cells (`cells`); `tree()` gives one derivation as
  a nested dict. On a chain it gives the cells `CompositionTable.derive(chain).cells` gives.
- **Limits.** Composition of two relations over pair facts only — no general rules, negation or aggregation; a learned
  identity relation is chained as an ordinary relation.

Measured on CLUTRR (Sinha et al., 2019), the clean split's training file only (the relation graph of each story, not its text): on 1,816 held-out stories (a fifth of the training file by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths, 111 cells) the store's reading of the query pair after chaining to the fixed point was compared with the table's derivation over the shortest paths: 1,675 equal (0.922), 11 disputed where the bracketings disagree (0.006), 130 with nothing derived where the table cannot derive (0.072), 0 differences; the fixed point took 0 rounds for 72 stories, 1 for 826 and 2 for 918; 2.6 s of chaining in all, 1.4 ms a story, on CPU; one run, the test file not read. On 120 random chains of 2–7 relations over 12 random tables whose bracketings can disagree: 0 differences (2 answers equal to the table's rest only on a disputed pair and stay hypotheses until a person answers it) (`benchmarks/relations_chain.py`, `tests/test_relations_chain.py`).

## Guarded rules: Datalog-lite with a negative stratum

`GuardedRules(majority=0.9, support=2)` learns rules over pair facts and unary facts by exact counting on labelled
query pairs, applies them by forward chaining to a fixed point, and answers a pair with the set of relations derived
on it — or the empty set, a refusal.

    from solvi.experimental.relations import GuardedRules, Story, chain_rules, pair_fact, unary_fact

    story = Story.from_edges(edges, types)               # a self-loop is_* / no_* is a unary fact
    rules = GuardedRules(0.9, 2, complements={"Not_living_in": "living_in"})
    rules.fit(examples, check=other_examples)           # (story, (x, y), set of relations) each
    rules.predict(story, (x, y))                        # frozenset of relations; ∅ = refused
    c = rules.closure(story); c.status(x, y, "living_in")   # "active" | "hypothesis" | "disputed"
    rules.to_knowledge(store, {"domain": "kin"})        # one "rule" item per rule
    chain_rules(store, scope={"story": 7}, rule_scope={"domain": "kin"}, complements=...)

- **Bodies.** An atom is a relation read forwards, s(x, y), or backwards, s⁻(x, y) = s(y, x). `B1` r ← s(x,y);
  `B1u` r ← s(x,y) ∧ u(x) (or u(y)); `B2` r ← s1(x,z) ∧ s2(z,y); `B2z` the same ∧ u(z); `BNE` r ← s(x,z) ∧ z ≠ y ∧
  u(y) («lives somewhere else»); `NEG` r ← u1(x) ∧ u2(y) ∧ ¬s(x,y). Unary predicates are the story's own (no invented
  predicates); depth comes from rounds — a derived fact feeds further bodies — never from longer bodies.
- **Learning.** A body true on a labelled pair is one vote; each relation of the pair's label set one hit. A rule
  needs hits ≥ `support` and a share ≥ `majority` of its votes; r ← r (∧ u) is never admitted, and a guarded body
  leaves when its unguarded generalisation is admitted. Rounds recount the votes on the fixed point of the rules
  admitted so far; the rule set only grows, so the rounds stop at its fixed point (or `rounds=`). `check=`: an
  independent labelled slice where a rule's share must also reach `majority` (and `check_min=` hits).
- **Rules wrong together** (`verify=True`). Each rule is counted on the fixed point of the rules before it, but rules
  admitted in one round can be wrong together: one right at the labelled pairs and wrong elsewhere derives facts that
  make others fire wrongly. After each round the new rules are recounted on the fixed point of the whole new set (on
  the check slice when given); the worse half of those it contradicts is retracted for good (`retracted`), and the
  recount repeats until none falls. When labels exist only on query pairs, a loose admission (a small support) admits
  many such rules and this recount then retracts most of a round, so prefer a support well above 2 there.
- **Residual votes** (`residual=True`, the composition table's residual fill generalised): within a round, the rules
  admitted on the direct votes are applied to the length-3 walks of each labelled pair; where one gives a middle fact
  that is not yet in the fixed point, the body that needs it gets the pair's vote. In the measurements so far this
  only reaches in one round what the next round's fixed point shows anyway.
- **The negative stratum.** After the positive fixed point, a `NEG` rule fires on (x, y) only when nothing of s and
  nothing of the head's own key holds there; its facts never feed positive rules. NEG bodies are learned on the
  final fixed point, counted on the pairs where the head's key is empty, and only for an s that holds on at least
  `support` labelled pairs of the same unary types.
- **Pairs carry sets; complements dispute.** Every relation is its own key on a pair, so two relations of one pair
  are two facts; relations declared complements (`complements={"Not_living_in": "living_in"}`) share a key, and two
  of them on one pair are a dispute: the pair is answered on neither. Statuses at the fixed point: disputed, active
  (given, or derived from active premises), hypothesis (derived only through disputed facts).
- **Two engines, one answer.** `closure(story)` (bitset joins, statuses) is what `predict` and the learner use;
  `closure(story, justify=True)` also lists every justification — what `chain_rules` writes. The tests check that both
  give the same facts and statuses, and that the store's active facts after `chain_rules` equal `answer`.
- **In a knowledge store.** `pair_fact(x, y, r, multi=True, family=...)` is a pair fact keyed by pair and family;
  `unary_fact(x, u)` a unary fact. `chain_rules` writes each derived fact through `store.add` with every
  justification (premise facts including the unary guards, and the rule; `inferred=True`, the weakest premise's
  source), then the NEG stratum with its absent relation in the evidence; a NEG fact whose condition ended is
  retracted on the next call; a second call writes nothing. `explain(store, id)` shows guards among the premises and
  `stratum` / `absent` for the negative stratum. `forward_chain(..., multi=True)` keeps several relations per pair
  for composition rules too.
- **Diagnostics.** `votes_for(head, body)` (the last round's counts, fit and check), `near_misses(heads)` (bodies
  often right about a head and not admitted), `templates_on(closure, pair, relations)` (which templates give an
  answer), `history` (per round: rules added, time).
- **Limits.** Two binary atoms per body; unary predicates only as given (no learned or invented ones, so a guard that
  is itself derived — «underage» from «school mates» — is out of reach); the store path writes every justification
  and is slow on dense stories (seconds to minutes for tens of thousands of derived facts); a wrong rule admitted at a
  small support spreads through the fixed point, so an independent check slice matters.

Measured on a planted world (`benchmarks/relations_rules_toy.py`: families of three generations, genders, homes,
colleagues; every ordered pair labelled by an independent oracle; 8 fit stories = 4,106 pairs, 4 check stories, 10
test stories = 4,992 pairs): 242 rules in 4 rounds (3.0 s on CPU), exact set match 1.000 on the test pairs, all 8
planted rules found (a guard over a derived relation, a chain through a derived relation, the «elsewhere» rule, the
negative stratum); without residual votes the same 242 rules; one round without them 35 rules, exact 0.975,
precision 0.941; the store path on 3 more stories (1,568 pairs, 3,110 facts) 0 differences from the in-memory fixed
point. One deterministic run.

::: solvi.experimental.relations.GuardedRules

::: solvi.experimental.relations.GuardedRule

::: solvi.experimental.relations.Story

::: solvi.experimental.relations.Closure

::: solvi.experimental.relations.chain_rules

::: solvi.experimental.relations.unary_fact

::: solvi.experimental.relations.forward_chain

::: solvi.experimental.relations.ChainReport

::: solvi.experimental.relations.explain

::: solvi.experimental.relations.Explanation

::: solvi.experimental.relations.CompositionTable

::: solvi.experimental.relations.Derivation

::: solvi.experimental.relations.Cell

::: solvi.experimental.relations

::: solvi.experimental.relations.library
