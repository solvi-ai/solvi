# `solvi.experimental.relations`

Relations over a state and rules found over them (experimental): packed-bit `Relations`, the relation algebra, a
bottom-up enumerator with observational equivalence (`enumerate_programs`, `synthesize`, `top_programs`), a `Library` of
accepted programs, compilation into plain set code or index arrays with a differential check (`compile_program`,
`check_program`), encoders (`encode`, `encode_paths`), `RelAlgProposer` for the laws' held-out test, `CompositionTable` (a composition
table of relations learned by counting, derivation over every bracketing, refusal of what it cannot derive) and
`forward_chain` (its rules applied to the facts of a knowledge store to a fixed point) and `RulesFirst` (the table
answers first, a model only what it refuses).

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

## Rules first, a model second

`RulesFirst(rules, *, fallback=None, render=None, by="relations:RulesFirst")`: `rules` is a `CompositionTable` (its
`derive_paths`) or a callable `(chains) -> Derivation`; `fallback(question, options)` returns `(answer, confidence)`,
`(answer, confidence, raw)` (raw kept in the record), a plain answer, or None (the model refuses).

    from solvi.experimental.relations import RulesFirst, decider_fallback, render_chains

    rf = RulesFirst(table, fallback=decider_fallback(model, task="What is Z to A?"))
    a = rf.answer([("father", "sister")], options, tail="Z")   # one chain or several chains of one query
    a.value, a.who, a.reason       # "aunt", "rules", "derived" — or the model's answer, "model", "underivable"
    rec = a.to_record()            # plain JSON: the path, the cells used or the model's output, options, confidence
    rf.replay(rec).ok              # derives again; re-reads the model's recorded output; never calls the model
    rf.stats                       # {"rules": n, "model": n, "refused": n, "by_reason": {...}}

- **Rules first.** A "derived" derivation is the answer; the question is not even rendered and the model is not
  called. On "disagree" / "underivable" (or "identity") the fallback answers; without a fallback the refusal stays a
  refusal (`value` None, `who` None) with the rules' reason.
- **The question.** `render_chains(chains, head="A", tail=None)` reads a chain the table's way: (father, sister) is
  "B is the father of A. C is the sister of B. What is C to A?"; several chains share the head and the tail, the
  people in between are new per chain. Pass your own `render(chains, head, tail)` for another wording.
- **The answer.** `who` is the path that answered; `reason` is always the rules' reason (why the model was asked on the
  model path); `confidence` is the model's — the rules give none per answer, their precision is measured. A model
  answer outside the options is not an answer (`note` says so) and counts as refused in `stats`.
- **Replay.** `replay(record)` requires the same derivation (relation, reason, candidates, cells), the same rendered
  question and the same value from the recorded model output; an edited record or a changed table is a problem
  (`Replay.ok` False, `Replay.problems`).
- **Limits.** A model answer is taken as given (no confidence gate). An empty reply (a reasoning model at its token
  cap) or a word outside the options counts as refused; no retry or budget check is built. Whole-graph questions over
  `forward_chain` and the text setting are not covered.

Measured once on the frozen test split of CLUTRR (Sinha et al., 2019), the clean split (the relation graph of each story, not its text; 1,146 queries, chains of 2–10 relations; the table fitted on the whole training file, the same 111 cells; the rules' answers byte-identical to the composition table's own measurement on this split): the rules answered 1,087 with 0 wrong and refused 59 (21 disagreeing bracketings, 38 underivable). With a small hosted LLM (z-ai/glm-5.3-flash) as the fallback (temperature 0, reasoning effort low, 2,000 output tokens; the first of two drafts answers; the refused queries rendered in words with the 18 relation names as options), 50 of the 59 refused queries got a readable option and 39 were right; 8 replies were empty because the model's reasoning used the whole output budget (finish reason "length"), and 1 named a relation outside the option list: overall 0.983 at coverage 0.992 (1,126 right of 1,137 answered; 11 wrong, 10 of them one chain of three relations whose gold in this split is itself split between mother and mother-in-law), against 0.949 at coverage 0.949 and 0 wrong for the rules alone. On an equal-size sample of queries the rules answered, the same LLM gave a readable option to 42 of 59 and was right on 41; on chains of 4 or more relations its precision among readable replies was the same on the refused and on the answered queries (34 of 35 each), the loss being empty replies (9 of 44 and 17 of 52). solvi-base as the fallback answered all 59 and was right on 2 (11 of 59 on the sample): it does not compose relations. Every answer replayed from its record without the model (0 failures). One run; $0.017 after the proxy's cache; the pre-registered bar of coverage 1.0 was missed. A pilot before the test run, on the training file only (1,816 held-out stories, a fifth of it by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths): the table answered 1,675 (4 wrong) and refused 141 (11 disagreeing bracketings, 130 underivable); the same LLM got 137 of the 141 right and solvi-base 11 of 141; one run, secondary to the test numbers (`tests/test_relations_first.py`).

::: solvi.experimental.relations.RulesFirst

::: solvi.experimental.relations.RulesFirstAnswer

::: solvi.experimental.relations.render_chains

::: solvi.experimental.relations.decider_fallback

::: solvi.experimental.relations.forward_chain

::: solvi.experimental.relations.ChainReport

::: solvi.experimental.relations.explain

::: solvi.experimental.relations.Explanation

::: solvi.experimental.relations.CompositionTable

::: solvi.experimental.relations.Derivation

::: solvi.experimental.relations.Cell

::: solvi.experimental.relations

::: solvi.experimental.relations.library
