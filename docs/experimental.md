# Experimental

`solvi.experimental` holds pieces that work and are tested but have not yet shown a measured gain, or a measured use,
of the kind the rest of solvi promises. Their API may change in any release. Each one either graduates or is removed by
**1.2** (two minor releases after 1.0).

```python
from solvi.experimental import STATUS
STATUS["lora"]       # {"since": "0.7", "what": "...", "missing": "...", "script": None, "deadline": "1.2"}
```

How you can tell you are using one:

- importing it warns once with a `solvi.ExperimentalWarning` (a `UserWarning`; silence it with
  `warnings.filterwarnings("ignore", category=solvi.ExperimentalWarning)`);
- a decision made by a System that uses one records it in the stored decision (`meta["experimental"]`, e.g.
  `["lora"]`), and `solvi report --overview` counts those decisions;
- nothing stable imports them, and `solvi` does not re-export them. One command loads one when you ask for it:
  `solvi hook` (hooks).

**Graduating** needs a bar fixed before the measurement, a script in `benchmarks/` that measures it, the bar met and
independently reviewed, no import of anything experimental, a conformance check, and a docs move; the old path then
keeps working for one release. **Removal**: a piece that has not graduated by its deadline, or that measures negative,
is removed and the negative result published.

Each section below gives the piece's `STATUS` entry: what it is, since when it exists, what it is missing to graduate,
the script that measures it (`changes`, `laws`, `skills`, `relations`) and where the guide describes it.

## `lora`

A LoRA adapter on the decider for one question (`solvi.experimental.lora.adapt_lora(part, examples)`; was
`part.adapt_lora(...)`; needs `solvi[lora]`). Since 0.7.

- **Missing:** a decision on real use, a benchmark script in the repo, and wiring as the adapter rung of an update (solvi.experimental.changes); measured gains over fit come with overconfidence.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [A LoRA adapter per question](guide.md#a-lora-adapter-per-question-adapt_lora-experimental);
  API: [solvi.experimental.lora](api/lora.md).

## `compile`

A written specification compiled by an LLM into catalog parts, accepted when two drafts agree and the specification's
tests pass (with its sandbox, `solvi.experimental.compile.sandbox`). Since 0.9.

- **Missing:** acceptance on larger specifications and a benchmark script in benchmarks/.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [A specification compiled into the catalog](guide.md#a-specification-compiled-into-the-catalog-solviexperimentalcompile);
  API: [solvi.experimental.compile](api/compile.md), [sandbox](api/sandbox.md).

## `hooks`

`solvi hook`: a coding agent's pre-edit rule checks and skill picker (Claude Code; Codex built from its documented
hook schema). Since 0.7.1.

- **Missing:** false deny and ask rates on real coding-agent sessions, with a script.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [solvi behind a coding agent's hooks](guide.md#solvi-behind-a-coding-agents-hooks);
  API: [solvi.experimental.hooks](api/hooks.md); runnable: `examples/22_coding_agent_hooks.py`.

## `charts`

Verified charts: every number quoted from the text, a deterministic SVG. Since 0.7.

- **Missing:** accuracy on real texts with the rule-based and the LLM proposer.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [Verified charts](guide.md#verified-charts-a-specialist-that-checks-every-number);
  API: [solvi.experimental.charts](api/charts.md); runnable: `examples/21_verified_chart.py`.

## `changes`

One change pipeline for four sources (a document version, verified experience, a person's correction, a search
proposal): pluggable stages, a journal of kind "change", promote into the knowledge store / compiled versions / a
labelled update (`LearningTarget`: the update ladder and its gates, which were the learning loop's until 1.1), versions
and rollback (`solvi.experimental.changes.Pipeline(stages, storage, store=...)`). Since 1.1.

- **Missing:** on a credit policy the pipeline carried a document version (144 of 144 changed decisions traced to their clause), admitted 12 of 12 exact exceptions, held every widening one for a person and refused 8 of 8 exceptions that touched other decisions, 0 extraneous changes, 14 versions replayed; on a bank-request stream with a person's labels every one of 22 candidates passed every stage (`nogate` = `gate`), so the stages added nothing there — the module is journal, impact analysis, approval levels and rollback; still missing: the narrows / widens classifier over rule conditions, a regression-scenario stage, and a script for the bank-request stream (the credit numbers: benchmarks/changes_credit.py).
- **Script:** `benchmarks/changes_credit.py` (the credit policy; needs the German Credit applications:
  `benchmarks/tasks/fetch.sh credit` and `prepare.py`). **Deadline:** 1.2.
- Guide: [The change pipeline](guide.md#the-change-pipeline-solviexperimentalchanges); design and API: the module
  docstring of `solvi.experimental.changes`; the library pieces it uses are stable:
  sensitive fields and citations in [solvi check](guide.md#checking-a-catalog-solvi-check), `only=` in
  [solvi diff](guide.md#catalog-fingerprint-solvi-diff-and-shadow-mode).

## `laws`

Laws of an environment from prediction mismatches, judged by the environment, never by a model
(`solvi.experimental.laws`): a grammar of boolean atoms you write, candidates "for this action, under this condition,
the effect is this" from each wrong prediction, acceptance only by a held-out test on other worlds, sessions or paths,
most specific first, retirement when the held-out precision falls, a probe planner for live systems (`split`,
`discriminate`: read-only before write) and export into the knowledge store (`LawSet.to_knowledge`). The acceptance is
sized by how many candidates were tested on the buffer (`sized=True`, the default; `sized=False` the fixed thresholds),
`LawSet.admit_by_probes` admits a law by asking the world where the candidates disagree and only when one class
survives, `predict(..., disagree="abstain")` abstains where laws that are not exceptions of one another disagree, and
`LawSet.cost` records the search cost. Since 1.1.

- **Missing:** on a game simulator with written mechanics the laws closed 94% of the gap between a learned map and the written mechanics, but random candidates under the same test came within 1.1 points, so the mismatch is a compact proposal heuristic and the test does the work; on a real MCP filesystem server with a path policy, probes planned from mismatches took refusals from 6 to 0 in one session (random probes: 48 to 29); still missing: a grammar too large to enumerate, retirement of weak laws nothing contradicts, wiring into a System's decisions, and a script for the two stands (the toys: benchmarks/laws_toy.py); a ranker of candidates for expensive checks (RankedProposer, a bag-of-atoms model trained on tasks drawn from the grammar) cut the checks to the true condition from 69.7 in the blind order to 2.1 on synthetic tasks of one grammar, in distribution only and never on a live system; the acceptance sized by the candidates tested on a buffer (on a synthetic buffer with no law the fixed thresholds accepted a law in 100 of 100 buffers, the sized rule in 0 of 100), admission by discriminating probes (on a synthetic world at 8 checks a random held-out sample accepted a wrong law in 0.593 of the trials, the probes in none) and abstention where laws that are not exceptions of one another disagree are measured on synthetic worlds only, not yet on the game simulator or the file server.
- **Script:** `benchmarks/laws_toy.py` (two deterministic toys that reproduce both findings in miniature; `pytest -m
  slow tests/test_laws_toy.py`). **Deadline:** 1.2.
- Guide: [Laws from an environment](best_practices.md#laws-from-an-environment-the-test-is-the-judge), [sized
  acceptance, probes, abstention](best_practices.md#size-the-acceptance-by-the-search-admit-by-probes-abstain-when-laws-disagree), [ranking
  candidates](best_practices.md#rules-over-relations-enumerate-with-a-library-rank-only-where-a-check-is-expensive); API:
  [solvi.experimental.laws](api/laws.md).

## `relations`

Relations over a state and rules found over them (`solvi.experimental.relations`): a state becomes named relations over
its nodes, stored as packed bits (a path tree: PARENT, SAME_DIR, IS_DIR, NAME=x; a grid seen from the player: EAST,
SOUTH, AT, FACE and the cells' materials and creatures); a relation-algebra enumerator builds programs bottom-up
(composition, converse, union, intersection, difference, a bounded complement, transitive closure, path counts) and
keeps one per behaviour — two programs are the same only when their full relations agree on every example; programs are
scored by the support and precision of a readout (does the subject relate to anything), noise-tolerant, and a
`Library` keeps accepted ones as terminals of later searches. `RelAlgProposer` hands the best programs to a law set as
atoms, and its unchanged held-out test judges them. A found rule runs as code: `compile_program` writes plain set code
(or index arrays for large graphs; dense answers refused) and `check_program` compares it with the interpreter pair for
pair; with `Atom(..., code=)`, `LawSet.to_module()` writes a law set as one module and `check_compiled` runs it against
`LawSet.predict` on every row (the bar: 0 differences). Composites by enumeration with a library
(`relations.library.find_rules`): enumerate small programs over the base relations, admit a program only when it passes
the held-out test on other worlds, and enumerate again with the admitted rules as new terminals, so a rule three levels
deep is a small program over earlier finds; a learned proposer has a fallback slot (`proposer=`), none ships. Since 1.1.

- **Missing:** on a pre-registered battery of 28 random rules over three relations the enumerator found 22 within seven nodes, every find exact on larger graphs of other families, and compiled rules matched hand-written set code pair for pair on a forest of 100,000 nodes within twice its time; on two toys (a path tree with "denied under a directory named secret at depth 3 or more", three levels; kinship on forests, depth 1 to 6; five seeds each) the library found every level within 2,556 and 10,817 enumerated programs in all, where the same search without it found neither top rule in 100,000, with 0 false accepts on larger fresh worlds, but the kinship ladder missed its bar of a tenth of that budget (1.08 times: the universal relation of the enumerator enlarges every level), and met it at 7,889 with the universal relation left out of the terminals (include_universal=False, an opt-in measured as an option, the bar unchanged); a rule the programs cannot express (x and y share a 4-clique) was never admitted on graphs of density 0.3 or 0.5, but on dense graphs (0.7) precision acceptance with held-out graphs of the same density admitted the plain edge relation, which held-out graphs of another density or exact acceptance rejected; still missing: a stand where relational atoms beat the flat grammar on real data (an MCP filesystem server with an ancestor rule, a game simulator with written mechanics), how deep the search reaches with the twenty-odd relations of a grid window, the library's intermediate concepts found rather than given as targets, a learned proposer for the fallback slot (none ships), and a way to drop proposed atoms no law uses and library entries that make other rules dearer.
- **Script:** `benchmarks/relations_battery.py` (the battery; the rule file is passed with `--rules`),
  `benchmarks/relations_compile.py` (compiled rules on a forest of 100,000 nodes, on worlds of 1,000 nodes and on
  100,000 law rows; `pytest -m slow tests/test_relations_gate.py`) and `benchmarks/relations_library_toy.py` (a path
  tree and kinship on forests, with and without the library; `pytest -m slow tests/test_relations_library_toy.py`).
  **Deadline:** 1.2.
- Guide: [Relational rules](best_practices.md#relational-rules-enumerate-let-the-test-judge-then-compile),
  [rules over relations with a library](best_practices.md#rules-over-relations-enumerate-with-a-library-rank-only-where-a-check-is-expensive);
  API: [solvi.experimental.relations](api/relations.md).

## `skills`

Learning in a world described by code (`solvi.experimental.skills`): System 2 plans on an exact model of the world (a
copy of the environment; the real one is never stepped by the planner), a plan counts only after the code re-simulated
it, verified plans are compiled into System 1 skills ("when these atoms hold, for this goal, do this sequence"; the
conditions keep the atoms every verified start shared), and a skill acts alone only while its outcomes keep an
anytime-valid certificate — before that System 2 re-verifies it at every use, and an alarm on its alone outcomes
retires it. `SkillLearner` runs the loop with a record per decision and per episode; `as_s2(learner)` makes the planner
System 2 of `solvi.Agent`; `SkillSet.to_knowledge` writes the skills into the knowledge store;
`SkillSet(disagree="abstain")` leaves to System 2 the states where certified skills of one goal disagree, and each
episode records its search cost. Since 1.1.

- **Missing:** on a deterministic gridworld (20 fresh layouts, five seeds) System 1's share reached 0.81 in the last five episodes with every goal reached, as when System 2 plans every goal, and 1.5% of System 1's decisions were in failed runs under a 20% promise pooled over the skills, while replayed cases never transferred and plans compiled without verification reached no goal; still missing: a stand beyond the toy (a game simulator with written mechanics, a service with a specification), a 5% promise (fifty clean uses per certified unit), a condition rule that specialises after a failure, and a planner with a heuristic; abstention where certified skills of one goal disagree and the search-cost record are tested on synthetic sets only, and the sized and probe admission of laws is not applied to skills.
- **Script:** `benchmarks/skills_toy.py` (the gridworld with four arms: skills, plan always, cases, scrambled; `pytest -m
  slow tests/test_skills_toy.py`). **Deadline:** 1.2.
- Guide: [Learning in a world described by code](best_practices.md#learning-in-a-world-described-by-code-plan-on-the-code);
  API: [solvi.experimental.skills](api/skills.md).

## Graduated and removed

- **Graduated in 1.1:** `counterfactual` — counterfactual explanations are the stable `solvi.core.explain`
  ([guide](guide.md#explaining-a-decision-the-smallest-change-that-flips-it)), with the cost of a text change a
  parameter (`text_cost=`, `costs=`); `solvi.experimental.counterfactual` re-exports it, with this warning, until 1.2.
- **Graduated in 1.1:** `mcp` — the MCP proxy with a guard is `solvi.serve`'s (`run_proxy`, `Proxy`, `Upstream`;
  [guide](guide.md#an-mcp-proxy)); `solvi.experimental.mcp` re-exports it, with this warning, until 1.2.
- **Graduated in 1.1:** `specialist` — the specialist contract is the stable `solvi.core.specialist`, its runs stored
  as records of kind "specialist" in a TraceStorage ([guide](guide.md#specialists-a-model-proposes-code-checks-the-store-keeps-the-run));
  `solvi.experimental.specialist` re-exports it, with this warning, until 1.2.
- **1.1:** `oncalib` (recalibrating a guarantee on the fly from outcomes: it broke the promise — recalibrate explicitly,
  `System.guarantee(..., corrections=True)`, on a schedule or after a drift flag) and `learning` (the learning loop: no
  measurable gain on streams of one kind of decision; a labelled update is now a candidate change of
  `solvi.experimental.changes`, `Change(artifact="update")` with `LearningTarget`). The changelog's 1.1 "Removed" table
  says why and what to use instead.

## Tried and left out

Some ideas were measured and did not make it into solvi, experimental or not; the
[changelog](../CHANGELOG.md#tried-and-left-out) lists them with what was measured.
