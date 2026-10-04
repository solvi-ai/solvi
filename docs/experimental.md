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
- nothing stable imports them, and `solvi` does not re-export them. Two commands load one when you ask for it:
  `solvi hook` (hooks) and `solvi serve --guard --upstream` (mcp).

**Graduating** needs a bar fixed before the measurement, a script in `benchmarks/` that measures it, the bar met and
independently reviewed, no import of anything experimental, a conformance check, and a docs move; the old path then
keeps working for one release. **Removal**: a piece that has not graduated by its deadline, or that measures negative,
is removed and the negative result published.

Each section below gives the piece's `STATUS` entry: what it is, since when it exists, what it is missing to graduate,
the script that measures it (none yet for any of them) and where the guide describes it.

## `learning`

The learning loop: trusted labels → a ladder of updates → gates → promote or roll back
(`solvi.experimental.learning.Learning(system, store)`; was `system.learning(...)` before 1.0). Since 0.7.

- **Missing:** a simulation on real data that decides the default gates; the gate's holdout must come from the audit,
  and a rollback during a drift flag must hold.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [Learning from corrections with gates and rollback](guide.md#learning-from-corrections-with-gates-and-rollback-experimental);
  API: [solvi.experimental.learning](api/learning.md).

## `lora`

A LoRA adapter on the decider for one question (`solvi.experimental.lora.adapt_lora(part, examples)`; was
`part.adapt_lora(...)`; needs `solvi[lora]`). Since 0.7.

- **Missing:** a decision on real use, a benchmark script in the repo, and wiring as the learning loop's adapter step;
  measured gains over fit come with overconfidence.
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

## `specialist`

The propose → check → render specialist pattern. Since 0.7.

- **Missing:** a measured specialist (charts has no accuracy number yet).
- **Script:** none yet. **Deadline:** 1.2.
- API: [solvi.experimental.specialist](api/specialist.md).

## `charts`

Verified charts: every number quoted from the text, a deterministic SVG. Since 0.7.

- **Missing:** accuracy on real texts with the rule-based and the LLM proposer.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [Verified charts](guide.md#verified-charts-a-specialist-that-checks-every-number);
  API: [solvi.experimental.charts](api/charts.md); runnable: `examples/21_verified_chart.py`.

## `mcp`

An MCP stdio proxy that puts the agent guard in front of `tools/call` (`solvi serve --guard --upstream`). Since 0.9.

- **Missing:** a measured run through the proxy against a real MCP server.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [An MCP proxy](guide.md#an-mcp-proxy); API: [solvi.solutions.guard](api/agents.md) (the proxy is rendered
  there).

## `oncalib`

Recalibrating a guarantee on the fly from outcomes, every N labels or after a drift flag. Since 1.0.

- **Missing:** a recalibration that keeps the guarantee's promise (it does not: see its module docs).
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [Outcomes as labels](guide.md#outcomes-as-labels-systemoutcome); API: [solvi.experimental.oncalib](api/oncalib.md).
  The stable path recalibrates explicitly, on a schedule or after a drift flag.

## `counterfactual`

The smallest change of a decision's inputs that changes its answer (adverse-action reasons;
`solvi.experimental.counterfactual.search(res, question)`, was `res.counterfactual(...)`). Since 0.7.

- **Missing:** one measured use, e.g. adverse-action reasons on the credit gallery task.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [Counterfactual explanations](guide.md#counterfactual-explanations-experimental-solviexperimentalcounterfactual);
  API: [solvi.experimental.counterfactual](api/counterfactual.md).

## Tried and left out

Some ideas were measured and did not make it into solvi, experimental or not; the
[changelog](../CHANGELOG.md#tried-and-left-out) lists them with what was measured.
