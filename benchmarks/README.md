# Benchmarks

The scripts behind the measured numbers in the [README](../README.md) and the docs (see Scripts below), and dataset
loaders for public document benchmarks: each loader turns a dataset into plain documents (`key`, `text`), reference
fields or clause spans, and typed questions with ground truth.

| Module | Dataset | Documents |
|---|---|---|
| `datasets/receipts.py` | SROIE receipts (ICDAR 2019) | 626 train / 361 test |
| `datasets/cord.py` | CORD v2 receipts (Indonesia) | 800 train / 100 validation / 100 test |
| `datasets/cuad.py` | CUAD v1 contracts | 408 train / 102 test (official `test.json`) |
| `datasets/kleister_nda.py` | Kleister-NDA agreements | 254 train / 83 dev-0 |

## Data layout

All loaders read from `$SOLVI_DATA` (default: `data/` relative to the working directory):

```
$SOLVI_DATA/
  sroie/{train,test}.jsonl
  cord/{train,validation,test}.jsonl
  cuad/CUADv1.json
  cuad/test.json
  kleister_nda/{train,dev-0}/in.tsv.xz
  kleister_nda/{train,dev-0}/expected.tsv
```

- **SROIE** ([ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13); the parquet copy used here is [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) on Hugging Face): one JSON object per line with `key`, `text` (OCR lines
  joined by newlines) and `gold` = `{company, date, address, total}`. As a fallback the loader reads
  `sroie/{split}.parquet` with columns `key`, `entities` (dict with the four fields) and `words` (list of OCR lines).
- **CORD v2** ([naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)): the `ground_truth`
  JSON column of each split, one record per line. The loader uses `valid_line` (words with `row_id`, `quad`, `is_key`
  and a `category` per group) to rebuild the text and exact field spans.
- **CUAD v1** ([CUAD release](https://github.com/TheAtticusProject/cuad), `data.zip`): `CUADv1.json` (all 510
  contracts, SQuAD format) and `test.json`. Contracts whose title is in `test.json` form the test split; the rest are
  train. `load()` covers five clause types; `load_all()` returns spans for all 41 categories.
- **Kleister-NDA** ([applicaai/kleister-nda](https://github.com/applicaai/kleister-nda)): the `train` and `dev-0`
  directories as in the repository. `in.tsv.xz` holds tab-separated rows with the document id in column 1 and the text
  in column 3; `expected.tsv` holds space-separated `key=value` pairs. Ground truth has no spans, so `match()` compares
  normalized values.

## Usage

```python
import os
os.environ["SOLVI_DATA"] = "/path/to/data"   # set before importing a loader

from benchmarks.datasets import receipts
docs = receipts.load("test")
rows = receipts.laya_rows(docs)              # typed questions with ground truth
```

The loaders need nothing beyond the standard library, except the parquet fallback of `receipts.py`, which needs
`pandas` and `pyarrow` (neither is a dependency of solvi: install them yourself, or use the JSON-lines files).

## Results

Every measured number in the README and the docs comes from a script below or from a published model card; the dataset
loaders above are kept as the exact reading of each dataset (the extraction numbers once measured with them are not
quoted any more: their scripts were not ported here).

## Scripts

| Script | What it measures |
|---|---|
| `perturb_injection.py` | `perturb=k` against instructions appended to Bitext support messages: how often they are followed, what it costs |
| `injection_review.py` | the guard's injection detector on a 1.0 review set: ordinary tool outputs flagged (11 of 12), reworded injections missed (14 of 20) |
| `guard_firewall.py` | the guard's call firewall (`Guard.firewall`) on a synthetic irrelevance set (60 relevant, 120 irrelevant, 60 near-miss requests, each with a proposed call): false calls made with no guard, the schema check, schema + grounded (1.000 / 0.750 / 0.217), and the wanted calls still made (1.000 / 1.000 / 0.833); deterministic, under a second |
| `fast_head.py`, `strategist_scale.py` | `fit` (the answer head and its selection) and the strategist at scale |
| `ask_speed.py` | milliseconds per `ask` on small and large inputs (the README quickstart, the gallery, random catalogs of 50 and 1 000 parts, a 3.7k-float stream input): the README's Speed table |
| `ask_overhead.py` | what solvi adds to one `ask` (every gallery entry and a decider project, no real model) |
| `journal_size.py` | the bytes a stored decision takes, full, compact and `sample:10` (`TraceStorage(record=...)`), on every gallery entry, a game-like loop and the credit task, and that each store verifies and replays: the guide's table under "Compact records and rotation" |
| `trace_signature.py` | what a signature (`solvi.core.store.signature`) adds over the hash chain when one stored record is edited |
| `octonion_signature.py` | the positional octonion signature, an experiment that left the package in 0.8 (`trace_signature.py` compares it with the syndrome code) |
| `changes_credit.py` | the change pipeline (`solvi.experimental.changes`) on the credit policy: v1 and v2 as document changes (144 of 144 changed history decisions traced to a clause), 20 corrections of a person (8 refused by impact, 7 held then promoted, 5 admitted, 0 extraneous changes), the `young` rule refused by static, decisions equal to the reference, 14 versions replayed, the journal verified; exits 1 when a count differs (needs `tasks/fetch.sh credit` and `tasks/prepare.py`) |
| `mcp_proxy_filesystem.py` | the MCP proxy with a guard (`solvi serve --guard … --upstream …`) in front of the reference filesystem server (node and `npx` on PATH): one scripted session through the proxy and the same allowed calls to a second server instance — blocked calls absent from the server's wire and from its folder, allowed results equal to the direct call's, malformed messages answered, every call stored, verified and replayed, the overhead per call; `--dry` runs 10 calls |
| `counterfactual_credit.py` | `solvi.experimental.counterfactual` on the credit policy v2 of `changes_credit.py`: for every application v2 refuses or sends to review, the one- and two-field changes that make it "approve" — validity (each explanation decided again), minimality of the amount and duration thresholds against a brute-force scan of the integer domain, coverage, determinism over two runs in fresh processes, evaluations and seconds per search; `--dry` on 20 applications by hash (needs `tasks/fetch.sh credit` and `tasks/prepare.py`) |
| `charts_real_texts.py` | `solvi.experimental.charts` on 50 paragraphs of US federal news releases (`datasets/charts_texts/`: texts, the hand reading of every number written before any proposer ran, every scanned release): every rendered value re-verified against the text, drawn points against the hand reading (label, unit, sign), charts drawn and refusal reasons, `solvi.testing.corrupt` on the recorded proposals, replay and `check_specialist` (the pre-registered charts-on-real-texts stand); `--dry` runs on the three gallery texts of `tests/test_charts.py` |
| `laws_rank_joint.py` | ranking candidate laws (`solvi.experimental.laws.RankedProposer`) on synthetic tasks of three settings (the rank test's grammar, deep conditions, correlated features): checks to the true condition for the blind order, the raw per-atom Jaccard, the trained bag of atoms, the joint statistic of the whole condition (`JointRanker`), a shuffled-rows control, the prior order that reads no rows (`PriorRanker`, with a shuffled-label control) and the hand-over from it to the joint statistic at 20 rows (`HandoverRanker`); `--curve` with the buffer cut to 0 / 5 / 20 / 100 / 300 rows (every setting); deterministic, a few minutes; one seed per run (`--seed`; the numbers in the docs are five seeds) |
| `drift_simulation.py` | what `solvi.core.guarantees.drift.DriftMonitor` flags on simulated streams: false flags on unchanged streams, and how many decisions after a change of the share answered alone or of the mix of answers it flags |
| `textin_extractors.py` | text in on the repository's own texts with typed fields (the shop requests of the guide, the e-mails of `examples/04`, the invoices of `examples/03`, the tickets of gallery 11, the claims of `examples/16`; and, counted apart, 30 texts written for it with string fields that have no pattern): each field read by `CueExtractor`, by the decider's span pointer and by the two in either order — right, wrong, missed against the values the hand-written code reads; no download beyond the decider (the numbers behind `TextIn`'s default extractor) |
| `textin_massive.py` | text in on MASSIVE (en-US, CC BY 4.0; `$MASSIVE_DIR`): routing to eight entry points of a home assistant, and each field read by `CueExtractor` and by the decider's span pointer — right, overlapping, wrong, missed |
| `knowledge/retraction.py` | the knowledge store's retraction exactness: 1,000 random retractions on a synthetic store of 10,000 items, each compared with the store rebuilt without the retracted items; the retraction time against the connected component it touches |
| `knowledge/taubench_action_model.py` | `ConservativeActionModel` on τ-bench retail probes (no LLM; a local clone of tau-bench): refusal precision and recall, abstention, the status effect, and two shorter vocabularies |
| `knowledge/toy_crafting.py` | the knowledge pieces together on a toy crafting world (the protocol of a crafting game, not Crafter): the action model carried across worlds, the agenda, the failure memory, a carried map dropped in a changed world |
| `knowledge/risk_dungeon.py` | `Protect` vs `RiskBudget`, and a learned gate with and without its bound, on a toy dungeon |
| `tasks/` | the task stand: nine public tasks (τ-bench retail, CUAD, RAGTruth, Banking77, German Credit, BIRD mini-dev, Abt-Buy, NATURAL PLAN, NAB), each with a scorer, a baseline without solvi and one solution with solvi; downloads, fixed splits, an OpenRouter client with a cache and a budget, the results table ([tasks/README.md](tasks/README.md)); the published run's model replies and the splits that may be passed on (`tasks/packed/`), and `tasks/stand.py`, which reruns every script from them offline and checks every quoted number against `tasks/results.json` |
| `vs_llm/` | solvi vs asking an LLM: the sets, the written policies, the runner for solvi / a model directly / a model inside solvi (LLMs and the decision models Jev and Jeeves), and every raw answer of the published run ([docs/vs_llm.md](../docs/vs_llm.md)) |
