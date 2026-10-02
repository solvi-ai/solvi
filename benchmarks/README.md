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
| `fast_head.py`, `strategist_scale.py` | `fit` (the answer head and its selection) and the strategist at scale |
| `ask_speed.py` | milliseconds per `ask` on small and large inputs (the README quickstart, the gallery, random catalogs of 50 and 1 000 parts, a 3.7k-float stream input): the README's Speed table |
| `ask_overhead.py` | what solvi adds to one `ask` (every gallery entry and a decider project, no real model) |
| `trace_signature.py` | what a signature (`solvi.signature`) adds over the hash chain when one stored record is edited |
| `octonion_signature.py` | the positional octonion signature, an experiment that left the package in 0.8 (`trace_signature.py` compares it with the syndrome code) |
| `drift_simulation.py` | what `solvi.drift.DriftMonitor` flags on simulated streams: false flags on unchanged streams, and how many decisions after a change of the share answered alone or of the mix of answers it flags |
| `textin_extractors.py` | text in on the repository's own texts with typed fields (the shop requests of the guide, the e-mails of `examples/04`, the invoices of `examples/03`, the tickets of gallery 11, the claims of `examples/16`; and, counted apart, 30 texts written for it with string fields that have no pattern): each field read by `CueExtractor`, by the decider's span pointer and by the two in either order — right, wrong, missed against the values the hand-written code reads; no download beyond the decider (the numbers behind `TextIn`'s default extractor) |
| `textin_massive.py` | text in on MASSIVE (en-US, CC BY 4.0; `$MASSIVE_DIR`): routing to eight entry points of a home assistant, and each field read by `CueExtractor` and by the decider's span pointer — right, overlapping, wrong, missed |
| `vs_llm/` | solvi vs asking an LLM: the sets, the written policies, the runner for solvi / a model directly / a model inside solvi (LLMs and the decision models Jev and Jeeves), and every raw answer of the published run ([docs/vs_llm.md](../docs/vs_llm.md)) |
