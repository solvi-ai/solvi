# Benchmarks

Dataset loaders for the public benchmarks behind the numbers in the [README](../README.md). Each loader turns a dataset
into plain documents (`key`, `text`), reference fields or clause spans, and typed questions with ground truth.

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

## Results

Setups, per-question numbers and caveats are in [docs/benchmarks.md](../docs/benchmarks.md).

The scripts that reproduce those numbers are being ported from the research repository and will land in this
directory before the first release.

## Scripts

| Script | What it measures |
|---|---|
| `perturb_injection.py` | `perturb=k` against instructions appended to Bitext support messages: how often they are followed, what it costs |
| `fast_head.py`, `strategist_scale.py` | `fit_fast` and the strategist at scale |
| `textin_massive.py` | text in on MASSIVE (en-US, CC BY 4.0; `$MASSIVE_DIR`): routing to eight entry points of a home assistant, and each field read by `CueExtractor` and by the decider's span pointer — right, overlapping, wrong, missed |
| `vs_llm/` | solvi vs asking an LLM: the sets, the written policies, the runner for solvi / a model directly / a model inside solvi (LLMs and the decision model Jev), and every raw answer of the published run ([docs/vs_llm.md](../docs/vs_llm.md)) |
