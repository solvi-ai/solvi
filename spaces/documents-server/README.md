---
title: solvi documents
emoji: 🧾
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
license: apache-2.0
short_description: Typed, cited answers from receipts and contracts
suggested_hardware: zero-a10g
models:
  - solvi-ai/extract-base
  - solvi-ai/extract-receipts
---

# solvi documents

A demo of [solvi](https://github.com/solvi-ai/solvi): typed answers (yes/no or a choice) from documents. Each answer comes
with a confidence, a reason you can check, quotes at exact character offsets and a hash-chained trace that is replayed after
every run.

- **Receipts**: company, date, total, tax, cash and change are found by `solvi-ai/extract-receipts`. A solvi catalog
  answers: reimbursable (amount within a limit, under the hard checks "not older than N days" and "issuing company
  named"), bought on a weekend, change correct (`cash - total = change`), tax shown.
- **Contracts: ask by description**: write a field name and a plain-English description, and `solvi-ai/extract-base`
  looks for it across the whole contract (overlapping windows), returning a quote or "absent". Fixed typed questions:
  governing law (Delaware / New York / California / other), termination notice of at least 30 days, liability capped,
  non-compete.
- **Compare**: benchmark numbers against a model that answers directly, and when to fine-tune.

All sample documents are synthetic.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SOLVI_RECEIPTS_MODEL` | `solvi-ai/extract-receipts` | receipts extractor (Hugging Face id or local directory) |
| `SOLVI_BASE_MODEL` | `solvi-ai/extract-base` | general extractor, fields by description |

The app runs on ZeroGPU (`@spaces.GPU`), on a regular GPU, or on CPU (much slower for ModernBERT-large).

## Run locally

```bash
pip install "gradio>=6" "solvi[model]"
python app.py                                  # from a clone of solvi: PYTHONPATH=../../src python app.py
```
