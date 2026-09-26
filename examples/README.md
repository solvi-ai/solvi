# Examples

Each file is a self-contained script. Run from the repository root with `uv run python examples/<file>`.

| file | domain | what it shows | needs |
|---|---|---|---|
| [01_leave_request.py](01_leave_request.py) | HR | rules and hard checks over a plain dict; the strategist skips catalog parts the questions do not need | core |
| [02_shop_order.py](02_shop_order.py) | e-commerce | two answers by rules, one ("suspicious?") learned from labeled history with `fit` | core |
| [03_invoices.py](03_invoices.py) | accounts payable | fields extracted from invoice text with quotes, checks on computed facts, a learned risk level | core |
| [04_refunds.py](04_refunds.py) | customer support | a learned yes/no that a hard "within 30 days" check always overrides | core |
| [05_tic_tac_toe.py](05_tic_tac_toe.py) | games | an agent from small functions over the board; never loses (checked exhaustively) | core |
| [06_learned_rules.py](06_learned_rules.py) | logistics | `learn_rule`: a readable if-then list learned from labeled addresses | core |
| [09_strategy_at_scale.py](09_strategy_at_scale.py) | insurance | a different generated plan per question set on a big catalog, early exit on hard checks, parallel slow services; timed | core |
| [07_receipts_model.py](07_receipts_model.py) | expenses | a receipts-tuned ModernBERT extractor cites each field; rules decide | `solvi[model]` |
| [08_contracts_by_description.py](08_contracts_by_description.py) | legal | fields defined only by description, read from a whole contract, cited or "absent" | `solvi[model]` |

Models for 07 and 08 load from Hugging Face (`solvi-ai/extract-receipts`, `solvi-ai/extract-base`) or from a local directory
given in `SOLVI_MODEL`.
