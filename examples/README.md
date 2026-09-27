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
| [10_learn_in_milliseconds.py](10_learn_in_milliseconds.py) | any | `fit_fast`: learn a question in milliseconds and absorb each correction instantly with `teach` | core |
| [11_answer_types_and_constraints.py](11_answer_types_and_constraints.py) | trust & safety | multi-label and ordinal answers, constraints between answers, joint decoding | core |
| [12_grounded_audit.py](12_grounded_audit.py) | expenses | one catalog with and without models: provenance, `res.audit()`, a hallucinated quote caught, a decision outside its options, a changed model, safeguard stats | core |
| [13_decide_model.py](13_decide_model.py) | customer support | a decider model as a catalog part: bias correction without labels, few-shot shift with `teach`, "other" as a threshold, abstention, constraints, audit, escalation for a target error rate, a JSON ticket | core (stand-in); the real model with `solvi[onnx]` or `solvi[model]` |
| [14_typed_catalog.py](14_typed_catalog.py) | customs | typed facts: type hints checked between producers and consumers at registration, a pydantic request (`System(inputs=...)`), answer types from the rules' return types, `type_rejected` → fallback / abstention, the response as JSON that loads back and replays | core |
| [15_typed_decisions.py](15_typed_decisions.py) | customer support | typed decisions: the questions as a pydantic model's fields (choice, ordinal score, yes/no, multi-label) about a pydantic ticket, four answers from one forward pass, act / escalate, a hard check, a constraint and a rule over the model, audit and stats | core (stand-in); the real model with `solvi[onnx]` or `solvi[model]` |
| [16_primitives.py](16_primitives.py) | insurance claims | answer primitives, each a value and a confidence: "not stated" (`Maybe[bool]`) vs abstain, a span parsed into a float, evidence quotes checked in the text (`require_evidence`), a ranking (`Rank`), an estimate with an interval (`Estimate`) — from plain rules and from a decider with the L14g contract; the confidence table, JSON round trip and replay | core (stand-in); an L14g checkpoint with `SOLVI_DECIDE_MODEL` |
| [17_model_strategist.py](17_model_strategist.py) | payments | the model strategist (experimental): a dead end the deterministic strategist cannot plan around, the cheapest verified plan with declared costs, a model's proposal checked (a bad one falls back), the plan in the trace; aliases for another team's names accepted by examples and targeted questions | core (stand-ins); the trained model with `SOLVI_STRATEGIST` |
| [07_receipts_model.py](07_receipts_model.py) | expenses | a receipts-tuned ModernBERT extractor cites each field; rules decide | `solvi[model]` |
| [08_contracts_by_description.py](08_contracts_by_description.py) | legal | fields defined only by description, read from a whole contract, cited or "absent" | `solvi[model]` |

Models for 07 and 08 load from Hugging Face (`solvi-ai/extract-receipts`, `solvi-ai/extract-base`) or from a local directory
given in `SOLVI_MODEL`. Examples 13, 15 and 16 use the decider from `SOLVI_DECIDE_MODEL` (a checkpoint folder or a Hugging Face id) when
it is set, and a keyword stand-in otherwise.
