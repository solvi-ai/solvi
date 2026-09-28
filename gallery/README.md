# solvi gallery

Decision tasks from twenty directions, each a small, runnable solvi catalog: Python functions and checks, typed questions,
and rules or learned heads. Every entry here has a `task.py` (the catalog), `state.json` (a default input), `cases.json`
(9–12 scenarios with the expected answers), `run.py` (runs them, prints the strategist's plan, verifies the trace and audits
every answer) and a README that shows what solvi does that an answer-only model cannot. All twelve entries also open in the
[browser playground](https://huggingface.co/spaces/solvi-ai/playground) (Gallery presets) — no install, no server.

```bash
uv run python gallery/07_kyc_aml/run.py          # any entry; exits non-zero if an answer differs from cases.json
uv run solvi test gallery/                       # every entry's cases.json as regression tests (or: uv run pytest gallery/)
```

**Audited (solvi 0.4).** Every runner calls `res.audit()` on every response — what each answer rests on (given inputs,
computed facts, quotes with offsets, learned parts with their fingerprints, checks, constraints) and which safeguards fired —
prints one line per case (support items, deterministic share, safeguards) and asserts the audit's invariants
([`_audit.py`](_audit.py)): every quote lies in its text, and is literally the text where the part is `exact=True` or
model-backed; answers are valid for their type; every forced answer names the hard check that decided it; a repaired answer
has a constraint-repair event; a catalog without learned parts is 100% deterministic. Over the twelve runners: 117 scenario
responses (plus 10 in 03's learned-verdict demo), all invariants hold. Learned parts are labelled as such: the rule lists
of 01 and 02 are registered with `model=`, so the audit counts them as learned (90–91% deterministic there) instead of
passing them off as plain code; 09's `fit_fast` head and 12's `learn_rule` list show up the same way (97–98%).

## Entries

| # | direction | decides | what it shows | solvi features |
|---|---|---|---|---|
| [01](01_support_triage) | customer support | intent, tags, urgency, refund request, priority, route | quoted evidence with negation ("I don't want a refund"); a 30-line intent rule list learned from 200 tickets in ~20 ms; legal threat and VIP SLA as hard rules; signals as one multi-label answer, priority ordinal, tied by a constraint | `cited` `hard checks` `strategist plan` `early exit` `trace replay` `learns in ms` `readable learned rules` `abstains` `multi-label` `ordinal` `constraints` `audited` |
| [02](02_email_routing) | email operations | team, needs a human | a readable routing table learned in ~15 ms; split votes abstain (500 fresh emails: 428 right, 72 abstained, 0 wrong); lookalike domain forces security | `cited` `hard checks` `strategist plan` `early exit` `trace replay` `learns in ms` `readable learned rules` `abstains` `audited` |
| [03](03_content_guard) | trust & safety | allow / review / block, harm types | e-mail, phone, card (Luhn), IBAN (mod-97), secrets and prompt injection found with their spans; a block no score can undo; harm types (multi-label) tied to the verdict by constraints that repair a learned verdict | `cited` `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `multi-label` `constraints` `audited` |
| [04](04_security_alert) | security | suspicious login, containment | impossible-travel speed (haversine), failed-login spike, new device; admin + impossible travel locks the account | `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `audited` |
| [05](05_agent_trace_audit) | AI-agent governance | approve / roll back / escalate | destructive commands, leaked secrets, cost overruns in an agent's tool log; a computed rollback plan; solvi's own trace as the audit record | `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `audited` |
| [06](06_release_rollout) | DevOps | promote / hold / roll back, page on-call | error increase, two-proportion z-test, p95 from histograms, SLO burn; an error ceiling rolls back and skips 9 steps | `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `audited` |
| [07](07_kyc_aml) | compliance | risk, file a SAR, freeze | fuzzy sanctions match plus birth date; structuring (deposits just under 10 000) computed; a sanctions hit skips the paid lookups | `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `audited` |
| [08](08_clinical_screening) | healthcare (demo) | escalation, NEWS2 band, sepsis screen | NEWS2 exactly per the RCP table, qSOFA; missing vitals abstain; escalation and band ordinal, tied by a constraint; a linear answerer trained on 4 000 patients missed 144–156 of 501 emergencies | `hard checks` `trace replay` `abstains` `ordinal` `constraints` `audited` |
| [09](09_credit_adverse_action) | lending | approve / decline / refer, adverse-action reasons | reasons in Regulation B wording as computed facts; "refer to underwriter" learned with `fit_fast`, corrected by `teach` in 0.17 ms | `hard checks` `early exit` `trace replay` `learns in ms` `abstains` `audited` |
| [10](10_procurement_3way_match) | procurement | pay / hold / reject, duplicate, approver | PO / receipt / invoice matched per line with tolerances in base currency (FX), duplicate "INV-001187" = "inv 1187"; the rate from the table, else from a same-day feed (a stale rate is rejected by `validate`) | `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `fallback producers` `typed` `audited` |
| [11](11_refund_double_charge) | payments support | double charge?, refund, reply | the customer's claim is quoted, the decision reads the ledger — they disagree in 4 of 9 cases | `cited` `hard checks` `early exit` `trace replay` `audited` |
| [12](12_predictive_maintenance) | industrial IoT | ok / watch / service / stop, likely fault | least-squares trends, z-scores, hours to the alert level; hard trips work with a sensor offline; 6 readable learned rules | `hard checks` `trace replay` `readable learned rules` `abstains` `audited` |

More directions in [examples/](../examples): HR leave approval (01), e-commerce fraud with a learned head (02), accounts payable
(03), refunds with a hard 30-day rule (04), a game agent that never loses (05), logistics routing with learned rules (06),
receipts with a fine-tuned model (07), contract review by field description (08), an insurance claim desk with generated
strategies, early exit and parallel services (09), learning in milliseconds (10). Games: the
[arcade](https://huggingface.co/spaces/solvi-ai/arcade).

## Against an answer-only model

Entries 01–06 were also run through [Laya](https://pypi.org/project/laya/) 0.3.20, a model that answers typed questions directly
(zero-shot, its own presets for 01–03 and the rules written into its questions; RTX 3060; scored on the answers both were
asked — 01's `tags` and 03's `harm`, added with solvi 0.4, are not scored):

| entry | solvi correct | Laya correct | time per decision, solvi (CPU) / Laya (GPU) |
|---|---|---|---|
| 01 support triage | 50 / 50 | 33 / 50 | ~1 ms / 63 ms |
| 02 email routing | 18 / 18 | 11 / 18 | 0.65 ms / 42 ms |
| 03 content guard | 30 / 30 | 18 / 30 | 0.76 ms / 56 ms |
| 04 security alert | 18 / 18 | 8 / 18 | 1.05 ms / 78 ms |
| 05 agent trace audit | 20 / 20 | 12 / 20 | 0.95 ms / 62 ms |
| 06 release rollout | 18 / 18 | 7 / 18 | 0.93 ms / 64 ms |

Read this as a demonstration, not a benchmark: the cases were written together with the catalogs, and the answer-only model saw
the rules only as text. The structural differences hold regardless of the numbers — solvi's hard rules cannot be overridden,
numbers are computed instead of guessed, every extracted value is quoted with its offsets, the trace re-verifies, and the
system abstains instead of guessing. Each entry's `compare_laya.py` and `compare_laya.out.txt` reproduce and record the run;
measured document benchmarks are in [docs/benchmarks.md](../docs/benchmarks.md).
