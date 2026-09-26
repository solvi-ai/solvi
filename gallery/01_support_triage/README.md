# 01 · Support triage

A customer message (plus the customer's tier and when the ticket arrived) goes in. Out come five answers: **intent** (refund,
technical_help, billing_question, information, cancellation, other), **urgent?**, **refund requested?**, **priority** (low / normal /
high) and **route** (billing, tech_support, retention, legal, general).

`cited` `hard checks` `strategist plan` `early exit` `trace replay` `learns in ms` `readable learned rules` `abstains` `runs in browser`

```bash
uv run python gallery/01_support_triage/run.py        # 10 scenarios, exits non-zero on a mismatch
```

## How it is built

- **Evidence with quotes.** `urgency`, `refund_ask`, `legal_threat`, `chargeback_threat` and `anger` are pattern extractors that
  return a `Quote` with offsets into the message. A phrase in a negated clause ("I don't want a refund", "not going to sue")
  counts as evidence for *no*, and the quote covers the negation. "What is your refund policy" is not a refund request.
- **Learned intent, readable.** A `RuleList` learned from 200 synthetic labeled tickets over `cue_words` (stop words dropped, light
  stemming, negated words prefixed with `not`). It takes about 20 ms, so `task.py` learns it at import, in the browser too. Every
  intent answer names the line of the list that fired: `intent_rule_line = "if cue_words has 'CHARG' -> refund (7/7)"`.
- **Hard checks.** `no_legal_threat` forces priority=high and route=legal. `no_chargeback_threat` and `vip_sla_ok` (a VIP waiting
  4 h or more) force priority=high. No rule output can change a forced answer.
- **Abstains.** Without a tier and timestamps, the VIP SLA cannot be checked, so priority abstains. The other four answers stand.

## Sample output (real run)

```
intent rules learned from 200 labeled tickets: 30 lines in 147 ms with System.learn_rule (21 ms fitting RuleList on cue_words alone, as task.py does); accuracy on 500 fresh synthetic tickets 0.906
 1. if cue_words has 'INVOICE' → billing_question   (16/16)
 2. if cue_words has 'FEEDBACK' → other   (10/10)
 ...
[2] negated refund                             1.69 ms   replay ok
    intent=technical_help  urgent=no  refund_requested=no  priority=normal  route=tech_support
    cited: refund_ask=False «don't want a refund» [2:21]
[4] legal threat                               1.31 ms   replay ok
    intent=refund  urgent=no  refund_requested=yes  priority=high (forced)  route=legal (forced)
    cited: legal_threat=True «lawyer» [112:118]
    cited: refund_ask=True «money back» [95:105]
    priority forced: hard check no_legal_threat is false
    route forced: hard check no_legal_threat is false
[7] VIP past the SLA                           1.06 ms   replay ok
    intent=information  urgent=no  refund_requested=no  priority=high (forced)  route=general
    priority forced: hard check vip_sla_ok is false
[10] missing customer record                    0.97 ms   replay ok
    intent=cancellation  urgent=no  refund_requested=no  priority=— (abstain)  route=retention
    priority abstain: cannot compute: tier, vip_sla_ok

10/10 cases match, replay ok on 10/10 traces, median decision 1.45 ms

early exit on 'legal threat': anger (not needed: hard check no_legal_threat failed), answer:priority (...), answer:route (...)
```

## vs an answer-only model

`compare_laya.py` runs the same 10 tickets through [Laya](https://huggingface.co/convaiinnovations/laya), a typed-answer model that
returns an answer and a probability. It uses Laya's own `triage_questions()` preset (intent, is_urgent, refund_requested). We add
priority and route questions in Laya's format, with our rules written into the instructions ("high: a legal threat, a
chargeback threat, a VIP customer who has waited 4 hours or more…"), and pass the tier and the hours waited in the state.

| | solvi | Laya (zero-shot, RTX 3060) |
|---|---|---|
| correct answers on the 10 cases | **50 / 50** | 33 / 50 |
| legal threat → route=legal, priority=high | forced by a hard check | route billing (0.94), priority normal (0.49) |
| VIP waiting 6.5 h → priority=high | forced by a hard check | normal (0.46) |
| "sorted today… finance close is tomorrow" → urgent | yes, cites «sorted today» | no (0.72) |
| "I don't want a refund, I want the Slack integration working" → intent | technical_help | refund (0.45) |
| missing tier and timestamps → priority | abstains, says why | normal (0.65) |
| evidence for an answer | quote with offsets, or the rule line that fired | none |
| median time per ticket | 1.0 ms, CPU | 63 ms, GPU |

Full per-case output of the run above: [compare_laya.out.txt](compare_laya.out.txt).

What this shows for triage:

- **The escalation rule always holds.** Writing it into the model's instructions did not make it hold: of the three forced
  escalations (legal, chargeback, VIP SLA), Laya got priority right only on the chargeback, at probability 0.50. In solvi it is a hard check, so it cannot lose to a probability.
- **Every answer can be checked.** A reviewer sees «lawyer» at [112:118] or the learned line `'CHARG' -> refund (7/7)`. With
  Laya you get a probability and nothing to check it against.
- **Missing data gives an abstain, not a guess.**
- **You can read and fix the learned part.** The 30 intent rules print as a list. Line 3, `'GREAT' -> other`, comes straight from
  one synthetic template, and you can see it.

Caveats: we wrote the 10 cases alongside the catalog, so solvi's 50/50 is not an unbiased accuracy. The learned intent rules score
0.906 on 500 fresh synthetic tickets, and hand-written text outside their vocabulary falls through to the default. Laya ran
zero-shot on its own preset, with no fine-tuning. To reproduce (needs the `laya` package and model; run from the new_kelly
repo, whose env has torch):
`cd exps_v2 && uv run --with laya==0.3.20 --with transformers --with-editable ../solvi python ../solvi/gallery/01_support_triage/compare_laya.py`
