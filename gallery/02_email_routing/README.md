# 02 · Email routing

An inbound email (sender, subject, body) goes in. Out come **team** (billing, technical, sales, security, hr) and
**needs_human**, which says whether a person should look before the email is routed.

`cited` `hard checks` `strategist plan` `early exit` `trace replay` `learns in ms` `readable learned rules` `abstains` `audited` `runs in browser`

```bash
uv run python gallery/02_email_routing/run.py        # learns the table, runs 9 scenarios, exits non-zero on a mismatch
```

## How it is built

- **A learned routing table you can read.** `RuleList` learns from 200 synthetic labeled emails, over the subject and body
  words (stop words dropped, light stemming). Learning takes 9–19 ms, so `task.py` does it at import, in the browser too.
- **Abstain instead of the default.** Every line of the table that matches the email votes for its team, weighted by the
  line's support. A team is chosen only if it has at least twice the votes of the runner-up. With a split vote or no matching
  line, `team` abstains (`why` shows the votes) and `needs_human` says yes. A plain decision list would have answered with its
  default, `else → billing`.
- **Hard check on the sender.** A domain that imitates ours (`acme-io.com`, `acrne.io`, `acme.io-secure.net`) forces
  team=security and needs_human=yes. It runs first; when it fails, the rest of the flow is skipped (early exit).
- **Cited risk signals.** `payment_change_ask` ("update my bank details") and `credential_ask` ("verify your password") come
  with quotes. A bank-detail change from outside the company needs a human even when the team is clear.

## What the audit shows

`run.py` asserts the audit's invariants on every email (see [`_audit.py`](../_audit.py)). `team_votes` is registered with
`model=ROUTING`, so the learned table shows up as **learned** with its fingerprint: 9/9 emails, 174 support items, 91%
deterministic, hard check decided ×2. A split vote makes the team rule return `None` — a deliberate abstention, which the audit reports
as its own safeguard (rule abstained ×2), separate from an answer outside the options — visible, not silent. From `res.audit("team")` on the split vote:

```
team = '—'  [abstain]  confidence 0.00  ← computed by team
  quoted      sender_domain = 'orbit.co'  sender[4:12] literal 'orbit.co'
  learned     team_votes = {'security': 6, 'technical': 4}  [RuleList RuleList #a66dde49]
  support     9 items (3 given, 4 computed, 1 quoted, 1 learned): 89% deterministic, 1 from models
  safeguards  rule abstained ×1
```

## Sample output (real run)

```
routing table learned from 200 labeled emails in 116 ms with System.learn_rule (19 ms fitting RuleList on the one fact, as task.py does):
 1. if words has 'LEAVE' → hr   (24/24)
 2. if words has 'INVOICE' → billing   (18/18)
 ...
16. if words has 'PAGE' → security   (6/6)
 ...
26. if words has 'PENSION' → hr   (3/3)
else → billing
on 500 fresh synthetic emails: 428 routed correctly, 72 abstained (sent to a person), 0 routed wrong

[6] payment page error (split vote)            0.71 ms   replay ok
    team=— (abstain)  needs_human=yes
    cited: sender_domain=orbit.co «orbit.co» [4:12]
    team abstain: rule returned None, not one of the answer options; team_votes = {'security': 6, 'technical': 4}; clear_winner = False
[8] lookalike CEO wire request                 0.29 ms   replay ok
    team=security (forced)  needs_human=yes (forced)
    cited: sender_domain=acme-io.com «acme-io.com» [24:35]
    team forced: hard check sender_not_lookalike is false
    needs_human forced: hard check sender_not_lookalike is false
[9] payroll bank change from Gmail             0.71 ms   replay ok
    team=hr  needs_human=yes
    cited: sender_domain=gmail.com «gmail.com» [20:29]
    cited: payment_change_ask=True «update my bank details» [28:50]
    team ok: team_votes = {'hr': 12}; clear_winner = True

9/9 cases match, replay ok on 9/9 traces, median decision 0.71 ms

early exit on 'lookalike CEO wire request': words (not needed: hard check sender_not_lookalike failed), sender_kind (...), team_votes (...), ...
```

The split vote in case 6 comes from line 16, `'PAGE' → security`. It was learned from one template ("a vulnerability in your
login page"). The table makes that visible, so you can fix the data instead of wondering why the email was not routed.

## vs an answer-only model

`compare_laya.py` runs the 9 emails through [Laya](https://huggingface.co/convaiinnovations/laya) with its own
`email_questions()` preset (category with an extra `other` option, is_phishing). We add a needs_human question in Laya's format
with our rules written into the instructions. Laya's `other` counts as a correct match wherever solvi abstains.

| | solvi | Laya (zero-shot, RTX 3060) |
|---|---|---|
| correct answers on the 9 emails | **18 / 18** | 11 / 18 |
| CEO wire request from `acme-io.com` | security + needs_human, forced | billing (1.00), needs_human no (0.70), is_phishing 0.10 |
| payment page error, unclear team | abstains, shows the split vote | billing (0.73), needs_human no (0.58) |
| "see the attached file", nothing to route on | abstains | other (0.84), but needs_human no (0.80) |
| parental leave from an employee | hr | other (0.39) |
| how the routing works | 26 printed lines, learned in 9–19 ms from 200 labels | weights inside the model |
| median time per email | 0.65 ms, CPU | 42 ms, GPU |

Full per-case output of the run above: [compare_laya.out.txt](compare_laya.out.txt).

- **The spoofing rule holds whatever the words say.** Laya read "invoice payment… new supplier… bank details" and answered
  billing at probability 1.00, with is_phishing at 0.10. In solvi the lookalike-domain check runs before anything else and
  decides.
- **You can see and change the routing.** The learned table is 26 lines. To move "payment page" emails to technical, you add
  labeled examples or edit a line, and replay still verifies every decision. You cannot audit a probability this way.
- **Unclear mail goes to a person, visibly.** solvi abstained on 72 of 500 fresh synthetic emails and routed none of them
  wrong. The answer-only model always picks a team.

Caveats: we wrote the 9 cases alongside the catalog, so 18/18 is not an unbiased accuracy. The 500-email check uses the same
generator as the training data. The table only knows the words its 200 examples contained. Reproduce the comparison from the repository root:
`uv run --with laya==0.3.20 --with torch --with transformers python gallery/02_email_routing/compare_laya.py`
