# 09 · Credit decision with adverse-action reasons

A consumer loan application goes in. Out come a **decision** (approve / decline / refer), the **principal
adverse-action reason** together with the full list of up to four reasons that a decline notice must state (ECOA /
Regulation B, FCRA style), and **"refer to a senior underwriter?"**, a question learned from past files.

`hard checks` `early exit` `trace replay` `learns in ms` `abstains` `audited` `runs in browser`

## How it decides

- **Scorecard.** Eight factors add up to 100 points: DTI after the new loan, credit score, delinquencies, length of
  history, utilisation, recent inquiries, length of employment, and loan-to-income. At 72 points or more the application
  is approved, from 58 to 71 it is referred, and below 58 it is declined. Three knock-outs always decline: DTI > 50 %,
  score < 580, and 3 or more delinquencies.
- **The adverse-action reasons are a computed fact.** Knock-outs come first, then the factors that lost the most points
  (at least a quarter of their maximum). At most four are listed, worded like the Regulation B sample notice. Because the
  reasons come from the same scorecard that made the decision, they always explain *that* decision.
- **Eligibility is a hard check**: age 18 or over (capacity to contract), and citizen or permanent resident. A failure
  forces decline, the matching reason, and "no referral". The scoring steps are skipped (early exit).
- **"Refer to underwriter?" has no written rule.** `run.py` learns it with `fit` from 200 synthetic past files;
  an underwriter then reviews 300 new files, each verdict absorbed by `teach`. In the playground (no history) it abstains.

## Run

```bash
uv run python gallery/09_credit_adverse_action/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

`cases.json` has 10 applications: a DTI knock-out, a thin file (refer), a decline with four reasons, two files $2 apart
on either side of the DTI 0.36 band edge, an underage applicant, one with temporary residence, a score knock-out despite
a high income, and two approved files that the learned habit still sends to a senior.
`state.json` is the thin file.

## What the audit shows

`run.py` asserts the audit's invariants on every application (see [`_audit.py`](../_audit.py)): 10/10 files, 520 support items,
98% deterministic — the only non-deterministic item is the learned underwriter head — hard check decided ×6. From
`res.audit("refer_to_underwriter")` on the self-employed thin file (the head's fingerprint changes with every `teach`):

```
refer_to_underwriter = 'no'  [ok]  confidence 0.53  ← learned by FastHead
  computed    dti = 0.1161
  learned     answer head (FastHead) = 'no'  (no 0.53, yes 0.47)  [FastHead FastHead #0604ac27]
  check       adult = True (hard)
  support     17 items (11 given, 5 computed, 1 learned): 94% deterministic, 1 from models
```

## Sample output (real run)

```
'refer_to_underwriter' learned from 200 past files in 331.3 ms (exact leave-one-out 0.920); accuracy on 400 new files 0.887

[3/10] decline_four_reasons  (44 points: the notice lists the four factors that lost the most points)
    decision             decline                                      ok      1.00
    principal_reason     Credit score insufficient                    ok      1.00
    points 44/100, DTI 0.243, knock-outs none
    adverse-action reasons: ['Credit score insufficient', 'Delinquent past or present credit obligations with others', 'Proportion of balances to credit limits is too high', 'Number of recent inquiries on credit bureau report']

[5/10] dti_just_over_36  (the same file with $2 more monthly debt: DTI 0.3601 scores 9, 69 points, refer)
    decision             refer                                        ok      1.00
    principal_reason     Excessive obligations in relation to income  ok      1.00

[6/10] underage  (hard check: no score can approve a 17-year-old)
    decision             decline                                      forced  1.00
    principal_reason     Applicant under the legal age to contract    forced  1.00
    refer_to_underwriter no                                           forced  1.00
    hard check adult is false; scoring skipped (9 steps not run)

── an underwriter reviews 300 new files; each verdict is taught to the head at once
   300 verdicts (30 of them corrections), each absorbed in 0.166 ms (median, max 0.906 ms)
   accuracy on the same 400 new files: 0.887 -> 0.897 (refitting from scratch on all 500 files gives 0.917)
   answers on the cases changed by these verdicts: decision 0; principal_reason 0; refer_to_underwriter 1 [self_employed_thin_file yes -> no (p 0.53)]
   (the learned answer may move; decisions and reasons come from rules and hard checks, which teach does not touch)

10/10 cases as expected; decision time median 1.64 ms, max 1.93 ms
```

## vs an answer-only model

An answer-only model returns `decline (0.83)`: a typed answer and a probability, with no rules behind it.

- **The regulation asks for reasons, and they have to be the real ones.** A decline notice must state the principal
  reasons. Here the reasons are part of the computation: the factors that cost the most points in the scorecard that
  declined the application. With an answer-only model, reasons can only be attached afterwards, for example by an
  attribution method, and nothing guarantees they match what drove the model.
- **Thresholds are exact and can be explained.** Adding $2 of monthly debt moves DTI from 0.3597 to 0.3601. The DTI
  factor drops from 15 to 9 points and the file goes from 75 points (approve) to 69 (refer). The reason list says why.
  A smooth model has no line at 0.36 to point to.
- **Eligibility and knock-outs are guaranteed.** Age is used only as capacity to contract, and it is a hard check, not a
  learned weight. A 17-year-old is declined for every input, and a score of 561 is declined even with a $140 000 income.
  A model can only make these outcomes likely.
- **The learned part stays inside its box.** Only "refer to underwriter?" is learned: 0.920 leave-one-out, and 0.887 →
  0.897 on 400 new files after 300 underwriter verdicts absorbed at 0.17 ms each (median). The verdicts moved one learned
  answer among the cases (the self-employed thin file, yes → no at p 0.53, so it is now a borderline call). They moved
  no decision and no reason. Retraining an answer-only model on corrections can change everything it outputs, decisions
  included.
- **An examiner can re-run the decision.** `replay` recomputes every step, from the payment to the reasons, and the
  hash chain shows that the record was not edited after the notice went out.

## Files

`task.py` (catalog and questions), `state.json`, `cases.json` (10 applications), `run.py` (history, cases, underwriter
corrections). The policy, thresholds and data are synthetic, not lending advice.
