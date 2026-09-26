# 07 · KYC / AML screening

A customer profile and 90 days of transactions go in; three decisions come out: the customer's **risk level**
(low / medium / high), whether to **file a suspicious activity report** (SAR), and whether to **freeze the account**.

`hard checks` `strategist plan` `early exit` `trace replay` `abstains` `runs in browser`

## How it decides

- **Sanctions screening is a function**, not a guess: names are normalised (accents, case, punctuation, word order) and
  compared with `difflib`; a hit needs a name score of at least 0.88 *and* no conflicting birth date. "Dmitriy Volkanov"
  born 1968-03-14 hits the listed "Dmitri Volkanov" (score 0.968); a "Dmitry Volkanov" born in 1991 scores 0.933 but is
  cleared by the birth date.
- **`customer_not_sanctioned` is a hard check** in every question's checkpoints, with `then = {risk: high, file_sar: yes,
  freeze: yes}`. No rule can override it. The strategist orders it first, so when it fails the rest of the flow, including
  the two paid lookups (news search, counterparty screening), is never run.
- **Computed facts** carry the evidence: `structuring` (most cash deposits of 9 000-9 999 inside any 7-day window, with
  dates and total), `velocity_ratio` (30-day volume / declared monthly volume), `pass_through` (wires sent on within 2 days
  for 90%+ of the amount), `high_risk_share`, PEP and adverse-media screening, counterparty screening.
- **Rules** are short and read those facts: `risk` sums named `risk_factors`; `file_sar` is "there are `sar_grounds`" (a
  PEP alone is risk, not suspicion); `freeze` needs a sanctioned party (a SAR alone must not freeze: that tips the
  customer off).

## Run

```bash
uv run python gallery/07_kyc_aml/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

Nine cases in `cases.json`: a clean salary account, structuring under the threshold, a transliterated sanctions hit, a
namesake with another birth date, a PEP with high-risk flows (high risk but no SAR), a cash business whose deposits are
spread out (no structuring), a money-mule pass-through, a sanctioned counterparty, and a profile without a declared
volume (risk abstains). `state.json` is the structuring case, for the playground.

## Sample output (real run)

```
[2/9] structuring_under_threshold  (four cash deposits of 9 400-9 950 in 6 days: ...)
    risk      high     ok       1.00  risk_factors = [('structuring', 3), ('volume 3x profile', 2)]
    file_sar  yes      ok       1.00  sar_grounds = ['structuring: 4 cash deposits just under 10000 (2026-09-11..2026-09-16, total 38900.0)']
    freeze    no       ok       1.00  counterparty_screening = []
    12 steps run, 0 skipped · replay ok (15 records) · 2.91 ms · expected ✓

[3/9] sanctions_hit_transliterated  ('Dmitriy' vs listed 'Dmitri', same birth date: ...)
    risk      high     forced   1.00  hard check customer_not_sanctioned is false
    file_sar  yes      forced   1.00  hard check customer_not_sanctioned is false
    freeze    yes      forced   1.00  hard check customer_not_sanctioned is false
    screening: 'Dmitri Volkanov' name score 0.968, birth date match True -> hit True
    2 steps run, 13 skipped (adverse_media, counterparty_screening, high_risk_share, pep_screen, ...) · replay ok (2 records) · 0.92 ms · expected ✓

[9/9] no_declared_profile  (the onboarding profile has no expected monthly volume: ...)
    risk      abstain  abstain  0.00  rule not computed: missing inputs: risk_factors; missing risk_factors, velocity_ratio
    file_sar  no       ok       1.00  sar_grounds = []

── the trace as audit evidence
   recorded: structuring = {'count': 4, 'total': 38900.0, 'from': '2026-09-11', 'to': '2026-09-16'}
   someone edits the record to count 2 after the fact -> replay: [(9, 'structuring', 'record modified after execution'), ...]

9/9 cases as expected; decision time median 3.49 ms, max 6.02 ms
```

## vs an answer-only model

An answer-only model here means one that returns `risk = high (0.91)`, `file_sar = yes (0.87)` with no citation and no
rules. What this task needs that such a model does not give:

- **The sanctions rule is guaranteed.** A confirmed hit forces all three answers with status `forced`; this holds for
  every input, not "usually". A probability of 0.97 for "freeze" still means some sanctioned customers are not frozen.
- **A SAR has to state its grounds.** Here they are computed facts with dates and totals: 4 deposits between 2026-09-11
  and 2026-09-16, total 38 900. The negative case is just as explicit: three deposits of 9 500 that are 5-7 days apart
  give at most 2 in any 7-day window, so no structuring. An answer-only model gives a label, and the analyst still has
  to rebuild the evidence by hand.
- **False positives are cleared by a stated reason**: the namesake's 0.933 name score is overruled by the birth date,
  and that fact is in the record.
- **It saves paid lookups.** In the sanctions case 13 of 15 steps are skipped, including both external lookups. Over the
  nine cases, 16 lookups were made and 2 skipped. An answer-only model needs all of its inputs fetched before it can answer.
- **The trace is audit evidence.** `replay` recomputes every step from the recorded input: editing the structuring
  count after the decision is caught and the step named. A model's logged probability cannot be re-derived.
- **It abstains on missing data.** Without a declared volume, risk abstains and names the missing fact, while SAR and
  freeze are still answered because they do not need it. An answer-only model outputs a risk level anyway.

## Files

`task.py` (catalog, questions, `prepare`), `state.json` (default input), `cases.json` (9 scenarios), `run.py`.
Names, lists and news items are synthetic; the high-risk country list is illustrative. Not compliance advice.
