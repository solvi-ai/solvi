# 10 · Procurement: 3-way match

A purchase order, a goods receipt and a supplier invoice (all JSON) go in. Out come **pay / hold / reject**, **already
paid (duplicate)?** and **needs a higher approver?**

`hard checks` `strategist plan` `early exit` `trace replay` `abstains` `fallback producers` `audited` `runs in browser`

## How it decides

- **Line-by-line 3-way match** (`line_match`). Each invoice line has to exist on the PO. Nothing may be billed beyond
  the received quantity. The unit price may differ from the PO price by at most **±2 %**, and the comparison is made
  after both prices are **converted to the base currency**, so a PO in EUR can be invoiced in USD. Every failing line
  becomes a sentence in `lines_failing`: `line 1 BX-4020: price +2.45% vs PO (tolerance ±2.0%)`.
- **Currency conversion before the approval limit.** The invoice total is converted at the day's rate and compared with
  the limit of the approver's role (clerk 10 000, manager 50 000, director 250 000 USD).
- **Duplicates** are matched on a normalised invoice number from the same supplier: `INV-001187` = `inv 1187`.
- **Three hard checks**, all checkpoints of `payment`, reject the invoice: supplier not active, invoice for another PO or
  supplier, already paid. The strategist puts them and their inputs first, so when one fails the line match is not
  computed at all (early exit).
- **Two producers for the invoice rate (solvi 0.4).** `fx_rate` has alternative producers, tried in order:
  `fx_rate_table` (the company's rate table, `validate=` rate > 0) and, if the table has no rate, `fx_rate_feed` (the daily
  reference feed in `fx_feed`, accepted only if `validate=` finds the feed dated the invoice day — a stale rate is rejected).
  The record of `fx_rate` says which producer was used and why the ones before it were rejected
  (`tried fx_rate_table (no value), fx_rate_feed (accepted)`); the audit shows it as a fallback, and replay re-checks both.
  The PO rate stays table-only.
- **No rate, no guess.** A missing rate is "not found", never 1.0: when neither producer is accepted, the answers that need
  amounts abstain, and the duplicate check, which does not need them, still answers.

## Run

```bash
uv run python gallery/10_procurement_3way_match/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

`cases.json` has 12 invoices: a clean EUR match, prices at +1.90 % (pay) and +2.45 % (hold, one penny apart), a short
delivery billed in full, GBP converted over the clerk's limit, a PO in EUR invoiced in USD, a duplicate with a
reformatted number, a supplier on hold, a currency without a rate in the table or the feed (abstain), a header total that
does not add up, and two JPY invoices with no JPY in the table: the feed's same-day rate is used (pay), and a feed rate nine
days old is rejected by `validate` (abstain).
`state.json` is the currency-over-limit invoice. Try setting `approver.role` to `manager` there.

## What the audit shows

`run.py` asserts the audit's invariants on every invoice (see [`_audit.py`](../_audit.py)): 12/12 invoices, 419 support items,
100% deterministic, hard check decided ×3, fallback producer ×2, rejected by validate ×2. From `res.audit("payment")` on the JPY
invoice paid with the feed's rate:

```
payment = 'pay'  [ok]  confidence 1.00  ← computed by payment
  computed    fx_rate = 0.00622
  computed    invoice_total_base = 8469.15
  check       within_approval_limit = True (soft)
  support     22 items (9 given, 13 computed): 100% deterministic
  safeguards  fallback producer ×1
              · fallback producer: fx_rate — fx_rate_feed used after fx_rate_table rejected
```

## Sample output (real run)

```
[3/12] price_over_tolerance  (one penny more per box: +2.45%, outside the tolerance -> hold that line)
    payment           hold     ok       1.00  lines_failing = ['line 1 BX-4020: price +2.45% vs PO (tolerance ±2.0%)']; header_total_ok = True; within_approval_limit = True
    lines: 1 BX-4020 qty 2500/2500 price +2.45% ✗; 2 TP-48 qty 600/600 price +0.00% ✓

[5/12] fx_pushes_over_limit  (8 200 GBP looks under the 10 000 limit, but at 1.27 it is 10 414 USD: a manager must approve)
    payment           hold     ok       1.00  lines_failing = []; header_total_ok = True; within_approval_limit = False
    escalate_approval yes      ok       1.00  within_approval_limit = False
    amount 8,200.00 GBP = 10,414.00 USD (limit 10,000 for clerk)

[7/12] duplicate_reformatted_number  ('INV-001187' is 'inv 1187', paid on 2026-09-02: reject; the line match is skipped)
    payment           reject   forced   1.00  hard check not_duplicate is false
    duplicate         yes      ok       1.00  duplicate_of = 'inv 1187'
    10 steps run, 5 skipped (header_total_ok, po_fx_rate, line_match, lines_failing, answer:payment) · replay ok · 0.85 ms · expected ✓

[9/12] missing_fx_rate  (invoiced in JPY and there is no JPY rate: amounts cannot be compared or limited, so no guess; ...)
    payment           abstain  abstain  0.00  rule not computed: missing inputs: lines_failing, within_approval_limit; missing fx_rate, ...
    duplicate         no       ok       1.00  duplicate_of = None
    fx_rate —: used none; tried fx_rate_table (no value), fx_rate_feed (no value)

[11/12] rate_from_fallback_feed  (no JPY in the rate table; the daily feed has JPY for the invoice date (0.00622): ...)
    payment           pay      ok       1.00  lines_failing = []; header_total_ok = True; within_approval_limit = True
    amount 1,361,600.00 JPY = 8,469.15 USD (limit 10,000 for clerk)
    fx_rate 0.00622: used fx_rate_feed; tried fx_rate_table (no value), fx_rate_feed (accepted)
    audit: 36 support items, 100% deterministic, 0 model outputs; safeguards: fallback producer ×2

[12/12] stale_feed_rate_rejected  (no JPY in the table, and the feed's JPY rate is from 2026-09-12, not the invoice date: ...)
    payment           abstain  abstain  0.00  rule not computed: missing inputs: lines_failing, within_approval_limit; missing fx_rate, ...
    fx_rate —: used none; tried fx_rate_table (no value), fx_rate_feed (rejected by validate)
    audit: 36 support items, 100% deterministic, 0 model outputs; safeguards: rejected by validate ×2

audit invariants hold on 12/12 responses: 419 support items, 100% deterministic, 0 model outputs; safeguards fired: fallback producer ×2, hard check decided ×3, rejected by validate ×2
12/12 cases as expected; decision time median 1.49 ms, max 1.99 ms
```

## vs an answer-only model

An answer-only model reads the three documents and returns `hold (0.78)`: no citation, no rules.

- **The outcome turns on arithmetic.** One penny per box moves the variance from +1.90 % to +2.45 % and the invoice from
  pay to hold. A conversion at 1.27 turns 8 200 GBP into 10 414 USD, over a 10 000 limit. Cross-currency prices
  (4.46 USD vs 4.10 EUR × 1.08) are compared only after conversion. Here each of these numbers is computed and shown
  for each line. An answer-only model has to do the same arithmetic implicitly, and its answer does not show whether it did.
- **A hold must say what to fix.** AP sends the supplier a line, a SKU and a figure (`billed 4000, received 3840`), not a
  probability. These are facts in `lines_failing`, and they are in the trace.
- **Controls are guaranteed.** A duplicate is never paid, and a supplier on hold is never paid. Hard checks force
  `reject` for every input, and the rest of the match is skipped: 5 of 15 steps in the duplicate and on-hold cases.
- **It abstains instead of assuming a rate.** Without a JPY rate in the table or a same-day rate in the feed, payment and
  approval abstain and name `fx_rate`, while the duplicate question is still answered. An answer-only model would output pay, hold or reject anyway.
- **The record can be audited.** `replay` recomputes each match, conversion and check from the recorded documents, so a
  controls audit can re-verify the decision instead of trusting a stored label.

## Files

`task.py` (catalog and questions), `state.json`, `cases.json` (12 invoices), `run.py`. Suppliers, rates and limits are
synthetic.
