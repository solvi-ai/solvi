# 10 · Procurement: 3-way match

A purchase order, a goods receipt and a supplier invoice (all JSON) go in. Out come **pay / hold / reject**, **already
paid (duplicate)?** and **needs a higher approver?**

`hard checks` `strategist plan` `early exit` `trace replay` `abstains` `runs in browser`

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
- **No rate, no guess.** A missing currency rate is an error, never 1.0: the answers that need amounts abstain, and the
  duplicate check, which does not need them, still answers.

## Run

```bash
uv run python gallery/10_procurement_3way_match/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

`cases.json` has 10 invoices: a clean EUR match, prices at +1.90 % (pay) and +2.45 % (hold, one penny apart), a short
delivery billed in full, GBP converted over the clerk's limit, a PO in EUR invoiced in USD, a duplicate with a
reformatted number, a supplier on hold, a currency without a rate (abstain), and a header total that does not add up.
`state.json` is the currency-over-limit invoice. Try setting `approver.role` to `manager` there.

## Sample output (real run)

```
[3/10] price_over_tolerance  (one penny more per box: +2.45%, outside the tolerance -> hold that line)
    payment           hold     ok       1.00  lines_failing = ['line 1 BX-4020: price +2.45% vs PO (tolerance ±2.0%)']; header_total_ok = True; within_approval_limit = True
    lines: 1 BX-4020 qty 2500/2500 price +2.45% ✗; 2 TP-48 qty 600/600 price +0.00% ✓

[5/10] fx_pushes_over_limit  (8 200 GBP looks under the 10 000 limit, but at 1.27 it is 10 414 USD: a manager must approve)
    payment           hold     ok       1.00  lines_failing = []; header_total_ok = True; within_approval_limit = False
    escalate_approval yes      ok       1.00  within_approval_limit = False
    amount 8,200.00 GBP = 10,414.00 USD (limit 10,000 for clerk)

[7/10] duplicate_reformatted_number  ('INV-001187' is 'inv 1187', paid on 2026-09-02: reject; the line match is skipped)
    payment           reject   forced   1.00  hard check not_duplicate is false
    duplicate         yes      ok       1.00  duplicate_of = 'inv 1187'
    10 steps run, 5 skipped (header_total_ok, po_fx_rate, line_match, lines_failing, answer:payment) · replay ok · 0.85 ms · expected ✓

[9/10] missing_fx_rate  (invoiced in JPY and there is no JPY rate: amounts cannot be compared or limited, so no guess; ...)
    payment           abstain  abstain  0.00  rule not computed: missing inputs: lines_failing, within_approval_limit; missing fx_rate, ...
    duplicate         no       ok       1.00  duplicate_of = None

10/10 cases as expected; decision time median 1.14 ms, max 4.39 ms
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
- **It abstains instead of assuming a rate.** Without a JPY rate, payment and approval abstain and name `fx_rate`,
  while the duplicate question is still answered. An answer-only model would output pay, hold or reject anyway.
- **The record can be audited.** `replay` recomputes each match, conversion and check from the recorded documents, so a
  controls audit can re-verify the decision instead of trusting a stored label.

## Files

`task.py` (catalog and questions), `state.json`, `cases.json` (10 invoices), `run.py`. Suppliers, rates and limits are
synthetic.
