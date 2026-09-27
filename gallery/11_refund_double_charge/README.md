# 11 · "I was charged twice": refund from the ledger, not from the ticket

A support ticket and the customer's ledger go in. Out come four answers: **does the customer say they were charged
twice?** (cited from the ticket), **does the ledger show a double charge still to refund?**, **refund: auto / manual /
none**, and **which reply to send**.

`cited` `hard checks` `early exit` `trace replay` `audited` `runs in browser`

## How it decides

- **The ticket is only the customer's claim.** Two extractors return a `Quote` with character offsets in the ticket: the
  phrase ("charged twice", "double charged", "Duplicate charge") and the amount mentioned. They answer "what does the
  customer say", and nothing else depends on them.
- **The ledger decides.** A double charge is two *settled charges* with the same merchant and amount, at most 10 minutes
  apart, where the second has not been refunded. A released card authorisation is not a charge. A renewal 31 days later
  is not a duplicate. The refund is the ledger amount.
- **Free balance** (the Jev "свободный остаток покрывает возврат?" step): the refund is automatic up to 500 when the
  payout account's balance minus reserved funds covers it. Otherwise a person handles it.
- **Hard check: no open chargeback.** When the bank is already returning the money, `refund` is forced to `none` and
  `reply` to `chargeback in progress`. The balance and limit steps are skipped.

## Run

```bash
uv run python gallery/11_refund_double_charge/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

`cases.json` has 9 tickets: a genuine double charge, a claim the ledger does not confirm (a released authorisation), a
monthly renewal, a claimed amount that differs from the ledger, a charge over the auto-refund limit, a free balance that
is too low, an open chargeback, a double charge the customer never mentioned, and one already refunded. `state.json` is
the unconfirmed claim.

## What the audit shows

`run.py` asserts the audit's invariants on every ticket (see [`_audit.py`](../_audit.py)). `claimed_amount` is an `exact=True`
extract, so the audit checks the number is literally what the customer wrote: 9/9 tickets, 258 support items, 100%
deterministic, hard check decided ×2. The claim and the ledger side by side (`res.audit([...])` on a claim the ledger does not
confirm):

```
customer_claims_double = 'yes'  [ok]  confidence 1.00  ← computed by customer_claims_double
  quoted      claimed_amount = 54.99  ticket[47:52] literal '54.99'
  quoted      claims_double_charge = True  ticket[4:18] converted from 'double charged'
is_double_charge = 'no'  [ok]  confidence 1.00  ← computed by is_double_charge
  computed    duplicate_pairs = []
  support     4 items (1 given, 3 computed): 100% deterministic
```

## Sample output (real run)

```
[2/9] claim_not_confirmed_by_ledger  (the customer is sure, but the first line is a card authorisation that was released: ...)
    ticket: 'You double charged me!! I see two payments of $54.99 to StreamFlix on my banking app. Fix this now.'
    claim (cited): claims_double_charge = True from ticket[4:18] 'double charged'; claimed_amount = 54.99 from ticket[47:52] '54.99'
    ledger: no double charge · refund 0.00, free balance 15250.00
    customer_claims_double yes                    ok      1.00
    is_double_charge       no                     ok      1.00
    refund                 none                   ok      1.00
    reply                  no duplicate found     ok      1.00

[4/9] claimed_amount_differs  (the customer quotes 49.99; the ledger has 54.99 (with tax) twice: the refund is the ledger's 54.99)
    claim (cited): claims_double_charge = True from ticket[16:32] 'charged me twice'; claimed_amount = 49.99 from ticket[47:52] '49.99'
    ledger: tx-9201 + tx-9202 ShopRight Online 54.99, 0.4 min apart · refund 54.99, free balance 15250.00

[6/9] free_balance_too_low  (420.00 is under the limit, but the payout account has 900 - 600 reserved = 300 free)
    refund                 manual                 ok      1.00

[7/9] chargeback_already_open  (a real double charge, but the bank's chargeback is open: the hard check forbids a second refund)
    refund                 none                   forced  1.00  hard check no_open_chargeback is false
    reply                  chargeback in progress forced  1.00  hard check no_open_chargeback is false
    replay ok (7 records, quotes checked against the ticket) · 1.11 ms · expected ✓

the ticket and the ledger disagree in 4 of 9 cases; every refund decision followed the ledger
9/9 cases as expected; decision time median 1.12 ms, max 1.61 ms
```

## vs an answer-only model

An answer-only model reads the ticket (and perhaps a ledger dump) and returns `refund = yes (0.88)`, with no citation
and no rules.

- **The claim and the fact are kept apart.** In 4 of the 9 cases the ticket and the ledger disagree: an authorisation
  the customer takes for a second charge, a monthly renewal, a duplicate that was already refunded, and a double charge
  the customer never mentioned. A model conditioned on "You double charged me!!" is pushed toward yes. Here the text
  cannot influence the refund at all, because no refund rule reads it.
- **The amount is computed, not generated.** The customer says 49.99 and the ledger shows 54.99 twice. The refund is
  54.99, the reply can quote both (`ticket[47:52]` against `tx-9201 + tx-9202`), and nobody has to trust a number that a
  model wrote.
- **Ledger semantics are rules.** Settled vs released, a 10-minute window, refunded or not, balance − reserved ≥ refund
  (900 − 600 = 300 < 420 → manual). Each of these is a line of Python that a reviewer can read, and it holds for every
  ticket.
- **No double refunds.** An open chargeback forces `none` for every input. The hard check runs first, and the balance
  and limit steps are skipped (7 of 13 records written).
- **Everything is checked again.** `replay` recomputes each step and verifies that every quote lies inside the ticket
  text. The reply sent to the customer can be traced to ledger ids.

## Files

`task.py` (catalog, questions, `prepare`), `state.json`, `cases.json` (9 tickets), `run.py`. Customers, merchants and
amounts are synthetic.
