# 11 · "I was charged twice": refund from the ledger, not from the ticket

A support ticket and the customer's ledger go in. Out come four answers: **does the customer say they were charged
twice?** (cited from the ticket), **does the ledger show a double charge still to refund?**, **refund: auto / manual /
none**, and **which reply to send**.

`cited` `hard checks` `early exit` `trace replay` `abstains` `audited` `runs in browser`

## How it decides

- **The ticket is only the customer's claim.** Two extractors return a `Quote` with character offsets in the ticket: what
  the customer says about being charged twice (`claimed`, `denied`, `unclear` or `not mentioned`, with the phrase it rests
  on) and the amount mentioned. They answer "what does the customer say", and no refund rule reads them.
- **The claim is read by a plain rule, and it abstains when the text is unclear.** Sentence by sentence (a sentence is cut
  again at "but", "however" and a dash), a claim is a word for taking money and a word for "twice" in the same clause:
  "charged twice", "billed me two times", "debited 2x", "duplicate transaction", "two identical charges", and "again" /
  "another" only next to "the same" or "right after" ("the same payment went through again"). "not", "never", "nobody"
  or "n't" up to five words before the phrase makes it a denial ("I was NOT charged twice"); "if", "whether", "maybe",
  "not sure", "thought" before it, or a yes/no question ("Did you take my payment twice?"), makes it unclear. Phrases that
  only look like one are skipped: "double-check", "twice a month", "can't believe", "not sure why". An unclear claim
  makes `customer_claims_double` abstain, and `reply` too when the ledger has nothing to refund (then the wording
  depends on what the customer meant). A person reads those tickets.
- **The ledger decides.** A double charge is two *settled charges* with the same merchant and amount, at most 10 minutes
  apart, where the second has not been refunded. A released card authorisation is not a charge. A renewal 31 days later
  is not a duplicate. The refund is the ledger amount.
- **Free balance**: the refund is automatic up to 500 when the payout account's balance minus reserved funds covers it.
  Otherwise a person handles it.
- **Hard check: no open chargeback.** When the bank is already returning the money, `refund` is forced to `none` and
  `reply` to `chargeback in progress`. The balance and limit steps are skipped.

## Run

```bash
uv run python gallery/11_refund_double_charge/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network). An earlier
version of `task.py` and `run.py` ran unchanged under Pyodide 0.27.2 (every case passed); the claim reader added since
uses only `re` and has not been re-run there.

`cases.json` has 16 tickets: a genuine double charge, a claim the ledger does not confirm (a released authorisation), a
monthly renewal, a claimed amount that differs from the ledger, a charge over the auto-refund limit, a free balance that
is too low, an open chargeback, a double charge the customer never mentioned, and one already refunded. Seven more are
about reading the text: three paraphrases ("billed me two times", "the same deposit was processed again", a renewal the
customer calls a "duplicate transaction"), two denials ("I was NOT charged twice", "I don't think I was billed double"
while the ledger does show a double charge, which is refunded anyway), and two unclear messages, where the claim
abstains. `state.json` is the unconfirmed claim.

## What the audit shows

`run.py` asserts the audit's invariants on every ticket (see [`_audit.py`](../_audit.py)). `claimed_amount` is an `exact=True`
extract, so the audit checks the number is literally what the customer wrote: 16/16 tickets, 468 support items, 100%
deterministic, hard check decided ×2, rule abstained ×3. The claim and the ledger side by side (`res.audit([...])` on a claim the ledger does not
confirm):

```
customer_claims_double = 'yes'  [ok]  confidence 1.00  ← computed by customer_claims_double
  quoted      claimed_amount = 54.99  ticket[47:52] literal '54.99'
  quoted      double_charge_claim = 'claimed'  ticket[4:18] derived from 'double charged'
is_double_charge = 'no'  [ok]  confidence 1.00  ← computed by is_double_charge
  computed    duplicate_pairs = []
  support     4 items (1 given, 3 computed): 100% deterministic
```

## Sample output (real run)

```
[2/16] claim_not_confirmed_by_ledger  (the customer is sure, but the first line is a card authorisation that was released: ...)
    ticket: 'You double charged me!! I see two payments of $54.99 to StreamFlix on my banking app. Fix this now.'
    claim (cited): double_charge_claim = 'claimed' from ticket[4:18] 'double charged'; claimed_amount = 54.99 from ticket[47:52] '54.99'
    ledger: no double charge · refund 0.00, free balance 15250.00
    customer_claims_double yes                    ok      1.00
    is_double_charge       no                     ok      1.00
    refund                 none                   ok      1.00
    reply                  no duplicate found     ok      1.00

[4/16] claimed_amount_differs  (the customer quotes 49.99; the ledger has 54.99 (with tax) twice: the refund is the ledger's 54.99)
    claim (cited): double_charge_claim = 'claimed' from ticket[16:32] 'charged me twice'; claimed_amount = 49.99 from ticket[47:52] '49.99'
    ledger: tx-9201 + tx-9202 ShopRight Online 54.99, 0.4 min apart · refund 54.99, free balance 15250.00

[6/16] free_balance_too_low  (420.00 is under the limit, but the payout account has 900 - 600 reserved = 300 free)
    refund                 manual                 ok      1.00

[7/16] chargeback_already_open  (a real double charge, but the bank's chargeback is open: the hard check forbids a second refund)
    refund                 none                   forced  1.00  hard check no_open_chargeback is false
    reply                  chargeback in progress forced  1.00  hard check no_open_chargeback is false
    replay ok (7 records, quotes checked against the ticket) · 0.93 ms · expected ✓

[15/16] unclear_claim_abstains  (the customer is unsure: the claim question abstains, and so does the reply ...)
    ticket: 'Not sure whether I was charged twice for UrbanRide (12.40) - the app shows something odd.'
    claim (cited): double_charge_claim = 'unclear' from ticket[23:36] 'charged twice'; claimed_amount = 12.4 from ticket[52:57] '12.40'
    customer_claims_double abstain                abstain 0.00  the rule abstained (returned None); double_charge_claim = 'unclear'
    is_double_charge       no                     ok      1.00
    refund                 none                   ok      1.00
    reply                  abstain                abstain 0.00  the rule abstained (returned None); double_charge_claim = 'unclear'; ...

the ticket and the ledger disagree in 6 of 16 cases (the claim is unclear in 2 more); every refund decision followed the ledger
16/16 cases as expected; decision time median 0.70 ms, max 117.48 ms
```

## What the rule cannot read

The rule reads the wordings above and nothing else. Measured on 80 messages written separately, by an LLM that had not
seen the rule (20 claims, 20 denials, 15 unclear, 25 about something else, with look-alikes such as "double-check my
address" and "I pay twice a month"): the claim question was right on 86%; it answered 84% of the messages by itself, and
15% of those answers were wrong (10 messages). What it got wrong:

- **Denials without a negation word it knows, right before the phrase**: "Nothing was charged twice", "Rather than a
  double charge, it was an add-on", a claim taken back later ("The double charge I reported earlier turned out to be a
  hold", "I was worried about a double charge earlier today, but everything is fine"). These are read as claims. ("I
  thought I'd been billed twice but I was wrong" abstains; "False alarm, only one charge went through" happens to come
  out right, as "not mentioned".)
- **"Twice" that belongs to another verb** in a sentence with a money word: "my card got declined twice", "the app
  crashed twice while I was updating my payment method" are read as claims.
- **Doubt without its usual words**: "I have a feeling I got charged twice", "Could I have been charged twice?" (a question
  it does not recognise) are read as claims; "There might be two charges here" is not seen at all.
- **Slang and idiom**: "double dipped into my account" is not seen.

None of this reaches the refund: `refund` and `is_double_charge` read only the ledger. What a misread changes is
`customer_claims_double` and, when there is nothing to refund, the reply's wording ("no duplicate found" or "not about a
charge").

## In production: a model reads the ticket, the rule is the fallback

Two producers of the same fact (`provides=`, see the guide's "Several producers of one fact"): a decider first, then the
rule. When the decider escalates (low confidence, an invalid reply, a server that does not answer), the rule's answer is
used; the trace records which producer was used and what the other one did. The options are the rule's own labels, so the
rest of the catalog does not change, and an `unclear` answer from the model abstains just like the rule's. Replace the
`double_charge_claim` extractor in `task.py` with:

```python
from solvi.llm import llm      # or a local decider: DecideModel.load("solvi-ai/solvi-large").decision(...)

model = llm("https://openrouter.ai/api/v1", "openai/gpt-oss-120b", api_key=os.environ["OPENROUTER_API_KEY"])
claim_llm = model.decision(
    "claim_llm", "What does the customer say about being charged twice?", "ticket",
    {"claimed": "they say they were charged twice for the same thing",
     "denied": "they mention a double charge only to say it did not happen",
     "unclear": "they ask, or are unsure, whether it happened",
     "not mentioned": "the message is not about being charged twice"},
    escalate_below=0.8)                           # better: claim_llm.act_guard(your_labelled_tickets, max_risk=0.05)
cat.fn(provides="double_charge_claim")(claim_llm)    # declared first: asked first


@cat.extract(provides="double_charge_claim")          # the rule: used when the model escalates or the server is down
def claim_rule(ticket):
    verdict, start, end = read_claim(ticket)
    return Quote(verdict, start, end, "ticket")
```

We ran this against a local stand-in server, not a real model: a valid reply is used, and an invalid one falls back to
the rule (`tried: [["claim_llm", "model escalated: invalid LLM output — the reply is not JSON"], ["claim_rule",
"accepted"]]`); replay passes either way. The gallery itself keeps only the rule, so it runs offline and in the browser.
Each ticket then costs one LLM request (0.3–5 s). The ledger decisions stay code.

## vs an answer-only model

An answer-only model reads the ticket (and perhaps a ledger dump) and returns `refund = yes (0.88)`, with no citation
and no rules.

- **The claim and the fact are kept apart.** In 6 of the 16 cases the ticket and the ledger disagree: an authorisation
  the customer takes for a second charge, two monthly renewals, a duplicate that was already refunded, a double charge
  the customer never mentioned, and one the customer denies. A model conditioned on "You double charged me!!" is pushed toward yes. Here the text
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
- **Where a strong LLM is better: reading the ticket.** A rule knows the wordings it was written for. A good LLM reads
  "false alarm, only one charge went through" or "double dipped" without a list of phrases. That is the one step in this
  task where a model earns its cost; the next section shows how to put one there without letting it touch the refund.

## Files

`task.py` (catalog, questions, `prepare`), `state.json`, `cases.json` (16 tickets), `run.py`. Customers, merchants and
amounts are synthetic.
