# 03 · Content guard

User text on its way into your app (an LLM prompt, a support chat, a public post) goes in. Out come a **verdict** (allow /
review / block), **sensitive_data?**, **harm** (every kind found: personal_data, bank_details, card_data, secret,
prompt_injection, abuse — a multi-label answer) and **prompt_injection?**. Each finding cites the exact span that triggered it.

`cited` `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `multi-label` `constraints` `audited` `runs in browser`

```bash
uv run python gallery/03_content_guard/run.py        # 10 scenarios, exits non-zero on a mismatch
```

## How it is built

- **Findings are extracts with quotes.** An e-mail address, a phone number, a card number, an IBAN, an API key or private key,
  an injection phrase, an insult. Each comes back as a `Quote` with offsets into the text.
- **Checksums, not look-alikes.** A card number must pass Luhn and an IBAN must pass mod-97. So `4539 1488 0343 6468` (an order
  number, last digit changed) is not a card.
- **Mention vs attack.** An injection phrase inside quotes, in a question *about* injections, counts as a mention. It gets
  review, not block, and prompt_injection=no.
- **Hard checks no score can override.** A card number or a secret anywhere, or a direct injection into an LLM prompt, forces
  verdict=block (and sensitive_data=yes for the first two). In every forced case the soft risk points would have allowed the
  text. `run.py` prints both.
- **Harm types and constraints (solvi 0.4).** `harm` is `Answer.multi`: one answer with every kind of harm, e.g.
  `('personal_data', 'card_data')`. Three constraints tie it to the other answers, each computed by its own rule:
  `block_if_card_or_secret` (a card or a secret among the harms is a block), `never_allow_serious_harm` (bank details, a direct
  injection or abuse are never simply allowed) and `sensitive_iff_data_harm` (sensitive_data is yes exactly when a data harm was
  found). With the rules here every response is feasible; `run.py` then swaps the verdict rule for a head learned from 6
  labelled texts and shows a constraint repairing its answer (below).
- **Surface-aware soft rules.** The same e-mail address is fine in a support chat and costs 2 risk points in a public post.
  2+ points gives review, 5+ gives block. Without a `surface`, the verdict abstains; a card or a secret is still blocked.

## What the audit shows

`run.py` asserts the audit's invariants on every text (see [`_audit.py`](../_audit.py)). `email_address`, `phone_number` and
`insult` are `exact=True` extracts, so their values must be literally the quoted text; the card, IBAN, secret and injection
findings return a label *derived from* the span, and the audit says so. 10/10 texts, 393 support items, 100% deterministic, hard
check decided ×6, every constraint satisfied. From `res.audit("verdict")` on the card number:

```
verdict = 'block'  [forced]  confidence 1.00  ← computed by no_card_number
  quoted      card_number = 'card ending 6467'  text[50:69] derived from '4539 1488 0343 6467'
  quoted      phone_number = '+1 415 555 0132'  text[122:137] literal '+1 415 555 0132'
  check       no_card_number = False (hard, decides the answer)
  constraint  block_if_card_or_secret: satisfied
  constraint  never_allow_serious_harm: satisfied
  not run     link_count, risk_points (not needed: hard check no_card_number failed)
  support     12 items (2 given, 3 computed, 7 quoted): 100% deterministic
```

With the verdict rule replaced by a head learned from 6 labelled texts, two runs: a weak head that sees only the surface — the
constraint repairs the IBAN case (constraint repair ×1 in 10 texts), and what no constraint covers stays the head's own, wrong
answer (a mention and a public-post contact detail come out `allow`); and a head that also reads the soft signals' total
(`risk_score`), which agrees with the rule on 10/10 and needs no repair. The audit lists what the head read and what it
ignored (solvi warns when a requested feature cannot be used, e.g. a dict):

```
verdict = 'review'  [ok]  confidence 0.33  ← learned by FastHead
  learned     answer head (FastHead) = 'allow'  (allow 0.67, review 0.33, block 0.00)  [FastHead FastHead #5ec33ca1]
  → answer    'review' — surface = 'support_chat' (+0.13); changed from 'allow' to satisfy never_allow_serious_harm
  safeguards  constraint repair ×1
```

## Sample output (real run)

```
[1] card number in an LLM prompt               2.19 ms   replay ok
    verdict=block (forced)  sensitive_data=yes (forced)  harm=('personal_data', 'card_data')  prompt_injection=no
    audit: 38 support items, 100% deterministic, 0 model outputs; safeguards: hard check decided ×2
    cited: card_number=card ending 6467 «4539 1488 0343 6467» [50:69]
    cited: phone_number=+1 415 555 0132 «+1 415 555 0132» [122:137]
    verdict forced: hard check no_card_number is false
    sensitive_data forced: hard check no_card_number is false
    the score alone would say: allow (risk_points = {})
[2] 16-digit order number                      0.80 ms   replay ok
    verdict=allow  sensitive_data=no  harm=()  prompt_injection=no
    verdict ok: risk_points = {}
[4] injection phrase only mentioned            0.76 ms   replay ok
    verdict=review  sensitive_data=no  harm=()  prompt_injection=no
    cited: injection=quoted «ignore previous instructions» [41:69]
    verdict ok: risk_points = {'injection phrase mentioned': 2}
[6] contact details in a public post           0.88 ms   replay ok
    verdict=review  sensitive_data=yes  harm=('personal_data',)  prompt_injection=no
    cited: email_address=marta.k@gmail.com «marta.k@gmail.com» [44:61]
    cited: phone_number=0176 555 01234 «0176 555 01234» [70:84]
    verdict ok: risk_points = {'e-mail in a public post': 2, 'phone in a public post': 2}
[9] role-play jailbreak                        0.67 ms   replay ok
    verdict=block (forced)  sensitive_data=no  harm=('prompt_injection',)  prompt_injection=yes
    cited: injection=direct «an AI with no rules» [39:58]
    verdict forced: hard check no_direct_injection is false
    the score alone would say: allow (risk_points = {})
[10] destination unknown                        0.60 ms   replay ok
    verdict=— (abstain)  sensitive_data=yes  harm=('personal_data',)  prompt_injection=no
    cited: phone_number=+44 20 7946 0958 «+44 20 7946 0958» [20:36]
    verdict abstain: cannot compute: no_direct_injection, risk_points

10/10 cases match, replay ok on 10/10 traces, median decision 0.80 ms
audit invariants hold on 10/10 responses: 393 support items, 100% deterministic, 0 model outputs; safeguards fired: hard check decided ×6

early exit on 'card number in an LLM prompt': link_count (not needed: hard check no_card_number failed), risk_points (...), answer:verdict (...), answer:sensitive_data (...)
```

In the playground, change the recorded value of step 2 (`no_card_number`) from False to True and recompute every hash after
it. Replay still catches the edit: `[2, 'no_card_number', 'value True ≠ recomputed False']`.

## vs an answer-only model

`compare_laya.py` runs the 10 texts through [Laya](https://huggingface.co/convaiinnovations/laya) with its own `guard_questions()`
preset (sensitive_data, prompt_injection, jailbreak), plus a verdict question with our policy written into the instructions.

| | solvi | Laya (zero-shot, RTX 3060) |
|---|---|---|
| correct answers on the 10 texts | **30 / 30** | 18 / 30 |
| card number in an LLM prompt | block, forced, cites «4539 1488 0343 6467» | allow (0.55), sensitive_data no (0.77) |
| "ignore all previous instructions…" | block, forced | prompt_injection yes (1.00), verdict allow (0.50) |
| injection phrase only quoted in a question | review, prompt_injection no | prompt_injection yes (0.84) |
| API key pasted into support chat | block, forced | block (0.39), but prompt_injection yes (0.71) |
| no `surface` given (phone number) | verdict abstains | allow (0.50), sensitive_data no (0.76) |
| where the finding is | span offsets for each finding | not given |
| median time per text | 0.76 ms, CPU | 56 ms, GPU |

Full per-case output of the run above: [compare_laya.out.txt](compare_laya.out.txt). The run predates the `harm` answer (solvi
0.4), which Laya has no counterpart for; `compare_laya.py` scores the three answers both were asked.

- **Detection is not the gap. The guarantee is.** Laya's own detector caught both direct injections at probability 1.00.
  Asked for a verdict under a written policy, it still answered allow on the direct injection and the role-play jailbreak,
  and on the card number. In solvi those are hard checks: a card number is blocked every time, at any score.
- **Checksums give exact answers where a model gives probabilities.** A Luhn-valid number is a card, and a number one digit off
  is not. A model sees two 16-digit strings that look almost the same.
- **The cited span lets you act, not just decide.** With the offsets you can redact exactly `[50:69]` and let the rest through.
  A probability gives you nothing to redact.

Caveats: we wrote the 10 cases alongside the catalog, so 30/30 is not an unbiased accuracy. Regular expressions miss what
they were not written for (obfuscated numbers, new token formats, injections in other languages). A model can generalize
there, so the two combine well: a learned detector as a soft signal, with solvi's hard checks on top. Reproduce from the
new_kelly repo:
`cd exps_v2 && uv run --with laya==0.3.20 --with transformers --with-editable ../solvi python ../solvi/gallery/03_content_guard/compare_laya.py`
