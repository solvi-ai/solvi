# 03 · Content guard

User text on its way into your app (an LLM prompt, a support chat, a public post) goes in. Out come a **verdict** (allow /
review / block), **sensitive_data?** and **prompt_injection?**. Each finding cites the exact span that triggered it.

`cited` `hard checks` `strategist plan` `early exit` `trace replay` `abstains` `runs in browser`

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
- **Surface-aware soft rules.** The same e-mail address is fine in a support chat and costs 2 risk points in a public post.
  2+ points gives review, 5+ gives block. Without a `surface`, the verdict abstains; a card or a secret is still blocked.

## Sample output (real run)

```
[1] card number in an LLM prompt               2.39 ms   replay ok
    verdict=block (forced)  sensitive_data=yes (forced)  prompt_injection=no
    cited: card_number=card ending 6467 «4539 1488 0343 6467» [50:69]
    verdict forced: hard check no_card_number is false
    sensitive_data forced: hard check no_card_number is false
    the score alone would say: allow (risk_points = {})
[2] 16-digit order number                      0.86 ms   replay ok
    verdict=allow  sensitive_data=no  prompt_injection=no
    verdict ok: risk_points = {}
[4] injection phrase only mentioned            0.77 ms   replay ok
    verdict=review  sensitive_data=no  prompt_injection=no
    cited: injection=quoted «ignore previous instructions» [41:69]
    verdict ok: risk_points = {'injection phrase mentioned': 2}
[6] contact details in a public post           0.87 ms   replay ok
    verdict=review  sensitive_data=yes  prompt_injection=no
    cited: email_address=marta.k@gmail.com «marta.k@gmail.com» [44:61]
    cited: phone_number=0176 555 01234 «0176 555 01234» [70:84]
    verdict ok: risk_points = {'e-mail in a public post': 2, 'phone in a public post': 2}
[9] role-play jailbreak                        0.42 ms   replay ok
    verdict=block (forced)  sensitive_data=no  prompt_injection=yes
    cited: injection=direct «an AI with no rules» [39:58]
    verdict forced: hard check no_direct_injection is false
    the score alone would say: allow (risk_points = {})
[10] destination unknown                        0.37 ms   replay ok
    verdict=— (abstain)  sensitive_data=yes  prompt_injection=no
    cited: phone_number=+44 20 7946 0958 «+44 20 7946 0958» [20:36]
    verdict abstain: cannot compute: no_direct_injection, risk_points

10/10 cases match, replay ok on 10/10 traces, median decision 0.86 ms

early exit on 'card number in an LLM prompt': email_address (not needed: hard check no_card_number failed), iban (...), insult (...), link_count (...), phone_number (...), risk_points (...), ...
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

Full per-case output of the run above: [compare_laya.out.txt](compare_laya.out.txt).

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
