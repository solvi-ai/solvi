# Best practices

What we learned while building on solvi, as advice. Each item says what to do and why. Every number on this page names
its source: a script in [benchmarks/](../benchmarks/) that re-runs it, or a published model card. A finding without such
a source is kept as advice, in plain words.

## The input a model reads

**Keep the keys of a state in one fixed order, and do not reorder them later.** A decision model reads a dict as lines
in the dict's order, and the order changes answers. This is not one model's quirk: every model we tried — the solvi
checkpoints, a hosted decision model and an LLM asked through `solvi.models.llm` — changed some answers when only the order of
the keys changed, and for some of them the accuracy moved with it.

So: build the state the same way every time — declare it as a pydantic model (`System(input_model=Model)`: the facts then
come in the model's field order whoever built the dict) — put what matters first, and
when you calibrate a threshold or fit a question, use states in the order production will send. solvi keeps the order
through its stores, so a stored decision replays on the text the model read.

**Know when the input was cut.** Without `long=`, a text beyond `max_len` is cut at the end; the decision says so in
`extra["truncated"]` and warns once. A category is usually clear from the start of a message; a fact near the end is
not read at all, and the model is no less confident. Many options with long descriptions crowd the input out the same
way: a long list of options can leave the text little room. Use `long="retrieve"` for documents, and keep option
descriptions short.

**For documents, give retrieval the document's words.** `long="retrieve"` finds sections by matching words. A field
written as a labelled line ("Invoice No.: INV-2542") shares almost none with a natural question, and a question in
English shares none with a Russian document. Put the labels' own words, in the document's languages, into
`retrieve_query` (`retrieve_query="Invoice No Contract No Ref Счёт №"`), and the line that holds the field is found far
more often.

**Ask for numbers and dates as typed spans, and let code read them.** `Span[float]` and `Span[date]` return the quote
and the parsed value ("EUR 18,851.12" → 18851.12, "21 July 2026"); what would be a guess ("03/04/2026") is rejected with
the reason. Do not ask a model to compare numbers or dates: on "is a greater than b", "which date is earlier", "count
the items" small decision models are unreliable. Compute the comparison in a catalog function and give the model
the finished fact.

**Give roles as finished pairs.** "From X to Y" asked as two questions ("from where?", "to where?") often gave the same
city twice. One question over ready pairs ("which route?") did much better.

## Choosing among options

**Up to a few dozen short options: ask directly.** When the options fit and leave the input at least half of the
window, one ordinary decision is the most accurate way; on a text game it beat both a shortlist and a tournament.

**More than fit: narrow by code first, then ask.** Where a number decides (a price within a budget, a level, a
distance), filter and rank in code and give the model what is left. The model-run shortlist and tournament
(`solvi.many`) were removed in 1.0: they lost to one direct decision.

**Do not score candidates one by one.** Asking "how good is this option?" per candidate does not separate them: the
model rates nearly everything "partly good", and a direct choice among them is far more accurate.

**Candidates that change every step: learn over their features.** `fit` and `teach` learn per fixed set of options,
so for an agent's ever-new candidates they learn nothing. A `CandidateHead` over what a candidate is (distance, kind,
was it a dead end) learns a hidden rule from corrections, and an update is cheap. It learns the rule it is shown; it
will not find a better one.

## Rules across many decisions

**A rule across items belongs after the items' decisions, not inside each one.** "One counterpart per product" over
product-matching pairs, enforced with `solvi.core.sets.decide_set` on the answers as given, raised F1 for every solver we
tried, and helped a weak solver most: on Abt-Buy (1,916 pairs) 0.872 → 0.909 for an LLM asked each pair, 0.860 → 0.870
for the same LLM through `solvi.models.llm`, 0.931 → 0.933 for a fitted head ([abtbuy/solution.py](../benchmarks/tasks/abtbuy/solution.py)
`--repair`, [abtbuy/llm_pair.py](../benchmarks/tasks/abtbuy/llm_pair.py)). Giving each request the other candidates as facts instead (how the pair ranks
among them) did worse. Prefer the exact method: it was at least as good as the greedy one, and fast on large
components.

**When the alternatives can be enumerated, search them instead of asking a model to propose.** On trip and meeting
planning problems a `solvi.core.slow.search` through the same checks found the right plan more often than an LLM's plans did: on
NATURAL PLAN 95 / 100 / 98 of 100 calendar, meeting and trip problems against 92 / 75 / 43 for gpt-oss-120b's own plans
([naturalplan/solution.py](../benchmarks/tasks/naturalplan/solution.py)). Make the checks that rule out a prefix (a missing
flight, a meeting out of reach) usable on partial candidates and list them in `prune=`: that keeps the search small
where the orders of the parts are too many to try — the largest 10-city trip took 11,414 asks, where 10 cities can be
visited in 10! ≈ 3.6 million orders.

## Guarantees and calibration

**Calibrate on your own stream.** A threshold shipped with a checkpoint holds on the checkpoint's data. The
[solvi-large card](https://huggingface.co/solvi-ai/solvi-large) shows it: at its shipped "≤ 10% error" threshold the
real error was 32–39% on three new data sets (10% on a fourth), while `act_guard(max_risk=0.10)` held on every set.
`act_guard(examples, risk)` on a few hundred of your own labelled cases is what makes the number a promise.

**The act signal falls with the number of options; accuracy may not.** As the number of options grows, accuracy can
stay level while the mean act probability falls steeply: with the default threshold the model then escalates most of
the cases it would have answered correctly. Calibrate per question, with the number of options production has.

**Put the promise where the decision is made.** When a head, a rule or a trust score answers, calibrate the question
itself (`system.guarantee`), not only the model under it — and on the signal that separates right from wrong: on CUAD
contract clauses a trust score built from three facts separated right from wrong answers with AUROC 0.83 on dev, the
LLM's own confidence with 0.63; at a 3% risk the trust score answered 67.6% of the eval questions alone with 4.0% wrong,
while the confidence answered 14% of dev ([cuad/solution.py](../benchmarks/tasks/cuad/solution.py)). Use `max_error=` for "≤ e of what
we answer is wrong" and `max_risk=` for "≤ r of all inputs": on a stream of mostly easy non-matches the risk held overall
while the matches an LLM gave alone were wrong far more often — `groups="answer"` puts the promise inside each answer.

**Size thresholds for inputs the decider has no answer for — when new kinds can come.** New kinds of input break every
promise calibrated without them: on a Banking77 stream where 40% of the requests come from 20 unseen intents after
request 1,000, a "≤ 5% wrong" threshold calibrated without them gave 16.4% wrong after the shift; an `OpenSetGate`
calibrated with `leave_out` kept 0.7% (57.6% answered alone before the shift, 13.7% after) and flagged the change 68
requests in ([banking77/solution.py](../benchmarks/tasks/banking77/solution.py), `--plain` for the first). An `OpenSetGate` with outside signals from `leave_out` follows their share without
labels: on public intent sets, with nothing tuned, it kept its promise in simulated streams where new intents or
out-of-scope queries grew gradually or jumped, where a plain threshold broke; the price is fewer requests answered
alone before any change. Pay for it only where new kinds are expected; for gradual ones `track=(200,)` costs less;
after a sudden jump to most of the traffic nothing keeps the promise for the first decisions — stop answering alone on
the flag. Its signal has to tell outside from known: an act head trained with left-out options did, a plain confidence
did not.

**Watch for drift.** A calibrated threshold keeps its promise under a shifted stream by escalating more, with no error
raised. `DriftMonitor()` with its defaults flags such changes within a few dozen decisions. In a simulation
([benchmarks/drift_simulation.py](../benchmarks/drift_simulation.py)) a fall of the share answered alone from 73% to
13% was flagged after 23–27 decisions, and 0.5% of unchanged streams were
flagged within 1,000 decisions (at most 1% promised). A change of the mix of the answers is seen
only by the window tests, which need about five decisions per answer in a window (`rep["not_tested"]` says when it is
not tested): the same script's change from 1:1:1 to 1:8:1 was flagged after 67 / 80 / 110 decisions with a window of
50 / 100 / 200. Take the reference from the stream itself unless the calibration set has its mix: a reference with another mix
differs from the stream from the start. Treat a quiet monitor as "no large change", not as
"no change".

**`fit` and `teach` move the scores, not the reading.** They shift and scale the logits of one question; they help
calibration and the mix of answers, and level off within a few dozen examples. When a question needs the model to read
the input differently, use a LoRA adapter (`adapt_lora`), a head over computed facts, or a rule.

**Let a second system's verified answers recalibrate a guarantee — and nothing else, unless you measured it.** When
an LLM handles what a fast system escalates, its answers that passed the checks and its own guarantee can be stored
as labels (`label_source="verified"`) and read by `System.guarantee(..., corrections=store, sources=...)`. We replayed
three stand tasks as a stream (the LLM's recorded answers; the fast system re-measured every 200 or so items): calibrating on
the human set plus every verified LLM answer raised the share answered alone in the second half of the stream from
90.7% to 96.7% on product matching (error among them 1.4%, promise 2%) and from 47.2% to 52.0% on contract clauses
(answered alone and wrong: 2.4% of the questions, promise 3%), within the promise both times. The same
answers fed to a head helped on contracts only (it already answered 90% of the pairs), fed to a memory of corrections
they helped on product matching but broke the contract promise (3.1% answered alone and wrong against 3%), and a per-answer "route" built from them
never helped. Feeding only the cases where the LLM disagreed with the fast system gave almost no material (5 verified
disagreements in 1,916 pairs) and no gain — feed everything it vouched for. Verification itself is the safeguard: on
bank requests with new intents the LLM was right on 75% and its guarantee vouched for none of its answers, so nothing
was learned; its unverified answers, fed to all four channels at once, raised the share answered alone after the shift
from 53% to 71% and the error among those from 16% to 20% (promise 5%), with more of the LLM's own mistakes copied.

**A person's labels on what you escalated are the hard cases — weight them, or they tighten the threshold.** On a
bank-request stream of 2,000 (20 of 77 topics unseen from request 1,000) a person labelled every escalated request and 5% of the
answers given alone; every 200 requests the guarantee was recalibrated on the human set plus those labels, the open-set gate
re-estimated with the labelled outside requests and the classifier refitted with a new topic as an option once it had 10 labels.
The promise held (0.47% / 2.64% wrong among the answers given alone before / after the shift), but the share answered alone
moved from 44.2% to 45.1%: before the shift it fell from 68.8% to 63.6% (the escalated requests are the ones the fast system
gets wrong more often, so the threshold rose from 0.75 to 0.83), after it the known topics gained 11 points (31.9% → 42.9%)
while the new topics, present in the head with 10–20 labels each, never passed the threshold calibrated on the old ones (7
answers, all wrong). Weight such labels by the probability they were sampled with (an audit label at 5% counts 20× an
escalation label), or calibrate on a representative audit alone, and give new options a threshold of their own with a
minimum support. A confirmation on a held-out 30% of the labels decided nothing here: a candidate answers 3–4 of the
held-out requests alone, which neither certifies nor refutes a 5% promise.

**Labels on a stream do not raise what a fixed System 1 answers alone — budget them for new answerers, not thresholds.**
On the bank-request stream of 2,000 (20 topics unseen from request 1,000) we fed one fixed classifier seven kinds of person
labels — every escalated request, a 5% or 10% random audit, both with inverse-sampling weights, and an active policy that
records each label's draw probability and moves the threshold only under an anytime-valid certificate — recalibrating the
guarantee, the open-set gate and a refitted head every 200 requests. The promise (≤ 5% wrong among the answers given alone)
held in every arm, and the share answered alone moved from 44.2% to at best 45.6%; the weights changed none of the 2,000
decisions; a shared threshold for the new topics never certified (at 5% error and 80% availability a group needs about 180
clean labels, the stream gave at most 174 with errors in them); the certificate never reached its bound at 10% or 30%
labels (log-wealth 0.35 of the 2.30 needed). What labels do buy: more answers on known topics after a shift (+11 points) at
the price of fewer before it (−5). If you want coverage on new kinds of input, plan for hundreds of labels per new option
and a new answerer, and keep the person on the rest.

**Fix a head's ridge strength when its answers go through a guarantee.** `fit` chooses λ by leave-one-out accuracy; a
refit on more labels can choose a much smaller one, and the head's ranking of its own answers gets worse even when the
added labels are right: on product matching, a head refitted on 808 instead of 300 pairs picked λ 0.1 and the error
among its surest half went from 0.6% to 2.1%, so no threshold met a 2% promise. With `lam=100, pairs=False` held fixed
the guarantee kept answering about 90%. Recalibrate the guarantee after every refit in any case.

## An LLM as a decider

**With a reasoning LLM, let it think: do not force a reply format on it.** A server that enforces `response_format`
by constrained decoding may apply the grammar from the first token and skip the thinking. On OpenRouter one of
gpt-oss-120b's providers answered every request that way under json_schema and json_object (no reasoning tokens) and
served a part of all requests (about a fifth of the judge's replies below had no reasoning under json_schema); a
yes/no hallucination judge asked through `solvi.models.llm` scored F1 0.744 on RAGTruth dev with the schema enforced and 0.791
with the contract in the prompt (the plain call to the model: 0.802), on eval 0.733 → 0.766 (plain 0.784)
([ragtruth/reply_format.py](../benchmarks/tasks/ragtruth/reply_format.py)); a product-matching question on Abt-Buy 0.837 → 0.860
(plain 0.872, [abtbuy/llm_pair.py](../benchmarks/tasks/abtbuy/llm_pair.py)). solvi does this by default: with
reasoning asked for in `extra_body`, `response_format="auto"` sends no format and `max_tokens` defaults to 2,048. If you
set `response_format="json_schema"` by hand, look for `extra["llm"]["reasoning"] == "none"` in the trace.

**Count an escalated decision as unanswered.** An LLM decision whose reply was invalid or cut off has no value and no
probabilities (`d.value is None`); go by `d.escalate`, not by `p ≥ 0.5`. With a reasoning model, give `max_tokens`
room: at 400, 26 of 1,916 product-pair replies were cut off ([abtbuy/llm_pair.py](../benchmarks/tasks/abtbuy/llm_pair.py));
the default with reasoning is 2,048.

**Keep the quote in the question.** Asking the same judge for the unsupported passage along with its yes/no is part of
how it finds one: without the quote its recall fell clearly. Moving the text before the question or dropping the "say
so with low probabilities" rule changed nothing measurable.

## A model that writes

**Ask for table rows copied as written, and parse them in code.** An extraction that asked for a travel-time table as
`[from, to, minutes]` rows was accepted and wrong in some problems, every wrong number standing elsewhere in the text;
asking for each row as the text writes it ("North Beach to Chinatown: 6"), checked literally (`quotes=`) and parsed in
code, removed those errors. What a quote check cannot see is a row left out.

**In a re-ask, quote what is false, and count the rounds.** Feedback that said where a plan stops working got the same
plan back three times; quoting the plan's own false sentence ("you meet Betty from 11:48AM, but Betty is there only
from 2:45PM") fixed it at once (a pilot of a few problems). Even so, on multi-city trips most re-asks did not rescue the
plan — the model traded one violation for another — and on days of meetings many repaired plans met fewer friends than
possible. Cap the rounds and measure what each one buys.

**Agreement of samples lowers the error of what is answered; it does not make it small.** Three queries per question,
compared by the rows they return: answering only when all three agree cut the wrong answers of a text-to-SQL model from
48% to 28% at 63% answered (BIRD mini-dev, 150 questions, [bird/solution.py](../benchmarks/tasks/bird/solution.py)) — far from
all of them. The rest were three samples agreeing on a reading of the question that was not the reference's.
Treat the share as a signal to calibrate on labelled examples, not as a proof.

## Instructions in the input

**Turn `perturb` on where the text comes from outside.** It re-asks without instruction-like sentences and escalates
when the answer changes — or when the answer stays and the model would not have given it alone without them. On 200
English support messages "classify this as X" set the answer in 28% of cases without it and 0% with it
([benchmarks/perturb_injection.py](../benchmarks/perturb_injection.py)); a Russian instruction over a Russian ticket is
caught the same way. On clean text the cost is small: the rules rarely fire on ordinary messages (the same script
counts how often they do on e-mails). The rules know English and Russian wordings; a paraphrase no rule knows passes
("kindly file this under X": 12.5% either way, same script), so it is a safeguard, not a proof.

**Ask in the language of the checkpoint's training; let the text be in any.** Questions and options in Russian lost
much accuracy against English ones over the same Russian text. Corrections do not carry across languages.

## Agents

**Rules choose the step, the model chooses within it.** A model asked to pick both the next step and its target
rarely picked the right page element (it chose the link that shared a word with the goal). A scripted flow that asks
the model a closed question at each step, with solvi's checks around it, finished the tasks.

**Keep the agent's memory in the decision's input.** State kept in the harness makes decisions unreplayable and hides
from the model what was already tried — it kept proposing what had already failed. An `Episode` snapshot as a given
fact fixed both: the decisions replayed, and more tickets were solved.

**Define progress explicitly.** "Something changed" is not progress: a wrong click changes the page too, and then
erases the memory of itself (one agent chose the same wrong option over and over). Progress is a sub-goal reached.

**Label outcomes by the sub-goal the step served.** One global measure ("closer to the goal") taught a head to skip
every step that does not move it — training, healing, shopping. With labels judged per sub-goal it agreed with the
intended policy far more often.

**Put the check on the action, not on who proposed it.** A guard written only for the model's proposals let the
fallback rule take every harmful action. Hard checks belong to the question, whoever answers it.

**Treat a loop detector as a signal.** Single detectors (the same action again, A ⇄ B) fire on honest repetition;
"no progress for a while AND one of them" was right far more often. Answer it softly: close the option for a while,
escalate.

**Keep a map of an environment the agent meets again.** Without one, every task re-discovers the structure: in the
command trees of real tools, an agent with a `WorldMap` kept across the tasks reached more of the named commands in far
fewer steps. Where everything is one step away (a site with a full sidebar) a map adds nothing, and it does not shorten
the first exploration.

**Charge a risk budget on every take, and expect no gain from it over a written agent's reflexes.** In a crafting game, `RiskBudget` placed where the written reflexes refuse the planner's action bought no achievements at any budget (−0.16 per episode at a budget of 1.0, −0.28 at 2.0, against the bar of +10%), and random takes at the same per-class rates did as well as the budgeted ones. With a key that charges a trade once per episode (the pattern of the agent example) the budget stopped bounding exposure: 29 takes for 0.56 charged, −23% steps lived. Pass no key, so every take is charged; and measure a risk policy against random takes before claiming it is justified.

**A script is a strong baseline — measure against it.** On simulated support tickets a model with memory solved as many
as a short script did; on incidents a runbook resolved far more than the model. What solvi adds to a model-driven agent
is that every step is checked, recorded and replayable, not that the model beats the rules.

## Specialists (a model proposes, code checks)

**Expect a second reader to buy exactness and cost half the coverage; let the arithmetic vouch where it can.** On receipts
(SROIE and CORD v2 test sets, 1,923 gold fields) an open OCR model alone read 88% of the fields right; accepting a field only
when a second OCR engine read the same digits in the same place and the receipt's sums held raised that to 96.4% with no
accepted total failing its arithmetic, but 64% of the fields went to "needs review" or "conflict": the second engine confirmed
39–60% of the first one's fields, and a sum can only vouch for a field that takes part in one (most item lines carry no unit
price). Reading the second engine on the first one's word boxes (41% coverage on calibration) and dropping it for money fields
with a passing sum (41%) moved coverage by a few points, not to the 70% we wanted. Budget for the review queue before promising
coverage, and put the budget where the fields have no equation neighbours (names, dates). With a second engine that
reads camera photos (PP-OCRv4 instead of Tesseract, on receipts no run had seen), agreement between the engines rose from
36–60% to 81–94% of fields and acceptance from 36% to 55% at 95.7% exact — and the misses left were the rule-based field
proposer picking the wrong line (a pre-rounding total, a card row as cash) with both engines reading the same digits. Measure
which of the two is your bottleneck before swapping readers: agreement between readers tells you.

**Normalise numbers the same way on both sides before judging a transcript, and look at what the two readers agreed on.**
Two speech models of different families (faster-whisper small, a Kaldi model) aligned in time gave 0 false words on 400
non-speech clips and caught 299 of 300 injected corruptions, yet among the numbers they agreed on only 90% (Russian) and 86%
(English) matched the reference text: 21 of the 22 misses were clock times ("10:00") and ratios ("3:2") that our rule
normaliser tokenised one way in the reference and another in the hypothesis; one per language was a real mishearing. A
threshold rule that demands 99% on a dev set with such misses picks a corner (English: 13% coverage at p ≥ 0.99). Fix the
tokeniser first; agreement of two readers is the check that works. With one tokeniser on both sides and a larger second
model, on sentences no run had seen, 97.2% (Russian, 81% coverage) and 97.6% (English, 70%) of the accepted numbers matched
the text; the rest split evenly between forms the rules still read differently (spoken fractions, spoken clock times,
"one point one") and numbers both readers misheard the same way — the second reader cannot catch those. Expect two
small readers plus rules to top out near 97–98% on read numbers, and budget a person for the rest.

**A certificate certifies execution, not reading: contract the type and the applicability.** A program that passes its
examples, a check that verifies a spec against its source, a replay that matches — each says the answerer ran right,
not that it was the right answerer for this input. Declare what reading the question right looks like
(`Catalog.contract`, a slow path's `contract=`, `Specialist.applicable`): the answer's type matches what the question
asks, the answer is not the input it was read from, and the answerer is called only when its precondition holds — not
because the input has the right form. These clauses are hard: they refuse whatever the confidence, with the reason
named.

## Tools (the agent guard)

**A call must be grounded in the request.** When the model may call a tool on a request that does not ask for one,
turn on `guard.firewall()`: every slot value of a call must be written in the user's request or be a fact your app
gives, numbers as digits, enumerations and booleans exempt by their declaration. On the synthetic irrelevance set of
[benchmarks/guard_firewall.py](../benchmarks/guard_firewall.py) the share of false calls made fell from 1.000 with no
guard to 0.750 with the schema check alone and 0.217 with the firewall, while 0.833 of the wanted calls were still made
(the rest — amounts said in words — went to a person). What gets through has all its values in the request; whether
the request asks for the call at all is a judgement the firewall leaves to `require_request`, the authorizer or a
person.

**Ground the arguments that matter in the user's words** (`ground=`, `ground_from=("user",)`), and in a long session
limit how far back a mention counts (`ground_last=1`): a file the user asked to read earlier otherwise grounds
deleting it now.

**Mark calls that must not repeat** (`once=True`): without it, an agent made a second refund of the same order.

**Require the user's own yes for actions a tool output could ask for** (`guard.require_confirmation`). With an
instruction planted in order lookups, an agent cancelled an order nobody asked about in most τ-bench runs without a
guard, in fewer with grounding and policies, and in none with confirmation. It costs turns: on the 30 clean τ-bench
tasks a guard with confirmation solved 14 against 18 without a guard, one run each with a simulated customer
([taubench/solution.py](../benchmarks/tasks/taubench/solution.py)), so keep it to actions that must be the user's decision.

**Write policies for what your backend does not check, not for what it does.** τ-bench's tools already refuse a wrong
status or a foreign payment method, and there a guard changed nothing. They do not check whose order is cancelled: a
planted note got another customer's order cancelled without a guard, and not with an ownership policy.

**Do not use a small decision model as the only authorizer.** Asked "did the user ask for this call with these
values?", solvi-base said yes to calls it should have refused. Grounding and policies in code did the work.

## Laws from an environment: the test is the judge

**Let the environment judge a law, on transitions it was not born from; keep the held-out buffer by group.** With
`solvi.experimental.laws` a wrong prediction proposes candidate laws ("for this action, under this condition, the effect
is this"), and a candidate is accepted only when it removes errors on held-out transitions of other worlds, sessions or
paths than the one that proposed it (`Buffer` keys rows by `group`; `test(..., exclude=group)`). On a game simulator
with written mechanics the laws closed 94% of the gap between a learned map and the written mechanics — and random
candidates from the same grammar under the same test came within 1.1 points: where the grammar can be enumerated, the
test does the work and the mismatch only keeps the set of laws compact (61 laws against 103). Retire a law when its
held-out precision falls (`retire_below`), and cap the set. Where each test is a call into a live system, plan the
probes: on a real MCP filesystem server with a path policy, probes chosen to tell the mismatch's candidates apart
(read-only calls first) learned every denial class in one session — refusals 6 → 0 in the next one, no allowed call
skipped — where random probes at the same budget went 48 → 29. **An LLM may propose, never judge:** under the same
test an LLM proposer ended 0.95 points below enumeration, and in the same study a model judged the truth of its own
claims about the path policy wrongly 18 times out of 40. Both findings in miniature, with their bars:
[laws_toy.py](../benchmarks/laws_toy.py) (`pytest -m slow tests/test_laws_toy.py`).

## Size the acceptance by the search; admit by probes; abstain when laws disagree

**Size the acceptance by how many candidates you tested.** A fixed bar ("support ≥ 5, precision ≥ 0.9") does not know
that a stand tried tens of thousands of candidates on the same buffer, and some random condition always passes it: on a
synthetic buffer with no law at all (2,000 rows, an effect at 70% independent of 40 sparse atoms, 821 conditions) the
fixed bar accepted a law in 100 of 100 buffers. `LawSet(sized=True)` — the default — tests each candidate at the level
δ / (N·(N + 1)), N the candidates scored on that buffer so far, so the chance of any false acceptance stays below δ:
0 of 100 on the same buffers. **Test against chance, not against 0.9.** The default null is the effect's base rate in
the buffer (`null="base"`: what a random condition achieves for that action and effect, floored at 0.05), with
`min_precision` kept as a separate filter: a deterministic law at a base rate of 3% was accepted in 100 of 100 buffers
on a median of 29 rows, where a fixed null of 0.9 (`null=0.9`) accepted it in none — beating 0.9 at tens of thousands of
candidates takes hundreds of rows even for a law that is never wrong. **The need:** a deterministic law needs
⌈ln(1/δ_N) / ln(1/null)⌉ rows, δ_N = δ / (N·(N + 1)): at 52,000 candidates and δ = 0.1, about 9 rows at a base rate of
3%, 68 at 70%, 228 against a fixed 0.9. **A buffer too small for its null accepts nothing — that is the rule working,
not failing.** Collect more held-out rows, test fewer candidates, declare `space=`, or pass `sized=False` knowingly and
treat the fixed bar's false accepts as the known price; the stands measured above ran with `sized=False`.

**Where the world answers questions, admit by discriminating probes, not by a random sample.** A neighbouring research
line measured that a hypothesis "certified" on 8 random held-out examples is wrong more often the larger the space
(0, 0.105, 0.193 at 17, 1,462, 124,287 hypotheses), and that 6 queries chosen to tell the survivors apart bring the
largest case to 0.030. `LawSet.admit_by_probes(candidates, ask, probes=, safe=, budget=)` does that for laws: it asks
the safe probe that splits the surviving classes most and accepts the shortest candidate only when one class is left;
otherwise it returns "undetermined" with the classes. On a synthetic world at 8 checks: a random held-out sample with
the shortest survivor accepted a wrong law in 0.593 of 300 trials, the probes in none (0.987 right, 0.013
undetermined). Ask only where the rest of your law set and base are right, or a right candidate is refuted with them.

**Abstain when consistent laws disagree.** Two laws that both passed their test, neither an exception of the other,
predicting different things for one state are a reason to abstain, not a tie to break by length or age:
`predict(state, action, disagree="abstain")` returns `Undetermined` there and counts it (`lawset.abstained`). Skills
do not abstain: two certified skills of one goal are both verified plans, and a different sequence is not a
disagreement about the world. **Report the search cost next to accuracy**:
`lawset.cost` and each skill episode's `cost` (candidates, tests, probes, simulated steps). Measured on synthetic worlds
only: [tests/test_laws_w2.py](../tests/test_laws_w2.py).

## Relational rules: enumerate, let the test judge, then compile

**When a rule is about structure — an ancestor, a neighbour, a path — give the grammar relations, not more flat atoms.**
`solvi.experimental.relations` encodes a state as relations over its nodes (a path tree: PARENT, SAME_DIR, IS_DIR,
NAME=x; a grid seen from the player: EAST, SOUTH, AT, FACE and the cells' materials) and enumerates small
relation-algebra programs bottom-up, one per behaviour. Two programs are one only when their full relations agree on
every example — never when only the readout agrees, or what composes from them is lost. On a pre-registered battery of
28 random rules over three relations it found 22 within seven nodes, each exact on larger graphs of other families, and
a rule that needs four variables (a common 4-clique) was never accepted. `RelAlgProposer` hands the best programs to a
law set as atoms and the law set's held-out test judges them: for a policy that denies reads under any directory named
"secret" (files may be named so too), on a synthetic path tree with that policy (`benchmarks/relations_compile.py`) the
found relational law was right on all 20,000 fresh rows, which the flat grammar's "the parent is named secret" cannot
be. On an MCP filesystem server behind a written guard, laws with a relational atom (ancestor directory named secret)
refused none of the 35 calls under a secret directory in the second session (the flat grammar refused all 35) and
wrongly skipped none of 130 allowed calls; one unrelated refusal remained (a listing of a file link, mispredicted by
both grammars), so the pre-registered bar of zero refusals was missed. On the game simulator +0.02 points (0.9882 vs
0.9880). One layout, one run each. **Equivalence holds on the examples only**: the smallest program of a class may not
generalise, so keep the held-out check on other groups, sizes or families. Expect programs of four or five nodes with
twenty-odd relations; with many or large examples search a subset and re-check on all (`synthesize(..., check=)`), and
with noisy labels rank by agreement (`top_programs`), not by an exact match.

**Compile a found rule, and accept the code only after the differential check.** Give atoms their code (`Atom(...,
code=)`): `LawSet.to_module()` writes the laws as one module that imports nothing from solvi, and
`check_compiled(module, buffer, lawset)` runs it against `LawSet.predict` on every row — accept it at 0 differences. A
relational rule on a large graph compiles to index arrays (`compile_program(ast, backend="arrays")`); refuse dense
answers (a complement over all pairs) unless the domain is bounded. On a forest of 100,000 nodes closure, grandparent
and sibling came out equal pair for pair; the index-array backend within 0.7–1.3× of hand code over a parent array
(output conversion excluded); the plain set-code backend 2–3× slower than hand-written set code over the same pairs. A
law set learned on the toy grid agreed with its interpreter on 100,000 rows. On five law sets learned on a game
simulator with written mechanics (61–103 laws each; one with 64 relational atoms over a 3×3 window compiled as a bounded
domain), `LawSet.to_module` + `check_compiled` agreed with `LawSet.predict` on every one of 308,510 logged transitions
(training, validation and test worlds), no atom refused. A differential check on logged rows certifies the module on
those rows, not equivalence in general — run it on your own rows. The compiled law part ran ≈ 14× (flat atoms) and ≈
120–165× (64 relational atoms) faster than the library's own row-by-row interpreter on 100,000 rows (single runs; the
interpreter evaluates every atom of the grammar per row, the compiled module only the atoms of each law's condition and
stops at the first hit). That is a speedup over the interpreter, not over hand-written code: on a forest of 100,000
nodes the index-array backend was within 0.7–1.3× of hand code over a parent array, and the set-code backend 2–3× slower
than hand-written set code over the same pairs. Scripts: [relations_battery.py](../benchmarks/relations_battery.py) and
[relations_compile.py](../benchmarks/relations_compile.py) (`pytest -m slow tests/test_relations_gate.py`).

## Learning in a world described by code: plan on the code

**Where the world is code, plan on the code and compile only what the code verified.** A simulator, a file system or a
service with a specification answers every "what happens if" exactly, so System 2 can search a copy of it
(`solvi.experimental.skills`: `ExactModel(env)` forks the environment, `plan` searches the forks, `verify` re-simulates
a plan on a fresh fork) and the real environment is never stepped by the planner. A plan becomes knowledge only after
that re-simulation, never because a model said it works: in the gridworld toy, sequences compiled without it reached no
goal. **A skill is used alone only while its outcomes keep the certificate** (`Certificate`): before it is certified,
System 2 re-verifies it on the exact model at every use, so every use has an outcome — a certificate fed only by the
decisions the system made alone can certify what it never tested. Budget the need: a fresh certificate takes 50 clean
outcomes at 5% and δ 0.1 (25 at 10%, 12 at 20%), so a skill used once per episode needs about fifty episodes before it
acts alone; pool the certificate by goal or over the set (the promise is then the pool's) or accept a looser promise
on short streams. In the toy (20 fresh layouts × 5 seeds) the set certified at 20% between the ninth and the thirteenth episode
and System 1's share reached 0.81 in the last five with every goal still reached; per skill nothing certified within
20 episodes.
**Cases do not transfer, rules do:** replaying the sequence that worked from the same start state gave System 1 nothing
on fresh layouts; a skill's condition keeps only the atoms its verified starts shared, so it reaches states no case has
seen. On Crafter (written mechanics, symbolic state, hand-written survival reflexes shared by every arm), an action
model learned from the agent's own attempts and carried across 30 worlds (10 independent runs) raised the unique
achievements unlocked per episode from 6.0 to 10.6 of 22 (+77%; the oracle 14.4) and System 1's share to 84% after about
twenty worlds, part of it by living longer; with the written rules given to the agent as gates, the learned memory added
nothing (−4.5%). Script for the gridworld:
[skills_toy.py](../benchmarks/skills_toy.py) (`pytest -m slow tests/test_skills_toy.py`).

**Measured on Crafter.** On Crafter (version 1.8.3; written mechanics as the exact model; symbolic state; hand-written
survival reflexes shared by every arm, 60% of all steps; the planner reads the copy's full map while the game shows an
agent a 9×7 window), in one run of 10 streams × 30 episodes, no agent seed (the loop is deterministic given the world
seeds). Achievements are unique achievements unlocked per episode (of 22). The skill set was certified at 5% / δ 0.1
(fifty clean uses, one e-process pooled over the stream's skills) in every stream by episode 1–2. In the last ten
episodes System 1 acted alone 1,350 times with an observed outcome and failed twice (0.15%, Wilson upper bound 0.54%; 4
of 2,882 over the whole run); 57 further alone runs were paused by a reflex and not resumed, so they carry no outcome
(counted as failures they would make 4.2%, Wilson 5.4%). The promise is the set's, not a single skill's; a skill's own
alarm retires it (4 retirements, each later re-certified). Goals that are stochastic by the written rules stayed with
System 2 and are outside the promise. With skills the planner was called 32 times per episode against 42 when it plans
every goal (−23%), and an episode took 84 s against 107 s on the same machine (−21%), with achievements kept: 15.2
against 15.7 unique achievements unlocked per episode of 22 (ratio 0.966, bootstrap lower bound 0.934). System 1's share
of the non-reflex steps rose from 7% in the first ten episodes to 23% in the last ten (9% of all steps); the
pre-registered bar of 60% was missed. Routes to a resource and plans for unseen footprints stay System 2 by design
(routes were 25% of non-reflex steps), and every certified skill was a one- or two-step gather sequence with a constant
footprint: in an open world the planner did most of the work and skills took the repeatable part. Hierarchical skills
and conditions with variables are untested. An arm that executed and compiled random action sequences of the plan's
length in place of verified plans unlocked 4.0 unique achievements per episode against 13.7 for the same agent with no
memory at all; none of its 5,497 unverified skills was ever certified, so System 1 never acted. The gain is the verified
plans', not the loop's bookkeeping, and the certificate held unverified skills out; what unverified memory does on its
own was not measured. The observation-only learner (the agent's own action memory with the written gates on) unlocked
13.3 unique achievements per episode, no higher than the same agent without memory (13.7); the planner arms' lead over
both is the planner's with the full map. The planner reads the whole map of the copy; a Crafter agent sees a 9×7 window
from pixels. No number here is comparable with published Crafter scores (a geometric mean of success rates over 1M
training steps).

## Self-change: guard refusal first

**Refusal is the first casualty of self-change; the regression check cannot be switched off.** A chain of model merges
has been reported to keep every skill and lose much of its refusal (a number from outside this repository, not
reproduced here): each change looks fine on what it was meant to improve, and the inputs the system must refuse or
hand over are answered a little more each time. In `solvi.experimental.changes` every
change whose target can answer passes `RefusalRegression` — the share refused on a held should-abstain set before and
after — and there is no flag to remove it. Declare the set (`Pipeline(refusals=...)`): outside or unknown inputs, the
cases your policy hands to a person; without one the change is held for a person — a vacuous pass would be a switch in disguise. Keep the
default tolerance (0: one input answered that was refused before refuses the change) unless your answerer is
nondeterministic. The set follows the policy version: a document version may rightly redefine what goes to a person, so
declare the set of the version in force, as `benchmarks/changes_credit.py` does: there a self-tuning threshold change that closed the review band
answered 137 of 137 hand-overs and was refused, while 12 exact corrections passed.

## Rules over relations: enumerate with a library, rank only where a check is expensive

**Find a rule over relations as follows** (`solvi.experimental.relations.library.find_rules`):
1. Enumerate small programs bottom-up with observational equivalence, from the base relations plus the library.
2. Accept only what passes the held-out test (exact, or support / precision / recall / net on other worlds).
3. Put accepted rules into the library as new terminals, so composites become small programs.
4. Use a gradient-trained learner only when the enumeration budget runs out or the labels are noisy, and only as a
   proposer whose program is then checked like any other candidate (`proposer=`).
5. Rank candidates with a learned or feature ranker only when a check is expensive (`RankedProposer`).

On two toys ([relations_library_toy.py](../benchmarks/relations_library_toy.py), five seeds each) the library found
every level — a path tree's "denied if an ancestor directory is named secret and the depth is at least 3" (three levels)
within 2,556 enumerated programs, kinship on forests to depth 6 (second cousin once removed) within 10,817 — where the
same search without the library found neither top rule in 100,000, with 0 false accepts on larger fresh worlds. The
kinship ladder misses the toy's bar of a tenth of that budget (1.08×): the universal relation U, a terminal of the
package's enumerator, enlarges every level: with `include_universal=False` (U left out of the terminals of both arms)
the ladder costs 7,889 and meets the bar, the path tree 2,174 (1,679 with the earlier minimal enumerator; the ladder
6,125). Leave U out when no rule you look for needs "every pair". A library also has a cost: every entry is a terminal,
so a rule it does not help gets dearer (sibling took 6,545 programs in its own round after grandparent and
great-grandparent were admitted, 951 without the library). The intermediate concepts were given as targets: the search
found which one was within reach and the programs, not the concepts. On a synthetic five-level path-tree policy (denied
if an ancestor directory is named secret and the depth is at least three; the four intermediate relations given as
named, unordered targets) `find_rules` with the library found every level within 4,111–4,132 enumerated programs over
five seeds; without the library the top level was not found within the cap of 1,000,000 programs, so the budget ratio is
at most 0.0042 and its true value unknown. Per level the library is not free (90 vs 71 and 2,951 vs 1,898 programs on
two of the shared levels; cheaper only on one). Two controls — a rule no program of the grammar can express, and one
that needs a 56-node program — were never admitted (0 of 20 runs). On the kinship ladder (nine rungs up to second cousin
once removed) the library found all nine within 3,370–3,378 programs; without it eight of nine — the ninth needs a
program of at least 17 nodes where the no-library search exhausted size 15 at ≈ 500,000 programs. 0 false accepts on
held-out forests of the same generator at a larger size. **Hold out other families, not only other draws:** a rule the
programs cannot express ("x and y share a 4-clique") was never admitted on graphs of density 0.3 or 0.5, exact or by
precision; on dense graphs (0.7) almost every edge is in a 4-clique, and precision / recall acceptance with held-out
graphs of the same density admitted the edge relation itself — held-out graphs of density 0.4, or exact acceptance,
rejected it (`tests/test_relations_library.py`). **The caveat on step 4:** that one-step rules are learned by gradient
is verified only on hand-picked rules over one relation; on random rules plain enumeration already finds every one-step
rule in seconds, and rules of two steps or more did not hold when learned by gradient — no gradient learner ships with
solvi. **On step 5:** a check on the logged rows is a few bit operations, so ranking pays only where each check is a
live probe or a planner call. On synthetic tasks drawn from one grammar (30 atoms, k = 3) the trained bag-of-atoms
ranker needed 2.1 checks to the true condition against 69.7 in the blind order (by length, then name), and the
raw-feature control 24.6 (the best raw threshold picked after the fact: 12.8) — in distribution only; measure the
control on your own grammar before trusting the trained one (`tests/test_laws_rank.py`). Measured once on a live file
server (8 mismatches; candidates contradicted by logged rows skipped free): every order spent 29 probes in all, because
1–5 candidates per mismatch were left to probe and the true conditions were single atoms — the shortest hypotheses,
which the length-first order meets first. Ranking can pay only where many consistent hypotheses remain after the free
checks; the control first.

## Learn composition tables by exact counting, derive by composition, refuse what the table cannot derive

Where a relation between two things follows from a chain of relations between them (kinship chains: a father's sister
is an aunt), learn the table of two-step compositions by counting labelled chains (`relations.CompositionTable`: a
cell is admitted when its majority relation has a share ≥ 0.9 and at least two votes), fill empty cells from chains of
three by residuals, and derive a longer chain by composing over every bracketing — answer only when every bracketing
that composes agrees, and refuse otherwise ("underivable", "disagree"); the cells used are the answer's reason. No model
is involved: the table is learned by counting and the derivation is code. The promise is precision among the answered,
not coverage. Measured on CLUTRR (Sinha et al., 2019), the clean split trained on chains of 2–3 relations and tested on 2–10; the input is the benchmark's relation graph — the chain of relations between the two people — not the story text; the text-to-graph step is not part of this measurement: a table learned by counting (majority 0.9, support 2, residual fill, all bracketings must agree; 111 cells from 9,074 training chains of length 2–3 in 0.07 s on CPU) gave 0 wrong among 1,087 answered of 1,146 test queries (38–190 per length; at lengths 2 and 3 only 2 and 7 distinct chains): precision among the answered 1.000 (0 of 1,087) at every length 2–10, coverage 0.949 overall, 0.975 at length 10 (116 of 119) and 0.916–0.993 at lengths 4–9; 59 refused (21 disagreeing bracketings, 38 underivable); one split, one run; the most frequent training relation per length answered 0.077 (0.134 at length 10). The benchmark is generated by closed-path composition rules, so an exact table can exist by construction; what is measured is whether counting with refusal recovers it without errors. The looser agreement rule costs errors: the agreement rule «majority of bracketings» answered 1,092 with 2 wrong (precision 0.998), so the strict rule bought 0 errors for 5 fewer answers.

Two settings to leave alone on a deterministic table: **keep `support=` explicit** — the sized acceptance with a base-rate null (δ 0.1) was identical to the default on the full training split (smallest admitted support 15); on a 10% slice (861 chains, 116 cells with votes) it refused 12 of 15 exact support-2 cells and 1 of 12 at support 3 (a 2-vote cell passes only for a relation with base rate ≤ 0.03), admitted 87 cells against 97, and answered more with lower precision (1,092 answered, 10 wrong against 1,083, 7 wrong) because residual votes lifted three further cells over 0.9 (two gender-dependent, one correct); and **leave the identity relation off**
(`identity=`) where a cell depends on what the chain does not carry (a gender): the opt-in identity relation, with no gender slot, gave 9 wrong answers at lengths 4–10. Residual fill after a
different table admits different cells; residual votes are not gated like direct votes yet.

## Derive to a fixed point, let disputes refuse

When the facts of a case and learned composition rules sit in one knowledge store, derive their consequences there
(`relations.forward_chain`) rather than in a side computation: every derived fact names its two premises and the rule,
is never ranked above the weaker premise, and leaves with a premise that is retracted. Do not pick a winner when two
derivations of one pair disagree — the store makes the pair disputed, which is the table's refusal by disagreement kept
as a question for a person; a person's answer (`store.resolve`) then settles the consequences of both sides without a
re-run. Keep the rules promoted before you rely on what follows from them: `rules="any"` chains hypothesis rules too,
and everything they derive stays a hypothesis until the rule is promoted. Measured on CLUTRR (Sinha et al., 2019), the clean split's training file only (the relation graph of each story, not its text): on 1,816 held-out stories (a fifth of the training file by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths, 111 cells) the store's reading of the query pair after chaining to the fixed point was compared with the table's derivation over the shortest paths: 1,675 equal (0.922), 11 disputed where the bracketings disagree (0.006), 130 with nothing derived where the table cannot derive (0.072), 0 differences; the fixed point took 0 rounds for 72 stories, 1 for 826 and 2 for 918; 2.6 s of chaining in all, 1.4 ms a story, on CPU; one run, the test file not read. On 120 random chains of 2–7 relations over 12 random tables whose bracketings can disagree: 0 differences (2 answers equal to the table's rest only on a disputed pair and stay hypotheses until a person answers it) (`benchmarks/relations_chain.py`).

## Synthesise chains by residuals first, enumerate second

When the rule you look for may be a chain of relations — kinship degrees, «the parent of the sibling of», a path of
fixed shape — search it top-down from the target first (`relations.synthesize_residual`, or
`synthesize(method="residual-first")`) and leave the enumerator for what a chain cannot express. Bottom-up enumeration
pays for every program of the size whether or not it could lead to the target; residuals prune a step as soon as a
pair of the target has no witness left. Measured without a library on the kinship ladder of
[relations_library_toy.py](../benchmarks/relations_library_toy.py) ([relations_residual.py](../benchmarks/relations_residual.py),
five seeds): residuals found all nine rungs on every seed (45 of 45; second cousin once removed in 0.40–0.55 s, 3 s for
all 45), the enumerator at 100,000 behaviours (size 12, U left out) found the four rungs up to great-grandparent and
none above (20 of 45, ≈ 6 s per rung at the cap); 0 false accepts in both on check worlds of another shape. The two
searches count different things (residual steps, distinct behaviours): compare what is found and the time. **Give the
search a held-out test** (`test=HeldOut(...)`): a candidate that fits the examples and fails it is counted and the search
goes on — on a path-tree policy the enumerator, without one, returned a program exact on three example trees and wrong
on the check trees. **Know the limit:** only chains (atoms and converses, ∖ I on segments); union, intersection,
closure and counts are out, so on the five-level path-tree policy the residuals found the one chain level and ruled
the count and intersection levels out in 0.04 s — there the enumerator with a library is the search (previous
section). Two masked segments side by side are out too, unless one of them is a library entry, which residuals take as
an atom.

## Ask why a rule did not derive, and learn what is missing

A refusal of a composition table says *that* the table cannot derive an answer; `relations.why_not(table, facts, x,
y, relation)` says *why*, over every path between the two: a missing cell r1 ∘ r2 → ? (ranked by the paths one cell
would unblock), a dispute between paths, another relation derived, or no path at all — a missing fact, not a missing
rule. Use the reason as the learning target: `Derivation.missing` carries the blocking cells on every "underivable"
refusal, and `WhyNot.to_agenda(agenda, table)` turns each gap into a goal «learn cell father ∘ sister» whose done check
reads the table, so the goal closes when the cell is observed and the table refitted. Keep a dispute where it already
is — a person question in the knowledge store — and a missing fact as a question about the story, not the rules. The
positive control is in the tests: remove one cell from a fitted table and `why_not` names exactly that cell; it has
not yet been measured on the refusals of a real split.

## Storage and audit

**One store, any number of writers — but check which backend.** SQLite and PostgreSQL take several processes;
`JSONLStorage` does on POSIX (a file lock per append), and on Windows one writing process.

**Verify against something kept elsewhere.** `verify()` catches edits; a rewrite of the whole store with its head is
caught only by a head or a signature you published (`verify(anchor=...)`, `verify(signature=...)`).

**Read the kind of a replay mismatch before raising an alarm.** `integrity` means the data changed after the run;
`missing_part`, `recompute` and `model_changed` mean the catalog or the model moved on. `summary` says which.

**Erase with `redact`, never by editing.** It removes a record's content, keeps the chain, and records the erasure;
editing or deleting a record is indistinguishable from tampering.

**For frequent decisions, store them compact and rotate the file.** A full record keeps every step's value and
inputs: 10 KB a decision on a game-like loop and on the credit task, up to 38 KB on the gallery. `record="compact"`
kept 3.7 KB and 1.9 KB of those, and every decision still verified and replayed (2.0–5.2× smaller,
[journal_size.py](../benchmarks/journal_size.py)); `rotate_bytes=` keeps each file small. What is left is mostly the
input: a long text given unchanged with every decision is stored with every decision — give it once, and a key or a
hash with each. Keep `"full"` (or `"sample:N"`) where you will want to see the old intermediate values after a change of
the catalog: a compact record tells you which step changed, not what it was.

**Give a model's spend a budget where it is spent.** A generator with `total=Budget(usd=...)` stops before the request
that would exceed it, and a refinement with `budget=` before the round; a budget written on top of the provider's API
cannot stop a request that solvi is about to send.

**After a change of the catalog, replay before you trust old traces.** A renamed part, a new input or a retrained
model each make old decisions non-reproducible; `solvi diff` shows which stored decisions would change and why.
