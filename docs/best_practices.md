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

**Counterfactuals over a few enumerated fields: try them all.** `solvi.core.explain.search(..., domains=...,
exhaustive=200)` re-runs every combination and returns every counterfactual (`complete`); on a three-input rule it was
1.5–1.6× faster than the probing search in one run. Numeric inputs keep the probing search.

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

**Take `procedure="fallback"` for a learn-then-test promise; keep the default when you want the realised error far
below delta.** The default tests every one of up to 64 thresholds at delta / 64; the fallback tests them from the top
down and lets a certified threshold pass its level on, at the same promise. On five intent classifiers (CLINC150,
Banking77; max-probability scores; delta = 0.1; error targets 1–10 %; 300 or 1,000 calibration examples) it answered
+0.10 more of the inputs on average than the default (+0.03 to +0.18 per classifier), never less in any cell, and
broke the promise in at most 5 % of calibration draws (the default: 0.2 % — it is far more conservative than its
delta). `procedure="fixed"` answers a little more — the fallback's price against it is −0.028 of the inputs pooled,
−0.055 at 300 examples and ≈ 0 at 1,000 — but can answer less than the default when a wrong case sits among the first
ones answered (3 of 40 cells, down to −0.08); take it only where e × n is about 3 or less (1 % at 300 examples), where
the fallback certifies nothing. Both need the error to fall with the score: on a signal where it does not, the descent
spends delta at the top and the default is the better choice. Measured on max-probability scores and delta = 0.1 only.

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

**The drift monitor needs 500 calibration examples.** `build` watches drift by default only when the guarantee has
at least 500 calibration examples (about 2 000 examples with the default split, 4 000 with a slow path); below that
nothing watches the stream unless `monitor=True`, and only `explain()` says so. Measured on four intent sets with
1 000-decision streams: 0–1 % false flags observed at 500 (95 % upper bound 3.6–5.4 %), 6–24 % on two sets at 25–200;
the shift caught in 89–100 % of streams at 500, 59–79 % on the other two sets at 50–200. A flag stays until
`reset_drift()`.

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
A regime change can raise the decider's confidence as well as lower it: on half of ten dataset pairs of intent
classification the mean confidence rose, where a monitor for a drop of confidence sees nothing. Test both directions,
or the whole spread of the answers' features (`ShiftWatch`, [experimental](experimental.md#ood)).

**`fit` and `teach` move the scores, not the reading.** They shift and scale the logits of one question; they help
calibration and the mix of answers, and level off within a few dozen examples. When a question needs the model to read
the input differently, use a LoRA adapter (`adapt_lora`), a head over computed facts, or a rule.

**If the zero-shot decider does not separate the classes, `fit` cannot help — in that case a LoRA adapter is needed.** `fit` only
moves the priors (a shift and a scale on the logits): when the logits carry no signal about the classes, it collapses
to one class. A user deciding how to close pop-up windows from short interface texts (window titles and the list of
actions, 375 labelled examples, judged on screens the model had not seen) got 8 answers given alone out of 375 from
solvi-base with `fit` and `act_guard`, and 257 with a LoRA adapter (88% of them right). Check first: if the
calibrated decider's accuracy on held-out labels is near the majority share, go to `adapt_lora` (or a head over
computed facts) rather than more examples for `fit` — and calibrate the adapter on held-out groups (the promise of an
adapter's confidence was optimistic on unseen screens: 14.3% against 10%).

**Split by source when examples share one.** Several examples from one screen, customer or document are not
independent: a random split calibrates on sources the model was fitted on, and the promise is optimistic on new ones.
`solvi.build(..., groups="screen")` (a key of the state, a function, or one group per example) keeps every group on one
side, so the guarantee is read on unseen groups; on a synthetic set where the model remembers its training screens the
random split promised ≤ 10% answered alone and wrong and gave about 25 % on new screens, the grouped split under 10 % (the generator of `tests/test_auto.py`, five seeds: 24–26 % against 3–9 %).

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

**Every score has a control that must fail.** In an external codebase we reviewed, a knowledge-graph ranker reported
100% Hits@10 that an untrained model also reached: the score measured the shape of the scoring, not the model. Run
`solvi honesty <set> --control`: the same answers against gold shuffled within each question. The set discriminates only
when the real confident-error rate is at least 0.05 below the control's and at most half of it (core set, one run: 4.9% against
54%). A system that always gives the majority answer scores the same against shuffled gold (`discriminates: false`), and a
question with one gold in every case cannot be controlled this way at all (`uncontrolled`): add cases where the answer differs.

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

**A fitted injection detector holds on the family it was fitted on, not beyond.** The fixed injection rules find about
a quarter of polite injections in tool outputs (23% of 248 held-out InjecAgent cases) and flag about 4% of ordinary
texts. A small classifier fitted on labelled outputs and calibrated on ordinary ones
(`solvi.experimental.injection.InjectionDetector`) found three times as many on that family at a lower flag rate
(1.8%) — but half of that gain was one address repeated across the dataset (+0.19 over the rules without it, 95% CI
+0.06…+0.38), a plain TF-IDF classifier did as well, and on AgentDojo-style injections the rules lead (78% against
53%). A detector is a per-deployment signal: fit and calibrate it on the outputs your own tools return. The grounding
and the policies are what hold.

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
accepted total failing its arithmetic, but 64% of the fields went to "needs review" or "conflict": the second engine
confirmed 36–60% of the first one's fields, and a sum can only vouch for a field that takes part in one (most item lines
carry no unit price). Reading the second engine on the first one's word boxes (41% coverage on calibration) and dropping
it for money fields with a passing sum (41%) moved coverage by a few points, not to the 70% we wanted. Budget for the
review queue before promising coverage, and put the budget where the fields have no equation neighbours (names, dates).
With a second engine that reads camera photos (PP-OCRv4 instead of Tesseract, on receipts no run had seen), agreement
between the engines rose from 36–60% to 63–94% of fields on the calibration receipts (81–94% on the blind sets) and
acceptance from 36% to 55% at 95.7% exact — and the misses left were the rule-based field proposer picking the wrong
line (a pre-rounding total, a card row as cash) with both engines reading the same digits. Measure which of the two is
your bottleneck before swapping readers: agreement between readers tells you.

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
with written mechanics (20 fresh worlds, 4,731 transitions; one run) the laws closed 94% of the gap between a learned
map and the written mechanics (exact-effect accuracy 0.892 → 0.988 against a ceiling of 0.994), but random candidates
from the same grammar under the same held-out test reached 0.977 — the laws' own gain over them is 1.1 points, and the
measurement's pre-registered bar for that control (random candidates at most +2 points over the base) was missed by
+8.4: where the grammar can be enumerated, the test does the work and the mismatch only keeps the set of laws compact
(61 laws against 103). Retire a law when its held-out precision falls (`retire_below`), and cap the set. Where each test
is a call into a live system, plan the probes: on a real MCP filesystem server with a path policy, probes chosen to tell
the mismatch's candidates apart (read-only calls first) learned every denial class in one session — refusals 6 → 0 in
the next one (200 calls per session, one run), no allowed call skipped — where random probes at the same budget went 48
→ 29. **An LLM may propose, never judge:** under the same test an LLM proposer ended 0.95 points below enumeration, and
in the same study a model judged the truth of its own claims about the path policy wrongly 18 times out of 40. Both
findings in miniature, with their bars: [laws_toy.py](../benchmarks/laws_toy.py) (`pytest -m slow
tests/test_laws_toy.py`).

## Size the acceptance by the search; admit by probes; abstain when laws disagree

**Size the acceptance by how many candidates you tested.** A fixed bar ("support ≥ 5, precision ≥ 0.9") does not know
that a setup tried tens of thousands of candidates on the same buffer, and some random condition always passes it: on a
synthetic buffer with no law at all (2,000 rows, an effect at 70% independent of 40 sparse atoms, 821 conditions) the
fixed bar accepted a law in 100 of 100 buffers. `LawSet(sized=True)` — the default — tests each candidate at the level
δ / (N·(N + 1)), N the candidates scored on that buffer so far, so the chance of any false acceptance stays below δ:
0 of 100 on the same buffers. **Test against chance, not against 0.9.** The default null is the effect's base rate in
the buffer (`null="base"`: what a random condition achieves for that action and effect, floored at 0.05), with
`min_precision` kept as a separate filter: a deterministic law at a base rate of 3% was accepted in 100 of 100 buffers
on a median of 29 rows, where a fixed null of 0.9 (`null=0.9`) accepted it in none — beating 0.9 at tens of thousands of
candidates takes hundreds of rows even for a deterministic law. **The need:** a deterministic law needs
⌈ln(1/δ_N) / ln(1/null)⌉ rows, δ_N = δ / (N·(N + 1)): at 52,000 candidates and δ = 0.1, about 9 rows at a base rate of
3%, 68 at 70%, 228 against a fixed 0.9. **A buffer too small for its null accepts nothing — that is the rule working,
not failing.** Collect more held-out rows, test fewer candidates, declare `space=`, or pass `sized=False` knowingly and
treat the fixed bar's false accepts as the known price; the measurements above ran with `sized=False`.

**The sized test assumes independent held-out rows; count support by distinct object when rows revisit the same ones.**
The bound treats every held-out row as a fresh draw. Rows from a random walk revisit the same objects with the same
persistent attributes, so twenty rows may be three objects seen many times: the support looks large, the binomial tail
looks small, and a law right only on those few objects passes. On relational laws over generated graphs with random-walk streams, a law
that passed on 14 held-out rows (13 right) came from 3 (world, object) pairs and was right on 41 % of fresh rows. Draw
the held-out states independently (fresh starts, or one row per object), or count a law's support by distinct object
rather than by row before you read δ as a promise. `Buffer(dedup=True)` (experimental) keeps one row per situation and
group, or `Buffer(dedup=lambda t: t.state["object"])` one per object: on a synthetic random walk over 3 objects per
world with no law at all, a law was accepted in 872 of 1,000 buffers by default and in 0 with dedup. It is not free:
on a second task (2,000 tasks of relational laws, dedup by world and actor) wrong laws per admission fell 0.415 →
0.112, but the share of true laws found fell 0.369 → 0.330, because the buffer shrank about sixfold and fewer right
laws passed the sized test. Turn it on where a few objects are revisited many times and a wrong law costs more than a
missed one.

**When two candidate conditions can match the same held-out rows, turn on `LawSet(rivals=True)`.** By default the
test cannot tell `mover OPEN & cell STOP -> pass` from `mover OPEN & cell SHUT -> pass` while no key was pushed into a
wall, and the candidates' order decides. With rivals on, a passing candidate is refused while another offered candidate
(or an active law) with the same effect and a condition neither wider nor narrower matches exactly its held-out rows.
On a grid-game-shaped world with levels arriving one at a time and the separating row on 10% of levels, the wrong law
was the first accepted in 198 of 1,000 streams by default and in 0 with rivals (right 763 → 816); at 30%, false
accepts 17 → 15 of 1,000; noise with no exception: 0 accepts either way. The price is delay: in 65 of the 1,000 streams
nothing was accepted within 20 levels, because the right law waits for a row that tells it apart. Probe for that row
(`admit_by_probes`) when you can ask the world. Covered by [tests/test_laws_rivals.py](../tests/test_laws_rivals.py).

**Where the world answers questions, admit by discriminating probes, not by a random sample.** The author's earlier
research line measured that a hypothesis "certified" on 8 random held-out examples is wrong more often the larger the
space (0, 0.105, 0.193 at 17, 1,462, 124,287 hypotheses), and that 6 queries chosen to tell the survivors apart bring
the largest case to 0.030. `LawSet.admit_by_probes(candidates, ask, probes=, safe=, budget=)` does that for laws: it
asks the safe probe that splits the surviving classes most and accepts the shortest candidate only when one class is
left; otherwise it returns "undetermined" with the classes. On a synthetic world at 8 checks: a random held-out sample
with the shortest survivor accepted a wrong law in 0.593 of 300 trials, the probes in none (0.987 right, 0.013
undetermined with `assume_one_law=True`; under the 1.2 default, which keeps witnesses against a second missing law,
0.897 right at 8 checks and 1.000 at 10, no wrong admission in the trials). Ask only where the rest of your law set and base are right, or a right candidate is refuted with them.

**To veto a probe admission with your own check, ask for a dry run: `admit_by_probes(..., commit=False)`.** It returns
the same decision without adding the law (status "proposed"); add it with `add` once your check agrees. Admitting and
then calling `retire` records a law that never fell as retired, which mixes "never used" with "fell later" in
`retired`. A planner that caches answers should key them on `LawSet.version`, which moves on every admission,
retirement and suspension; and a law set or buffer that is pickled to resume a run drops its `id()`-keyed caches.

**When two encodings of one effect agree by construction, declare it: `LawSet(equivalent=[(e1, e2, guard_atom)])`.**
"One cell forward" and "one cell east" agree on every state where the agent faces east, so no probe can split them
and a learner that admits a rule only when one hypothesis is left spends its budget and gives up. Declared, the two
are one class before probing, and the merged twin stays a rival until the probes refute it on every encoding it
carries. On MiniGrid (one grid family, 3 layouts, 20 seeds, probes answered by the simulator, budget 8): position
rules 15 from 2,908 probes (2 wrong) → 24 from 84 (none wrong on all 3,116 local states), accuracy 0.93 → 0.99;
facing rules none in 22,845 probes → 34 from 193 (none wrong), accuracy 0.68 → 0.95. Declare only what is true: a
false declaration is caught only on logged rows (`check_equivalent(buffer)`) or when the rule later fails. Measured in
a deterministic world with exact answers.

**Use `FactoredModel` for accuracy, not to learn from fewer transitions.** One `LawSet` per channel with
`FactoredModel(action=...)` (every channel law names the action) was more accurate on MiniGrid than one joint
`LawSet` over the whole observation (next-state accuracy 0.999 vs 0.984, 10.8 vs 13.4 laws, 5 seeds), but needed
about as many environment transitions (0.94× and 1.05× the joint model's in two runs). Both models admitted false laws
on states the environments never reach (facing a key while already carrying one): a law is only as general as the
states your walks produce, so check it on the states that matter to you.

**When two laws may be missing at once, use `admit_by_probes(..., max_new=2)` and give it at least 12 probes.** It is
a known method — a version space (Mitchell 1982) over law sets with up to two new laws, split by discriminating
probes as in generalized binary search (Golovin et al. 2010). On two synthetic boolean two-law worlds (10 atoms, a
pool of 200 states, 1,000 trials each, budget 16) every accepted law was correct in 1,000 of 1,000 trials against
5–6.5 % wrong for the default, and both laws were found in about 42 %. Where only one law is missing it needs about
1.5× the probes (11.7 against 7.6 on a single-law benchmark; at a budget of 8 it is undetermined in 97.5 % of trials).
Three or more missing laws are not measured, and the `coverage` warning is off in this mode.

**When the laws of a stream change, give the law set a window: `Buffer(window=N)` with `LawSet(recent=M,
readmit="grown")`; size N to a fraction of how long a regime lasts.** On synthetic streams whose laws change abruptly
(two conjunctive laws over 12 atoms, one flipped and one replaced at each of five switches, phases of 1,500–2,500 rows),
`Buffer(window=150)` with `LawSet(recent=50, readmit="grown")` re-learned the new laws in 605 rows on average (30 of 500
switches censored), where a periodic reset tuned under the same rule took 1,080 (83 censored) and the library default
did not re-learn within any phase (an oracle with known switch points: 238); prequential accuracy 0.796 against 0.755
(reset) and 0.692 (default), oracle 0.813, with 0 wrong laws among 1,221 accepted (reset 3 of 1,258); no arm accepted a
law on streams without laws (a property of the sized rule, shared by all arms). The window length was tuned to this
phase length: another drift rate needs another window, and an adaptive window is not built. Synthetic abrupt drift only
— no gradual, incremental or recurring drift, no real stream. The first measurement of this feature was mis-scored (a
cache bug: 31 wrong laws that were in fact 1) and the bar was then relaxed to at most 3 wrong laws; this run meets it at
0, as it meets the original bar of 0. The score memo (`memo=True`) and the proposals count are built and unit-tested,
not measured. A sliding window with forgetting and a retirement on recent performance is a standard answer to concept
drift (FLORA, Widmer and Kubat 1996; ADWIN, Bifet and Gavaldà 2007; Gama et al. 2014); what is the library's is only how
the sized rule's N behaves under a window (it never decreases; a candidate counts again after a full turnover) and
`readmit="grown"`.

**When the world's answers can be wrong, give the probes the noise level: `admit_by_probes(..., noise=ε, level=L)`.**
With the default one wrong answer refutes the right law: at 10 % noise the default admission was wrong in about 41 % of
synthetic trials. With `noise=` the hypotheses keep likelihoods and the probes are chosen by expected information gain,
so a small answer budget is spent where it decides most. On a synthetic single-law benchmark at 10 % noise and 24
answers it accepted the right law in 92 % of trials (0.4 % wrong), where the best repetition scheme under a 5 %
wrong-accept cap — a majority vote per probe — finished in 40 % and ran out of answers in 58 %; at 48 answers a
per-probe sequential test was as right (99 % against 100 %) and the noisy admission used 23 % fewer answers; at 5 %
noise the two were equally right and it used 24–40 % fewer answers. Prefer it to repeating probes when answers are
scarce; when they are plentiful and the noise is low, a per-probe sequential test is as right and simpler. Give ε as
you know it (both the method and the repetition schemes it was compared with knew the true ε) and pick `level` for
your wrong-accept cap: 0.99–0.999 held a 5 % cap at 5–10 % noise, and at 20 % noise with 24 answers no level did —
give it more answers there (at 48 it was right in 91 % with 0.4 % wrong). A wrong ε, noise that depends on the state
and several missing laws are not measured; the mechanism is covered by
[tests/test_laws_noise.py](../tests/test_laws_noise.py), the numbers come from synthetic worlds only.

**The likelihood weight only ranks the candidates you offered; the fit check stops a wrong one being accepted when the
true law is missing.** `noise=` compares the offered hypotheses with each other, so when the true law is not among the
candidates the least-bad wrong one can hold all of the weight — in 1.2.0 a logged buffer alone gave it there, with no
probe asked. The admission now also checks that the chosen law fits ε where it applies (logged rows and asked probes):
more misses than Binomial(n, ε) allows at `fit_level` (default 0.01), or fewer than `fit_support` rows (default 1),
returns "undetermined" with a reason that says the law does not fit the noise. When you see that reason, offer more
candidates (a larger `k`, other atoms) rather than lowering `level`. Few rows give the check little power (two misses
in five at ε = 0.05 pass), so raise `fit_support` if a handful of answers can decide; `fit_level=None` turns it off.
Covered by [tests/test_laws_noise_fit.py](../tests/test_laws_noise_fit.py).

**When a law may stop holding for a while and come back, suspend it instead of retiring it: `LawSet(on_fall="suspend")`;
keep the default `reactivate_min=10` unless your observations are clean.** On synthetic streams a returning law came
back in about 40 rows against 263 for the best retire-and-readmit setting that kept false admissions at the cap (at most
0.25 of the baseline's rows against every setting tried; faster-forgetting windows came back in ~160 rows but admitted
4–24 false laws per 80 streams); m = 5 was faster but sat at the false-admission
cap of its tuning, m = 10 is the safer default. A law whose states the agent stops visiting is never re-checked.

**Add history atoms (`History(grammar, depth=1)`) only when a law may depend on the previous step.** Where one did, they
learned it and the present-state grammar could not; on worlds that need no history the larger grammar costs a slower
start (−0.017 closed learning-curve area; the same final level from about 1,250 rows). Depth 1 only is measured.

**Treat `LawConfidence` as an estimate and pick its threshold on your own data.** It answered 78 % of law-covered states
against 47 % for a per-law precision threshold at the same ≈ 2 % risk when an atom tracks the hidden cause, but it gives
no risk guarantee. Its `exact` flag needs rows: at the default `exact_rows=60` a law right on 97 % of its rows is flagged
exact about 1 time in 5; raise `exact_rows` where an exact law matters.

**Abstain when consistent laws disagree.** Two laws that both passed their test, neither an exception of the other,
predicting different things for one state are a reason to abstain, not a tie to break by length or age:
`predict(state, action, disagree="abstain")` returns `Undetermined` there and counts it (`lawset.abstained`). Skills
do not abstain: two certified skills of one goal are both verified plans, and a different sequence is not a
disagreement about the world. **Report the search cost next to accuracy**:
`lawset.cost` and each skill episode's `cost` (candidates, tests, probes, simulated steps). Measured on synthetic worlds
only: [tests/test_laws_sized_probes.py](../tests/test_laws_sized_probes.py).

## Relational rules: enumerate, let the test judge, then compile

**When a rule is about structure — an ancestor, a neighbour, a path — give the grammar relations, not more flat atoms.**
`solvi.experimental.relations` encodes a state as relations over its nodes (a path tree: PARENT, SAME_DIR, IS_DIR,
NAME=x; a grid seen from the player: EAST, SOUTH, AT, FACE and the cells' materials) and enumerates small
relation-algebra programs bottom-up, one per behaviour. Two programs are one only when their full relations agree on
every example — never when only the readout agrees, or what composes from them is lost. On a pre-registered battery of
28 random rules over three relations (the rule file was written in the author's earlier research line before any search
ran and is not shipped — `relations_battery.py --rules` takes it; the bars were fixed from that line's reference run, so
this is a port-fidelity check, one deterministic run) it found 22 within seven nodes, each exact on larger graphs of
other families, and a rule that needs four variables (a common 4-clique) was never accepted. `RelAlgProposer` hands the
best programs to a law set as atoms and the law set's held-out test judges them: for a policy that denies reads under
any directory named "secret" (files may be named so too), on a synthetic path tree with that policy
(`benchmarks/relations_compile.py`) the found relational law was right on all 20,000 fresh rows, which the flat
grammar's "the parent is named secret" cannot be. On an MCP filesystem server behind a written guard, laws with a
relational atom (ancestor directory named secret) refused none of the 35 calls under a secret directory in the second
session (the flat grammar refused all 35) and wrongly skipped none of 130 allowed calls; one unrelated refusal remained
(a listing of a file link, mispredicted by both grammars), so the pre-registered bar of zero refusals was missed. On the
game simulator +0.02 points (0.9882 vs 0.9880). One layout, one run each. **Equivalence holds on the examples only**:
the smallest program of a class may not generalize, so keep the held-out check on other groups, sizes or families.
Expect programs of four or five nodes with twenty-odd relations; with many or large examples search a subset and
re-check on all (`synthesize(..., check=)`), and with noisy labels rank by agreement (`top_programs`), not by an exact
match.

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
outcomes at 5% and δ 0.1 (25 at 10%, 12 at 20%; with the default bet 0.9 — the count depends on it), so a skill used
once per episode needs about fifty episodes before it acts alone; pool the certificate by goal or over the set (the
promise is then the pool's) or accept a looser promise on short streams. In the toy (20 fresh layouts × 5 seeds) the set
certified at 20% between the ninth and the thirteenth episode and System 1's share reached 0.81 in the last five with
every goal still reached; per skill nothing certified within 20 episodes; the 20% promise pooled over the set was chosen
after these runs — at 5% nothing certified within the toy's 20 episodes (as the changelog says). **Cases did not
transfer, rules did (the gridworld toy):** replaying the sequence that worked from the same start state gave System 1
nothing on fresh layouts; a skill's condition keeps only the atoms its verified starts shared, so it reaches states no
case has seen. On Crafter (written mechanics, symbolic state, hand-written survival reflexes shared by every arm), an
action model learned from the agent's own attempts and carried across 30 worlds (10 independent runs) raised the unique
achievements unlocked per episode from 6.0 to 10.6 of 22 (+77%; the oracle 14.4) and System 1's share to 84% after about
twenty worlds, part of it by living longer; with the written rules given to the agent as gates, the learned memory added
nothing (−4.5%). Script for the gridworld: [skills_toy.py](../benchmarks/skills_toy.py) (`pytest -m slow
tests/test_skills_toy.py`).

**Measured on Crafter.** On Crafter (version 1.8.3; written mechanics as the exact model; symbolic state; hand-written
survival reflexes shared by every arm, 60% of all steps; the planner reads the copy's full map while the game shows an
agent a 9×7 window), in one run of 10 streams × 30 episodes, no agent seed (the loop is deterministic given the world
seeds). Achievements are unique achievements unlocked per episode (of 22). The skill set was certified at 5% / δ 0.1
(fifty clean uses, one e-process pooled over the stream's skills) in every stream by episode 1–2. In the last ten
episodes System 1 acted alone 1,350 times with an observed outcome and failed twice (0.15%, Wilson upper bound 0.54%; 4
of 2,882 over the whole run); 57 further alone runs were paused by a reflex and not resumed, so they carry no outcome
(counted as failures they would make 4.2%, Wilson 5.4%). The promise is the set's, not a single skill's; a skill's own
alarm retires it (4 retirements, each later re-certified; measured with the alarm before 1.2, `alternatives=None`). Goals that are stochastic by the written rules stayed with
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

**Generalize a footprint only against the code of the world** (1.1.1, `lift="lattice"`). A skill whose footprint
is all constants («the cell to the left is stone») matches rarely in an open world; widening it is a claim about the
world, so let the world's code judge it: write the relations and lifts once (`Vocabulary`: «a resource within 2 steps in
direction D», «has a tool for R», a lower bound for a count), let the lattice propose the wider footprints most general
first, and keep one only when the plan, instantiated on states that match the wider footprint but not the original,
reaches its goal on the exact copy — never because it looks right or a model says so. Feed it both kinds of state:
recorded states of the stream (what the skill will meet) and the world's own perturbations (bindings the stream has not
shown yet); check every goal a variable can take (a use is asked for a goal, so a stone-gathering footprint without the
pickaxe atom must meet a state without the pickaxe). The verification is a sample, so keep the certificate on: it
catches what the sample missed. On the small grid of the tests (`tests/test_skills_footprint.py`: a tree next to the
player in a direction drawn per layout, 30 layouts, verify_n=8, the certificate at 20% pooled over the set) one turned skill «move
D, do» replaced the constant ones (3 skills against 5) and System 1 decided 30% of the steps of the last ten layouts
against 0 with the condition rule alone; in the same grid with «do» refused in one direction the turned candidate was
refuted on the copy every time and no alone run failed. Measured once in an open world (the simulator benchmark of 1.1, 10
streams × 30 episodes): System 1's share of non-reflex steps 0.343 / 0.486 / 0.385 by thirds against 0.071 / 0.175 / 0.233
with constant footprints, failures 0.51 % under the certificate, achievements 0.918 of planning always; the share fell in
the last third while the pooled certificate's alarms (the alarm before 1.2) retired skills — if shares fall late, check the certificate's
restart policy first (a hypothesis; not tested by a separate arm). Generalizing a verified plan's conditions is old
(explanation-based macro-operators, lifted action models learned from traces); what is this library's is the judge — the
plan re-run on an exact copy of the world for the states the wider footprint admits.

## Self-change: guard refusal first

**Refusal is the first casualty of self-change; the regression check cannot be switched off.** Each change looks fine on
what it was meant to improve, and the inputs the system must refuse or hand over are answered a little more each time.
In `solvi.experimental.changes` every change whose target can answer passes `RefusalRegression` — the share refused on a
held should-abstain set before and after — and there is no flag to remove it. Declare the set
(`Pipeline(refusals=...)`): outside or unknown inputs, the cases your policy hands to a person; without one the change
is held for a person — a vacuous pass would be a switch in disguise. Keep the default tolerance (0: one input answered
that was refused before refuses the change) unless your answerer is nondeterministic. The set follows the policy
version: a document version may rightly redefine what goes to a person, so declare the set of the version in force, as
`benchmarks/changes_credit.py` does: there a self-tuning threshold change that closed the review band answered 137 of
137 hand-overs and was refused, while 12 exact corrections passed.

## Rules over relations: enumerate with a library, rank only where a check is expensive

**Find a rule over relations as follows** (`solvi.experimental.relations.library.find_rules`):
1. Enumerate small programs bottom-up with observational equivalence, from the base relations plus the library.
2. Accept only what passes the held-out test (exact, or support / precision / recall / net on other worlds).
3. Put accepted rules into the library as new terminals, so composites become small programs.
4. Use a gradient-trained learner only when the enumeration budget runs out or the labels are noisy, and only as a
   proposer whose program is then checked like any other candidate (`proposer=`).
5. Rank candidates only when a check is expensive (`RankedProposer`): by the joint statistic of the whole condition
   first (`JointRanker`, no training), a learned ranker only after it.
6. With negated literals (`Grammar(negations="refine")`) and a candidate budget, give the negated candidates their
   own budget (`BudgetedProposer(inner, 2000, neg_budget=...)`; the default is a separate budget of the same size).
   Under one shared budget they used it up before the mismatch that proposes the true positive law was reached: on a
   second task found fell 0.64 → 0.55 and 0.25 → 0.11 on two families that need no negation; ordering positives first
   or a tie-break against negations recovered none of 21 lost tasks, the separate budget all 21 (a replay; the cost is
   about 1.5–1.9× the candidates per task, all counted in the sized bound's N).

On two toys ([relations_library_toy.py](../benchmarks/relations_library_toy.py), five seeds each) the library found
every level — a path tree's "denied if an ancestor directory is named secret and the depth is at least 3" (three levels)
within 2,556 enumerated programs, kinship on forests to depth 6 (second cousin once removed) within 10,817 (the mean
over seeds; 10,813–10,819 by seed) — where the same search without the library found neither top rule in 100,000, with 0
false accepts on larger fresh worlds. The kinship ladder misses the toy's bar of a tenth of that budget (1.08×): the
universal relation U, a terminal of the package's enumerator, enlarges every level: with `include_universal=False` (U
left out of the terminals of both arms) the ladder costs 7,889 and meets the bar, the path tree 2,174 (1,679 with the
earlier minimal enumerator; the ladder 6,125). Leave U out when no rule you look for needs "every pair". A library also
has a cost: every entry is a terminal, so a rule it does not help gets dearer (sibling took 6,545 programs in its own
round after grandparent and great-grandparent were admitted, 951 without the library). The intermediate concepts were
given as targets: the search found which one was within reach and the programs, not the concepts. On a synthetic
five-level path-tree policy (denied if an ancestor directory is named secret and the depth is at least three; the four
intermediate relations given as named, unordered targets) `find_rules` with the library found every level within
4,111–4,132 enumerated programs over five seeds; without the library the top level was not found within the cap of
1,000,000 programs, so the budget ratio is at most 0.0042 and its true value unknown. Per level the library is not free
(90 vs 71 and 2,951 vs 1,898 programs on two of the shared levels; cheaper only on one). Two controls — a rule no
program of the grammar can express, and one that needs a 56-node program — were never admitted (0 of 10 runs: five
seeds, with and without the library). On the kinship ladder (nine rungs up to second cousin once removed) the library
found all nine within 3,370–3,378 programs; without it eight of nine — the ninth needs a program of at least 17 nodes
where the no-library search exhausted size 15 at ≈ 500,000 programs. 0 false accepts on held-out forests of the same
generator at a larger size. **Hold out other families, not only other draws:** a rule the programs cannot express ("x
and y share a 4-clique") was never admitted on graphs of density 0.3 or 0.5, exact or by precision; on dense graphs
(0.7) almost every edge is in a 4-clique, and precision / recall acceptance with held-out graphs of the same density
admitted the edge relation itself — held-out graphs of density 0.4, or exact acceptance, rejected it
(`tests/test_relations_library.py`). **The caveat on step 4:** that one-step rules are learned by gradient is verified
only on hand-picked rules over one relation; on random rules plain enumeration already finds every one-step rule in
seconds, and rules of two steps or more did not hold when learned by gradient — no gradient learner ships with solvi.
**On step 5:** a check on the logged rows is a few bit operations, so ranking pays only where each check is a live probe
or a planner call. On synthetic tasks drawn from one grammar (30 atoms, k = 3) the trained bag-of-atoms ranker needed
2.1 checks to the true condition against 69.7 in the blind order (by length, then name; another draw of the same grammar
— over five seeds the blind order needed 85), and the raw-feature control 24.6 (the best raw threshold picked after the
fact: 12.8) — in distribution only; measure the control on your own grammar before trusting the trained one
(`tests/test_laws_rank.py`). Measured once on a live file server (8 mismatches; candidates contradicted by logged rows
skipped free): every order spent 29 probes in all, because 1–5 candidates per mismatch were left to probe and the true
conditions were single atoms — the shortest hypotheses, which the length-first order meets first. Ranking can pay only
where many consistent hypotheses remain after the free checks; the control first.

**Rank by the joint statistic of the whole condition.** `JointRanker` ranks candidate conditions by the joint statistic
of the whole condition over the logged rows — the Jaccard between the rows the whole condition matches and the rows
where its effect predicts right, equal Jaccard → fewer atoms — with no training. A per-atom scorer (the raw
`FeatureRanker`, the trained `BagOfAtomsRanker`) adds atom scores, so it cannot see two atoms that pick out the effect
only together, or a correlated twin of a true atom. On three synthetic settings drawn from the library's own grammars
(five seeds × 60 tasks each, 300 logged rows per task; [laws_rank_joint.py](../benchmarks/laws_rank_joint.py) runs one
seed): on the grammar above (about 330 candidates per task) 1.00 checks to the true condition on average (median 1)
against 85 in the blind order (a different draw of tasks: five seeds against the one seed above), 27 for the raw
per-atom Jaccard and 2.1 for the trained bag of atoms; on deep conditions (3–5 atoms of k = 5, about 6,000 candidates
per task, 5% label noise) 17 (median 1) against 2,182 / 1,310 / 591 (medians 1,480.5 / 823.5 / 86); on correlated
features (three feature pairs copied with probability 0.8, about 350 candidates) 1.01 (median 1) against 126 / 47 / 7.7.
The mean on the deep setting hides a tail: single tasks took up to 836 checks (17 of 300 above 50, five above 170), and
in 112 of 300 tasks a condition with a strictly higher Jaccard under the label noise came before the true one — the only
room a trained scorer over the whole condition could have, and it is not measured. A control scoring against the
effect's rows shuffled among the allowed rows (the count kept) was worse than the blind order (112 / 2,434 / 166): the
gain is in which rows the whole condition shares with the effect, not in how many. It is a statistic over the logged
rows: with no logged rows every score is equal and the order is the blind one — the statistic needs rows; on the deep
setting with the buffer cut to its first 5 / 20 / 100 / 300 rows it needed 1,997 / 1,569 (median 522) / 534 (median 9.5)
/ 17 checks against 2,182 blind (at 5 rows the trained bag of atoms was worse than the blind order, 2,224), so the gain
shows from about 20 rows and is full by 300 (`--curve`). Ranking does not change what is accepted: the held-out test
accepted the same law under every order in 894 of 900 tasks (by construction: the acceptance scores every candidate; six
exact ties), and on the deep noisy setting it accepted nothing in 180 of 300 tasks at 300 rows — acceptance needs more
rows than ranking does. The trained scorer it beats is the library's linear bag of atoms above, which sums per-atom
scores and so cannot represent two atoms that matter only together; in distribution only — the tasks come from the same
generators as the settings. Use it where the journal has rows; a trained scorer over the whole condition is a possible
later extra, only if a multi-seed measurement on deep conditions shows a gain over it.

**Before the journal has rows, rank by a prior; let the joint statistic take over as rows arrive.** `PriorRanker`
scores a condition by its own features — its number of atoms, its atoms' kinds, each atom's prior usefulness and the
co-occurrence of its features in true conditions — learned from synthetic tasks of your grammar (`fit(grammar, sample,
tasks=, rows=, sizes=)`; or `fit_tasks(grammar, tasks)` on mismatches whose law you know); it reads no rows.
`HandoverRanker(prior, JointRanker())` lets the evidence take over gradually: score = the joint Jaccard + 20 / (20 +
rows) × the prior's rank among the candidates in [0, 1] — the prior's order with no rows, the joint statistic's as rows
grow (`mode="switch"` is a fixed switch at `min_rows`; `last` / `RankedProposer.scored_by` say what scored). **A prior only knows what the task family favours:** on a planted grammar whose sampler
draws three features 30 times as often, the prior needed 10.3 checks at 0 rows against 43.1 blind (a prior fitted on shuffled labels: 206 — worse than blind; mean over 40 tasks, one run, `tests/test_laws_rank_prior.py`); on a skewed family of deep conditions (three
features drawn 8 times and one comparison type 4 times as often; 5,112 candidates per task; five seeds, 300 tasks, training and test tasks
from the same sampler) 208 against 1,245 blind; on the three settings above, where the conditions are drawn uniformly,
0.91 / 0.83 of the blind checks on the deep and the correlated setting, and 0.93 on the first, where it fell back to the
blind order in 2 of 5 seeds — the prior's large gain is where the family is skewed, and the skew here was chosen. The fallback is built in:
unless the prior's held-out checks on its own training tasks are 5% below both the blind order's and a shuffled-label
fit's, `learned` is False and it scores as the blind order. Fit the prior on tasks that look like yours (`sizes=`, the
features your laws use, or `fit_tasks` on past mismatches whose law you know), not on the grammar alone. **Blend, do not switch:** on the skewed family the prior still beat the joint statistic at 100 rows (208 against 361;
at 20 rows 208 against 1,019), so a fixed switch at 20 rows cost checks; the blend needed 184 / 98.6 / 8.9 at 20 / 100 /
300 rows — better than either part there. Its price is on the deep uniform setting at many rows: 27.2 against 17.2 for
the joint statistic at 300 rows (111 of 300 tasks worse, 31 better, medians 2 and 1; five seeds — a tail under label
noise, the mechanism not tested) — check the curve (`benchmarks/laws_rank_joint.py --curve`); only `mode="switch"` was
measured as the alternative there (the joint statistic's number at 300 rows), a smaller `n0` was not. The weight
n0 / (n0 + rows) has the form of a pseudo-count shrinkage, but the blend adds a Jaccard score to a rank and is not a
posterior.

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

## Rules with guards and a negative stratum: count on labelled pairs, confirm on a check slice, recount after each round

When a case is a graph and its labels are sets of relations on a few query pairs, learn guarded rules by counting (`relations.GuardedRules`) and keep three habits. Count on the labelled pairs, and keep an independent check slice (`check=`, `check_min=2`): a rule counted only where it was learned is too easily right by chance. Let the rounds recount every vote on the fixed point of the rules admitted so far, and keep the recount of each round's new rules on the joint fixed point (`verify=True`): rules admitted together can be wrong together, because labels exist only on the query pairs and a derived fact cannot be checked where it lands. Prefer a strict admission there: in a first run on the held-out part of the benchmark's training file (NoRA, Das et al., NeurIPS 2025, the no-ambiguity configuration; two splits by a hash of the row index, about 9,250 fit / 2,000 check / 1,970 held-out query pairs each; one run per split), exact set match / precision among answered / coverage: the strict setting (majority 1.0, support 10 on the labelled pairs plus two confirmations on the check slice) 0.328 / 0.681 / 0.482 with 469 rules and 0.485 / 0.743 / 0.652 with 436 rules; the default counting setting (majority 0.9, support 2) 0.274 / 0.568 / 0.482 with 720 rules and 0.398 / 0.609 / 0.653 with 838 rules; the most frequent label set per entity kinds 0.335 and 0.341; (benchmarks/relations_rules_nora.py, --salt nora and nora-b). On the test files the strict setting gave 0.734 / 0.734 / 0.999 on the off-path split and 0.477 / 0.961 / 0.496 on the depth split, where a label-frequency control answers 0.491; the default setting 0.593 on the off-path split and 0.007 on the depth split; one run, the setting chosen on the training file's held-out part (numbers and limits on the API page). Limits: labels exist only on the query pairs, so a derived fact cannot be checked where it lands, and counting at the looser setting admits rules right on the labelled pairs and wrong elsewhere (the joint recount after each round keeps the set from collapsing); the stricter setting does better here, but which chain rules it keeps depends on the split (0.328 against 0.485: whether living_in ← living_in_same_place ∘ living_in survives the strict share on derived facts); the «elsewhere» relation (Not_living_in, about a third of the rows) is refused by every setting without a declaration — a closed-world default per entity type was not built in 1.1.0 (since 1.1.1: `functional=`); residual votes only bring forward what the next round's fixed point shows anyway (identical rule sets on the planted world); two binary atoms per body, unary predicates only as given (none learned or invented), and a store path that writes every justification (slow on dense stories). On a toy with every pair labelled the default setting is exact: on a planted world with an independent oracle (families, genders, homes; every ordered pair labelled) exact set match 1.000 on 4,992 test pairs, all 8 planted rules found, the store path equal to the in-memory fixed point on 1,568 pairs (benchmarks/relations_rules_toy.py).

**Declare functional relations; the closed world is a declaration, not a finding.** When a relation has at most one object per subject of a type (one home per person), say so (`GuardedRules(..., functional={"living_in": "person"})`, with its complement declared): «lives elsewhere» is then derived for every other place of a subject whose home is known, and nothing is guessed for a subject whose home is not. Nothing learns which relations are functional — a wrong declaration makes false negatives — so keep it a stated fact of the domain, kept in the store as a spec item you can retract. The stratum is only as good as the known positives: in a first run on the held-out part of the benchmark's training file (the two hash splits above, one run each) it answered 373 of 588 and 17 of 590 «elsewhere» rows exactly, depending on whether the split's strict rules derive the home (the stories never state it for a queried person), with 4–5 more wrong answers on about 1,380 other rows; check how many subjects have a known positive (`why_not` names the missing one) before expecting the relation answered. On the benchmark's three test files with the declaration (one run, the bars pre-registered): D-na 0.477 → 0.914 at precision 0.979 — 1,463 closed answers, all right, from derived homes; BL-na and OPEC-na unchanged; 0 new wrong answers of 70,943; the unanswered «elsewhere» rows are subjects with no known home or with two derived ones, and keep `closed_world_in_rounds` off (110 right answers lost with it) — one relation, one benchmark, one run.

## Read the graph with code where the phrasing is regular; refuse the rest

When the rules work on a relation graph and the input is text whose relation phrases are regular (the benchmark family
of kinship stories: «[X]'s son [Y]», «[Y] is the son of [X]», «[X] took her son [Y]»), read the graph with code
(`relations.TextFrontend`) rather than a model, and let a sentence no pattern reads give no edge: a missed edge costs a
refusal downstream (no path, or the table refuses), and an edge read wrongly costs a wrong answer — on the benchmark's test split 9 of the 13 wrong answers rest on a reversed label and 4 on a pair the labels do not mark, none on a path of labelled edges alone. Keep the global check (`GraphCheck`, the default)
as a safety net that names what it drops — an edge whose relation contradicts a person's gender, two relations on one
ordered pair, or a second father / mother / husband / wife; every edge involved is dropped, the check never picks a
winner — and do not expect it to raise precision: on the test split it dropped 70 stated edges (50 of them false by the
labels) and changed no wrong answer, while 4 right answers went with the dropped edges (precision among the answered
0.960 with it, 0.961 without). Give the declared genders when you have them (`genders=`): the owner of «his / her» is
the nearest name of the pronoun's gender, and the inverse edge is chosen by the owner's gender; without them the genders
are inferred from the relation words and an edge whose owner's gender stays unknown gets no inverse. Hand the graph to
the rules as it is (`TextGraph.paths(x, y)` → `CompositionTable.derive_paths`; `TextGraph.story()` → `GuardedRules`, or
its facts → `pair_fact` → `forward_chain`). Measured once on CLUTRR (Sinha et al., 2019), the clean split's test file
(1,146 queries, chains of 2–10 relations; the composition table fitted on the whole training file — the same table as
on the labelled graph): the graph read from the text gave precision among the answered 0.960 (314 of 327; Wilson 95 %
0.933–0.977; 0.984 / 0.951 / 0.964 at chain lengths 2–3 / 4–7 / 8–10) at coverage 0.285 (0.526 at length 2, 0.185 at
length 10), against 1.000 at 0.949 from the benchmark's own relation graph; edges precision 0.957 / recall 0.744 against
labels that mark only the chain's edges (of 229 false stated edges, 173 are reversed in the labels, 47 lie on unlabelled pairs and 9 are other); of the 13 wrong answers 9 rest on a reversed label and 4 on an unlabelled pair, none on labelled edges alone; 752 of the
819 refusals had no path in the read graph; 0.09 ms a story on CPU; the bars were pre-registered before the run — the
precision and coverage bars met, the check's bar (precision higher with the check than without on chains of 4 and more)
not met. Two limits of this number: the five patterns were written, in the author's earlier research line, while
reading this benchmark's test stories, so it measures the shipped frontend on this split and not generalization to
unseen phrasings (the port is equal to that line's reference edge for edge on 1,816 training stories, and nothing was
tuned here); and coverage is bounded by the patterns' recall — in that line a model on the pairs no pattern covers
raised edge recall to about 0.87–0.89 at a lower edge precision, and it is not shipped (torch, and trained on data under
a non-commercial license).

## Rules first, a model second

Where rules derive most answers exactly and refuse the rest, let the rules answer first and ask a model only what they refuse (`relations.RulesFirst(table, fallback=...)`): the rules' share keeps its precision and costs no call, the refused share goes to the fallback, which answers or refuses, and every answer says which path gave it (`who`). The rules answer every query they derive and the model is not called for it; only a refusal (disagreeing bracketings or an underivable chain) goes to the fallback, with the question rendered in words and the list of options; every answer records which path gave it and what it used (the cells, or the model's output); a replay derives the rules' answers again and re-reads the recorded model output through the same option check without calling the model. Measure the fallback on the refused queries and on a sample of the ones the rules answered: the two shares can differ: here the refused ones held a chain whose gold is itself split, and long chains where the model returned nothing. Pick the fallback by that measurement, not by its size — solvi-base, a decider over options, did not compose relations here. The fallback must be able to do the task and to return an answer in the required form: a decider over options that cannot compose relations gives coverage without accuracy, and a reasoning model whose output budget runs out on long chains returns no answer at all; the dispatcher records both as what they are (a model path with a wrong answer; a refusal) and repairs neither. Measure the fallback on the refused queries and on a sample of the answered ones before trusting it, and check the finish reason of every reply. Measured once on the frozen test split of CLUTRR (Sinha et al., 2019), the clean split (the relation graph of each story, not its text; 1,146 queries, chains of 2–10 relations; the table fitted on the whole training file, the same 111 cells; the rules' answers byte-identical to the composition-table measurement above): the rules answered 1,087 with 0 wrong and refused 59 (21 disagreeing bracketings, 38 underivable). With a small hosted LLM (z-ai/glm-5.3-flash) as the fallback (temperature 0, reasoning effort low, 2,000 output tokens; the first of two drafts answers; the refused queries rendered in words with the 18 relation names as options), 50 of the 59 refused queries got a readable option and 39 were right; 8 replies were empty because the model's reasoning used the whole output budget (finish reason "length"), and 1 named a relation outside the option list: overall 0.983 at coverage 0.992 (1,126 right of 1,137 answered; 11 wrong, 10 of them one chain of three relations whose gold in this split is itself split between mother and mother-in-law), against 0.949 at coverage 0.949 and 0 wrong for the rules alone. On an equal-size sample of queries the rules answered, the same LLM gave a readable option to 42 of 59 and was right on 41; on chains of 4 or more relations its precision among readable replies was the same on the refused and on the answered queries (34 of 35 each), the loss being empty replies (9 of 44 and 17 of 52). solvi-base as the fallback answered all 59 and was right on 2 (11 of 59 on the sample): it does not compose relations. Every answer replayed from its record without the model (0 failures). One run; $0.017 after the proxy's cache; the pre-registered bar of coverage 1.0 was missed. A first run before the test run, on the training file only (1,816 held-out stories, a fifth of it by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths): the table answered 1,675 (4 wrong) and refused 141 (11 disagreeing bracketings, 130 underivable); the same LLM got 137 of the 141 right and solvi-base 11 of 141; one run, secondary to the test numbers.

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
caught only by a head or a signature you published (`verify(anchor=...)`, `verify(signature=...)`). A frequent
check of a large store can run `verify(deep=False)` (every chain check, no re-check of the stored responses; 1.15–1.41×
faster, measured once) with a full `verify()` now and then.

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

**Index a large store before repeated audits, and diff only what moved.** For repeated `quarantine` / `where_is` on a
large store, build `store.provenance_index()` once: it reads only the records that read or hold the fact. For a catalog
change that touches a few parts, `diff(..., incremental=True)` re-runs only the decisions those parts sit under; use the
full diff when parts are non-deterministic, call external sources, or only `Catalog(combine=)` changed.

## Checks, drift and knowledge over time (from a second external review)

**A check from the same model call is not a check.** In the review's measurement a check made by the same call
agreed with the answer in 260 of 260 cases, every error included; three checks by one model were no better than one
answer. Declare `Catalog.check(shares_call_with=...)` and `family=` so `solvi check` flags it, and put a check of
another family, or executed code, next to a model. Copies of one model as a "jury" let 18–38 % errors through among
accepted answers at a promised ≤ 2 %. A weak check of another family is still worth having when it is weighted rather
than given a veto: weighting raised its AUROC from 0.667 to 0.793, while "any one check against refuses" dropped
precision from 1.0 to 0.14.

**A re-derivation is a slow path, not a boolean check.** Wire it as the `Dispatcher`'s `slow=` path so it can take its answer;
on competition maths (AIME) "always answer slowly" beat every answer-or-abstain policy at every cost of error tried.

**A calibration does not survive a new error regime.** A calibrator fitted on other sets lost 0.28 of utility at an
error cost of 30 on a new year of AIME; an agent gate calibrated on reference traces had AUROC 0.69 on new ones.
Recalibrate on a few of your own labelled cases; watch the agreement rate when labels are absent.

**Drift on numbers:** `NumericDrift` with a reference of at least ~50 values from calibration data, not the stream's
start; under 20 values the false-alarm bound does not hold (it warns). Whiten autocorrelated residuals first.

**Facts with `valid=`:** ask with `snapshot(valid_at=<the decision's world time>)`; without it every interval's
value is a fact at once, and `Knowledge.value(snapshot, s, r)` reads whichever comes first — in a rule, read
`Knowledge.value(snapshot, s, r, valid_at=<the same time>)`. After retracting a wrong item and telling the correction, `redecide(..., upto="now")` gives
the corrected answers; the default answers «what would the decision have been without the bad item at the time».

**Mutable registries read by rules are state, not code:** pass them as inputs, or freeze them before the first
`ask`; the fingerprint keeps the content first seen in the process.

**Keep regression cases from your records:** `testing.cases_from_store(storage, n=...)` plus
`testing.boundary(cases, system=...)` before changing a catalog; hand-written cases rarely sit on a threshold's edge.

**Many cases at once (mutation runs):** `solvi test --fast` gives the same answers and statuses about 3.7× faster on a
small catalog; keep the full run, with its replay check, for the committed suite.

**Isolate cases that can crash or hog memory:** `solvi test --isolate --max-mem 2G` costs about 90 ms per case on a
small task (process start, imports, System build); leave room above the imports — numpy with a threaded BLAS reserves
about 1 GB of address space at import.

**Laws on few rows:** on 3 groups × 6 rows a deterministic one-atom law was learned 0/20 with the sized default and
20/20 with `acceptance="exact", min_support=2`; the exact rule has no false-accept bound over many candidates, so
pair it with `confirm=` (an independent check). Measured values need `tol=`: with exact equality, 1e-6 noise refuted
every candidate of y = c·x. Values over several orders of magnitude need `rtol=` (relative), and give it to the
`LawSet` itself (`LawSet(..., rtol=)`) so that `step` / `test` / `observe` compare within it too: y = 2.5·x over x from
1 to 1e5 with relative noise 1e-7 learned no law in 60 steps under exact equality and learned it with rtol=1e-5, with no
mismatch after; in `admit_by_probes`, tol=1e-4 refuted every candidate where the noise reached 2e-3…0.2 and rtol=1e-4
admitted the right one.

**A copied environment is an oracle only if it is deterministic.** Before using a clone or a replay as the judge
(`admit_by_probes(ask=...)` on a `deepcopy` of the environment, a simulator that plans on a copy, `boundary(system=...)`
pinning answers), run the same actions twice from the same state and compare. An environment whose step iterates a
`set` (or a dict keyed by objects hashed by address) is not: in Crafter 1.8 objects are despawned in the order of a set
of objects, so the same seed gives different worlds from run to run and a `deepcopy` diverges from the original —
probes asked on it are noisy observations, not exact answers, and a refutation from one of them can be wrong. Fix the
iteration order in the environment, or treat the copy as noisy (a tolerance, repeated probes, `confirm=`). `boundary`
already leaves out inputs whose answers differ when asked twice.

**Paid checks pay only when a wrong accept is dear, the check is cheap and independent, and wrong answers occur.**
Where a check costs more than the error, or every check is the same model, deciding without checks did as well or
better in the review's test setups. Where the error was cheap, an agent without checks beat the full layer by +0.032 [+0.024, +0.041];
"don't know" paid only when a miss cost more than 1.5–1.7× an escalation; a cost of error given as a point estimate
inside the noise failed, and a cost prior without exploration never tried anything.

**A hard veto without a cost of error costs more than it saves.** A game advisor's 428 vetoes cost 10.25 points
[−16, −4.5]; a gate that can only stop lowered a cautious agent's utility by 0.031 (a prompt penalty took refusals
from 0 to 8 %). Give the gate a "check" or "ask" action.

**A budget holds the price, not the risk:** the average cost was held at 12.9–14.4 of 15 with utility unchanged.

**Calibrate novelty on real out-of-scope examples, not perturbations.** Imitated novelty gave 28 % coverage, mixing
it with real cases 30 %; about 100 real cases were needed for 60 %. `OpenSetGate.calibrate(novel_scores=)` takes them,
and `build(novel=[state, ...])` feeds System 1's signal on them to the gate in place of the simulation (it warns below
100 and when labelled examples are among them).

**For detection, restart the test.** An e-process started once adapts slowly after a long quiet prefix (detected in
0 of 200 seeds, a delay of 185 of 187 steps); `DriftMonitor`'s CUSUM restarts — use it, or a Shiryaev–Roberts
detector with an ARL bound.

**Wrap a strong agent only at the decision points where it is unsure; without evidence the agent's action stands.**
The full layer over a tuned classic policy passed 0 of 22 primary checks; the parts chosen on a small preliminary run, 7 of 22.

**Taint on the whole context costs honest tasks.** On AgentDojo utility fell from 0.97 to 0.70 (−0.27 [−0.43,
−0.14]), 62 of 92 interventions coming from context-wide taint, at an attack rate the agent itself held at 0 of 190.
Scope injections per tool (`injections="off"` with policies over the values) where the outputs are trusted.

**The change pipeline pays where a wrong change is dear; where labels are cheap, relabel.** A pipeline on an
agent's own strategy cost −0.011 [−0.015, −0.007]; fresh labels were 12.7 % cheaper over 40 epochs; with no owner of
the shipped version there were 0 rollbacks in 40 epochs.

**The knowledge store is for retraction and provenance, not retrieval.** A strong RAG scored 1.000 on the same facts;
a bitemporal journal writes about 1,000× dearer. A signature proves the sender, not the truth (112 forged readings per
run passed with a stolen key); disagreement with outcomes should down-weight a source, and only evidence of tampering
make it a forger (a rule on outcomes accused an honest lab in 52 % of runs).

**Votes of one model family are not independent.** Multiplying the probabilities of a base and a large model of one
family as if they were separate witnesses is badly calibrated: on Banking77 the naive product had ECE 0.118. A single
temperature on the product, tuned on a dev set, brought it to 0.026 — better than a combiner that models the
correlation (0.053). Tune one temperature on the combined score, or add a check of another family.

**Order questions to a person by how many items each answer unblocks, not first-in-first-out.** On a queue of
escalations with dependencies, ranking by the decisions an answer unblocks resolved about 4× more items per answer than
FIFO; a simple count of each question's downstream items got most of that (the full ranking was only 1.18× better),
and re-ranking after every answer added nothing over ranking once.

**`Certificate`'s default alarm (1.2) bets on breaks of a stated size; keep it unless the true rate sits well below the
promise.** The default alarm (two CUSUMs against 1.5× and 2× the promise, level 50) is quiet well inside the promise and
catches a doubling within about 150 outcomes at a promise of 0.05 (about 30 at 0.2). It is not a δ-guarantee: a skill
living at its promise will be retired within a few thousand uses and must re-certify; if your promise is loose relative
to the true rate, that is intended. A per-promise k-of-W window detects a doubling faster at small promises (by about 50
outcomes at 0.05) but alarms far more near the promise; pick it only when the true rate is known to sit well below the
promise. The old alarm (`alternatives=None`, level 1/δ) is a burst detector whose level must grow as the promise
shrinks: it fired on 67 % of 300 stable streams at a failure rate of 5 % under a promise of 20 %, and the smallest level
that kept stable streams quiet was about 300 for a promise of 0.2–0.3, 1000 for 0.1 and 3000 for 0.05, where it then
missed real breaks. The level 50 (= 5/δ at δ 0.1; other δ untested) and the alternatives were chosen on simulated
Bernoulli streams; the default is one setting for all promises and was 1–2 outcomes slower than a per-promise tuned
level of the old alarm in one setting (promise 0.2, break ×3), a loss accepted after a first run showed it and
re-tested on fresh seeds.

**Laws cover little where outcomes are stochastic** (about 2 % coverage in Crafter; in Freeciv 0.576 / 0.860 against
0.963 for a lookup table), and short matches (30 steps or fewer) give no laws at all. The gain of "measure until
resolved" comes in the first ~20 measurements. `learn_order` saved 7 % of the cost at 2.9× the time on cheap checks —
use it only where parts are expensive. A `RuleList` as the compiler's System 1 took 0–12 % of the traffic, against
48–81 % for a distilled logistic head.

**A slow or local LLM backend: raise `timeout`, lower `retries`.** A server that does not answer after its retries
makes the question abstain and is counted as `no_answer` in `System.stats`, apart from the model's own doubt; watch
that share, not only escalations.

**Check an exact model's clone hook once** with `testing.conformance.check_environment(make, clone=)`: a clone that
shares state with the original, or restores the wrong state, makes plans that do not hold. Nondeterminism found by
accident is what `check_environment` is for.

**Labels from corrections are not a random sample** — people correct what looks wrong. On a toy dispatcher the stored
corrections showed 7.2 % error against 1.8 % on the truth. `Dispatcher(audit=0.05–0.25)` and labelling the marked
decisions gives an unbiased reading (within 5 points on 1,200 decisions at audit=0.25). A random audit also catches a
check that broke silently: in a drift simulation the current monitor missed a broken check in 98 of 100 runs, while a 10 %
random audit with `AuditWatch` found it in 96 of 100 (4 of 100 false flags on unchanged streams; the audit's merit is
that it held its false-alarm level — see experimental.md).

**Refit on the hash split when examples keep arriving.** `build(split="hash")` (the default since 1.2; 1.1 shuffled)
keeps every old example in its share when examples are added (600 + 60 examples, 5 seeds: 0 old examples moved in or out of the fit share, against 101–120 with the
shuffle). Routes still move with the thresholds; `solvi diff` shows them. Moving from the shuffle to the hash puts
about 41% of the examples in another share (2,000 refund requests in `examples/24_one_entry_point.py`, 5 seeds, measured once: 40.1–42.1%; 84–89% of the shuffle's
calibration examples), so a system rebuilt on 1.2 is calibrated on other examples: pass `split="shuffle"` to rebuild
1.1's exactly.

**Check a change pipeline's gates once with `testing.conformance.check_pipeline`** on your own change schema. An
aggregate gate passes harm in a small slice and salami (eight steps of +0.012 on a 40-case slice each pass against the
version in force); add `SliceGuard` with the original pinned (`pin_file=` when proposals run in separate processes, with its hash
`pinned=` next to the evaluator's pins), and give `KnownFailures(key=...)` what the change *does*,
not a seed or salt it carries, so a replay under new identities is refused after its first rejection.

**Run the expensive stages last.** Put `IsolatedTests`, `FreshEnvironments`, `SliceGuard` and `Canaries` after the
cheap stages, or use `Pipeline(order="cheap_first")`, so expensive evidence is not paid for a change a cheap stage
refuses.

**Watch a shipped change per slice, against its promise.** `pipe.watch(version, promise)` reverts only that version
when one slice's evidence is enough; on synthetic rollouts it reverted 60 of 60 harmful ones and 0 of 60 clean ones,
where a mean-with-tolerance rule checked after every outcome reverted 22 of 60 clean ones.

**A Shadow halt needs evidence:** about 10 discordant pairs when the candidate is always the wrong one, hundreds when it
is wrong on a tenth of them. Label the shadowed answers so that the outcome does not depend on which answer was served.

**Turn on `cache=` only for catalogs whose parts are pure** (no clock, randomness or I/O). A long input makes hashing
most of a hit (0.43–0.74× the time of an uncached ask on 3.7k floats, against at most 0.17× on small inputs). Run
`replay_all` on the store now and then: an impure part shows there as a recompute mismatch on the `cache_of` records.

## A compact decider: a ~70M encoder with distillation

**For a pointer-head decider (encoder + heads, code executes the program), start from Ettin-68m and distil from the larger
model.** On TAT-QA arithmetic (test, 3 seeds) Ettin-68m fine-tuned with soft targets from a 149M ModernBERT-base teacher gave
72.1 % against 72.4 % for the 149M model (within 0.3 points, not significant), with 70.25M / 44.47M parameters (total / without
the embedding matrix) against 153.75M / 115.07M and 0.45–0.47× its CPU latency (ONNX Runtime, 4 threads, batch 1, two quiet
sessions on one Intel Xeon type; 0.50× at 1,024 tokens). On a second task (typed decisions over JSON states, 2,000 test
questions, 3 seeds, no distillation) the same 68m encoder gave 75.7 % against 74.6 % for the 150m one.

- **Distillation pays most for a mid-size student.** +2.1 points over the same encoder without it at 32M (33.07M total), +0.6
  (not significant) at 68M, +1.1 (not significant) at 17M — the known capacity-gap pattern, not a monotone trend.
- **Below ~40M the parity is not reached.** Ettin-32m with distillation gave 70.1 % on TAT-QA arithmetic (0.17× latency) and
  71.8 % against 74.6 % on typed decisions — about 0.96 of the larger model, short of a statistically clear 0.95.
- **Dynamic int8 on CPU keeps accuracy but did not speed this up** (0.91–0.93× latency, 0.87× memory; loss 0.05–0.14 points).
- **A table-structure prior** (row and column embeddings, permutation-equivariant row choice) gave no gain over an
  equal-parameter dense head on either size.
- Arithmetic on this pipeline saturates around 72–73 % from 68M to 395M encoders, so part of the parity is the pipeline's
  ceiling; one task family per number, no transfer claim.
