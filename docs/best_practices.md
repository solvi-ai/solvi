# Best practices

What we learned while building on solvi, as advice. Each item says what to do and why. Every number on this page names
its source: a script in [benchmarks/](../benchmarks/) that re-runs it, or a published model card. A finding without such
a source is kept as advice, in plain words.

## The input a model reads

**Keep the keys of a state in one fixed order, and do not reorder them later.** A decision model reads a dict as lines
in the dict's order, and the order changes answers. This is not one model's quirk: every model we tried — the solvi
checkpoints, a hosted decision model and an LLM asked through `solvi.llm` — changed some answers when only the order of
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
product-matching pairs, enforced with `solvi.sets.decide_set` on the answers as given, raised F1 for every solver we
tried, and helped a weak solver most: on Abt-Buy (1,916 pairs) 0.872 → 0.909 for an LLM asked each pair, 0.860 → 0.870
for the same LLM through `solvi.llm`, 0.931 → 0.933 for a fitted head ([abtbuy/solution.py](../benchmarks/tasks/abtbuy/solution.py)
`--repair`, [abtbuy/llm_pair.py](../benchmarks/tasks/abtbuy/llm_pair.py)). Giving each request the other candidates as facts instead (how the pair ranks
among them) did worse. Prefer the exact method: it was at least as good as the greedy one, and fast on large
components.

**When the alternatives can be enumerated, search them instead of asking a model to propose.** On trip and meeting
planning problems a `solvi.search` through the same checks found the right plan more often than an LLM's plans did: on
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
yes/no hallucination judge asked through `solvi.llm` scored F1 0.744 on RAGTruth dev with the schema enforced and 0.791
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

**A script is a strong baseline — measure against it.** On simulated support tickets a model with memory solved as many
as a short script did; on incidents a runbook resolved far more than the model. What solvi adds to a model-driven agent
is that every step is checked, recorded and replayable, not that the model beats the rules.

## Tools (the agent guard)

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

## Storage and audit

**One store, any number of writers — but check which backend.** SQLite and PostgreSQL take several processes;
`JSONLStorage` does on POSIX (a file lock per append), and on Windows one writing process.

**Verify against something kept elsewhere.** `verify()` catches edits; a rewrite of the whole store with its head is
caught only by a head or a signature you published (`verify(anchor=...)`, `verify(signature=...)`).

**Read the kind of a replay mismatch before raising an alarm.** `integrity` means the data changed after the run;
`missing_part`, `recompute` and `model_changed` mean the catalog or the model moved on. `summary` says which.

**Erase with `redact`, never by editing.** It removes a record's content, keeps the chain, and records the erasure;
editing or deleting a record is indistinguishable from tampering.

**After a change of the catalog, replay before you trust old traces.** A renamed part, a new input or a retrained
model each make old decisions non-reproducible; `solvi diff` shows which stored decisions would change and why.
