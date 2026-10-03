# Mission

solvi builds decision systems you can check.

A decision system here is made of two kinds of parts. **Fast solvers** — rules, plain code, small models — answer
where they are sure. **LLMs and search** deliberate where the fast ones are not sure. A **person** decides what
neither can. Every decision is recorded, can be explained, and can be replayed later to confirm it.

## What solvi gives you

**Accountability.** For each task you can see how it was solved: what the system read, which part answered, what it
promised about that answer, and whether the promise held. This works the same for a rule, a small model or an LLM.

**Safety.** Any model can be wrapped in checks that keep it from doing harm: hard checks that a model's confidence cannot
override, guards on an agent's tool calls, and an error promise that sends a case to a person instead of guessing.

**Accumulated knowledge.** A system keeps verified knowledge about the environment it works in: rules, the conditions
under which an action works, a map of the world it moves in, and its goals — an agenda whose items are marked done by
code checks and opened by gates. Where a written specification exists, it is compiled into rules. Where none exists,
rules are learned from experience, and each one records where it came from. Every item can be checked and can be
retracted, and what was built on a retracted item goes with it.

**Any model.** solvi works with any model: an LLM behind any OpenAI-compatible server, your own classifier, or plain
code. A small local model is available, so much of a system runs without the cloud. There is no quality promise for that
model; measure it on your task first.

## Where to use it

- **Single decisions**: routing tickets, reading documents and contracts, policy decisions (approve, refuse, escalate).
- **Sequential decisions in an environment**: agents that call tools, planning, games — where what the system learned
  about the world on one step is used on the next.

## What it is not

- Not a text generator. solvi decides and checks; writing prose is the job of the model you plug in.
- Not a hosted service. It is a Python library; you run it where your data is.
- Not a replacement for LLMs. It uses them where they are needed and checks what they return.
- Where a guarantee cannot be given, solvi does not pretend: it hands the case to a person.

## Honest limits

- On a stream of one kind of decision (classification, for example), solvi keeps its error promise, but it does not
  promise that the system gets better over time by itself.
- The error promise does not hold in the window between an abrupt shift in the inputs and the moment a drift check
  notices it. After a flag, stop answering alone until the thresholds are calibrated again
  ([the guide explains this](guide.md#thresholds-with-a-guarantee-act_guard-learn-then-test-conformal-sets)).

## Honesty

- Only what measurably helps enters the library. A feature whose gain is not clear in general stays out, or stays
  marked experimental.
- Negative results are published alongside positive ones.
- A promise is a number you can check: each one names its script or test, and you can run it yourself.

## Where it stands

The decision runtime, the error promises, the traces and replay, the guards and the checks are in the library today
(see the [guide](guide.md)). The world map and an agent's episodes are there too. A knowledge journal with sources and
retraction, and an agenda with done checks and gates, are the next step toward 1.0 — see the [roadmap](../ROADMAP.md).
