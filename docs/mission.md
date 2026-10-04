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
code checks and opened by gates. Every item records where it came from (a person, an outcome, a written
specification, a verified answer — never the system's own guess), can be checked, and can be retracted exactly: what was
built on a retracted item goes with it, and the decisions that rested on it are listed. By default this knowledge
protects: an action it predicts will fail is not taken, and a gate is a hard check. Taking a justified risk against a
prediction, within a budget, is an option for when protection costs too much. In an environment it meets again, a
system with this knowledge was shown to need far fewer slow decisions than the first time.

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

- Growth with accumulated knowledge was shown only in environments the system meets again: a crafting game and the
  Pokémon world map. It was not shown on streams of one kind of decision (classification, matching) or for a support
  agent with tools. There solvi keeps its error promise and the knowledge gives accountability — sources, retraction,
  disputes for a person — but it does not promise that the system gets better over time by itself.
- Justified risk lowers the cost of protection; it does not promise to do as well as a system without the knowledge.
- The error promise does not hold in the window between an abrupt shift in the inputs and the moment a drift check
  notices it. After a flag, stop answering alone until the thresholds are calibrated again
  ([the guide explains this](guide.md#thresholds-with-a-guarantee-act_guard-learn-then-test-conformal-sets)).

## Honesty

- Only what measurably helps enters the library. A feature whose gain is not clear in general stays out, or stays
  marked experimental.
- Negative results are published alongside positive ones.
- A promise is a number you can check: each one names its script or test, and you can run it yourself.

## Where it stands

In 1.0 the library has two levels. Ready systems you configure: `solvi.build` for decisions from labelled examples
with a promise, `solvi.Agent` for acting in an environment, `solvi.Guard` for an agent's tool calls, and
`solvi.Knowledge` for what they know. Under them, the building blocks in `solvi.core` — the decision runtime, the
error promises, the traces and replay, the checks, the knowledge store — each replaceable by your own part. Knowledge
as protection, with sources and exact retraction, ships in 1.0; so do the agenda with done checks and gates, and
justified risk as an option ([Using solvi: agents and knowledge](agent.md)). What works but has not yet shown a
measured gain is kept apart in `solvi.experimental`, each piece with what it is missing and a deadline; see the
[roadmap](../ROADMAP.md) for what comes next.
