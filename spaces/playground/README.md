---
title: solvi playground
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Decision tasks in Python, running in your browser
---

# solvi playground (runs in your browser)

Interactive demo of [solvi](https://github.com/solvi-ai/solvi) ([PyPI](https://pypi.org/project/solvi/)): decision systems
from a catalog of Python functions, checks and rules, with a strategist that plans the flow, typed answers with reasons, and a
hash-chained trace that can be replayed.

This is a **static Space**: [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite` 5.45.0) loads Python
(Pyodide) into the visitor's browser and installs `solvi==0.8.0` from PyPI there (pinned in `index.html`). There is no server: every decision, including
code typed in the Playground, runs on the visitor's machine. The first visit downloads about 35-40 MB (Pyodide, Gradio, pandas,
numpy/scipy) once; later visits come from the browser cache.

Tabs:

- **Playground**: write your own catalog module (`cat`, `QUESTIONS`, optional `prepare(state)` and `setup(system)`) and
  `init_state`, run it, and see the answers, the **audit** (`res.audit()`: given → computed → quoted → decided → learned →
  checks / rule / constraints → answer, the safeguards that fired, the deterministic vs model share, `res.feasible` and
  `System.stats`), the planned flow (and what was not taken), `computed_state` with provenance, the trace replay, and what
  happens when you tamper with the trace or replace a model after the decision. Presets: two for solvi 0.5 (typed facts
  checked by pydantic; answer primitives — "not stated", a span, evidence quotes, a ranking, an estimate), three for 0.4 (a
  lying model caught by grounding, rules between answers, multi-label + ordinal), nine small tasks and the fifteen gallery
  tasks (13–15: helpers for coding agents, with keyword stand-ins for the deciders). The
  System is kept while the code is unchanged, so `setup` (e.g. `fit`) runs once and the stats accumulate. The code runs in-process (`sandbox.py`: `exec` in a fresh
  module) with a 5 s time guard (`sys.settrace` on the visitor's own frames), so an infinite loop is stopped.
- **solvi vs LLM**: cases from the public benchmark ([docs/vs_llm.md](../../docs/vs_llm.md),
  [benchmarks/vs_llm](../../benchmarks/vs_llm)), 15 refund and 15 3-way-match cases picked to be instructive (values at
  a limit, currency conversion, injected instructions, missing and conflicting facts, plain cases). For a case: its input,
  the right answer, what each LLM answered when asked directly (the saved answers of the published run, with their
  confidence, marked right or wrong; no API call, no key), what solvi decides (the gallery catalog runs live, with the
  rule or hard check behind each answer and the audit), and what the same LLMs answered inside solvi (alone or sent to a
  person). "Reorder the options" runs solvi again with its options in another order and the input's keys shuffled (the
  same answers) next to the LLMs' saved answers to a reordered request, where the run has one; "Replay" re-executes the
  trace, and a tampered copy fails. Below, the benchmark's main table. An honest summary on top: strong LLMs follow
  these rules nearly perfectly and solvi is not more accurate there; the differences are cost, speed, repeatability,
  replay and the guarantee.
- **New in 0.7**: small live demos, with keyword stand-ins in place of models and a scripted agent (no model runs in
  the browser): escalation with a guarantee (`act_guard` on 300 labelled emails: the answered share, the error, the risk
  on new emails, `must_escalate_at_least`, and the guarantee line of the audit; a slider sets the risk), a vote of two
  model families under one guarantee (`solvi.multi.Vote` with `act_guard`: disagreement escalates with both proposals;
  the audit's guarantee line), text in (`solvi.textin.TextIn` with `CueExtractor`: a message → the question it asks and
  its fields, each with a quote; `system.ask_text` answers it), the agent guard (preview: `solvi.agents.Guard` allows a
  refund to the account the user wrote, denies one to an account found only in a tool output, escalates the same call
  with `tool_values="escalate"`; the URL matcher accepts the address the user named and refuses look-alike hosts), a
  verified chart (preview: `solvi.charts.chart(text)` → an SVG where every number is quoted from the text; a careless
  proposal checked value by value; replay and an edited record), which record changed (preview: `store.signature()`
  names the one rewritten decision of six after every hash and the head were recomputed), learning from corrections
  (`fit` on 10 tickets, then 290 `teach` corrections, with 0.7's refit on doubling and without it) and a report for
  people (`res.report()` as Markdown and as the self-contained HTML page). The Playground tab also shows the report of
  every run. Each feature is detected before use: with an older solvi a demo says which version it needs instead of
  failing. The gated learning loop, LoRA adapters and `long="full"` are not shown (experimental, or they need torch or
  a model).
- **Strategy**: an insurance claim desk with 21 parts and 5 rules, six of them slow on purpose. The generated plan as a graph,
  early exit on failed hard checks, and live timings: plain script vs solvi one by one. Pyodide has no threads, so the
  parallel run (`workers=8`) is shown as the native numbers from the solvi README. Plus a scale test on catalogs of 100 to
  10 000 parts.
- **Business decisions**: leave request and invoice approval as forms, no code; extracted quotes are highlighted.
- **Learn from examples**: learn a readable rule list from labeled addresses, check it on fresh data, try your own.
- **About**.

Another Space: [solvi arcade](https://huggingface.co/spaces/solvi-ai/arcade).

## Files

- `index.html`: loads `@gradio/lite` from jsDelivr and lists the Python files (`<gradio-file url=...>`) and requirements.
- `vs_llm.py`: the "solvi vs LLM" tab (pure Python, no gradio): renders a case, runs the gallery catalog live, reorders
  the options, replays the trace. Its arms and groups come from `vs_llm.json`, so a new model or a new kind of arm shows
  up without a code change.
- `vs_llm.json`: that tab's data (curated cases, the saved answers of every arm on them, the summary table), built by
  `benchmarks/vs_llm/make_playground_bundle.py` (rerun it after the benchmark changes; `--check` says whether it is stale).
- `new07.py`: the "New in 0.7" demos (pure Python, no gradio; each feature detected before use; `run(name, text, risk)`
  returns the markdown, the audit, the report as Markdown and HTML, and a picture).
- `audit_view.py`: renders the audit panel from `Response.audit().to_dict()`.
- `app.py` (entrypoint), `sandbox.py`, `demos.py`, `strategy_demo.py`, `presets/`: the app, ported from the Gradio 6 server
  Space in `../playground` to the Gradio 5 API that Gradio-Lite ships.

## Dependency pinning (important)

Gradio-Lite installs its bundled gradio 5.45 wheel with micropip, which takes the newest version of every dependency and does
not backtrack. With today's PyPI that fails at start-up (huggingface-hub 2.x vs gradio's `<1.0`; filelock 4.x does not import
in Pyodide). A small script at the top of `index.html` wraps the Pyodide web worker so that its view of the PyPI simple index
only shows wheels uploaded before 2025-10-15 (like `uv --exclude-newer`), plus caps huggingface-hub/starlette/filelock; `solvi`
itself is exempt from the date filter, and its version is pinned in `<gradio-requirements>` (bump it with each release). If Gradio-Lite is upgraded, revisit `CUTOFF` and `CAPS` there.

The page also appends a `?v=` stamp to the Python file URLs so browsers never run a stale cached `app.py` after an update.

## Run locally

```bash
cd spaces/playground
python -m http.server 8000      # then open http://localhost:8000/
```

Opening `index.html` as a `file://` URL does not work (the Python files are fetched over HTTP). The same `app.py` also runs as
a normal server app: `pip install "gradio>=5,<6" solvi && python app.py`.

## Limitations in the browser

- No threads: solvi falls back to running steps one by one; `workers=8` timings are the native ones from the README.
- `time.sleep` does wait in Gradio-Lite's worker (checked in Chrome), so the Strategy timings are real. If it ever returns
  at once in some runtime, the six slow parts are simulated and the Strategy tab reports the
  simulated service time separately from the measured compute time.
- The time guard cannot interrupt a single long C call (for example `sum(range(10**12))`), and there is no memory limit
  other than the browser's.
- Python in Pyodide is slower than native (roughly 1.5-3x), so the timings are higher than on a server.
- The browser console shows two harmless lines at start-up: huggingface_hub's "only soft file lock is available" warning
  (printed on stderr while gradio is imported) and a `postMessage` origin warning when the page is not inside huggingface.co.
