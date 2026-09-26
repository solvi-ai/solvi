---
title: solvi playground
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Write a decision task in Python, get verifiable answers
---

# solvi playground

Interactive demo of [solvi](https://github.com/solvi-ai/solvi): decision systems from a catalog of Python functions, checks
and rules, with a strategist that plans the flow, typed answers with reasons, and a hash-chained trace that can be replayed.

Tabs:

- **Playground**: write your own catalog module (`cat`, `QUESTIONS`, optional `prepare(state)`) and `init_state`, run it,
  and see the answers, the planned flow (and what was not taken), `computed_state` with provenance, the trace replay, and
  what happens when you tamper with the trace. Nine presets. User code runs in a separate sandboxed process
  (`runner.py`: 5 s timeout, 5 s CPU, ~1 GB memory, temporary directory, empty environment).
- **Strategy**: an insurance claim desk with 21 parts and 5 rules, six of them slow on purpose. Shows the generated plan as
  a graph, early exit on failed hard checks, and live timings: plain script vs solvi one by one vs solvi parallel. Plus a
  scale test on catalogs of 100 to 10 000 parts.
- **Business decisions**: leave request and invoice approval as forms, no code; extracted quotes are highlighted.
- **Learn from examples**: learn a readable rule list from labeled addresses, check it on fresh data, try your own.
- **About**.

Run locally from a clone of solvi:

```bash
cd spaces/playground
pip install "gradio>=6"
PYTHONPATH=../../src python app.py
```

`strategy_demo.py` and `demos.py` vendor catalogs from `examples/03`, `06`, `09` and `benchmarks/strategist_scale.py` so the
Space is self-contained.
