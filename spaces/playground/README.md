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
(Pyodide) into the visitor's browser and installs `solvi>=0.4.0` from PyPI there. There is no server: every decision, including
code typed in the Playground, runs on the visitor's machine. The first visit downloads about 35-40 MB (Pyodide, Gradio, pandas,
numpy/scipy) once; later visits come from the browser cache.

Tabs:

- **Playground**: write your own catalog module (`cat`, `QUESTIONS`, optional `prepare(state)` and `setup(system)`) and
  `init_state`, run it, and see the answers, the **audit** (`res.audit()`: given → computed → quoted → decided → learned →
  checks / rule / constraints → answer, the safeguards that fired, the deterministic vs model share, `res.feasible` and
  `System.stats`), the planned flow (and what was not taken), `computed_state` with provenance, the trace replay, and what
  happens when you tamper with the trace or replace a model after the decision. Presets: three for solvi 0.4 (a lying model
  caught by grounding, rules between answers, multi-label + ordinal), nine small tasks and the twelve gallery tasks. The
  System is kept while the code is unchanged, so `setup` (e.g. `fit_fast`) runs once and the stats accumulate. The code runs in-process (`sandbox.py`: `exec` in a fresh
  module) with a 5 s time guard (`sys.settrace` on the visitor's own frames), so an infinite loop is stopped.
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
- `audit_view.py`: renders the audit panel from `Response.audit().to_dict()`.
- `app.py` (entrypoint), `sandbox.py`, `demos.py`, `strategy_demo.py`, `presets/`: the app, ported from the Gradio 6 server
  Space in `../playground` to the Gradio 5 API that Gradio-Lite ships.

## Dependency pinning (important)

Gradio-Lite installs its bundled gradio 5.45 wheel with micropip, which takes the newest version of every dependency and does
not backtrack. With today's PyPI that fails at start-up (huggingface-hub 2.x vs gradio's `<1.0`; filelock 4.x does not import
in Pyodide). A small script at the top of `index.html` wraps the Pyodide web worker so that its view of the PyPI simple index
only shows wheels uploaded before 2025-10-15 (like `uv --exclude-newer`), plus caps huggingface-hub/starlette/filelock; `solvi`
itself is exempt, so new solvi releases are picked up. If Gradio-Lite is upgraded, revisit `CUTOFF` and `CAPS` there.

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
