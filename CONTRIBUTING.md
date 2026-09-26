# Contributing

- Set up: `uv sync --group dev`; run the tests with `uv run pytest`.
- Keep the core (`solvi` without extras) free of heavy dependencies: numpy and scipy only. Model code goes behind `solvi[model]`.
- A change to answers, the strategist or the trace needs a test. Numbers in the README come from `benchmarks/`; update both.
- New examples: one self-contained script in `examples/`, a line in `examples/README.md`, and it must run in CI (core only) or
  say which extra it needs.
