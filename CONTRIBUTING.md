# Contributing

Thanks for helping. Questions and ideas are welcome in
[Discussions](https://github.com/solvi-ai/solvi/discussions); bugs and concrete feature requests in
[Issues](https://github.com/solvi-ai/solvi/issues). Security problems: see [SECURITY.md](SECURITY.md), not a public issue.
By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setting up

```bash
git clone https://github.com/solvi-ai/solvi && cd solvi
uv sync --group dev
uv run pytest -q                                    # core tests (model tests skip without the extras)
for f in gallery/*/run.py; do uv run python "$f" > /dev/null || echo "FAIL $f"; done
```

With the model extras: `uv run --with torch --with transformers --with onnxruntime --with tokenizers --with huggingface_hub pytest -q`.
A decider checkpoint end to end, both backends: `uv run --with ... python tools/smoke_decide.py <checkpoint folder>`.
The documentation site: `uv sync --group docs`, then `uv run mkdocs serve` (preview) or `uv run mkdocs build --strict` (what
CI runs on every pull request: a broken link, a missing anchor or a docstring the API reference cannot render fails it).

## Rules of the road

- Keep the core (`solvi` without extras) light: numpy, scipy and pydantic only, and `import solvi` must not import any of
  them eagerly beyond numpy — the browser Spaces (Pyodide) depend on it. Model code goes behind `solvi[model]` / `solvi[onnx]`
  and is imported lazily.
- A change to answers, the strategist, the trace or its hashes needs a test. Traces written by an earlier version must keep
  replaying; if a hash changes on purpose, say so in the CHANGELOG.
- Numbers in the README and the docs come from `benchmarks/` or a model card; update both, and never round in your favour.
- New examples: one self-contained script in `examples/`, a line in `examples/README.md` and in the README table; it must run
  in CI (core only, a stand-in for any model) or say which extra it needs.
- New gallery entries: `gallery/NN_name/` with `task.py`, `state.json`, `run.py` and a README; then
  `uv run python tools/sync_gallery.py` to copy it into the playground Space.
- Style: plain functions, short docstrings that say what a thing returns and why; line length 120.

## Pull requests

Small and focused is best. Describe what changes for a user, how you tested it, and anything that changes a trace hash or
an answer. CI runs the tests on Python 3.10–3.13, the gallery and the core examples.
