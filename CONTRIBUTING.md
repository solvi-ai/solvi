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
- Messages solvi writes for people (audit labels, a `why`, a rejection or escalation reason) are recorded in English; a
  new or changed one needs its Russian template in `solvi/i18n.py` (`RU` for labels, `MESSAGES_RU` for messages).
  English output must not change by accident: `tests/test_i18n.py` compares it with `tests/i18n/en_golden.json`; when the
  change is intended, regenerate it with `uv run python tests/i18n/render.py` and say so in the CHANGELOG.
- Style: plain functions, short docstrings that say what a thing returns and why; line length 120.

## Releasing

1. Move the CHANGELOG's "Unreleased" section under the new version and date; bump the version in `pyproject.toml`.
2. Publish a GitHub release (tag `vX.Y.Z`). The `publish` workflow builds and uploads to PyPI; the `docs` workflow
   deploys the documentation site.
3. The browser Spaces (`spaces/playground`, `arcade`, `realms`, `documents-web`) install solvi from PyPI when a page
   loads, so they pick up the release by themselves; upload a Space's folder only when its own files changed. Before
   uploading, check the new version locally in a real browser:

   ```bash
   cd spaces/playground && python -m http.server 8000 &
   uv run --with playwright python -m playwright install chromium          # once
   uv run --with playwright python tools/smoke_spaces.py --only playground --url playground=http://localhost:8000/index.html
   ```

4. After the upload to PyPI, the `smoke-spaces` workflow runs `tools/smoke_spaces.py` against the public Spaces (it can
   also be started by hand from the Actions tab). It opens each Space in headless Chromium, waits for Pyodide to install
   solvi (a cold load takes one to several minutes), runs one preset and checks its output: the playground's
   "Trace replay: OK" and its "New in 0.7" tab, a tic-tac-toe move in the arcade, a turn in realms, and Python + solvi
   ready in documents (`--with-model` also downloads the 790 MB extractor and runs a use case). It prints the solvi
   version each Space loaded; the results and a screenshot of every failing Space are kept as the run's artifact.
   Without Playwright the script skips (exit 0; `--require` makes it exit 2). A cold load now and then stalls on a CDN,
   so a failing Space is tried once more (`--retries`).

## Pull requests

Small and focused is best. Describe what changes for a user, how you tested it, and anything that changes a trace hash or
an answer. CI runs the tests on Python 3.10–3.13, the gallery and the core examples.
