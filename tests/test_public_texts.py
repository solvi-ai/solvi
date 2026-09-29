"""The published trees (README, docs, the package, the gallery, the examples, the benchmarks) do not point into the private
research repo and carry no research experiment codes."""
import gzip
import pathlib
import re

ROOT = pathlib.Path(__file__).parents[1]
PRIVATE = re.compile(r"new_kelly|exps_v2|kelly", re.IGNORECASE)
TREES = ["README.md", "docs", "src", "gallery", "examples", "benchmarks"]
TEXT = {".md", ".py", ".json", ".yml", ".yaml", ".toml", ".txt", ".html", ".csv"}


def test_no_private_repo_paths_in_published_trees():
    hits = []
    for tree in TREES:
        top = ROOT / tree
        for p in [top] if top.is_file() else sorted(top.rglob("*")):
            if p.is_file() and p.suffix in TEXT and "__pycache__" not in p.parts:
                for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if PRIVATE.search(line):
                        hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


# Research experiment codes (L14g, L25, ...) mean nothing to a reader. Allowed: math (L0 L1 L2, L2 regularization), model
# names (MiniLM-L6), lowercase format strings ('l14g typed v2') and the alt text of an image that shows such a label.
CODE = re.compile(r"\bL[0-9]{1,2}[a-z]?\b")
CODE_OK = re.compile(r"MiniLM-L[0-9]+|\bL[0-2]\b")
CODE_SKIP = {"docs/images/ALT.md"}


def test_no_experiment_codes_in_published_trees():
    hits = []
    for tree in TREES:
        top = ROOT / tree
        for p in [top] if top.is_file() else sorted(top.rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and p.suffix in TEXT and "__pycache__" not in p.parts and rel not in CODE_SKIP \
                    and "realms" not in rel:
                for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if CODE.search(CODE_OK.sub("", line)):
                        hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def _gz_texts():
    """The compressed data and raw answers under benchmarks/ (JSON lines, gzipped): (relative path, text)."""
    for p in sorted((ROOT / "benchmarks").rglob("*.gz")):
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            yield p.relative_to(ROOT).as_posix(), f.read()


def test_benchmark_data_has_no_private_references_or_codes():
    """The shipped sets and raw answers too: no private paths, no project code names, no experiment codes."""
    private = re.compile(r"new_kelly|exps_v2|kelly", re.IGNORECASE)
    hits = []
    for rel, text in _gz_texts():
        for n, line in enumerate(text.splitlines(), 1):
            if private.search(line) or CODE.search(CODE_OK.sub("", line)):
                hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits[:50])
