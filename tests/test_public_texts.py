"""The published trees (README, docs, the package, the gallery, the examples) do not point into the private research repo."""
import pathlib
import re

ROOT = pathlib.Path(__file__).parents[1]
PRIVATE = re.compile(r"new_kelly|exps_v2")
TREES = ["README.md", "docs", "src", "gallery", "examples"]
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
