"""What 1.1 removed (CHANGELOG, 1.1.0 "Removed"): importing a removed module raises ModuleNotFoundError, what replaces it
works, and no doc page, nav entry, STATUS entry or old path is left behind."""
import importlib
import random
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GONE = ["solvi.experimental.oncalib", "solvi.experimental.learning", "solvi.learning"]


@pytest.mark.parametrize("module", GONE)
def test_the_module_is_gone(module):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # no old-path warning on the way: the path is not kept
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


def test_outcomes_are_still_labels_and_recalibration_is_explicit(tmp_path):
    """oncalib is gone; System.outcome stores the label and System.guarantee(corrections=True) reads it."""
    from solvi import Answer, Catalog, Question, System
    from solvi.core.store import JSONLStorage
    cat = Catalog()

    @cat.fn
    def score(x: float) -> float:
        return x
    rng = random.Random(0)

    def draw(n):
        return [({"x": (x := rng.random())}, x + rng.gauss(0, 0.1) > 0.5) for _ in range(n)]
    s = System(cat, [Question("label", "label?", Answer.yes_no())], storage=JSONLStorage(str(tmp_path / "d.jsonl")))
    s.fit("label", draw(200), features=["score"])
    for st, y in draw(20):
        s.outcome(s.ask(st), y)
    rep = s.guarantee("label", draw(100), max_risk=0.1, corrections=True)
    assert rep["labels"]["corrections"] == {"outcome": 20}
    doc = System.outcome.__doc__ + System.teach.__doc__
    assert "oncalib" not in doc and "solvi.experimental.learning" not in doc
    assert "corrections=True" in System.outcome.__doc__


def test_the_learning_loop_is_gone_and_the_pipeline_takes_updates():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental import changes
    for name in ("Learning", "ShadowGate", "UpdateReport"):
        assert not hasattr(changes, name), name
    assert "update" in changes.ARTIFACTS and changes.LearningTarget.artifact == "update"
    from solvi.core.system import System
    with pytest.raises(AttributeError, match="solvi.experimental.changes"):
        System.learning.fget(None)


def test_nothing_is_left_behind():
    from solvi import _deprecate
    from solvi.experimental import STATUS
    assert "oncalib" not in STATUS and "learning" not in STATUS
    assert "solvi.experimental.learning" not in _deprecate.MOVED and "solvi.learning" not in _deprecate.old_paths()
    nav = (ROOT / "mkdocs.yml").read_text()
    index = (ROOT / "docs" / "api" / "index.md").read_text()
    for mod in ("solvi.experimental.oncalib", "solvi.experimental.learning"):
        name = mod.rsplit(".", 1)[1]
        assert not (ROOT / "docs" / "api" / f"{name}.md").exists(), mod
        assert f"{mod}:" not in nav and f"[`{mod}`]" not in index, mod
    assert not (ROOT / "src" / "solvi" / "experimental" / "oncalib.py").exists()
    assert not (ROOT / "src" / "solvi" / "experimental" / "learning.py").exists()
    docs = "".join(p.read_text() for p in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
    for mod in ("solvi.experimental.oncalib", "solvi.experimental.learning"):
        assert mod not in docs, mod
    src = "".join(p.read_text() for p in (ROOT / "src" / "solvi").rglob("*.py"))
    assert "experimental.oncalib" not in src and "experimental.learning" not in src
    guide = (ROOT / "docs" / "guide.md").read_text()
    assert "### Learning from corrections with gates and rollback (experimental)" not in guide
    assert "a labelled update is a candidate change" in guide.lower()


def test_the_changelog_says_what_was_removed_and_what_to_use_instead():
    text = (ROOT / "CHANGELOG.md").read_text()
    block = text.split("## 1.1.0")[1].split("\n## ")[0]
    removed = block.split("### Removed")[1].split("\n### ")[0]
    assert "| removed | why | use instead |" in removed
    row = next(r for r in removed.splitlines() if r.startswith("| `solvi.experimental.oncalib`"))
    assert "broke the promise" in row and "corrections=True" in row
    row = next(r for r in removed.splitlines() if r.startswith("| `solvi.experimental.learning`"))
    assert "no measurable gain" in row and re.search(r"LearningTarget", row) and 'artifact="update"' in row
