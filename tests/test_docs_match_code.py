"""Sentences of the README and the guide that restate the code — a signature, a list of keys, what a published checkpoint
does — checked against the code, so they cannot drift apart again."""
import inspect
import json
import re
from pathlib import Path

from solvi.audit import STATS
from solvi.decide import DecideModel

ROOT = Path(__file__).resolve().parents[1]
GUIDE = (ROOT / "docs" / "guide.md").read_text()
README = (ROOT / "README.md").read_text()


def _flat(text):
    return re.sub(r"\s+", " ", text)


def test_the_guide_gives_the_whole_signature_of_model_decision():
    sig = str(inspect.signature(DecideModel.decision)).replace("(self, ", "(").replace("'", '"')
    assert f"`model.decision{sig}`" in _flat(GUIDE)


def test_the_guide_lists_every_lifetime_stat():
    para = _flat(GUIDE[GUIDE.index("### Lifetime stats"):GUIDE.index("`system.safeguard_report()` prints them")])
    assert [k for k in STATS if f"`{k}`" not in para] == []


def test_the_readme_does_not_promise_one_forward_pass_for_the_published_checkpoints():
    """Both published checkpoints declare their multi-question pass and ship it switched off."""
    flat = _flat(README)
    assert "several in one forward pass when the checkpoint can" not in flat and "four answers from one forward pass" not in flat
    assert "one question per forward pass with the published checkpoints" in flat and "multi_question=True" in flat
    assert "the shared pass, what escalated" not in flat           # the audit of that snippet prints no shared pass
    example = (ROOT / "examples" / "15_typed_decisions.py").read_text()
    assert "model.batchable" in example                              # the heading says what the model does
    assert json.dumps("four answers from one forward pass")[1:-1] not in (ROOT / "examples" / "README.md").read_text()
