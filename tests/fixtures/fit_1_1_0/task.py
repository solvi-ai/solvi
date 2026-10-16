"""A System whose decision part was fitted (part.fit) and taught (part.teach), written by solvi 1.1.0 — whose
adaptations were keyed by the question alone — to prove that 1.1.1, which keys a second part asking the same question
apart, verifies and replays the journal unchanged and computes the same fingerprints while one part per question holds
an adaptation (tests/test_decide.py). make.py wrote the files with the 1.1.0 adaptation keys."""
import os

import numpy as np

from solvi import Catalog, System
from solvi.core.deciders import DecideModel

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIONS = ["travel", "meals", "software"]


class Words:
    """A stand-in decider: a keyword count per option is its logit, plus a bias towards "travel"."""

    def fingerprint(self):
        return "words-fit-v1"

    def logits(self, items):
        out = []
        for it in items:
            t = it.text.lower()
            out.append({"logits": np.array([2.0 * t.count(o) + (1.0 if o == "travel" else 0.0) for o in it.options])})
        return out


EXAMPLES = ([(f"{o} {o} item {i}", o) for i in range(12) for o in OPTIONS]
            + [(f"a receipt, number {i}", OPTIONS[i % 3]) for i in range(12)])
STATES = [{"note": "train to the client, travel"}, {"note": "meals with the team"}, {"note": "a receipt"},
          {"note": "software licence and meals"}]


# None (the catalog default) when make.py wrote the files; the tests set "first", the rule 1.1 decided by (solvi 1.2
# defaults to "strictest" and marks it in the catalog fingerprint)
COMBINE = None


def fitted_part(fit=False):
    """The part on a fresh model with the adaptation 1.1.0 wrote (adaptations.json, read back as a user's file is —
    the fit's numbers are the file's, not this machine's); fit=True refits on EXAMPLES and teaches one correction
    (make.py, which then writes the file)."""
    model = DecideModel(Words(), {"format": "solvi_decide v2", "act": False})
    part = model.decision("category", "What kind of expense is it?", "note", OPTIONS)
    if fit:
        part.fit(EXAMPLES)
        part.teach("a receipt for a licence", "software")
    else:
        model.load_adaptations(os.path.join(HERE, "adaptations.json"))
    return part


def system(storage=None):
    cat = Catalog() if COMBINE is None else Catalog(combine=COMBINE)
    part = fitted_part()
    return System(cat, [part.question(cat)], storage=storage)
