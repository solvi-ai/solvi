"""The honesty suite's model-backed subset: the published decider (solvi-ai/solvi-base, ONNX) on support messages.

The checkpoint is $SOLVI_DECIDE_MODEL (a checkpoint folder, or a Hugging Face id — downloaded when it is not in the
cache), else solvi-ai/solvi-base when it is already in the Hugging Face cache (`solvi models pull solvi-ai/solvi-base`);
`model_dir()` returns None when there is none, and the tests skip — nothing is downloaded unless the variable names an
id. The committed baseline (model_v1.baseline.json, its "model" line) is solvi-ai/solvi-base's: another checkpoint is
compared with those numbers. Needs `solvi[onnx]`.

  team    which team handles the message; "other" is the decider's abstain threshold, and a calibrated confidence below
          0.5 escalates
  refund  does the customer ask for money back (yes / no)"""
import os

from solvi import Catalog, System, models
from solvi.decide import DecideModel

TEAMS = {"billing": "payments, invoices, charges, refunds", "technical": "bugs, crashes, errors, the app or site not working",
         "shipping": "delivery, parcels, tracking, couriers", "other": "anything else"}


PUBLISHED = "solvi-ai/solvi-base"


def model_dir():
    src = os.environ.get("SOLVI_DECIDE_MODEL")
    if src and not os.path.isdir(os.path.expanduser(src)):
        return str(models.cached_path(src) or src)          # a Hugging Face id: the opt-in to a download
    p = os.path.expanduser(src) if src else models.cached_path(PUBLISHED)
    if p and os.path.isfile(os.path.join(p, "solvi_decide.json")) and os.path.isdir(os.path.join(p, "onnx")) \
            and any(f.endswith(".onnx") for f in os.listdir(os.path.join(p, "onnx"))):
        return str(p)
    return None


def system():
    path = model_dir()
    if path is None:
        raise FileNotFoundError(f"no decider checkpoint: `solvi models pull {PUBLISHED}`, or set SOLVI_DECIDE_MODEL to a "
                                "directory with solvi_decide.json and onnx/ (or to a Hugging Face id)")
    m = DecideModel.load(path, backend="onnx")
    cat = Catalog()
    qs = [m.decision("team", "Which team should handle this customer message?", "doc", TEAMS,
                     min_confidence=0.5).question(cat),
          m.decision("refund", "Does the customer ask for their money back?", "doc", bool).question(cat)]
    return System(cat, qs)
