"""The honesty suite's model-backed subset: the real decider (solvi-decide ONNX checkpoint) on support messages.

The checkpoint is found in $SOLVI_DECIDE_MODEL or ~/.cache/solvi_release/decide-base (a directory with solvi_decide.json
and onnx/); `model_dir()` returns None when there is none, and the tests skip. Needs `solvi[onnx]`.

  team    which team handles the message; "other" is the decider's abstain threshold, and a calibrated confidence below
          0.5 escalates
  refund  does the customer ask for money back (yes / no)"""
import os

from solvi import Catalog, System
from solvi.decide import DecideModel

TEAMS = {"billing": "payments, invoices, charges, refunds", "technical": "bugs, crashes, errors, the app or site not working",
         "shipping": "delivery, parcels, tracking, couriers", "other": "anything else"}


def model_dir():
    for p in (os.environ.get("SOLVI_DECIDE_MODEL"), os.path.expanduser("~/.cache/solvi_release/decide-base")):
        if p and os.path.isfile(os.path.join(p, "solvi_decide.json")) and os.path.isdir(os.path.join(p, "onnx")) \
                and any(f.endswith(".onnx") for f in os.listdir(os.path.join(p, "onnx"))):
            return p
    return None


def system():
    path = model_dir()
    if path is None:
        raise FileNotFoundError("no solvi-decide checkpoint: set SOLVI_DECIDE_MODEL to a directory with solvi_decide.json "
                                "and onnx/")
    m = DecideModel.load(path, backend="onnx")
    cat = Catalog()
    qs = [m.decision("team", "Which team should handle this customer message?", "doc", TEAMS,
                     escalate_below=0.5).question(cat),
          m.decision("refund", "Does the customer ask for their money back?", "doc", bool).question(cat)]
    return System(cat, qs)
