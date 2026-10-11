"""Write the journal with 1.1.0's adaptation keys (run from this folder with the 1.1.0 sources on the path):

    PYTHONPATH=<1.1.0 src> python make.py

decisions.jsonl: four decisions of the fitted part; fingerprints.json: the model's, the part's and the system's."""
import json
import os

from solvi.core.store import JSONLStorage
from task import HERE, STATES, fitted_part, system

for f in os.listdir(HERE):
    if f.startswith("decisions."):
        os.remove(os.path.join(HERE, f))
store = JSONLStorage(os.path.join(HERE, "decisions.jsonl"))
s = system(store)
for st in STATES:
    s.ask(st)
store.close()
part = fitted_part()
with open(os.path.join(HERE, "fingerprints.json"), "w") as fh:
    json.dump({"model": part.model.fingerprint(), "part": part.fingerprint(), "system": system().fingerprint()},
              fh, indent=1, sort_keys=True)
