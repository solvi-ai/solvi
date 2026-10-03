"""Write the 0.8.0 files: run with the 0.8.0 sources on the path, from this folder:

    git archive v0.8.0 src | tar -x -C /tmp/v080 && PYTHONPATH=/tmp/v080/src python make.py

decisions.jsonl / decisions.db: four decisions and one correction; calibration.json: the category part calibrated
(calibrate_for, max_error=0.1); fingerprints.json: the fingerprints 0.8.0 computed for the system and the part."""
import json
import os

import solvi
from solvi.storage import JSONLStorage, SQLiteStorage
from task import CALIBRATION, STATES, category_part, system

assert solvi.__version__ == "0.8.0", solvi.__version__
HERE = os.path.dirname(os.path.abspath(__file__))
part = category_part()
print("calibrate_for", part.calibrate_for(CALIBRATION, max_error=0.1))
part.save_calibration(os.path.join(HERE, "calibration.json"))
cal = os.path.join(HERE, "calibration.json")
for store in (JSONLStorage(os.path.join(HERE, "decisions.jsonl")), SQLiteStorage(os.path.join(HERE, "decisions.db"))):
    s = system(store, cal)
    ids = [s.ask(st).stored_id for st in STATES]
    s.teach("category", STATES[3], "software", by="reviewer", of=ids[3])
    store.close()
    print(type(store).__name__, len(ids), "decisions, 1 correction")
s = system(None, cal)
with open(os.path.join(HERE, "fingerprints.json"), "w") as fh:
    json.dump({"system": s.fingerprint(), "part": category_part().fingerprint(),
               "calibrated_part": category_part(cal).fingerprint()}, fh, indent=1, sort_keys=True)
