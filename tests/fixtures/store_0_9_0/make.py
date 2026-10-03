"""Write the 0.9.0 files: run with the 0.9.0 sources on the path, from this folder:

    git archive v0.9.0 src | tar -x -C /tmp/v090 && PYTHONPATH=/tmp/v090/src python make.py

decisions.jsonl / decisions.db: four decisions and one correction; calibration.json: the category part calibrated
(calibrate_for, max_error=0.1); dispatch.jsonl: thirty decisions of solvi.build's dispatcher (the policy record first);
fingerprints.json: the fingerprints 0.9.0 computed for the system, the part and System 1 of the build."""
import json
import os

import solvi
from solvi.storage import JSONLStorage, SQLiteStorage
from task import ASKED, CALIBRATION, STATES, category_part, dispatch, system

assert solvi.__version__ == "0.9.0", solvi.__version__
HERE = os.path.dirname(os.path.abspath(__file__))
for f in os.listdir(HERE):
    if f.startswith(("decisions.", "dispatch.", "calibration.")):
        os.remove(os.path.join(HERE, f))
part = category_part()
print("calibrate_for", part.calibrate_for(CALIBRATION, max_error=0.1))
cal = os.path.join(HERE, "calibration.json")
part.save_calibration(cal)
for store in (JSONLStorage(os.path.join(HERE, "decisions.jsonl")), SQLiteStorage(os.path.join(HERE, "decisions.db"))):
    s = system(store, cal)
    ids = [s.ask(st).stored_id for st in STATES]
    s.teach("category", STATES[3], "software", by="reviewer", of=ids[3])
    store.close()
    print(type(store).__name__, len(ids), "decisions, 1 correction")
auto = dispatch(os.path.join(HERE, "dispatch.jsonl"))
by = [auto.ask(st).by for st in ASKED]
print("dispatch", {b: by.count(b) for b in sorted(set(by))})
auto.storage.close()
s = system(None, cal)
with open(os.path.join(HERE, "fingerprints.json"), "w") as fh:
    json.dump({"system": s.fingerprint(), "part": category_part().fingerprint(),
               "calibrated_part": category_part(cal).fingerprint(), "dispatch_system": dispatch().system.fingerprint()},
              fh, indent=1, sort_keys=True)
