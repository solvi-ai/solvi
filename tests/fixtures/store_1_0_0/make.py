"""Write the 1.0.0 files: run with solvi 1.0.0 from PyPI, from this folder:

    uv venv -p 3.12 /tmp/v100 && VIRTUAL_ENV=/tmp/v100 uv pip install solvi==1.0.0 && /tmp/v100/bin/python make.py

decisions.jsonl / decisions.db: the four STATES of task.py stored with record="compact" (four compact records)."""
import os

import solvi
from solvi.core.store import JSONLStorage, SQLiteStorage
from task import STATES, system

assert solvi.__version__ == "1.0.0", solvi.__version__
HERE = os.path.dirname(os.path.abspath(__file__))
for f in os.listdir(HERE):
    if f.startswith("decisions."):
        os.remove(os.path.join(HERE, f))
for store in (JSONLStorage(os.path.join(HERE, "decisions.jsonl"), record="compact"),
              SQLiteStorage(os.path.join(HERE, "decisions.db"), record="compact")):
    s = system(store)
    for st in STATES:
        r = s.ask(st)
        print(type(store).__name__, {q: (a.answer, a.status) for q, a in r.results.items()})
    print("replay_all:", [(b["seq"], b["kinds"]) for b in store.replay_all(s)])
    store.close()
