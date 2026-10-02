"""Write the 0.7.1 stores: run with the 0.7.1 sources on the path, from this folder:

    git archive v0.7.1 src | tar -x -C /tmp/v071 && PYTHONPATH=/tmp/v071/src python make.py
"""
import os

import solvi
from solvi.storage import JSONLStorage, SQLiteStorage
from task import STATES, system

assert solvi.__version__ == "0.7.1", solvi.__version__
HERE = os.path.dirname(os.path.abspath(__file__))
for store in (JSONLStorage(os.path.join(HERE, "decisions.jsonl")), SQLiteStorage(os.path.join(HERE, "decisions.db"))):
    s = system(store)
    ids = [s.ask(st).stored_id for st in STATES]
    s.teach("category", STATES[3], "software", by="reviewer", of=ids[3])
    print(type(store).__name__, len(ids), "decisions, 1 correction")
