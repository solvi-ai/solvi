"""Set-valued facts hash the same in every process and replay after a JSON round trip."""
import os
import subprocess
import sys

from solvi import Answer, Catalog, Question, System
from solvi.system import Response

CODE = """
from solvi import Answer, Catalog, Question, System
cat = Catalog()
@cat.fn
def tags(text):
    return set(text.split())
@cat.rule("has_refund")
def has_refund(tags):
    return "refund" in tags
r = System(cat, [Question("has_refund", "Refund?", Answer.yes_no())]).ask({"text": "please refund my broken mug now"})
print(r.trace.records[-1].hash)
"""


def _system():
    cat = Catalog()

    @cat.fn
    def tags(text):
        return set(text.split())

    @cat.rule("has_refund")
    def has_refund(tags):
        return "refund" in tags

    return System(cat, [Question("has_refund", "Refund?", Answer.yes_no())])


def test_trace_hash_does_not_depend_on_the_process():
    hashes = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env, check=True)
        hashes.add(out.stdout.strip())
    assert len(hashes) == 1


def test_set_fact_replays_after_json_round_trip():
    s = _system()
    r = s.ask({"text": "please refund my broken mug now"})
    back = Response.from_json(r.to_json(), catalog=s)
    assert back.trace.replay(s.catalog)["ok"]


def test_a_failed_step_hashes_the_same_in_every_process():
    import subprocess
    import sys
    code = "from solvi.runtime import vhash, MISSING; print(vhash(MISSING))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
            for _ in range(3)}
    assert len(outs) == 1
