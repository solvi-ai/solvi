"""ExactModel.fork without clone= deep-copies the environment: a copy that holds a torch module or is large is warned
about once, pointing at clone=."""
from __future__ import annotations

import warnings


from solvi.core import Outcome

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.skills import ExactModel

# a stand-in for torch.nn.Module without importing torch
_Module = type("Module", (), {"__module__": "torch.nn.modules.module"})
_Linear = type("Linear", (_Module,), {"__module__": "torch.nn.modules.linear"})


class Walk:
    def __init__(self, payload=None):
        self.pos, self.payload = 0, payload

    def reset(self, seed=None):
        self.pos = 0
        return {"pos": 0}

    def actions(self, state):
        return ["right"]

    def step(self, action):
        self.pos += 1
        return Outcome({"pos": self.pos}, effect=None)


def _forks(m, n=3):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(n):
            m.fork()
    return [x for x in w if "clone=" in str(x.message)]


def test_a_small_plain_environment_is_copied_silently():
    assert _forks(ExactModel(Walk())) == []


def test_a_torch_module_in_the_environment_is_warned_once():
    got = _forks(ExactModel(Walk(payload={"policy": _Linear()})))
    assert len(got) == 1 and "torch" in str(got[0].message)


def test_a_large_copy_is_warned_once(monkeypatch):
    monkeypatch.setattr(ExactModel, "FORK_WARN_BYTES", 1000)
    got = _forks(ExactModel(Walk(payload=list(range(5000)))))
    assert len(got) == 1 and "bytes" in str(got[0].message)


def test_a_clone_hook_is_never_warned_about():
    m = ExactModel(Walk(payload={"policy": _Linear()}), clone=lambda env, state: Walk())
    assert _forks(m) == []
