"""Private parameter views and bounded sequential batches; execution stays in System."""
from dataclasses import dataclass
import json
import math

from .types import is_model, model_facts

__all__ = []


def _json_defaults(value):
    def check(item, depth=0):
        if depth > 32:
            raise ValueError("parameter JSON nesting exceeds 32")
        if type(item) in (str, int, bool, type(None)):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is dict and all(type(k) is str for k in item):
            children = item.values()
        elif type(item) is list:
            children = item
        else:
            raise ValueError("parameters require finite JSON with string object keys")
        for child in children:
            check(child, depth + 1)
    if type(value) is not dict:
        raise TypeError("parameters must be a dict")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _input(system, state):
    if is_model(state):
        state = model_facts(state)
    if not isinstance(state, dict):
        raise TypeError("batch/view input must be a dict or BaseModel")
    if not all(isinstance(k, str) for k in state):
        raise TypeError("input fact names must be strings")
    clashes = sorted(set(state) & set(system.catalog.parts))
    if clashes:
        raise ValueError(f"given facts cannot replace computed parts: {clashes}")
    if "__solvi_hypothetical_probe__" in state:
        raise ValueError("the hypothetical probe marker is reserved for System.probe")
    return dict(state)


def _ask_many(system, states, questions=None, *, max_items=1000, defaults=None, **options):
    if type(max_items) is not int or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    if isinstance(states, (dict, str, bytes)):
        raise TypeError("ask_many expects an iterable of input dicts or BaseModels")
    inputs = []
    for state in states:
        if len(inputs) == max_items:
            raise ValueError(f"ask_many exceeds max_items={max_items}")
        merged = json.loads(defaults) if defaults is not None else {}
        merged.update(_input(system, state))
        checked = _input(system, merged)
        inputs.append(state if defaults is None and is_model(state) else checked)
    names = system._names(questions)
    return [system.ask(state, names, **options) for state in inputs]


@dataclass(frozen=True, eq=False)
class _ParameterView:
    """Immutable defaults over an explicitly shared, mutable System runtime."""
    _system: object
    _defaults: str

    @property
    def params(self):
        """A detached copy of the private defaults."""
        return json.loads(self._defaults)

    @property
    def shared_system(self):
        """The shared backend: guards, calibration, learning, stats, cache and storage."""
        return self._system

    def fingerprint(self):
        return {"system": self._system.fingerprint(), "params": self.params}

    def _merged(self, init_state):
        state = self.params
        state.update(_input(self._system, init_state))
        return _input(self._system, state)

    def ask(self, init_state, questions=None, **options):
        return self._system.ask(self._merged(init_state), questions, **options)

    async def aask(self, init_state, questions=None, **options):
        return await self._system.aask(self._merged(init_state), questions, **options)

    def ask_many(self, states, questions=None, *, max_items=1000, **options):
        return _ask_many(self._system, states, questions, max_items=max_items, defaults=self._defaults, **options)


def _with_params(system, params=None, **facts):
    values = {} if params is None else params
    if type(values) is not dict:
        raise TypeError("parameters must be a dict")
    values = {**values, **facts}
    encoded = _json_defaults(values)
    _input(system, values)
    return _ParameterView(system, encoded)
