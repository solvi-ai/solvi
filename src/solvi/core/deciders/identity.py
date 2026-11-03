"""Opt-in deployment leases. The provider, not this client, guarantees generation pinning.

identity_lease is a zero-argument context manager factory yielding a nonnegative,
monotonically increasing integer generation for the entire deployment/router.
It must pin that generation until context exit, including cached operations.
No built-in remote adapter currently supplies this contract.

Canary accept() never resets unknown/stale lease state. Loaded calibration bindings
are audit provenance only: a fresh successful calibration is required. Checkpoint
metadata calibration is unsupported in this opt-in mode. Combinations fail closed
until their own binding/lease transaction is implemented.
"""
import contextlib
import functools
import threading
import copy

from ..catalog import Decision


class IdentityUnavailable(Exception):
    pass


class IdentityGuard:
    def __init__(self, model, lease):
        self.model, self.lease = model, lease
        self.generation = None
        self.state = "unknown"
        self.epoch = 0
        self.stale = False
        # No trusted provider binding exists for checkpoint-level calibration.
        self.unsupported_calibration = (any(key in model.meta for key in
                                            ("calibration", "temperature", "temperature_multi", "thresholds",
                                             "escalate_below", "other_threshold", "multi_threshold", "act")) or
                                        any(t != 1.0 for t in model.temperatures.values()) or
                                        bool(model.act_thresholds) or model.escalate_below is not None or
                                        bool((model.caps.get("act") or {}).get("calibrator")))
        self.lock = threading.RLock()
        self.local = threading.local()

    def invalidate(self):
        self.epoch += 1
        self.model.clear_cache()
        self.model.adaptations.clear()
        self.state = "unknown"
        # Only a new successful calibration may recover this state.
        self.stale = True

    @contextlib.contextmanager
    def operation(self, calibration=False):
        with self.lock:
            if getattr(self.local, "active", False):
                yield
                return
            try:
                manager = self.lease() if callable(self.lease) else None
                if manager is None:
                    raise IdentityUnavailable()
                generation = manager.__enter__()
            except Exception:  # noqa: BLE001 -- a provider lease failure is an unknown identity
                self.invalidate()
                raise IdentityUnavailable() from None
            try:
                if type(generation) is not int or generation < 0:
                    self.invalidate()
                    raise IdentityUnavailable()
                if self.unsupported_calibration:
                    self.invalidate()
                    raise IdentityUnavailable()
                if self.generation is None and self.model.adaptations:
                    self.invalidate()
                if self.generation is not None and generation != self.generation:
                    self.invalidate()
                    if generation < self.generation:
                        raise IdentityUnavailable()
                self.generation = generation
                self.state = "confirmed"
                if self.stale and not calibration:
                    raise IdentityUnavailable()
                self.local.active = True
                try:
                    yield
                finally:
                    self.local.active = False
            finally:
                try:
                    manager.__exit__(None, None, None)
                except Exception:  # noqa: BLE001 -- loss of lease invalidates the completed output
                    self.invalidate()
                    raise IdentityUnavailable() from None


def abstain(guard):
    return Decision(None, {}, confidence=0.0, escalate="provider identity unconfirmed or calibration stale",
                    extra={"provider_identity": {"state": guard.state, "epoch": guard.epoch,
                                                  "generation": guard.generation, "stale": guard.stale}})


def guarded(fn):
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        guard = self.identity_guard
        if guard is None:
            return fn(self, *args, **kwargs)
        try:
            with guard.operation():
                return fn(self, *args, **kwargs)
        except IdentityUnavailable:
            # Preserve model.decide batch shape without executing an unleased scorer.
            text = args[0] if args else kwargs.get("text")
            from .state import _single
            if fn.__name__ == "decide_pass":
                parts = args[1] if len(args) > 1 else kwargs["parts"]
                return [abstain(guard) for _ in parts]
            return abstain(guard) if _single(text) else [abstain(guard) for _ in text]
    return call


def part_gate(part, result):
    guard = part.model.identity_guard
    if guard is None:
        return result
    if isinstance(result, list):
        return [part_gate(part, d) for d in result]
    if getattr(guard.local, "calibrating", False):
        return result
    binding = getattr(part, "_identity_binding", None)
    configured = any(getattr(part, k, None) is not None for k in
                     ("escalate_below", "act_threshold", "guarantee", "conformal_set", "groups"))
    if (not getattr(guard.local, "active", False) or guard.stale or
            (configured and binding != (guard.generation, guard.epoch))):
        return abstain(guard)
    return result


def part_operation(fn):
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        guard = self.model.identity_guard
        if guard is None:
            return fn(self, *args, **kwargs)
        try:
            with guard.operation():
                return part_gate(self, fn(self, *args, **kwargs))
        except IdentityUnavailable:
            from .state import _single
            text = args[0] if args else kwargs.get("text")
            if fn.__name__ == "decide" and not _single(text):
                return [abstain(guard) for _ in text]
            return abstain(guard)
    return call


def safe_finish(fn):
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        return part_gate(self, fn(self, *args, **kwargs))
    return call


def calibration_operation(fn):
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        guard = self.model.identity_guard
        if guard is None:
            return fn(self, *args, **kwargs)
        fields = ("escalate_below", "act_threshold", "guarantee", "conformal_set", "groups")
        previous = {key: copy.copy(getattr(self, key)) for key in fields}
        self._identity_binding = None
        try:
            with guard.operation(calibration=True):
                guard.local.calibrating = True
                try:
                    # No old threshold, set, or grouped promise survives a fresh fit.
                    for key in fields:
                        setattr(self, key, None)
                    self.model.clear_cache()
                    result = fn(self, *args, **kwargs)
                    binding = (guard.generation, guard.epoch)
                    self._identity_binding = binding
                    guard.stale = False
                finally:
                    guard.local.calibrating = False
            # The model lock spans lease exit; exit failure invalidates below.
            return result
        except BaseException:
            for key, value in previous.items():
                setattr(self, key, value)
            guard.invalidate()
            self._identity_binding = None
            raise
    @functools.wraps(call)
    def locked(self, *args, **kwargs):
        guard = self.model.identity_guard
        with guard.lock if guard is not None else contextlib.nullcontext():
            return call(self, *args, **kwargs)
    return locked


def combination_guard(fn):
    """Combinations have no lease binding yet: fail closed before any leaf scoring."""
    def find(combo):
        for member in combo.members:
            part = getattr(member, "part", None)
            if part is not None:
                guard = getattr(getattr(part, "model", None), "identity_guard", None)
                if guard is not None:
                    return guard
            nested = member if hasattr(member, "members") else None
            if nested is not None:
                guard = find(nested)
                if guard is not None:
                    return guard
        return None

    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        guard = find(self)
        if guard is None:
            return fn(self, *args, **kwargs)
        return abstain(guard)
    return call
