"""Deprecated (0.8; removed in 0.9): the segment model is solvi.segment_model (experimental; no checkpoint is published).
Every name still reads from here, with a SolviDeprecationWarning."""
from . import _deprecate


def __getattr__(name):
    from . import segment_model
    if name.startswith("__") or not hasattr(segment_model, name):
        raise AttributeError(f"module 'solvi.strategy_model' has no attribute {name!r}")
    _deprecate.renamed(f"solvi.strategy_model.{name}", f"solvi.segment_model.{name}")
    return getattr(segment_model, name)


__all__ = []
