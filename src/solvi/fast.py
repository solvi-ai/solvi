"""Deprecated (0.8; removed in 0.9): FastHead, VecFeaturizer, CandidateHead and select_features are in solvi.heads, next
to Head — one module for the learned answer heads. Every name still reads from here, with a SolviDeprecationWarning."""
from . import _deprecate

__getattr__ = _deprecate.module_getattr("solvi.fast", {n: "solvi.heads:" + n for n in (
    "FastHead", "VecFeaturizer", "CandidateHead", "select_features")})


__all__ = []
