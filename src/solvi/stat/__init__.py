"""Sequential evidence, multiple testing and descriptive agreement diagnostics.

MeanBound tests a conditional bounded-mean null; Mixture and MirrorBound test
that conditional mean loss is at least their alpha. Updates and evidence are in
log space. Certification and change-detector alarms have different guarantees;
see Certificate. Reserve all online allocations before their outcomes are read.
"""
from ._mean import MeanBound
from ._process import DeltaBudget, MirrorBound, Mixture, clean_need
from ._certificate import Certificate, EProcess
from ._multiple import ELOND, e_bh
from ._agreement import agreement_diagnostics
from ._checks import Check, CheckRegistry, error_correlation, NaiveCombiner, CorrelationAwareCombiner
from ._source_trust import SourceTrust, SourceOpinion
from ._episode_priors import EpisodePriors, EpisodePrior

__all__ = ["MeanBound", "Mixture", "MirrorBound", "DeltaBudget", "Certificate", "EProcess", "clean_need", "ELOND", "e_bh", "agreement_diagnostics"]

__all__ += ["Check", "CheckRegistry", "error_correlation", "NaiveCombiner", "CorrelationAwareCombiner"]
__all__ += ["SourceTrust", "SourceOpinion"]

__all__ += ["EpisodePriors", "EpisodePrior"]
