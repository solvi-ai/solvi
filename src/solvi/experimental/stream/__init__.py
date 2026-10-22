"""solvi.experimental.stream — a promise earned on the stream (experimental; stage 1: the certificate alone).

    from solvi.experimental.stream import StreamPromise, DeltaBudget

    sp = StreamPromise(max_risk=0.05, delta=0.05)            # joint: P(answered alone and wrong) ≤ 5 %
    system.guarantee("intent", promise=sp, signal="act")      # installed as OpenSetGate is; +inf until certified
    sp.learn(score, would_be_answer, label)                   # one labelled sample (label None: missing, a loss)
    sp.threshold                                              # the loosest certified grid point (only loosens)
    sp.restart("alarm")                                       # next (v, δ_v) from the budget; uncertified again

An anytime-valid learn-then-test certificate on a threshold grid: one e-process per grid point (a mixture of constant
bets, or bets="agrapa" — `experimental._evalue.MeanBound` mirrored), fixed-sequence over the grid from strict to loose,
δ spent over restarts by `DeltaBudget` (δ_v = `experimental.skills.Certificate.delta_at(v)`). See
`stream.certificate` for the statement and what is recorded. Clean samples a fresh certificate needs at α = δ = 5 %:
85 (v = 1), 116 (v = 2), 134 (v = 3); 74 for the mixture at the full δ, against 59 for a one-shot binomial test.

Unmeasured: no stand has run yet. The sampling coin, the change detector, the Dispatcher wiring and the store records
are later stages; until then the caller draws the samples and calls `learn` and `restart`."""
from __future__ import annotations

from .. import warn_on_import
from ._eprocess import DeltaBudget, MirrorBound, Mixture, clean_need
from .certificate import StreamPromise, fixed_grid

warn_on_import(__name__)

__all__ = ["DeltaBudget", "MirrorBound", "Mixture", "StreamPromise", "clean_need", "fixed_grid"]
