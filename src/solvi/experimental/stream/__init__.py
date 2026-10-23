"""solvi.experimental.stream — a promise earned on the stream (experimental; stages 1–3a: the certificate, its detector,
the sampling inside the Dispatcher and the agent as the oracle).

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

Stage 2: `RiskDetector` (a Shiryaev–Roberts e-detector on the deployed risk, `stream.detector`) and the calibration
coin and rate (`stream.sampling`), run by `Dispatcher(audit="adaptive", promise=sp, detector=det,
on_drift="recertify", probation=300, secret=key)`: the coin draws the samples, the slow path labels them, an alarm
restarts the certificate and opens probation, every decision records its state and `Dispatcher.replay_stream()`
rebuilds it. Standalone, the caller still draws the samples and calls `learn` and `restart`.

Stage 3a: `stream.tickets` — an agent outside the process as the oracle (`AgentOracle`, `TicketDesk`; `solvi serve
--mcp --agent-oracle`): blind tickets, durable in the store, an expired ticket a recorded loss.

Unmeasured: no stand has run yet. The learning loop is stage 3b."""
from __future__ import annotations

from .. import warn_on_import
from ._eprocess import DeltaBudget, MirrorBound, Mixture, clean_need
from .certificate import StreamPromise, fixed_grid
from .detector import RiskDetector
from .sampling import Rate, StreamDriver, coin_message
from .tickets import AgentOracle, TicketDesk, TicketError

warn_on_import(__name__)

__all__ = ["AgentOracle", "DeltaBudget", "MirrorBound", "Mixture", "Rate", "RiskDetector", "StreamDriver", "StreamPromise",
           "TicketDesk", "TicketError", "clean_need", "coin_message", "fixed_grid"]
