"""solvi.experimental.stream — a promise earned on the stream (experimental; stages 1–3b: the certificate, its detector,
the sampling inside the Dispatcher, the agent as the oracle and the closed learning loop).

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

Stage 3b: `stream.versions.LearningLoop` — a live learner (`stream.learner`: `HashedTextLearner`, `FastHeadLearner`)
takes the oracle's labels; frozen versions, each with its own StreamPromise calibrated only on samples drawn after its
freeze and δ from one DeltaBudget, are promoted when certified and answering alone on ≥ min_share (5 %) of the same
samples and ≥ the served version's share + margin (1 point); version 0 may be the fitted System 1 (initial=), a
RiskDetector with probation watches the served version, a resolved ticket's verdict is a label (ticket_verdict); every
freeze / promote / discard / alarm is in the chain and `replay()` rebuilds which version served each decision. The gain is unproven; stage 4's stand measures it.

The adapter (LoRA) loop, stage 1: `stream.policy` — `LoopPolicy` (the abstract class whose hooks decide what the loop
takes, when it trains, what it freezes, promotes, discards and does on an alarm), `LoopState` (what a hook reads),
`Budget`, and `DefaultPolicy`; `stream.gates` — the content-key split and the gate arithmetic. Not wired into a loop
yet.

Unmeasured: no stand has run yet."""
from __future__ import annotations

from .. import warn_on_import
from ._eprocess import DeltaBudget, MirrorBound, Mixture, clean_need
from .certificate import StreamPromise, fixed_grid
from .detector import RiskDetector
from .policy import Budget, DefaultPolicy, LabelRow, LoopPolicy, LoopState, Paired, TrainReport, VersionInfo
from .learner import FastHeadLearner, Frozen, FrozenError, FrozenFast, HashedTextLearner, Learner
from .sampling import Rate, StreamDriver, coin_message
from .tickets import AgentOracle, TicketDesk, TicketError
from .versions import LearningLoop, Verdict, ticket_verdict

warn_on_import(__name__)

__all__ = ["AgentOracle", "Budget", "DefaultPolicy", "DeltaBudget", "FastHeadLearner", "Frozen", "FrozenError",
           "FrozenFast", "HashedTextLearner", "LabelRow", "Learner", "LearningLoop", "LoopPolicy", "LoopState",
           "MirrorBound", "Mixture", "Paired", "Rate", "RiskDetector", "StreamDriver", "StreamPromise", "TicketDesk",
           "TicketError", "TrainReport", "Verdict", "VersionInfo", "clean_need", "coin_message", "fixed_grid",
           "ticket_verdict"]
