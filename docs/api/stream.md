# `solvi.experimental.stream`

`StreamPromise`: an anytime-valid learn-then-test certificate on a threshold grid, earned on the stream (stages 1–2
of 4): one e-process per grid point (a mixture of constant bets, or the aGRAPA bet), fixed-sequence from strict to
loose, the threshold only loosens, a missing label counted as a loss; `DeltaBudget` spends δ over restarts
(δ_v = 6δ/(π²v²)). Installed with `System.guarantee(question, promise=sp)`; `RiskDetector` and the calibration coin
(`sampling`) run inside `Dispatcher(audit="adaptive", promise=sp, detector=det)`; `tickets` (stage 3a) makes an
agent the oracle over MCP — `TicketDesk`, `AgentOracle`, `solvi serve --mcp --agent-oracle`; `versions` (stage 3b)
is the closed learning loop — `LearningLoop` over a `learner` (`HashedTextLearner`, `FastHeadLearner`). Measured once,
with documented limits (see [experimental](../experimental.md#stream)).

::: solvi.experimental.stream

## `solvi.experimental.stream.certificate`

`StreamPromise` and `DeltaBudget`: the certificate itself — one e-process per grid point, fixed-sequence from strict to
loose, the threshold only loosens; δ spent over restarts.

::: solvi.experimental.stream.certificate

## `solvi.experimental.stream.detector`

`RiskDetector`: a Shiryaev–Roberts e-detector on the deployed risk; its bound is an expected run length to a false
alarm, not a per-horizon probability.

::: solvi.experimental.stream.detector

## `solvi.experimental.stream.sampling`

The calibration coin (keyed on the question, run and decision number, never on the input), the sampling rate, and the
per-decision step the Dispatcher runs.

::: solvi.experimental.stream.sampling

## `solvi.experimental.stream.tickets`

::: solvi.experimental.stream.tickets

## `solvi.experimental.stream.versions`

::: solvi.experimental.stream.versions

## `solvi.experimental.stream.learner`

::: solvi.experimental.stream.learner

## `solvi.experimental.stream.policy`

The adapter loop's policy (stage 1, not wired into a loop yet): `LoopPolicy`, the abstract class whose hooks decide
what the loop takes, when it trains, what it freezes, promotes and discards and what an alarm does; `LoopState`, what
a hook reads; `Budget`; `DefaultPolicy`. Unmeasured.

::: solvi.experimental.stream.policy

## `solvi.experimental.stream.gates`

The content-key split and the gate arithmetic the default policy uses: the held-out gate, the honesty gate, label
mode's paired Wilson bound, the memorisation probe, the improvement signal.

::: solvi.experimental.stream.gates
