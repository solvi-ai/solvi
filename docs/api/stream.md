# `solvi.experimental.stream`

`StreamPromise`: an anytime-valid learn-then-test certificate on a threshold grid, earned on the stream (stages 1–2
of 4): one e-process per grid point (a mixture of constant bets, or the aGRAPA bet), fixed-sequence from strict to
loose, the threshold only loosens, a missing label counted as a loss; `DeltaBudget` spends δ over restarts
(δ_v = 6δ/(π²v²)). Installed with `System.guarantee(question, promise=sp)`; `RiskDetector` and the calibration coin
(`sampling`) run inside `Dispatcher(audit="adaptive", promise=sp, detector=det)`. Unmeasured (see
[experimental](../experimental.md#stream)).

::: solvi.experimental.stream
