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

The adapter loop's policy (run by `AdapterLoop`, stage 2): `LoopPolicy`, the abstract class whose hooks decide
what the loop takes, when it trains, what it freezes, promotes and discards and what an alarm does; `LoopState`, what
a hook reads; `Budget`; `DefaultPolicy`. Unmeasured.

::: solvi.experimental.stream.policy

## `solvi.experimental.stream.loop`

The adapter loop (stage 2): `AdapterLoop`, the lane's loop driven by a `LoopPolicy` — labels split by content key,
online or batch learners, promotion on samples whose key the candidate never trained on with the memorisation probe
and a Wilson bound on the error there, the RiskDetector OR-ed with a score monitor, controls; a hook that raises after
the state changed stops the loop (status "error") with the decision recorded. Stage 3: a batch learner's job also
carries the calib labels, the budget, α and the live job status; a training refused by its time estimate is recorded
as "refused", a runner that dies stops the loop (status "error", replay exact). Stage 3b: deferred apply — a
batch learner's training runs in a thread of its own (its runner beside it) and `step` / `train_now` return at once;
the loop goes on deciding on the served version, and the result is applied on the first decision after it is in
(gates, then freeze or discard, a failure, a refusal or a stop), recorded there (`started_at`, `applied` in the
"train" event) so that `replay()` applies it at the same place without reading a clock. `wait=True` (the constructor's,
or `train_now(wait=True)`) keeps the blocking form; `close()` cancels a training in flight; `wait_training(timeout)`
waits for a result. Lock order: the part's version lock, then the loop's (no deadlock with a decision of the part that
reaches the loop). `Budget.max_versions_kept` evicts the oldest retired versions ("evict" events, `release()` called),
never the served one, the rollback target or version 0. No checkpoint or attach yet. Unmeasured.

::: solvi.experimental.stream.loop

## `solvi.experimental.stream.lora_learner`

The LoRA learner (stage 3): `LoraLearner`, a batch learner that trains a LoRA adapter on the decider from the base
checkpoint for each version, in a runner, and runs `act_guard` on the calib labels; `LoraVersion`, an adapter with its
calibration — scored in shadow on the scoring thread only, installed and rolled back with
`solvi.experimental.lora.swap` (the adapter and its thresholds change together under the part's version lock), saved
as `<v>.lora.safetensors` and `<v>.calib.json`. Needs `solvi[lora]`. Unmeasured.

::: solvi.experimental.stream.lora_learner

## `solvi.experimental.stream.runner`

Where a training runs (stage 3): `SubprocessRunner` (the default — nice 10, threads capped at `Budget.cpu_threads`, the
device picked by `pick_device` with a CPU fallback, the time estimate refused above `Budget.max_train_s`, a heartbeat
log, the result file) and `ThreadRunner` (tests, tiny data); `TrainingRefused`, `TrainingFailed`, `RunnerError` and
`TrainingCancelled` (`run(..., cancel=event)`: a subprocess is killed, a thread runner stops waiting). Unmeasured.

::: solvi.experimental.stream.runner

## `solvi.experimental.stream.serving`

How a frozen version is put in front of a System: the `Served` protocol, `HeadVersion`, `BaseVersion`.

::: solvi.experimental.stream.serving

## `solvi.experimental.stream.gates`

The content-key split and the gate arithmetic the default policy uses: the held-out gate, the honesty gate, label
mode's paired Wilson sign test (safe only together with the held-out gate), the memorisation probe, the improvement
signal, the fewest calib labels a calib threshold needs.

::: solvi.experimental.stream.gates
