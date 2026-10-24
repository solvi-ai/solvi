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

`DefaultPolicy`'s improvement signal reads coverage as well as error: with `target_share=0.5` (the default) the served
version is short when it answers alone on less than half of its last `window` held-out labels, or errs alone there
above α. On error alone a weak but certified incumbent would never be retrained — certification keeps its error alone
at most α, so it is short in coverage, not in error. The share is read on held-out labels, a selection-biased sample
that is mostly deferred decisions, so it reads low (0.09 there against 0.30 on the whole stream in one smoke run whose
script is not in the repository); a strong incumbent will usually be retrained too, and the promotion gates (the
novel-only paired comparison, the memorisation probe, the novel-risk bar) keep a worse version out.
`target_share=None` restores the error-only signal. Unmeasured; see [experimental](../experimental.md#stream).

`DefaultPolicy`'s `stale` and `trial` (default 3 000 each) count **decisions since the candidate's freeze**
(`LoopState.since_freeze`), not coin samples — changed in 1.4.0. A candidate not certified `stale` decisions after its
freeze, or certified and not promotable `trial` decisions after it, is discarded, and training can start again. At a
sampling rate of 0.08–0.17 the old 1 500 samples were 9 000–18 000 decisions, so a stalled candidate blocked training
for a whole stream. 3 000 leaves room to certify: the grid opens 200 decisions after the freeze and a point with
0–1 % risk needs about 120–280 samples at 6–12 decisions per sample (arithmetic from the certificate's constants, not
measured). The clock runs from the freeze; an alarm's restart does not reset it.

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
never the served one, the rollback target or version 0. A result whose training saw the served version change (a
rollback) is recorded "stale" and discarded, not gated on a comparison with a version no longer served
(`served_at_start` in "train_start" and "train"). A batch learner's `train(job)` runs on another thread than
`update`, `seen` and `keys`: it must read the job only (or its own state under its own lock). Stage 4: `save(dir)` /
`AdapterLoop.load(dir, learner, ...)` and `cancel_training()` (see `stream.checkpoint`). Stage 5: attach.
`AdapterLoop.attach(host, learner=None, *, question=None, dir=None, policy=None, initial="auto", part=None, **kw)`
builds a loop and attaches it — to a `Dispatcher` (stream mode: every `ask` is one decision of the loop) or to a
`System` alone (label mode, `mode="label"`: `teach` / `outcome` call `label(x, y)`; no coin, no promise, no alarm; a
version serves at its own act_guard threshold; a frozen candidate is promoted when it has that threshold and
`policy.promote(st, None)` says so — the `promote()` control and the automatic promotion check the same; `DefaultPolicy(
pre_freeze=False)` is refused, since the pre-freeze gates are the only measurement before a version is served;
`status()["guarantee"]` is "fixed-set (act_guard on calib labels); no anytime certificate"). Label mode's watch: from a
promotion until the served version changes again, each calib or held-out label scores the served version and the one
served before it on the same input (+1 alone and right, −1 alone and wrong, 0 not alone); `policy.watch(st, Watch)`
(`DefaultPolicy`: `watch_rollback` — ≥ `watch_min` (100) labels, the e-process `policy.watch_eprocess()` (a
`Mixture(0.5)` fed 1 when the served version wins a discordant label, 0 when the earlier one does) at wealth
≥ 1 / (1 − `wilson`), better > worse, and the margin per label ≥ `min_gain`; the base `LoopPolicy.watch` returns
False: opt-in) returns True to roll back to the earlier version: a "rollback" event with `by: "policy"`, why "watch: …"
and the counts under `watch`, replayed from the recorded scores; `status()["watch"]` shows the counts and the
e-process's log wealths (`lw`; `Watch.log_e`), saved by the checkpoint. Anytime-valid: P(a false rollback) ≤
1 − `wilson` per promotion at any horizon (Ville); simulated, not a stand: false rollbacks with both versions equally good 0.4–1.4 % within 300 watched labels,
0.5–1.9 % within 1 000 and 0.8–2.1 % within 3 000; a served version truly worse (it wins 40 % of the discordant
labels, 10 % of the labels discordant) rolled back in 10 / 28 / 70 % of runs by those horizons, a truly better one in
≤ 0.2 %; watched labels are the calib and held-out ones, about 30 % of all labels at the default split. One more scoring per calib or
held-out label while it runs. Without a learner it is a
`LoraLearner` on the question's decision part (dir= required). `loop.attach_to(host)` attaches a loop built by hand;
`Dispatcher(..., loop=loop)` calls it. `detach()` cancels a training in flight, puts version 0 back on the System,
removes the hooks and closes the loop ("detach" event, replayed). `serving(v)`: version v installed for the duration
(the Dispatcher's replay). Refused: a question answered by a combination or a part that is a member of one, a
Dispatcher with its own `promise=`, a second loop on a System or a Dispatcher, `start` / `complete` with a loop (use
`ask`), a stream-mode attach whose `DefaultPolicy` refuses the slow path's labels (the slow path's `label_source`
attribute, "s2" when it has none) — the loop would have no label source and every decision would escalate: give
`s2_labels=True`, a trusted oracle as the slow path (its Thoughts carry `{"oracle": {"label_source": "human"}}`), or use
label mode; refused before any state is set; at run time the labels the policy refuses are counted
(`status()["labels"]["refused"]`) and `warn_refused` (200) of them in a row with none accepted warn once — and, while a loop serves a part, a Cascade,
Vote or Route built over it (refused at construction until `detach()`). In stream mode the decision is split around the dispatch: the loop reads it (scores, coin, whether its rule
answers alone) before System 1 is asked and takes its step after, with the slow path's accepted answer as the label —
neither inside the part's call. A Dispatcher's loop decisions run one at a time under its own lock, taken before the
part's and the loop's and never while either is held (order: dispatcher → part → loop), the slow path's call included:
throughput ≤ 1 / the slow path's latency (measured once on a stand-in: 40 decisions in four threads with a 50 ms slow
path, 3.7 × as long as without a loop). The controls take the same lock first: a control waits for the decision in
flight. The candidate is scored in shadow only while its grid is not fixed and on coin-sampled decisions (`shadow` is
None in the record otherwise). The dispatch escalates what the
loop's rule does not answer alone and checks what its coin draws; each `Dispatched` records `learning` (the loop's
decision number, the version served and its fingerprint, the candidate, the coin, whether the loop's rule answered
alone, the label it took, its events); `Dispatcher.replay` / `replay_all` replay each decision with its version
installed. A label whose source the policy refuses is a missing label for the loop (the default policy refuses the
slow path's "s2" unless `s2_labels=True`). A loop that stops or fails leaves the dispatch as it would be without it
(`learning: {"stopped": ...}` or `"error"`). `teach` in stream mode only stores the correction; without storage a
UserWarning says it is lost. Unmeasured.

::: solvi.experimental.stream.loop

## `solvi.experimental.stream.checkpoint`

The checkpoint (stage 4): `AdapterLoop.save(dir, cancel=False)` writes the loop's whole state — every version (its
state, fingerprint, paired samples, the keys it trained on, its `StreamPromise` through `dump()`) and the files of the
versions that still have a model (through `Served.save`, version 0 included), the δ budget's index, the training
budget's counters, the policy (now and as built), the `RiskDetector` (`dump()`), the score monitor, the rate's window,
the label store `labels.jsonl` and the records `records.jsonl` (or, with a `TraceStorage`, its chain head) — into
`dir/ckpt-<n>-<id>/`, then replaces `dir/CURRENT` atomically, so a crash leaves the previous checkpoint usable.
`manifest.json` lists every file's SHA-256 and is signed (`manifest.sig`, HMAC-SHA256 under the loop's secret,
SHA-256 without one). Threat model: the pickles in a checkpoint are run only after the HMAC and every file's hash are
verified, so with a secret a file changed or added by someone without it is refused; without a secret there is no
tamper protection against a writer of the folder (a plain SHA-256 anyone can recompute), and `load` warns
(`CheckpointTrustWarning`). `load` refuses a tampered, missing or unlisted file, a checkpoint saved with a secret
loaded without one, another secret, another learner class, a part with other weights, a version file
with another fingerprint, a store without the chain head or with records after it; the versions are loaded from the
verified bytes (a private copy), never from the folder again after the check. The learner is given fresh and fed
the saved train labels again. A training in flight refuses the save unless `cancel=True`. A loaded loop continues with
the same decisions and records as the loop it was saved from and `replay()` rebuilds every decision before and after
the load. A save holds the part's version lock and the loop's lock for all its writes, so the part's decisions wait
for it; one loop saves into a directory. Unmeasured.

::: solvi.experimental.stream.checkpoint

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
