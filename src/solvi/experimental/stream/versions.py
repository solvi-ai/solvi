"""The closed learning loop: frozen versions, each with its own stream certificate, promoted under δ-spending
(experimental; stage 3b).

    from solvi.experimental.stream import HashedTextLearner, LearningLoop

    loop = LearningLoop(HashedTextLearner(), question="intent", max_risk=0.05, delta=0.05, secret=key,
                        storage=store, initial=model)
    out = loop.step(text, oracle)          # oracle: () → the label (None: missing), called only when one is needed
    out["answer"], out["by"]               # "local" (the served version alone), "oracle", or "person" (no label)
    loop.versions()                        # [{"version", "state", "certified", "threshold", ...}]
    loop.replay()                          # every decision and event rebuilt from the records and compared

Additive coverage is the default: it needs stream mode and an `initial` model (version 0, above), as in the example;
`additive=False` is the replacement protocol, which lost on the stand (worse than no loop by 11–16 points on 3 of 4
intent datasets; additive coverage gained on 4 of 4).

Versions. Version 0 is "no model" by default: it answers nothing alone, has no certificate and spends no δ (it never
certifies). With `initial=` (opt-in) version 0 is an existing frozen model — the fitted System 1, e.g.
`FrozenFast.of_system(system, question)` over its fitted head — with its own `StreamPromise` (the first (v, δ_v) of the
budget), fed from decision 1 (its freeze is before the stream); its fingerprint is the "start" event of decision 1. The
live learner starts empty either way. The live learner (`stream.learner.Learner`) takes every accepted oracle label and
is never served. When no candidate exists and the live learner has taken ≥ max(freeze_min, growth × the served version's
labels at its freeze, the refreeze floor) labels (300, 2: constants from the adapted design), a snapshot is frozen as the candidate, with
its own `StreamPromise` drawing the next (v, δ_v) from the loop's one `DeltaBudget` at its first sample (lazy charge;
Σ δ_v ≤ δ over every version ever frozen). Each decision is scored by the served version and, in shadow, by the candidate, before its label is read.

Calibration. A version's promise reads only samples drawn after its freeze: the samples of decision n go to the
served version's promise and the candidate's — each frozen before n — and only then does the label reach the live
learner; a version frozen at decision n first reads decision n + 1. Nothing is ever scored by a model that saw the
label. Samples are drawn by the calibration coin of stage 2 — HMAC(secret, "sample|question|seed|n"), no input
bytes — at the rate from the past (1 while the served version is not certified, else min(1, max(floor, budget /
the share answered alone))); a decision the served version does not answer alone goes to the oracle anyway and its
label trains the live learner, but it reaches a promise only when the coin drew it (escalation-only calibration
breaks the promise: the stage-2 control).

Promotion. Once the candidate is certified, on the candidate's own samples (all after its freeze and after the last
alarm): ℓ_k = the share with the candidate's score ≥ its threshold, ℓ_s = the share with the served version's score
≥ the threshold of its comparison certificate — a paired comparison on the same samples, both versions certified the
same way: the comparison certificate is a `StreamPromise` for the served version on exactly the candidate's decisions
and samples, same grid rule, the candidate's δ_v (1.4.0; before, the served version's own threshold, from an older
certificate loosened at a larger δ_v, so a fresh candidate lost on the certificate's age). It only ranks and spends no
δ: nothing is answered at its threshold. Promote when ℓ_k ≥ `min_share` (0.05) AND
ℓ_k ≥ ℓ_s + `margin` (0.01: one point) — the author's rule (2026-10-23): under the joint promise a model of noise
certifies at a strict threshold that almost nothing passes, and with margin 0 over a version that answers nothing it
was promoted; the floor keeps such a version out (the shuffled-label control: shuffled labels never promote, both kinds). Discard a candidate not certified `stale` decisions
after its freeze (3 000) or certified and not better `trial` decisions after it (default: stale) — decisions, not coin
samples (1.4.0; before, 1 500 samples, at ρ ≈ 0.1 more decisions than a stream holds); the refreeze floor becomes `refreeze` × the
live learner's labels (1.5). A discarded version's δ_v stays spent. Every version's certificate holds together under the
budget, so the rule only affects utility, not validity.

The detector (opt-in, `detector=RiskDetector(alpha=<the promised rate>)`, stage 2's class). Each sampled decision
whose served version had a threshold in force, outside probation, feeds it the deployed loss as the Dispatcher's
driver does (joint: answered alone and wrong; selective: wrong, fed only when answered alone; a missing label is
wrong). On an alarm: a "stream" event "alarm" (R, the detector's n, why), the detector is reset, the served
version's promise restarts (the next (v, δ_v) from the loop's budget at its first post-alarm sample, threshold
+inf), and so does the candidate's,
certified or not, its paired samples dropped: the alarm is evidence that the conditional loss moved, so a certificate
earned before it is about another regime (seen in the stage-4 smoke: a candidate certified before a shift was promoted
at the alarm, ℓ_s being 0, and served the moved world at a true risk above α). Each restart is a "restart" event (why
"alarm"). A retired version keeps its certificate until it is rolled back to; a rollback after an alarm restarts it
(a retired version reads no samples, so restarting it earlier would only spend δ). Probation opens: the next
`probation` (300) decisions are not answered alone whatever the threshold (ρ = 1), and after that until the served
version is certified again — the stricter of the two. Probation belongs to the question, not the version: a certified
candidate is promoted at once (ℓ_s is 0 under +inf) but answers alone only after probation. A promotion or a rollback
resets the detector.

Labels. Accepted by default: the trusted sources ("human", "outcome", "rule", "audit"; solvi.core.sources). A slow
path's answer ("s2", or "verified") is refused (UntrustedLabel, before anything changes) unless `s2_labels=True`; with
it the label is taken and recorded as "s2". The loop is not a `Head` a System's `teach` reaches. The oracle may return
a `Verdict(label, source, label_of)` instead of a bare label: its source replaces step()'s and label_of (the stored id
of the record that holds the label) is recorded. `ticket_verdict(desk, resolved)` reads one from a ticket resolved on
a stage-3a `TicketDesk` (the agent's verdict: source "s2", so `s2_labels=True`; a person's: "human"; an expired or
refused ticket: no label).

In the chain. Each decision is a record of kind "loop": "stream" (kept by a redaction) holds n, the served and candidate
versions, ρ, sampled, alone, by, label_source, label_of, probation (at issue) and the decisions of it left, the served
score and threshold, the candidate's shadow score, both promises' states, the detector's state and the live learner's
labels; "loop" (not kept) holds the input, the would-be answers, the label and the answer. Each state change is a record
of kind "stream" ({"event": "freeze" | "promote" | "discard" | "rollback" | "start" | "alarm" | "restart" | the
promises' "grid" / "certified" / "loosened" / "recertify", "question", "version", "v_index", "delta_v", ...}); a freeze
carries the frozen model's fingerprint, its decision and the labels it had seen. With `versions_dir=`, each frozen
model is pickled as `<versions_dir>/<version>.pkl`. `replay()` re-runs the protocol from the records alone (the coin recomputed with the
secret, the scores read from the records, the live learner replaced by its label count) and compares every decision —
which version served it and the certificates' states — and every event; with `versions_dir` each pickle's fingerprint is
checked and, with `rescore=True`, every recorded score is recomputed by the frozen model that gave it.

With `system=` a promoted or rolled-back version is installed as the question's answer head
(`solvi.core.versions.serve_head`; version 0 puts back the head the System had when the loop was built, or removes
it), so the System's fingerprint and every trace name it.

Since the adapter loop's stage 2 `LearningLoop` is `stream.loop.AdapterLoop` driven by `DefaultPolicy` with the
constants above (`lane_policy`: every label trains — calib and holdout shares 0 —, no pre-freeze gates, no
memorisation probe, no novel-risk bar, the refreeze floor, no budget limit, no alarm rollback) and the detector as its only alarm source
(the lane's behaviour; `AdapterLoop` itself never takes the detector alone). Its records gain the decision's content
key, split and wall clock, and a "train" event per freeze. The paired comparison counts only samples whose key the
candidate never trained on; on streams without repeated inputs that is every sample, as before.

Not built here: the Dispatcher's `loop=` (the loop drives its own coin and oracle; a ticket's verdict reaches it
through `ticket_verdict`). Unmeasured: the gain is unproven (solvi's prior evidence is against
it); the stand (stage 4) measures it."""
from __future__ import annotations

from .loop import S2_SOURCES, AdapterLoop, Verdict, ticket_verdict
from .policy import Budget, DefaultPolicy


def lane_policy(*, freeze_min=300, growth=2.0, stale=3000, trial=None, min_share=0.05, margin=0.01, refreeze=1.5,
                s2_labels=False) -> DefaultPolicy:
    """The lane LearningLoop's protocol as a DefaultPolicy: every label trains, every training freezes (the stream
    certificate decides), promotion on certified and ℓ_k ≥ min_share and ℓ_k ≥ ℓ_s + margin, discard `stale`
    decisions after the freeze uncertified or `trial` decisions certified (not coin samples), retrain at ≥ max(freeze_min, growth × the served version's labels,
    refreeze × the labels at the latest discard), no training budget, no alarm rollback."""
    return DefaultPolicy(Budget(per_day=2 ** 62, min_interval_s=0.0), s2_labels=s2_labels, calib=0.0, holdout=0.0,
                         first=int(freeze_min), growth=float(growth), refreeze=float(refreeze), min_novel=1,
                         min_share=min_share, margin=margin, stale=int(stale),
                         trial=int(stale) if trial is None else int(trial), probation=0, probe=False,
                         pre_freeze=False, novel_risk=False)


class LearningLoop(AdapterLoop):
    """See the module docstring. learner: the live Learner; question: the question's name (in the coin and the
    records); max_risk= / max_error=, delta: each version's StreamPromise (promise_options: its other keywords — grid,
    grid_from, bets ...); freeze_min, growth, stale, trial, refreeze: the protocol's constants; min_share, margin: the
    promotion rule (ℓ_k ≥ min_share and ℓ_k ≥ ℓ_s + margin); initial: version 0's frozen model (None: no model);
    detector, probation: a RiskDetector on the served version and the decisions escalated after its alarm;
    s2_labels: take slow-path labels; source: the default source of the oracle's labels; secret, seed: the coin's key
    and run; sampling: {"floor", "budget", "window"} of the rate; storage: a TraceStorage for the records (None: kept
    in memory for replay); versions_dir: where frozen models are pickled; system, feature: install the served version
    as the question's answer head (feature: the text fact a text model reads)."""

    def __init__(self, learner, *, question, max_risk=None, max_error=None, delta=0.05, freeze_min=300, growth=2.0,
                 stale=3000, trial=None, min_share=0.05, margin=0.01, refreeze=1.5, initial=None, detector=None,
                 probation=300, s2_labels=False, source="human", secret=None, seed=0, sampling=None,
                 promise_options=None, storage=None, versions_dir=None, system=None, feature="text", additive=True,
                 max_layers=3):
        policy = lane_policy(freeze_min=freeze_min, growth=growth, stale=stale, trial=trial, min_share=min_share,
                             margin=margin, refreeze=refreeze, s2_labels=s2_labels)
        super().__init__(learner, question=question, policy=policy, max_risk=max_risk, max_error=max_error,
                         delta=delta, initial=initial, detector=detector, monitor=None, probation=probation,
                         source=source, secret=secret, seed=seed, sampling=sampling, promise_options=promise_options,
                         storage=storage, versions_dir=versions_dir, system=system, feature=feature,
                         additive=additive, max_layers=max_layers, _detector_alone=True, _batch=False)

    freeze_min = property(lambda self: self.policy.first)
    growth = property(lambda self: self.policy.growth)
    stale = property(lambda self: self.policy.stale)
    trial = property(lambda self: self.policy.trial)
    min_share = property(lambda self: self.policy.min_share)
    margin = property(lambda self: self.policy.margin)
    refreeze = property(lambda self: self.policy.refreeze)
    s2_labels = property(lambda self: self.policy.s2_labels)

    @property
    def refreeze_floor(self) -> float:
        """refreeze × the live learner's labels at the latest discard (0 before any)."""
        last = max((v["discarded_at"] for v in self._vs.values() if v["discarded_at"] is not None), default=None)
        return 0.0 if last is None else self.policy.refreeze * last


__all__ = ["LearningLoop", "S2_SOURCES", "Verdict", "lane_policy", "ticket_verdict"]
