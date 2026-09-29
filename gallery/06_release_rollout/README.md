# 06 · Release rollout

Canary metrics against the baseline go in: request and error counts, latency histograms, the SLO and the rollout
thresholds. Out come a **decision** (promote / hold / roll_back) and **page_oncall?**.

`hard checks` `strategist plan` `early exit` `trace replay` `abstains` `audited` `runs in browser`

```bash
uv run python gallery/06_release_rollout/run.py        # 9 scenarios, a narrower plan, early-exit timing
```

## How it is built

- **Numbers are computed from raw counts.** Error rates, the relative error increase (%), and a two-proportion z-test that
  says whether the increase is more than noise. Also p95 latency from Prometheus-style cumulative histograms (linear
  interpolation, as `histogram_quantile` does), the latency increase (%), and the SLO burn rate (how many times faster than
  allowed the canary spends the error budget).
- **Hard checks run first.** A canary error rate above the 2% ceiling forces roll_back and a page. Fewer than 2 000 canary
  requests forces hold (never promote on too little data). When the ceiling check fails, the histograms, the z-test and the
  latency work are not run: 9 steps are skipped, and the decision takes 0.36 ms instead of 0.82 ms (median of 200 runs).
- **Rule for the rest.** Roll back on a significant jump (+50% and z > 3), p95 over the SLO, or burn ≥ 10×. Hold on +20%
  errors, +15% latency, or burn ≥ 1×. Otherwise promote. Page when burn ≥ 3×.
- **Plan per question set.** Asking only `page_oncall` plans 4 steps instead of 12, with no histograms and no z-test.
- **Abstains.** A missing latency histogram (a delayed metrics pipeline) makes the decision abstain. The page decision needs
  only error counts, so it still answers.

## What the audit shows

`run.py` asserts the audit's invariants on every snapshot (see [`_audit.py`](../_audit.py)): 9/9 snapshots, 187 support items,
100% deterministic, hard check decided ×3. From `res.audit("decision")` on the error ceiling — one computed number and a hard
check, seven steps not run:

```
decision = 'roll_back'  [forced]  confidence 1.00  ← computed by canary_errors_under_ceiling
  computed    canary_error_pct = 3.163
  check       canary_errors_under_ceiling = False (hard, decides the answer)
  not run     baseline_error_pct, baseline_p95_ms, canary_p95_ms, error_increase_pct, error_z, latency_increase_pct, slo_burn_rate (not needed: hard check canary_errors_under_ceiling failed)
  support     7 items (4 given, 3 computed): 100% deterministic
```

## Sample output (real run)

```
[1] healthy canary                             1.25 ms   replay ok
    decision=promote  page_oncall=no
    computed: canary_error_pct=0.143, error_increase_pct=10.0, error_z=0.35, canary_p95_ms=266.7, latency_increase_pct=0.0, slo_burn_rate=0.29
[3] error rate above the ceiling               0.47 ms   replay ok
    decision=roll_back (forced)  page_oncall=yes (forced)
    decision forced: hard check canary_errors_under_ceiling is false
    page_oncall forced: hard check canary_errors_under_ceiling is false
    computed: canary_error_pct=3.163
    early exit: 9 steps skipped
[4] too little canary traffic                  0.51 ms   replay ok
    decision=hold (forced)  page_oncall=no
    decision forced: hard check enough_canary_traffic is false
    computed: canary_error_pct=0.125, slo_burn_rate=0.25
    early exit: 7 steps skipped
[6] latency up 27%, within SLO                 0.80 ms   replay ok
    decision=hold  page_oncall=no
    computed: canary_error_pct=0.122, error_increase_pct=-6.2, error_z=-0.2, canary_p95_ms=340.0, latency_increase_pct=27.5, slo_burn_rate=0.24
[7] small error bump, not significant          0.70 ms   replay ok
    decision=hold  page_oncall=no
    computed: canary_error_pct=0.184, error_increase_pct=41.5, error_z=1.42, canary_p95_ms=266.7, latency_increase_pct=0.0, slo_burn_rate=0.37
[9] latency histogram missing                  0.63 ms   replay ok
    decision=— (abstain)  page_oncall=no
    decision abstain: rule not computed: missing inputs: canary_p95_ms, latency_increase_pct; missing canary_p95_ms, latency_increase_pct
    computed: canary_error_pct=0.143, error_increase_pct=10.0, error_z=0.35, slo_burn_rate=0.29

9/9 cases match, replay ok on 9/9 traces, median decision 0.70 ms

asking only page_oncall: 4 steps instead of 12:
 1. fn      canary_error_pct         ← canary   [input for slo_burn_rate; input for canary_errors_under_ceiling]
 2. check   canary_errors_under_ceiling ← canary_error_pct, max_error_pct   [checkpoint page_oncall]
 3. fn      slo_burn_rate            ← canary_error_pct, slo   [rule page_oncall]
 4. rule    answer:page_oncall       ← slo_burn_rate   [answer page_oncall]
'healthy canary': median 0.821 ms over 200 runs
'error rate above the ceiling': median 0.357 ms over 200 runs
```

## vs an answer-only model

`compare_laya.py` gives the same 9 metric snapshots, as JSON, to [Laya](https://huggingface.co/convaiinnovations/laya). Laya has no
canary preset, so decision and page_oncall are written in its format, with our thresholds spelled out in the instructions.

| | solvi | Laya (zero-shot, RTX 3060) |
|---|---|---|
| correct answers on the 9 snapshots | **18 / 18** | 7 / 18 |
| what it answered | follows the computed numbers | page=yes on all 9 (0.72–0.85), roll_back on 7 of 9 |
| healthy canary (+10% errors, z 0.35) | promote, no page | roll_back (0.42), page yes (0.72) |
| +42% errors but z 1.42 | hold: not proven, not ignorable | roll_back (0.44) |
| latency histogram missing | decision abstains | promote (0.37) |
| p95 from histogram buckets, z-score, burn rate | computed, in the trace | not produced |
| median time per decision | 0.93 ms, CPU | 64 ms, GPU |

Full per-case output of the run above: [compare_laya.out.txt](compare_laya.out.txt).

- **A rollout gate is arithmetic.** A z-score, an interpolated percentile and a burn rate are formulas over counts. solvi
  computes them and every value is in the trace. Laya answered page=yes for every snapshot, including the healthy canary.
- **The ceiling is guaranteed and cheap.** Above 2% errors it is roll_back and a page, decided before the expensive work
  runs, and no other signal can promote it.
- **"Not enough data" is a real answer.** Too little canary traffic gives hold, forced. Missing latency data gives an
  abstain with the reason. An answer-only model promoted the snapshot whose latency data was missing.
- **The decision can be re-verified.** After an incident, replay recomputes the error rates and the p95 from the recorded
  counts and names any step whose stored value does not match.

Caveats: we wrote the 9 snapshots alongside the catalog. The thresholds (2% ceiling, +50% with z > 3, burn rates) are
example policy, so set your own. Laya was asked zero-shot, outside its presets. Reproduce from the repository root:
`uv run --with laya==0.3.20 --with torch --with transformers python gallery/06_release_rollout/compare_laya.py`
