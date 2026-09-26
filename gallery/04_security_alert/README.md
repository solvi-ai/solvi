# 04 · Security alert

A login event plus the account's recent login history goes in. Out come **suspicious?** and **action** (allow /
require_mfa / lock_account).

`hard checks` `strategist plan` `early exit` `trace replay` `abstains` `runs in browser`

```bash
uv run python gallery/04_security_alert/run.py        # 9 scenarios + a tampered trace, exits non-zero on a mismatch
```

## How it is built

- **Numbers are computed, not guessed.** The last successful login, the great-circle distance to it (haversine), the hours
  between the two logins, the implied travel speed (`impossible_travel` means over 500 km at more than 900 km/h), whether the
  device is new, and the spike in failed logins against the 30-day hourly average (in %).
- **Hard checks.** An admin account with impossible travel, or a login from a denylisted IP, forces suspicious=yes and
  action=lock_account. Hard checks and their inputs run first. When one fails, the device and spike analysis are skipped
  (early exit, 4 of 12 steps).
- **Rules** for everyone else: impossible travel from a new device → lock; impossible travel from a known device (a VPN), or a
  new device with a failed-login spike or without MFA → require_mfa; otherwise allow.
- **Abstains.** Without coordinates, the distance cannot be computed, so both answers abstain and name the missing facts.
- **The trace is the incident record.** Replay recomputes the whole trace. If someone edits the stored speed and recomputes
  every hash, replay still names the altered step.

## Sample output (real run)

```
[1] admin, Berlin -> Lagos in 2 h              0.97 ms   replay ok
    suspicious=yes (forced)  action=lock_account (forced)
    suspicious forced: hard check admin_travel_plausible is false
    action forced: hard check admin_travel_plausible is false
    computed: distance_km=5195.8, hours_between=2.25, travel_kmh=2309.2, impossible_travel=True
    early exit: 4 steps skipped
[3] impossible travel, known laptop (VPN)      0.83 ms   replay ok
    suspicious=yes  action=require_mfa
    computed: distance_km=9916.3, hours_between=1.25, travel_kmh=7933.0, impossible_travel=True, new_device=False, failed_spike_pct=260.0
[5] fast but possible (Amsterdam)              0.89 ms   replay ok
    suspicious=no  action=allow
    computed: distance_km=576.7, hours_between=1.5, travel_kmh=384.5, impossible_travel=False, new_device=False, failed_spike_pct=50.0
[6] failed-login spike + new device            0.80 ms   replay ok
    suspicious=yes  action=require_mfa
    computed: distance_km=0.0, hours_between=1.08, travel_kmh=0.0, impossible_travel=False, new_device=True, failed_spike_pct=360.0
[9] location unknown                           1.13 ms   replay ok
    suspicious=— (abstain)  action=— (abstain)
    suspicious abstain: rule not computed: missing inputs: impossible_travel; missing admin_travel_plausible, distance_km, impossible_travel, travel_kmh
    action abstain: rule not computed: missing inputs: impossible_travel; missing admin_travel_plausible, distance_km, impossible_travel, travel_kmh
    computed: hours_between=1.08, new_device=True, failed_spike_pct=160.0

9/9 cases match, replay ok on 9/9 traces, median decision 0.83 ms

early exit on 'admin, Berlin -> Lagos in 2 h': failed_spike_pct (not needed: hard check admin_travel_plausible failed), new_device (...), answer:suspicious (...), answer:action (...)

tampered trace of 'same login, regular user' (travel_kmh 2309.2 -> 80.0, hashes recomputed): {'ok': False, 'steps': 12, 'mismatches': [(4, 'travel_kmh', 'value 80.0 ≠ recomputed 2309.2'), (5, 'impossible_travel', 'input travel_kmh does not match the recorded one'), (5, 'impossible_travel', 'value True ≠ recomputed False')]}
```

## vs an answer-only model

An answer-only model reads the event and the history and answers directly. `compare_laya.py` measures this with
[Laya](https://huggingface.co/convaiinnovations/laya). Laya has no preset for login alerts, so both questions are written in its
format, with our thresholds and denylist spelled out in the instructions, and the case JSON is passed as the state.

| | solvi | Laya (zero-shot, RTX 3060) |
|---|---|---|
| correct answers on the 9 logins | **18 / 18** | 8 / 18 |
| what it answered | follows the computed facts | suspicious=yes on all 9 (0.58–0.81), action=allow on all 9 |
| Berlin → Lagos, admin account | lock_account, forced | allow (0.39) |
| denylisted IP | lock_account, forced | allow (0.46) |
| no coordinates | abstains, names the missing facts | suspicious yes (0.72), allow (0.44) |
| 2 309 km/h, 360% spike | computed, in the trace | not produced |
| median time per login | 1.05 ms, CPU | 78 ms, GPU |

Full per-case output of the run above: [compare_laya.out.txt](compare_laya.out.txt).

- **Haversine and a percentage are arithmetic, not language.** Laya's answers did not change with the numbers: yes and allow
  whatever the distance, speed or spike. solvi computes 5 195.8 km in 2.25 h = 2 309 km/h and decides from that.
- **The admin lock is guaranteed.** It is a hard check, so no other signal can talk it down to require_mfa or allow.
- **You can defend the decision later.** Every number that led to a lock is in a hash-chained trace. Replay recomputes it, and
  if the stored speed is edited to 80 km/h, replay names step 4 even after all hashes are recomputed. A model's probability
  cannot be re-derived like that.
- **Unknown location gives an abstain, not a guess.**

Caveats: we wrote the 9 cases alongside the catalog. The comparison asks Laya zero-shot, outside its presets, so it is a
measurement of an answer-only model asked to do arithmetic. It is not a claim about what Laya does after fine-tuning. Reproduce
from the new_kelly repo:
`cd exps_v2 && uv run --with laya==0.3.20 --with transformers --with-editable ../solvi python ../solvi/gallery/04_security_alert/compare_laya.py`
