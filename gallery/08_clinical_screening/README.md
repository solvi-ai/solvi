# 08 · Clinical screening: NEWS2, qSOFA, escalation

> **Demo only, not medical advice.** This entry is a software demonstration on synthetic patients. It is not a clinical
> tool, has not been validated, and must not be used to guide the care of anyone.

Vital signs (and a lactate result) go in. Three answers come out: the **NEWS2 clinical risk band** (low / low-medium /
medium / high), a **sepsis screen** (yes / no) and an **escalation level** (routine / urgent review / emergency).

`hard checks` `trace replay` `abstains` `runs in browser`

## How it decides

- **NEWS2 is computed from the published table** (Royal College of Physicians, 2017). There is one function per parameter:
  respiration, SpO2 (scale 1, or scale 2 for hypercapnic patients), air/oxygen, systolic BP, pulse, ACVPU and temperature.
  `news2` adds them up and lists any parameter that scored 3. The band follows the RCP thresholds: 0-4 is low, a single
  red parameter is low-medium, 5-6 is medium, 7 or more is high.
- **qSOFA** (Sepsis-3) counts RR ≥ 22, altered mentation and SBP ≤ 100. The sepsis screen is positive at qSOFA ≥ 2, or at
  qSOFA 1 with lactate ≥ 2 mmol/L.
- **Two hard checks** are checkpoints of `escalation`: SpO2 < 85 %, and SBP < 90 with altered consciousness. If either
  fails, the answer is forced to emergency, whatever the score.
- **A missing vital is never guessed.** `prepare` drops fields sent as `null`, the strategist marks the facts that
  depend on them as impossible to compute, and those answers abstain. A hard check that *can* still be evaluated still
  decides.

## Run

```bash
uv run python gallery/08_clinical_screening/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

`cases.json` has 10 patients: stable, a single red parameter, a COPD patient on scale 2, two patients 0.1 °C apart on
either side of the 4/5 boundary, septic, hypoxic (hard check), shocked with no SpO2 reading, no respiration rate, and a
patient with low NEWS2 but positive qSOFA. `state.json` is the hypoxic patient.

## Sample output (real run)

```
[4/10] four_points_routine  (on four band edges: RR 21 -> 2, SpO2 95 -> 1, pulse 91 -> 1, SBP 111 -> 0, temp 38.0 -> 0; total 4)
    escalation    routine       ok       1.00  news2 = {'total': 4, 'red': [], 'by_parameter': {'resp_rate': 2, 'spo2': 1, 'pulse': 1}}; qsofa = 0
[5/10] five_points_urgent  (the same patient 0.1 °C warmer: temperature 38.1 scores 1, total 5 = medium)
    escalation    urgent review ok       1.00  news2 = {'total': 5, 'red': [], 'by_parameter': {'resp_rate': 2, 'spo2': 1, 'pulse': 1, 'temperature': 1}}; qsofa = 0

[7/10] hypoxic_hard_check  (NEWS2 6 (medium) would mean urgent review; SpO2 83% trips the hard check and forces emergency)
    escalation    emergency     forced   1.00  hard check spo2_not_critical is false
    news2_band    medium        ok       1.00  news2 = {'total': 6, 'red': ['spo2'], 'by_parameter': {'spo2': 3, 'oxygen': 2, 'pulse': 1}}

[8/10] shock_spo2_missing  (no SpO2 (probe off): NEWS2 abstains, but SBP 84 with a V response is an emergency anyway)
    escalation    emergency     forced   1.00  hard check not_shocked is false
    news2_band    abstain       abstain  0.00  cannot compute: news2
    sepsis_screen yes           ok       1.00  qsofa = 3; lactate_high = True

[9/10] resp_rate_missing  (respiration rate not recorded: nothing is guessed, every answer that needs it abstains)
    escalation    abstain       abstain  0.00  cannot compute: news2, qsofa

── answer-only baseline (ridge on the 8 raw vitals, trained on 4 000 synthetic patients, tested on 2 000)
   fitted in 628 ms; escalation accuracy 0.763 (solvi's table + rules: 2000/2000 by construction)
   emergencies missed: 144 of 501; hard-rule emergencies (SpO2 < 85 or shock) missed: 6 of 273
   routine patients sent to emergency: 4

10/10 cases as expected; decision time median 1.02 ms, max 1.23 ms
```

## vs an answer-only model

- **NEWS2 is an exact table, and a learned answerer only approximates it.** We measured this: a ridge answerer
  (solvi's `fit_fast` on the eight raw vitals, with pairwise terms, but no table and no rules) was trained on 4 000
  synthetic patients labelled by the exact rules. On 2 000 new patients it gets 76.3 % of escalations right, misses 144
  of 501 emergencies, and sends 4 routine patients to emergency. The table and rules match the ground truth by
  construction. Band edges explain the gap: 38.0 °C scores 0 and 38.1 °C scores 1; SpO2 89 % scores 0 on scale 2 and 3
  on scale 1. A smooth model blurs exactly these steps. A larger model would narrow the gap, but it would still be an
  approximation of a table that can simply be computed.
- **The safety rules are guaranteed.** The same baseline missed 6 of 273 hard-rule emergencies (SpO2 < 85 or shock).
  With solvi the hard checks force emergency for every such input. In case 7, the table alone says urgent review; the
  hard check overrides it.
- **Every point is visible.** `by_parameter` shows where the score came from (`{'spo2': 3, 'oxygen': 2, 'pulse': 1}`),
  so a clinician can check it against the chart in seconds. An answer-only model returns "urgent review (0.71)".
- **Missing data leads to abstention, not a guess.** With no respiration rate, all three answers abstain and name what
  is missing. With no SpO2, the band abstains, but the shock rule still fires and the sepsis screen is still computed.
  An answer-only model must be given some value to fill each gap, and then it answers anyway.
- **The record can be re-verified.** `replay` recomputes every parameter score from the recorded vitals, so an audit can
  confirm that the escalation followed the table.

## Files

`task.py` (catalog, questions, `prepare`), `state.json` (default input), `cases.json` (10 patients), `run.py` (cases +
measured baseline). All patients are synthetic.
