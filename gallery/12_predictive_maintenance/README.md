# 12 · Predictive maintenance

Forty-eight hours of sensor readings from a pump go in: bearing temperature, vibration RMS and motor current. Out come
**health** (ok / watch / service / stop) and the **likely fault** (none / bearing / imbalance / electrical / cooling).

`hard checks` `trace replay` `readable learned rules` `abstains` `runs in browser`

## How it decides

- **Computed signals.** For each sensor: the last three readings averaged, a z-score against the machine's own baseline,
  and a **least-squares slope** over the last 24 hours. When vibration is rising, `hours_to_vib_alert` gives the time
  left until the ISO 10816 alert level of 4.5 mm/s at the current rate.
- **Health rule.** Service when any signal is 4 or more sd off baseline, or when the alert level is at most 72 hours
  away. Watch when a signal is 2 or more sd off, or rising. Otherwise ok.
- **Two hard safety checks** force `stop`: vibration of 7.1 mm/s or more, and bearing temperature of 95 °C or more.
  They are checkpoints of `health`, so no rule and no missing sensor can delay them.
- **An offline sensor is missing, not normal.** `prepare` drops a signal that contains nulls. The strategist then marks
  what depends on it as impossible to compute, and those answers abstain.
- **"Likely fault" is learned as a readable rule list.** `run.py` gives `learn_rule` 150 labelled past incidents
  (synthetic, with 5 % wrong labels, as in real maintenance logs). The resulting list is installed as the question's
  rule. In the playground, without the incidents, this question abstains.

## Run

```bash
uv run python gallery/12_predictive_maintenance/run.py
```

`task.py` with `state.json` loads as a playground preset (plain Python + numpy, no threads or network); `task.py`
and `run.py` were also run unchanged under Pyodide 0.27.2 (every case passed).

`cases.json` has 10 machines: healthy, bearing wear (service), the same wear early (watch, 78 h to the alert), an
imbalance step, a vibration trip, a cooling failure, an electrical overload, a current sensor offline (abstain), a trip
while that sensor is offline, and a temperature trip. `state.json` is the bearing-wear pump.

## Sample output (real run)

```
'fault' learned from 150 labeled incidents in 303 ms: 6 rules
 1. if vib_trend has 'RISING' → bearing   (31/32)
 2. if temp_level has 'ELEVATED' → electrical   (19/20)
 3. if vib_level has 'HIGH' → imbalance   (27/30)
 4. if temp_trend has 'FLAT' → none   (27/28)
 5. if current_level has 'NORMAL' → cooling   (34/34)
 6. if vib_trend has 'FLAT' → electrical   (6/6)
else → cooling
accuracy on 300 new incidents (true fault): 0.957

[2/10] bearing_wear_service  (vibration climbing for 30 hours and accelerating; plan the bearing change)
    health  service   ok       1.00  vib_z = 8.37; temp_z = 1.8; current_z = -0.03; hours_to_vib_alert = 7.4; vib_trend = 'rising'; temp_trend = 'flat'
    fault   bearing   ok       1.00  vib_level = 'high'; vib_trend = 'rising'; temp_level = 'normal'; temp_trend = 'flat'; current_level = 'normal'
    vib 3.893 mm/s (z 8.37, slope +1.96/day) · temp 65.6 °C (z 1.8, slope +3.3/day) · current 40.967 A (z -0.03) · vib alert in 7.4 h
    fault rule fired: if vib_trend has 'RISING' -> bearing

[3/10] bearing_early_watch  (the same wear pattern, early: rising, but the alert level is more than 72 hours away)
    health  watch     ok       1.00  vib_z = 2.08; temp_z = 0.5; current_z = 0.06; hours_to_vib_alert = 78.0; vib_trend = 'rising'; temp_trend = 'flat'

[9/10] trip_with_sensor_offline  (current still offline, but vibration is 7.8 mm/s: missing data does not delay a stop)
    health  stop      forced   1.00  hard check vibration_below_trip is false
    fault   abstain   abstain  0.00  cannot compute: current_level

10/10 cases as expected; decision time median 1.87 ms, max 3.10 ms
```

Rule 6 is a tie: several literals cover the same six incidents, and which one is printed can vary between runs (string
hashing). Accuracy and every case answer do not change.

## vs an answer-only model

An answer-only model looks at the three series and returns `service (0.81)`: no citation, no rules.

- **A planner needs the number, not the label.** "Vibration reaches 4.5 mm/s in 7.4 h at +1.96 mm/s per day" is what
  lets someone schedule a crew. Here it is a computed fact from a least-squares fit, and the watch/service line is
  explicit: 78 h is watch, 72 h or less is service.
- **Stops are guaranteed.** At 7.1 mm/s or 95 °C, the answer is `stop` for every input, even with the current sensor
  offline (case 9). A model's probability of "stop" is high, not certain.
- **It abstains instead of assuming normal.** A dead current clamp makes health and fault abstain and name `current_z` /
  `current_level`. A model fed a gap, a zero or an imputed value still answers.
- **The learned part can be read and corrected.** The fault classifier is six lines, 0.957 on 300 new incidents, and
  every answer names the rule that fired. The list also shows its weak spots: rule 2 recognises electrical faults by
  moderate heat, not by current, and rule 5 calls anything with normal current "cooling" once rules 1-4 have passed. An
  engineer can see this and fix the labels or the catalog. The weights of an answer-only classifier cannot be reviewed
  this way.
- **Replay.** Every z-score, slope and threshold decision is recomputed from the recorded readings, so a post-incident
  review can confirm what the system knew and when.

## Files

`task.py` (catalog, questions, `prepare`), `state.json`, `cases.json` (10 machines), `run.py` (incident history,
`learn_rule`, cases). Sensor data and incidents are synthetic; the thresholds are illustrative, not a maintenance
standard.
