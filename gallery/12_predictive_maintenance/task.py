"""Predictive maintenance: 48 hours of sensor readings from a pump (bearing temperature, vibration RMS, motor current) ->
health (ok / watch / service / stop) and the likely fault.

Each signal is compared with the machine's own baseline (z-score of the last three readings), and its trend is a least-squares
slope over the last 24 hours. A rising vibration gives the hours left until the ISO 10816 alert level (4.5 mm/s) at the
current rate. Vibration at or above 7.1 mm/s, or bearing temperature at or above 95 °C, is a hard safety check: stop now,
whatever the rest says. A sensor that is offline (null readings) is not guessed: the answers that need it abstain.
"Likely fault" has no hand-written rule: run.py learns a readable rule list from 150 labeled past incidents (learn_rule);
here, without them, it abstains.
Try: append 7.4 to vib_rms, or set current_a to null."""
from __future__ import annotations

import numpy as np

from solvi import Answer, Catalog, Question

cat = Catalog()

SIGNALS = ("temp_c", "vib_rms", "current_a")
VIB_ALERT = 4.5                 # mm/s, ISO 10816 class II zone C/D boundary: plan service before this
VIB_TRIP = 7.1                  # mm/s: stop the machine
TEMP_TRIP = 95.0                # °C bearing temperature: stop the machine
SERVICE_WITHIN_H = 72           # alert level reached within 3 days at the current rate -> service
FAULT_FACTS = ["vib_level", "vib_trend", "temp_level", "temp_trend", "current_level"]


def prepare(state):
    """Readings arrive as columns; a signal with any null (sensor offline) is dropped so the strategist sees it is missing."""
    r = state.pop("readings")
    state["hours"] = r["hours"]
    for s in SIGNALS:
        if r.get(s) is not None and all(v is not None for v in r[s]):
            state[s] = r[s]
    return state


def _last(y):
    return round(float(np.mean(y[-3:])), 3)


def _slope(hours, y):
    """least-squares slope per hour over the last 24 hours"""
    h, v = np.asarray(hours[-24:], float), np.asarray(y[-24:], float)
    return round(float(((h - h.mean()) * (v - v.mean())).sum() / ((h - h.mean()) ** 2).sum()), 5)


def _level(z):
    return "high" if z >= 4 else "elevated" if z >= 2 else "normal"


def _trend(slope, sd):
    day = slope * 24
    return "rising" if day > 2 * sd else "falling" if day < -2 * sd else "flat"


# ---------- per-signal facts
@cat.fn
def vib_latest(vib_rms):
    return _last(vib_rms)


@cat.fn
def temp_latest(temp_c):
    return _last(temp_c)


@cat.fn
def current_latest(current_a):
    return _last(current_a)


@cat.fn
def vib_slope(hours, vib_rms):
    return _slope(hours, vib_rms)


@cat.fn
def temp_slope(hours, temp_c):
    return _slope(hours, temp_c)


@cat.fn
def vib_z(vib_latest, baseline):
    return round((vib_latest - baseline["vib_rms"]["mean"]) / baseline["vib_rms"]["sd"], 2)


@cat.fn
def temp_z(temp_latest, baseline):
    return round((temp_latest - baseline["temp_c"]["mean"]) / baseline["temp_c"]["sd"], 2)


@cat.fn
def current_z(current_latest, baseline):
    return round((current_latest - baseline["current_a"]["mean"]) / baseline["current_a"]["sd"], 2)


@cat.fn
def hours_to_vib_alert(vib_latest, vib_slope):
    """hours until vibration reaches VIB_ALERT at the current rate (0 if already there, None if not rising)"""
    if vib_latest >= VIB_ALERT:
        return 0.0
    return round((VIB_ALERT - vib_latest) / vib_slope, 1) if vib_slope > 0 else None


# ---------- categorical signals (what the learned fault rules read)
@cat.fn
def vib_level(vib_z):
    return _level(vib_z)


@cat.fn
def temp_level(temp_z):
    return _level(temp_z)


@cat.fn
def current_level(current_z):
    return _level(current_z)


@cat.fn
def vib_trend(vib_slope, baseline):
    return _trend(vib_slope, baseline["vib_rms"]["sd"])


@cat.fn
def temp_trend(temp_slope, baseline):
    return _trend(temp_slope, baseline["temp_c"]["sd"])


# ---------- hard safety checks
@cat.check(hard=True, then={"health": "stop"})
def vibration_below_trip(vib_latest):
    """hard: vibration at or above 7.1 mm/s -> stop now"""
    return vib_latest < VIB_TRIP


@cat.check(hard=True, then={"health": "stop"})
def temperature_below_trip(temp_latest):
    """hard: bearing temperature at or above 95 °C -> stop now"""
    return temp_latest < TEMP_TRIP


# ---------- answers
@cat.rule("health")
def health(vib_z, temp_z, current_z, hours_to_vib_alert, vib_trend, temp_trend):
    """service: any signal 4+ sd off baseline, or the vibration alert level within 72 h; watch: 2+ sd off, or rising"""
    if max(vib_z, temp_z, current_z) >= 4 or (hours_to_vib_alert is not None and hours_to_vib_alert <= SERVICE_WITHIN_H):
        return "service"
    if max(vib_z, temp_z, current_z) >= 2 or "rising" in (vib_trend, temp_trend):
        return "watch"
    return "ok"


QUESTIONS = [
    Question("health", "Machine health", Answer.choice(["ok", "watch", "service", "stop"]),
             requires=["vibration_below_trip", "temperature_below_trip"]),
    Question("fault", "Likely fault", Answer.choice(["none", "bearing", "imbalance", "electrical", "cooling"]), uses=FAULT_FACTS),
]
